
import os
import shutil
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from config import UPLOAD_DIR
from meeting_runner import process_meeting, parse_meeting_output
from db import meetings, conversations, users
from chat_agents import chat_agent
from meeting_agents import master_agent, live_points_agent
from tools import live_transcribe_chunk, transcribe_audio, translate_text, optimize_text, generate_meeting_notes
from agents import Runner
import uuid
import json
import io
from pydub import AudioSegment
from docx import Document
import csv
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
# Replace with the actual path to your Firebase Admin SDK JSON file, e.g., "path/to/serviceAccountKey.json"
# Alternatively, set the path in an environment variable for security
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "oops.json")
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    except FileNotFoundError:
        raise Exception(f"Firebase credentials file not found at {FIREBASE_CREDENTIALS_PATH}. Please provide the correct path.")

# Initialize FastAPI app
app = FastAPI(title="TalkToText Pro API")

# Allow requests from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dependency to get authenticated user ID from Firebase token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Live Session Manager for WebSocket
class LiveSessionManager:
    def __init__(self):
        self.sessions = {}  # run_id -> {'websocket': WebSocket, 'transcript': '', 'accumulated_audio': b'', 'language': str, 'user_id': str}

    async def handle_live_session(self, websocket: WebSocket, run_id: str, language: str, user_id: str):
        self.sessions[run_id] = {
            'websocket': websocket,
            'transcript': '',
            'accumulated_audio': b'',
            'language': language,
            'user_id': user_id
        }
        await websocket.accept()
        
        try:
            while True:
                data = await websocket.receive_bytes()
                session = self.sessions[run_id]
                session['accumulated_audio'] += data
                
                if len(session['accumulated_audio']) > 10240:
                    chunk_transcript = await live_transcribe_chunk(session['accumulated_audio'], session['language'])
                    optimized = await optimize_text(chunk_transcript)
                    if session['language'] != 'en':
                        optimized = await translate_text(optimized, 'en')
                    
                    session['transcript'] += optimized + ' '
                    
                    runner = Runner()
                    context = {"transcript": session['transcript']}
                    result = await runner.run(
                        live_points_agent,
                        input="Generate updated key points and action items from the live transcript.",
                        context=context
                    )
                    points = parse_meeting_output(result.final_output)
                    
                    await meetings.update_one(
                        {"run_id": run_id},
                        {"$set": {"partial_transcript": session['transcript'], "partial_points": points}}
                    )
                    
                    await websocket.send_json({
                        "run_id": run_id,
                        "transcript": optimized,
                        "key_points": points.get("key_points", []),
                        "action_items": points.get("action_items", []),
                        "sentiment": points.get("sentiment", "neutral")
                    })
                    
                    session['accumulated_audio'] = b''
        except WebSocketDisconnect:
            if run_id in self.sessions:
                transcript = self.sessions[run_id]['transcript']
                del self.sessions[run_id]
                await meetings.update_one(
                    {"run_id": run_id},
                    {"$set": {"status": "done", "result": parse_meeting_output(transcript)}}
                )

live_manager = LiveSessionManager()

@app.websocket("/live-transcribe")
async def live_transcribe(websocket: WebSocket, language: str = Query("en"), user_id: str = Depends(get_current_user)):
    run_id = str(uuid.uuid4())
    await meetings.insert_one({
        "run_id": run_id,
        "user_id": user_id,
        "status": "live",
        "file_path": None,
        "language": language,
        "filename": f"Live Meeting {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "created_at": datetime.now(timezone.utc)
    })
    await live_manager.handle_live_session(websocket, run_id, language, user_id)

@app.post("/upload-audio")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    language: str = Query("en", description="Language of the meeting audio")
):
    valid_extensions = (".mp3", ".wav", ".m4a", ".mp4")
    if file.content_type.split("/")[0] not in ["audio", "video"] and not file.filename.endswith(valid_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload audio/video only.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    unique_filename = f"{user_id}_{timestamp}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, unique_filename)
    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    run_id, _ = await process_meeting(saved_path, user_id, language)
    background_tasks.add_task(process_meeting, saved_path, user_id, language)
    return {
        "status": "processing",
        "run_id": run_id,
        "message": "File uploaded successfully and processing started.",
        "file_info": {
            "filename": file.filename,
            "saved_as": unique_filename,
            "content_type": file.content_type,
            "path": saved_path,
            "language": language
        }
    }

@app.post("/chat/{run_id}")
async def chat_with_meeting(
    run_id: str,
    request: dict,
    user_id: str = Depends(get_current_user)
):
    message = request.get("message", "")
    if not message:
        raise HTTPException(400, "Message is required")
   
    meeting = await meetings.find_one({"run_id": run_id, "user_id": user_id})
    if not meeting:
        raise HTTPException(404, "Meeting not found")
   
    if meeting.get("status") != "done":
        raise HTTPException(400, "Meeting processing not complete yet")
   
    context = {
        "meeting_transcript": meeting.get("result", {}).get("final_output", ""),
        "meeting_summary": meeting.get("result", {}).get("summary", ""),
        "meeting_key_points": meeting.get("result", {}).get("key_points", []),
        "meeting_decisions": meeting.get("result", {}).get("decisions", []),
        "meeting_action_items": meeting.get("result", {}).get("action_items", []),
        "original_file": meeting.get("file_path", ""),
        "language": meeting.get("language", "en")
    }
   
    runner = Runner()
    result = await runner.run(
        chat_agent,
        input=message,
        context=context
    )
   
    chat_id = str(uuid.uuid4())
    await conversations.insert_one({
        "chat_id": chat_id,
        "run_id": run_id,
        "user_id": user_id,
        "message": message,
        "response": result.final_output,
        "timestamp": datetime.now(timezone.utc)
    })
   
    return {
        "chat_id": chat_id,
        "response": result.final_output,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/conversations/{run_id}")
async def get_conversations(run_id: str, user_id: str = Depends(get_current_user)):
    cursor = conversations.find({"run_id": run_id, "user_id": user_id}).sort("timestamp", 1)
    chats = await cursor.to_list(length=50)
   
    for chat in chats:
        chat["_id"] = str(chat["_id"])
        if chat.get("timestamp"):
            chat["timestamp"] = chat["timestamp"].isoformat()
   
    return {
        "run_id": run_id,
        "conversations": chats
    }

@app.delete("/conversations/{run_id}/{chat_id}")
async def delete_conversation(run_id: str, chat_id: str, user_id: str = Depends(get_current_user)):
    result = await conversations.delete_one({"chat_id": chat_id, "run_id": run_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Conversation not found")
    return {"message": "Conversation deleted successfully"}

@app.delete("/conversations/{run_id}")
async def delete_all_conversations(run_id: str, user_id: str = Depends(get_current_user)):
    result = await conversations.delete_many({"run_id": run_id, "user_id": user_id})
    return {"message": f"Deleted {result.deleted_count} conversations"}

@app.put("/conversations/{run_id}/{chat_id}")
async def update_conversation(run_id: str, chat_id: str, request: dict, user_id: str = Depends(get_current_user)):
    message = request.get("message")
    response = request.get("response")
    if not message and not response:
        raise HTTPException(400, "At least one of 'message' or 'response' must be provided")
    
    update_data = {}
    if message:
        update_data["message"] = message
    if response:
        update_data["response"] = response
    update_data["updated_at"] = datetime.now(timezone.utc)
    
    result = await conversations.update_one(
        {"chat_id": chat_id, "run_id": run_id, "user_id": user_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Conversation not found")
    return {"message": "Conversation updated successfully"}

@app.get("/conversations/{run_id}/download/word")
async def download_conversations_word(run_id: str, user_id: str = Depends(get_current_user)):
    meeting = await meetings.find_one({"run_id": run_id, "user_id": user_id})
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    
    cursor = conversations.find({"run_id": run_id, "user_id": user_id}).sort("timestamp", 1)
    chats = await cursor.to_list(length=50)
    
    doc = Document()
    doc.add_heading(f"Meeting: {meeting.get('filename', 'Untitled Meeting')}", 0)
    doc.add_heading("Meeting Details", level=1)
    doc.add_paragraph(f"Date: {meeting.get('created_at', '').isoformat()}")
    doc.add_paragraph(f"Language: {meeting.get('language', 'en')}")
    doc.add_paragraph(f"Status: {meeting.get('status', 'unknown')}")
    
    if meeting.get("result"):
        doc.add_heading("Transcript", level=2)
        doc.add_paragraph(meeting["result"].get("final_output", ""))
        doc.add_heading("Summary", level=2)
        doc.add_paragraph(meeting["result"].get("summary", ""))
        doc.add_heading("Key Points", level=2)
        for point in meeting["result"].get("key_points", []):
            doc.add_paragraph(f"- {point}")
        doc.add_heading("Action Items", level=2)
        for item in meeting["result"].get("action_items", []):
            doc.add_paragraph(f"- {item}")
    
    if chats:
        doc.add_heading("Conversations", level=1)
        for chat in chats:
            doc.add_heading(f"Chat ID: {chat['chat_id']}", level=2)
            doc.add_paragraph(f"Timestamp: {chat['timestamp'].isoformat()}")
            doc.add_paragraph(f"User Message: {chat['message']}")
            doc.add_paragraph(f"AI Response: {chat['response']}")
            doc.add_paragraph("-" * 50)
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    filename = f"meeting_{run_id}.docx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/conversations/{run_id}/download/csv")
async def download_conversations_csv(run_id: str, user_id: str = Depends(get_current_user)):
    meeting = await meetings.find_one({"run_id": run_id, "user_id": user_id})
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    
    cursor = conversations.find({"run_id": run_id, "user_id": user_id}).sort("timestamp", 1)
    chats = await cursor.to_list(length=50)
    
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Meeting ID", "Meeting Name", "Date", "Language", "Status", "Type", "Content"])
    
    # Meeting details
    writer.writerow([
        meeting["run_id"],
        meeting.get("filename", "Untitled Meeting"),
        meeting.get("created_at", "").isoformat(),
        meeting.get("language", "en"),
        meeting.get("status", "unknown"),
        "Transcript",
        meeting.get("result", {}).get("final_output", "")
    ])
    writer.writerow(["", "", "", "", "", "Summary", meeting.get("result", {}).get("summary", "")])
    for point in meeting.get("result", {}).get("key_points", []):
        writer.writerow(["", "", "", "", "", "Key Point", point])
    for item in meeting.get("result", {}).get("action_items", []):
        writer.writerow(["", "", "", "", "", "Action Item", item])
    
    # Conversations
    for chat in chats:
        writer.writerow([
            meeting["run_id"],
            meeting.get("filename", "Untitled Meeting"),
            chat["timestamp"].isoformat(),
            "",
            "",
            "User Message",
            chat["message"]
        ])
        writer.writerow([
            meeting["run_id"],
            meeting.get("filename", "Untitled Meeting"),
            chat["timestamp"].isoformat(),
            "",
            "",
            "AI Response",
            chat["response"]
        ])
    
    buffer.seek(0)
    filename = f"meeting_{run_id}.csv"
    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.delete("/meetings/{run_id}")
async def delete_meeting(run_id: str, user_id: str = Depends(get_current_user)):
    """
    Delete a specific meeting and its associated conversations
    """
    meeting = await meetings.find_one({"run_id": run_id, "user_id": user_id})
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    
    # Delete associated file if it exists
    if meeting.get("file_path"):
        try:
            os.remove(meeting["file_path"])
        except OSError:
            pass  # Ignore if file doesn't exist
    
    # Delete meeting and conversations
    await meetings.delete_one({"run_id": run_id, "user_id": user_id})
    await conversations.delete_many({"run_id": run_id, "user_id": user_id})
    return {"message": "Meeting and associated conversations deleted successfully"}

@app.put("/meetings/{run_id}")
async def rename_meeting(run_id: str, request: dict, user_id: str = Depends(get_current_user)):
    """
    Rename a meeting's display name
    """
    filename = request.get("filename")
    if not filename:
        raise HTTPException(400, "Filename is required")
    
    result = await meetings.update_one(
        {"run_id": run_id, "user_id": user_id},
        {"$set": {"filename": filename}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Meeting not found")
    return {"message": "Meeting renamed successfully"}

@app.get("/meetings/{user_id}")
async def get_all_meetings(user_id: str = Depends(get_current_user)):
    cursor = meetings.find({"user_id": user_id}).sort("created_at", -1)
    meetings_list = await cursor.to_list(length=100)
   
    simplified_meetings = []
    for meeting in meetings_list:
        simplified_meetings.append({
            "run_id": meeting.get("run_id"),
            "filename": meeting.get("filename", meeting.get("file_path", "").split("/")[-1] if meeting.get("file_path") else "Live Meeting"),
            "created_at": meeting.get("created_at").isoformat() if meeting.get("created_at") else "",
            "status": meeting.get("status"),
            "has_result": meeting.get("status") == "done",
            "language": meeting.get("language", "en")
        })
   
    return {
        "user_id": user_id,
        "meetings": simplified_meetings
    }

@app.get("/status/{run_id}")
async def get_status(run_id: str, user_id: str = Depends(get_current_user)):
    doc = await meetings.find_one({"run_id": run_id, "user_id": user_id})
    if not doc:
        raise HTTPException(404, "Run ID not found")
    return {
        "run_id": run_id,
        "status": doc.get("status", "unknown"),
        "result": doc.get("result"),
        "language": doc.get("language", "en")
    }

@app.get("/history/{user_id}")
async def history(user_id: str = Depends(get_current_user)):
    cursor = meetings.find({"user_id": user_id}).sort("created_at", -1).limit(20)
    docs = await cursor.to_list(length=20)
   
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        if doc.get("created_at"):
            doc["created_at"] = doc["created_at"].isoformat()
   
    return {
        "user_id": user_id,
        "history": docs
    }

@app.get("/")
async def root():
    return {"message": "Server is running successfully!"}

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "TalkToText API is live 🚀"}

@app.get("/test-gemini")
async def test_gemini():
    from meeting_agents import master_agent
    from agents import Runner
    runner = Runner()
    test_result = await runner.run(
        master_agent,
        input="Hello Gemini! Summarize this meeting: 'We discussed project timelines and assigned tasks.'"
    )
    return {"output": test_result.final_output}

@app.post("/auth/save-user")
async def save_user(user: dict = Body(...)):
    """
    Save or update user info in backend DB after Firebase login.
    """
    user_id = user.get("uid")
    if not user_id:
        raise HTTPException(400, "Missing uid")

    existing_user = await users.find_one({"uid": user_id})
    if existing_user:
        # Update login timestamp
        await users.update_one(
            {"uid": user_id},
            {"$set": {
                "email": user.get("email"),
                "displayName": user.get("displayName"),
                "photoURL": user.get("photoURL"),
                "provider": user.get("provider", "email"),
                "last_login": datetime.now(timezone.utc)
            }}
        )
        return {"message": "User updated", "uid": user_id}
    else:
        # Insert new user
        user["created_at"] = datetime.now(timezone.utc)
        user["last_login"] = datetime.now(timezone.utc)
        await users.insert_one(user)
        return {"message": "User created", "uid": user_id}