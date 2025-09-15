import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends
from datetime import datetime, timezone
from config import UPLOAD_DIR
from meeting_runner import process_meeting
from db import meetings

router = APIRouter()

@router.post("/upload-audio")
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

@router.get("/{user_id}")
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

@router.get("/status/{run_id}")
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

@router.delete("/{run_id}")
async def delete_meeting(run_id: str, user_id: str = Depends(get_current_user)):
    meeting = await meetings.find_one({"run_id": run_id, "user_id": user_id})
    if not meeting:
        raise HTTPException(404, "Meeting not found")
   
    if meeting.get("file_path"):
        try:
            os.remove(meeting["file_path"])
        except OSError:
            pass
   
    await meetings.delete_one({"run_id": run_id, "user_id": user_id})
    await conversations.delete_many({"run_id": run_id, "user_id": user_id})
    return {"message": "Meeting and associated conversations deleted successfully"}

@router.put("/{run_id}")
async def rename_meeting(run_id: str, request: dict, user_id: str = Depends(get_current_user)):
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