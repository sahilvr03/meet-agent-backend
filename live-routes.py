from fastapi import APIRouter, WebSocket, Query, HTTPException
import uuid
from datetime import datetime, timezone
from db import meetings
from firebase_admin import auth
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/live-transcribe")
async def live_transcribe(
    websocket: WebSocket,
    language: str = Query("en"),
    token: str = Query(None),
    user_id: str = Query(None)
):
    try:
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        if decoded_token["uid"] != user_id:
            logger.error(f"User ID mismatch: token UID {decoded_token['uid']}, provided user_id {user_id}")
            await websocket.close(code=1008, reason="User ID mismatch")
            return

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
        logger.info(f"Starting live transcription for run_id: {run_id}, user_id: {user_id}")
        await live_manager.handle_live_session(websocket, run_id, language, user_id)
    except Exception as e:
        logger.error(f"Authentication or initialization failed: {str(e)}")
        await websocket.close(code=1008, reason=f"Authentication failed: {str(e)}")