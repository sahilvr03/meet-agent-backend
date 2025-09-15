from fastapi import APIRouter, Body, HTTPException, Depends
from datetime import datetime, timezone
from db import users

router = APIRouter()

@router.post("/save-user")
async def save_user(user: dict = Body(...)):
    user_id = user.get("uid")
    if not user_id:
        raise HTTPException(400, "Missing uid")
    existing_user = await users.find_one({"uid": user_id})
    if existing_user:
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
        user["created_at"] = datetime.now(timezone.utc)
        user["last_login"] = datetime.now(timezone.utc)
        await users.insert_one(user)
        return {"message": "User created", "uid": user_id}