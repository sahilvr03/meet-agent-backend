from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI

client = AsyncIOMotorClient(MONGO_URI)
db = client.meto
meetings = db.meetings
users = db.users
conversations = db.conversations  # 👈 Add this