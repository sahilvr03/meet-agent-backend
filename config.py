import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Prefer GEMINI key, fallback to OpenAI key
API_KEY = os.getenv("GEMINI_API_KEY") 

TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "whisper")  # whisper or google
MODEL = os.getenv("MODEL", "gemini-2.0-flash")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
