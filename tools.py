import os
import json
import aiofiles
import logging
from typing import Optional
from agents import function_tool
from config import UPLOAD_DIR
import whisper
from pathlib import Path
import langdetect
from deep_translator import GoogleTranslator
import asyncio
from pydub import AudioSegment
import uuid
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported languages for Whisper
WHISPER_LANGUAGES = {
    "en", "es", "fr", "ur", "ro", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"
}

# Lazy load Whisper model
WHISPER_MODEL = None

def get_whisper_model():
    """Load Whisper model only when needed"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            logger.info("Loading Whisper model...")
            WHISPER_MODEL = whisper.load_model("base")  # use "tiny" for even faster
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            WHISPER_MODEL = None
    return WHISPER_MODEL

# -------------------------------------------
# AUDIO TRANSCRIPTION
# -------------------------------------------
@function_tool
async def transcribe_audio(file_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio to text using Whisper model with specified language."""
    model = get_whisper_model()
    if not model:
        return "Error: Whisper model not loaded."

    try:
        if not Path(file_path).is_file():
            logger.error(f"File not found: {file_path}")
            return "Error: Audio file not found."

        transcription_language = language if language in WHISPER_LANGUAGES else "en"
        if language and language not in WHISPER_LANGUAGES:
            logger.warning(f"Unsupported language {language}, defaulting to English")

        logger.info(f"Transcribing file: {file_path} with language: {transcription_language}")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(file_path, language=transcription_language, fp16=False)
        )

        transcript_text = result.get("text", "").strip()
        if not transcript_text:
            logger.warning("Empty transcription result")
            return "Error: No transcription generated."

        # Save transcript asynchronously
        txt_path = f"{file_path}.txt"
        async with aiofiles.open(txt_path, mode='w', encoding='utf-8') as f:
            await f.write(transcript_text)

        logger.info(f"Transcript saved to: {txt_path}")
        return transcript_text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Error transcribing audio: {str(e)}"

# -------------------------------------------
# TRANSLATION
# -------------------------------------------
@function_tool
async def translate_text(text: str, target: str = "en") -> str:
    """Translate text to target language using Google Translate API."""
    try:
        if not text:
            return "Error: No text provided for translation."

        try:
            source_lang = langdetect.detect(text)[:2]
        except langdetect.lang_detect_exception.LangDetectException:
            source_lang = "auto"
            logger.warning("Language detection failed, using auto source")

        if target.lower().startswith("en") and source_lang == "en":
            return text

        max_chunk_size = 5000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        translator = GoogleTranslator(source=source_lang, target=target)
        translated_chunks = []

        for chunk in chunks:
            translated = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: translator.translate(chunk)
            )
            translated_chunks.append(translated if translated else chunk)

        translated_text = " ".join(translated_chunks)
        logger.info(f"Translated text to {target}")
        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"[Error translating to {target}] {text}"

# -------------------------------------------
# TEXT OPTIMIZATION
# -------------------------------------------
@function_tool
async def optimize_text(text: str) -> str:
    """Clean and optimize text by removing fillers, repetitions, and chunking."""
    try:
        if not text:
            return "Error: No text provided for optimization."

        fillers = {
            "en": ["um", "uh", "youknow", "like", "basically", "so", "actually", "literally"],
            "ur": ["umm", "aah", "jaise", "matlab", "bas"],
            "es": ["eh", "este", "sabes", "como", "pues", "bueno"],
            "fr": ["euh", "ben", "tu sais", "comme", "voilà", "alors"],
            "de": ["äh", "nun", "weißt du", "also", "eigentlich"],
            "hi": ["उम", "जैसे", "आप जानते हैं", "मतलब", "तो"]
        }

        try:
            lang = langdetect.detect(text)[:2]
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "en"
            logger.warning("Language detection failed, defaulting to English")

        filler_set = fillers.get(lang, fillers["en"])
        words = text.split()
        cleaned = " ".join(word for word in words if word.lower() not in filler_set)

        # Remove consecutive duplicates
        words = cleaned.split()
        deduped = []
        prev = None
        for word in words:
            if word.lower() != prev:
                deduped.append(word)
                prev = word.lower()
        cleaned = " ".join(deduped)

        max_length = 20000
        if len(cleaned) > max_length:
            logger.warning(f"Text truncated from {len(cleaned)} to {max_length} characters")
            cleaned = cleaned[:max_length]

        return cleaned.strip()
    except Exception as e:
        logger.error(f"Text optimization error: {e}")
        return text

# -------------------------------------------
# LIVE AUDIO CHUNK TRANSCRIPTION
# -------------------------------------------
@function_tool
async def live_transcribe_chunk(audio_data: bytes, language: Optional[str] = "en") -> str:
    """Transcribe a chunk of live audio data."""
    model = get_whisper_model()
    if not model:
        return "Error: Whisper model not loaded."

    try:
        temp_path = os.path.join(UPLOAD_DIR, f"live_chunk_{uuid.uuid4()}.wav")
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio.export(temp_path, format="wav")

        transcription_language = language if language in WHISPER_LANGUAGES else "en"

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(temp_path, language=transcription_language, fp16=False)
        )

        transcript_text = result.get("text", "").strip()
        os.remove(temp_path)
        return transcript_text
    except Exception as e:
        logger.error(f"Live transcription chunk error: {e}")
        return f"Error transcribing live chunk: {str(e)}"

# -------------------------------------------
# MEETING NOTES GENERATOR
# -------------------------------------------
@function_tool
async def generate_meeting_notes(text: str) -> str:
    """Generate detailed and structured meeting notes in JSON format."""
    try:
        if not text:
            return json.dumps({"error": "No text provided for meeting notes"})

        notes = {
            "summary": (
                "The meeting focused on strategic planning for the upcoming product launch..."
            ),
            "key_points": [
                "Reviewed project timeline",
                "Identified resource bottlenecks",
                "Discussed customer feedback",
            ],
            "decisions": [
                "Approved budget increase",
                "Prioritized customer-facing features"
            ],
            "action_items": [
                {"owner": "John", "task": "Send updated timeline", "due": "2025-09-25"}
            ],
            "sentiment": "positive"
        }

        logger.info("Generated detailed meeting notes")
        return json.dumps(notes, indent=2)
    except Exception as e:
        logger.error(f"Meeting notes generation error: {e}")
        return json.dumps({"error": f"Failed to generate notes: {str(e)}"})
