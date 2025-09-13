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

# Load Whisper model (use 'medium' for better accuracy)
try:
    WHISPER_MODEL = whisper.load_model("base")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    WHISPER_MODEL = None

@function_tool
async def transcribe_audio(file_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio to text using Whisper model with specified language."""
    if not WHISPER_MODEL:
        return "Error: Whisper model not loaded."
   
    try:
        # Validate file existence
        if not Path(file_path).is_file():
            logger.error(f"File not found: {file_path}")
            return "Error: Audio file not found."
        
        # Validate language
        transcription_language = language if language in WHISPER_LANGUAGES else "en"
        if language and language not in WHISPER_LANGUAGES:
            logger.warning(f"Unsupported language {language}, defaulting to English")
        
        logger.info(f"Transcribing file: {file_path} with language: {transcription_language}")
        
        # Transcribe with Whisper
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: WHISPER_MODEL.transcribe(file_path, language=transcription_language, fp16=False)
        )
       
        # Extract transcript
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

@function_tool
async def translate_text(text: str, target: str = "en") -> str:
    """Translate text to target language using Google Translate API."""
    try:
        if not text:
            return "Error: No text provided for translation."
       
        # Attempt language detection with fallback
        try:
            source_lang = langdetect.detect(text)[:2]
        except langdetect.lang_detect_exception.LangDetectException:
            source_lang = "auto"
            logger.warning("Language detection failed, using auto source")
        
        if target.lower().startswith("en") and source_lang == "en":
            return text
       
        # Split text into chunks to handle API limits (max 5000 chars per request)
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

@function_tool
async def optimize_text(text: str) -> str:
    """Clean and optimize text by removing fillers, repetitions, and chunking."""
    try:
        if not text:
            return "Error: No text provided for optimization."
        
        # Define filler words for supported languages
        fillers = {
            "en": ["um", "uh", "youknow", "like", "basically", "so", "actually", "literally"],
            "es": ["eh", "este", "sabes", "como", "pues", "bueno"],
            "fr": ["euh", "ben", "tu sais", "comme", "voilà", "alors"],
            "ur": ["umm", "aah", "jaise", "matlab", "bas"],
            "ro": ["ă", "î", "ști", "cum", "deci", "adică"],
            "de": ["äh", "nun", "weißt du", "also", "eigentlich"],
            "it": ["ehm", "tipo", "sai", "allora", "cioè"],
            "pt": ["hum", "tipo", "sabe", "então", "bem"],
            "ru": ["э", "ну", "знаешь", "вот", "так сказать"],
            "zh": ["嗯", "那个", "你知道", "就是", "然后"],
            "ja": ["あの", "えっと", "ね", "じゃ", "みたいな"],
            "ko": ["음", "그", "알다시피", "그러니까", "저기"],
            "ar": ["أم", "يعني", "حسنا", "طيب"],
            "hi": ["उम", "जैसे", "आप जानते हैं", "मतलब", "तो"]
        }
       
        # Detect language
        try:
            lang = langdetect.detect(text)[:2]
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "en"
            logger.warning("Language detection failed, defaulting to English")
        
        filler_set = fillers.get(lang, fillers["en"])
        # Clean text
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
       
        # Truncate to avoid token limits (20k chars as guardrail for longer summaries)
        max_length = 20000
        if len(cleaned) > max_length:
            logger.warning(f"Text truncated from {len(cleaned)} to {max_length} characters")
            cleaned = cleaned[:max_length]
       
        return cleaned.strip()
    except Exception as e:
        logger.error(f"Text optimization error: {e}")
        return text

# tools.py (add this new tool for chunked live transcription)
@function_tool
async def live_transcribe_chunk(audio_data: bytes, language: Optional[str] = "en") -> str:
    """Transcribe a chunk of live audio data."""
    if not WHISPER_MODEL:
        return "Error: Whisper model not loaded."
    
    try:
        # Save chunk to temporary file for Whisper processing
        temp_path = os.path.join(UPLOAD_DIR, f"live_chunk_{uuid.uuid4()}.wav")
        audio = AudioSegment.from_file(io.BytesIO(audio_data))  # Use pydub to handle bytes
        audio.export(temp_path, format="wav")
        
        transcription_language = language if language in WHISPER_LANGUAGES else "en"
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: WHISPER_MODEL.transcribe(temp_path, language=transcription_language, fp16=False)
        )
        
        transcript_text = result.get("text", "").strip()
        
        # Clean up temp file
        os.remove(temp_path)
        
        return transcript_text
    except Exception as e:
        logger.error(f"Live transcription chunk error: {e}")
        return f"Error transcribing live chunk: {str(e)}"
    

@function_tool
async def generate_meeting_notes(text: str) -> str:
    """Generate detailed and structured meeting notes in JSON format."""
    try:
        if not text:
            return json.dumps({"error": "No text provided for meeting notes"})
        
        # Detailed prompt for comprehensive summarization
        prompt = (
            "You are an AI assistant tasked with generating detailed meeting notes from the provided transcript. "
            "Produce a comprehensive summary (200-300 words) that includes:\n"
            "- The meeting's purpose and context\n"
            "- Key discussion topics and their significance\n"
            "- Identified participant roles or key contributors (if detectable)\n"
            "- Major outcomes and their implications\n"
            "Also, extract:\n"
            "- Key points: At least 5-7 specific points discussed\n"
            "- Decisions: Clear decisions made during the meeting\n"
            "- Action items: Specific tasks with owners and deadlines\n"
            "- Sentiment: Overall tone (positive, neutral, negative) based on the discussion\n"
            f"Transcript (first 4000 characters):\n{text[:4000]}"
        )
        
        # Placeholder for AI model call (replace with actual async model call in production)
        # For now, using an enhanced mock response
        notes = {
            "summary": (
                "The meeting focused on strategic planning for the upcoming product launch, involving key stakeholders from the product development, marketing, and operations teams. The primary objective was to align on project milestones, resource allocation, and marketing strategies to ensure a successful Q4 launch. Discussions covered the project timeline, identifying critical deadlines for product testing and marketing campaigns. The team highlighted resource constraints in the development team, prompting a detailed review of staffing needs. Key contributors included the project manager, who outlined the timeline, and the marketing lead, who proposed a multi-channel campaign strategy. The meeting also addressed customer feedback from the beta phase, emphasizing the need for feature enhancements. Outcomes included a revised timeline with accelerated testing phases and approval for additional budget to hire contract developers. The team agreed to prioritize customer-facing features and delay non-critical updates to the next sprint. The overall sentiment was collaborative and solution-oriented, with a positive outlook on meeting the launch goals despite initial challenges."
            ),
            "key_points": [
                "Reviewed project timeline, setting key milestones for Q4 product launch.",
                "Identified resource bottlenecks in the development team.",
                "Discussed customer feedback from beta testing, focusing on feature priorities.",
                "Proposed a multi-channel marketing strategy for the launch.",
                "Evaluated budget needs, highlighting additional staffing requirements.",
                "Addressed risk mitigation strategies for delayed deliverables.",
                "Confirmed alignment on product feature prioritization."
            ],
            "decisions": [
                "Approved a 15% budget increase for Q4 to support hiring contract developers.",
                "Prioritized customer-facing features for the launch, postponing non-critical updates.",
                "Set a revised timeline with accelerated testing phases to meet launch deadlines."
            ],
            "action_items": [
                {"owner": "John (Project Manager)", "task": "Send updated project timeline to all stakeholders", "due": "2025-09-25"},
                {"owner": "Sarah (Marketing Lead)", "task": "Finalize multi-channel campaign plan", "due": "2025-09-22"},
                {"owner": "Alex (Operations Lead)", "task": "Recruit two contract developers", "due": "2025-09-30"},
                {"owner": "Team", "task": "Conduct follow-up meeting to review testing progress", "due": "2025-10-05"}
            ],
            "sentiment": "positive"
        }
        
        logger.info("Generated detailed meeting notes")
        return json.dumps(notes, indent=2)
    except Exception as e:
        logger.error(f"Meeting notes generation error: {e}")
        return json.dumps({"error": f"Failed to generate notes: {str(e)}"})