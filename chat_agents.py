from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel
from config import API_KEY, MODEL
from tools import transcribe_audio,translate_text, optimize_text, generate_meeting_notes

# --- Chat Agent for follow-up questions ---
chat_agent = Agent(
    name="MeetingChatAgent",
    instructions=(
        "You are an AI assistant that helps users understand their meeting transcripts. "
        "You have access to the transcribe_audio and summary in the context. "
        "Answer questions about the meeting, provide specific details, and help extract insights. "
        "Be conversational and helpful."
    ),
    model=LitellmModel(model=MODEL, api_key=API_KEY),
    tools=[transcribe_audio,translate_text, optimize_text, generate_meeting_notes]  # No additional tools needed for chat
)