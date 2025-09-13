from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from config import API_KEY, MODEL
from tools import transcribe_audio, translate_text, optimize_text, generate_meeting_notes


# --- Step 1: Base Transcription Agent ---
piaic_agent = Agent(
    name="TranscriptionAgent",
    instructions=(
        "Your task is to accurately transcribe the uploaded audio or video file provided in context['file_path']. "
        "Use the `transcribe_audio` function with the specified language from context['language'] to generate a high-quality transcript. "
        "If the transcript is not in English, use the `translate_text` function to translate it to English. "
        "Finally, optimize the transcript using `optimize_text` to remove filler words, repetitions, and improve clarity. "
        "Ensure the transcript is precise and captures all spoken content, including nuanced details."
    ),
    model=LitellmModel(model=MODEL, api_key=API_KEY),
    tools=[transcribe_audio, translate_text, optimize_text]
)
# meeting_agents.py (add this new agent for live points generation)

# --- Step 2: Translation Agent ---
translation_agent = piaic_agent.clone(
    name="TranslationAgent",
    instructions=(
        "Translate the provided transcript to English if it is in another language, using the `translate_text` function. "
        "Ensure the translation preserves the meaning and context of the original text, handling idiomatic expressions accurately."
    ),
    tools=[translate_text]
)

# --- Step 3: Summarizer Agent ---
summarizer_agent = piaic_agent.clone(
    name="SummarizerAgent",
    instructions=(
        "Generate detailed and structured meeting notes from the optimized transcript using the `generate_meeting_notes` function. "
        "Produce a comprehensive summary (200-300 words) that captures the meeting's purpose, key discussions, participant roles (if identifiable), "
        "and major outcomes. Include detailed key points, decisions, and action items with clear ownership and deadlines. "
        "Analyze the sentiment of the meeting to reflect the tone and mood accurately."
    ),
    tools=[generate_meeting_notes]
)

# --- Step 4: Master Orchestrator Agent ---
master_agent = Agent(
    name="TalkToTextAgent",
    instructions=(
        "You are an AI assistant that processes meeting recordings step-by-step:\n"
        "1. Transcribe the audio using the TranscriptionAgent with the language specified in context['language'].\n"
        "2. Translate the transcript to English using the TranslationAgent if necessary.\n"
        "3. Summarize and extract detailed action points and decisions using the SummarizerAgent.\n"
        "Ensure each step is executed with high accuracy and the final output is comprehensive and well-structured."
    ),
    model=LitellmModel(model=MODEL, api_key=API_KEY),
    tools=[transcribe_audio, translate_text, optimize_text, generate_meeting_notes],
    handoffs=[translation_agent, summarizer_agent]
)

live_points_agent = summarizer_agent.clone(
    name="LivePointsAgent",
    instructions=(
        "You are an AI assistant for live meetings. Use the accumulated transcript to generate real-time key points, "
        "decisions, and action items as the meeting progresses. Update incrementally without repeating previous points. "
        "Keep summaries concise for live updates."
    ),
    tools=[generate_meeting_notes]
)
# --- Step 5: Run Everything ---
def run_pipeline(file_path: str, language: str = "en"):
    context = {"file_path": file_path, "language": language}
    result = Runner.run_sync(master_agent, input=context)
    return result.final_output