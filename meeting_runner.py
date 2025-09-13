import uuid
import datetime
from datetime import timezone
import json
import re
from agents import Runner
from db import meetings
from meeting_agents import master_agent

async def process_meeting(file_path: str, user_id: str, language: str = "en"):
    run_id = str(uuid.uuid4())
    # Insert initial state into MongoDB
    await meetings.insert_one({
        "run_id": run_id,
        "user_id": user_id,
        "status": "queued",
        "file_path": file_path,
        "language": language,
        "created_at": datetime.datetime.now(timezone.utc)
    })
    runner = Runner()
    context = {
        "file_path": file_path,
        "user_id": user_id,
        "language": language
    }
    try:
        # Run the agent pipeline
        result = await runner.run(
            master_agent,
            input=f"Please process the uploaded meeting recording located at {file_path}. "
                  f"Perform transcription with language set to {language}, translation if needed, and generate a structured summary.",
            context=context
        )
        # Extract and structure the result
        structured_result = parse_meeting_output(result.final_output)
        # Update meeting with results
        await meetings.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "done",
                    "result": structured_result,
                    "processed_at": datetime.datetime.now(timezone.utc),
                    "language": language
                }
            }
        )
        return run_id, structured_result
    except Exception as e:
        # Update with error status
        await meetings.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "error",
                    "error_message": str(e)
                }
            }
        )
        raise e

# parse_meeting_output in meeting_runner.py (updated for better parsing)
def parse_meeting_output(text: str) -> dict:
    result = {
        "final_output": text,
        "summary": "",
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "sentiment": "neutral"
    }
    
    # Improved summary extraction: everything before first section
    summary_match = re.search(r'^(.*?)(?=KEY POINTS:|DECISIONS:|ACTION ITEMS:|SENTIMENT:)', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()
    
    # Key points (unchanged)
    key_points_section = re.search(r'(?i)key points[:\s]*(.*?)(?=decisions|action items|sentiment|$)', text, re.DOTALL)
    if key_points_section:
        points = re.findall(r'(?:[-•*]|\d+\.)\s*(.*)', key_points_section.group(1))
        result["key_points"] = [p.strip() for p in points if p.strip()]
    
    # Decisions (unchanged)
    decisions_section = re.search(r'(?i)decisions[:\s]*(.*?)(?=action items|sentiment|$)', text, re.DOTALL)
    if decisions_section:
        decisions = re.findall(r'(?:[-•*]|\d+\.)\s*(.*)', decisions_section.group(1))
        result["decisions"] = [d.strip() for d in decisions if d.strip()]
    
    # Improved action items
    action_section = re.search(r'(?i)action items[:\s]*(.*?)(?=sentiment|$)', text, re.DOTALL)
    if action_section:
        action_text = action_section.group(1)
        action_items = []
        # New pattern for "Owner (Role): Task (Due: date)"
        structured_pattern = r'-\s*([^:]+):\s*([^\(]+)\s*\(Due:\s*([^\)]+)\)'
        matches = re.finditer(structured_pattern, action_text)
        for match in matches:
            owner = match.group(1).strip()
            task = match.group(2).strip()
            due = match.group(3).strip()
            action_items.append({"owner": owner, "task": task, "due": due})
        
        if not action_items:
            simple_actions = re.findall(r'(?:[-•*]|\d+\.)\s*(.*)', action_text)
            action_items = [{"owner": "Unassigned", "task": a.strip(), "due": "No due date"} for a in simple_actions if a.strip()]
        
        result["action_items"] = action_items
    
    # Sentiment (unchanged)
    sentiment_match = re.search(r'(?i)sentiment[:\s]*(\w+)', text)
    if sentiment_match:
        result["sentiment"] = sentiment_match.group(1).lower()
    
    return result
def extract_summary(text: str) -> str:
    """Extract summary from the result"""
    parsed = parse_meeting_output(text)
    return parsed["summary"] if parsed["summary"] else text[:500] + "..." if len(text) > 500 else text

def extract_action_items(text: str) -> list:
    """Extract action items from text"""
    parsed = parse_meeting_output(text)
    return parsed["action_items"]

def extract_decisions(text: str) -> list:
    """Extract decisions from text"""
    parsed = parse_meeting_output(text)
    return parsed["decisions"]