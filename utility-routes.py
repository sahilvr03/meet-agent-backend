from fastapi import APIRouter
from meeting_agents import master_agent
from agents import Runner

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Server is running successfully!"}

@router.get("/ping")
async def ping():
    return {"status": "ok", "message": "TalkToText API is live 🚀"}

@router.get("/test-gemini")
async def test_gemini():
    runner = Runner()
    test_result = await runner.run(
        master_agent,
        input="Hello Gemini! Summarize this meeting: 'We discussed project timelines and assigned tasks.'"
    )
    return {"output": test_result.final_output}