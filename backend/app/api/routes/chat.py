from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.qa import QAService

router = APIRouter()
service = QAService()


@router.post("/ask", response_model=ChatResponse)
async def ask_question(payload: ChatRequest) -> ChatResponse:
    try:
        return await service.answer(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

