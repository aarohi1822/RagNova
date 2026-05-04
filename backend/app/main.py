from fastapi import FastAPI

from app.api.routes import chat, documents, health
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Enterprise-style hybrid RAG QA platform for portfolio-grade GenAI engineering.",
)

app.include_router(health.router, tags=["health"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}

