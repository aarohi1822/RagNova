from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.documents import IngestionResponse
from app.services.ingestion import IngestionService

router = APIRouter()
service = IngestionService()


@router.post("/upload", response_model=IngestionResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> IngestionResponse:
    try:
        return await service.ingest(files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

