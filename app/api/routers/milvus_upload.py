from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.core.auth import get_current_active_user
from app.models.auth_models import User
from pymilvus import MilvusClient
import os

from dotenv import load_dotenv
load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")

router = APIRouter(prefix="/milvus", tags=["Milvus Upload"])

@router.post("/query")
async def query_milvus(
    query: str,
):
    return {"message": "Milvus query is not yet implemented."}

@router.post("/upload-doc")
async def upload_document_milvus(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    # Check role admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    return {"message": "Document upload to Milvus is not yet implemented."}