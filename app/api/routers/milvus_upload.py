from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.core.auth import get_current_active_user
from app.models.auth_models import User
from app.services.milvus_service import normalize_docx_to_chunks, upload_chunks_to_milvus
from app.utils import logger

router = APIRouter(prefix="/milvus", tags=["Milvus Upload"])

@router.post("/test_query_milvus")
async def query_milvus(
    query: str,
    collection_name: str,
    current_user: User = Depends(get_current_active_user)
):
    # Check role admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    try:
        from app.services.milvus_service import query_milvus
        results = query_milvus(collection_name=collection_name, query=query, top_k=3)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Milvus: {str(e)}")

@router.post("/upload-doc")
async def upload_document_milvus(
    file: UploadFile = File(...),
    collection_name: str = Form("collection_in_milvus"),
    current_user: User = Depends(get_current_active_user)
):
    # Check role admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    allowed_suffixes = {".docx", ".pdf"}
    # Check file type
    suffix = "." + file.filename.split(".")[-1].lower()
    if suffix in allowed_suffixes:
        try:
            file_bytes = await file.read()

            # Normalize and process the document
            chunks = normalize_docx_to_chunks(file_bytes, suffix)
            logger.info(f"Normalized document into {len(chunks)} chunks.")
            # Upload to Milvus
            result = upload_chunks_to_milvus(chunks, collection_name=collection_name, file_name=file.filename)
            logger.info(f"Upload result: {result}")
            return {"message": f"Uploaded {len(chunks)} chunks to Milvus with collection name {collection_name}."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only .docx are allowed." )
        
    
    