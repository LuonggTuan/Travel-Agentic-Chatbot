from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    requires_approval: bool = False
    approval_data: dict | None = None