from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    requires_approval: bool = False
    approval_data: dict | None = None

class ApprovalRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = None

class ApprovalResponse(BaseModel):
    response: str = Field(..., description="Response từ AI sau khi xử lý approval")
    completed: bool = Field(..., description="True nếu tất cả actions đã hoàn thành")
    requires_approval: Optional[bool] = Field(False, description="True nếu còn action khác cần approval")
    approval_data: Optional[Dict[str, Any]] = Field(None, description="Thông tin action tiếp theo (nếu có)")
