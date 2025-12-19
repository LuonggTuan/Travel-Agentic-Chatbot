from fastapi import APIRouter, Request, Depends, HTTPException
from langchain_core.messages import HumanMessage
from app.utils import logger
from app.models.chat_models import ChatRequest, ChatResponse
from app.models.auth_models import User
from app.core.auth import get_current_active_user

router = APIRouter(prefix="/chat", tags=["Chatbot"])

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    app_request: Request = None,
):
    graph = app_request.app.state.graph

    thread_id = str(current_user.user_id)
    logger.info(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": current_user.user_id
        }
    }

    result = graph.invoke(
        {"messages": [("user", request.message)]},
        config=config
    )

    messages = result.get("messages", [])

    response_text = "Tôi sẵn sàng hỗ trợ bạn."
    requires_approval = False
    approval_data = None

    for msg in reversed(messages):
        # interrupt / approval
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            requires_approval = True
            approval_data = {
                "action": msg.tool_calls[0]["name"],
                "details": msg.tool_calls[0].get("args", {})
            }

        if hasattr(msg, "content") and msg.content:
            response_text = msg.content
            break

    return {
        "response": response_text,
        "requires_approval": requires_approval,
        "approval_data": approval_data
    }
