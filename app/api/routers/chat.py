from fastapi import APIRouter, Request, Depends, HTTPException
from langchain_core.messages import HumanMessage
from app.utils import logger
from app.models.chat_models import ChatRequest, ChatResponse, ApprovalRequest, ApprovalResponse
from app.models.auth_models import User
from app.core.auth import get_current_active_user
from langchain_core.messages import ToolMessage

def _log_event(event: dict, _log: set):
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _log:
            msg_repr = message.pretty_repr(html=True)
            logger.info(msg_repr)
            _log.add(message.id)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    app_request: Request = None,
):
    graph = app_request.app.state.graph

    # ===== THREAD / MEMORY CONFIG =====
    thread_id = str(current_user.user_id)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": current_user.user_id,
        }
    }

    logger.info("=" * 80)
    logger.info(f"[NEW EVENT START]")
    logger.info(f"[THREAD_ID] {thread_id}")
    logger.info("[USER_ID]" + current_user.user_id)
    logger.info(f"[USER INPUT] {request.message}")

    requires_approval = False
    approval_data = None

    _log = set()
    try:
        # ===== STREAM VÀ LƯU EVENT CUỐI =====
        for event in graph.stream(
            {"messages": [("user", request.message)]},
            config=config,
            stream_mode="values",
        ):
            _log_event(event, _log)
            last_event = event  # ✅ Cập nhật event cuối
        
        # ===== CHECK INTERRUPT =====
        snapshot = graph.get_state(config)
        
        if snapshot.next:
            # ⏸️ CẦN APPROVAL
            logger.info("⏸️ Need Approval")
            requires_approval = True
            
            messages = snapshot.values.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tc = msg.tool_calls[0]
                    approval_data = {
                        "tool_call_id": tc["id"],
                        "action": tc["name"],
                        "details": tc.get("args", {}),
                    }
                    response_text = (
                        f"Tôi muốn thực hiện hành động **{tc['name']}**. "
                        "Bạn có đồng ý không?"
                    )
                    break
        
        else:
            # ✅ GRAPH HOÀN THÀNH
            logger.info("✅ Graph completed successfully")
            
            # Lấy response từ snapshot
            messages = snapshot.values.get("messages", [])
            for msg in reversed(messages):
                # Chỉ lấy AI message có content và không có tool_calls
                if hasattr(msg, "content") and msg.content:
                    # Skip nếu message chỉ có tool_calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        continue
                    response_text = msg.content
                    break
            
            # Fallback nếu không tìm thấy
            if not response_text:
                response_text = "Tôi đã xử lý xong yêu cầu của bạn."
        
        logger.info(f"[FINAL RESPONSE] {response_text[:200]}...")
        logger.info("=" * 80)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "response": response_text,
        "requires_approval": requires_approval,
        "approval_data": approval_data,
    }

@router.post("/approval")
async def handle_approval(
    request: ApprovalRequest,
    current_user: User = Depends(get_current_active_user),
    app_request: Request = None,
):
    graph = app_request.app.state.graph

    thread_id = str(current_user.user_id)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": current_user.user_id,
        }
    }

    if request.feedback:
        logger.info(f"[FEEDBACK] {request.feedback}")
    
    response_text = ""
    completed = False
    requires_approval = False
    approval_data = None

    _log = set()

    try:
        # ===== GET CURRENT STATE =====
        snapshot = graph.get_state(config)
        
        if not snapshot or not snapshot.next:
            raise HTTPException(
                status_code=400,
                detail="Không có action nào đang chờ approval"
            )

        if request.approved:
            # Customer approval
            logger.info("Customer approval!!!")

            for event in graph.stream(
                None,
                config=config,
                stream_mode="values"
            ):
                _log_event(event, _log)
            # Kiểm tra state sau khi resume
            updated_snapshot = graph.get_state(config)

            if updated_snapshot.next:
                logger.info("⏸️ Another action needs approval")
                requires_approval = True
                completed = False
                
                messages = updated_snapshot.values.get("messages", [])
                for msg in reversed(messages):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tc = msg.tool_calls[0]
                        approval_data = {
                            "tool_call_id": tc["id"],
                            "action": tc["name"],
                            "details": tc.get("args", {}),
                        }
                        response_text = (
                            f"Action trước đã hoàn thành. "
                            f"Tôi muốn thực hiện thêm: **{tc['name']}**. "
                            "Bạn có đồng ý không?"
                        )
                        break
            else:
                # TẤT CẢ ACTIONS ĐÃ HOÀN THÀNH
                logger.info("✅ All actions completed")
                completed = True
                
                # Lấy response cuối cùng
                messages = updated_snapshot.values.get("messages", [])
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content:
                        # Skip tool_calls message
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            continue
                        response_text = msg.content
                        break
                
                if not response_text:
                    response_text = "✅ Action đã được thực hiện thành công!"

        else:
            # Customer Rejected
            logger.info("Customer REJECTED!!!")
            
            # Tìm tool call ID để reject
            messages = snapshot.values.get("messages", [])
            last_ai_message = None
            
            for msg in reversed(messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    last_ai_message = msg
                    break
            
            if not last_ai_message or not last_ai_message.tool_calls:
                raise HTTPException(
                    status_code=400,
                    detail="Không tìm thấy action để reject"
                )
            
            tool_call_id = last_ai_message.tool_calls[0]["id"]
            feedback_message = request.feedback or "Khách hàng đã từ chối thực hiện action này."
            
            logger.info(f"Sending rejection with tool_call_id: {tool_call_id}")
            
            # Gửi ToolMessage với rejection
            for event in graph.stream(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=(
                                f"Action bị từ chối bởi khách hàng.\n"
                                f"Lý do: {feedback_message}\n"
                                f"Vui lòng tiếp tục hỗ trợ khách hàng theo cách khác."
                            )
                        )
                    ]
                },
                config=config,
                stream_mode="values",
            ):
                _log_event(event, _log)
            
            # Lấy response sau khi reject
            final_snapshot = graph.get_state(config)
            messages = final_snapshot.values.get("messages", [])
            
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    # Skip tool_calls message
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        continue
                    response_text = msg.content
                    break
            
            if not response_text:
                response_text = "Tôi hiểu rồi. Tôi có thể giúp bạn bằng cách nào khác?"
            
            completed = True

        logger.info(f"[APPROVAL END]")
        logger.info(f"[COMPLETED] {completed}")
        logger.info(f"[REQUIRES_APPROVAL] {requires_approval}")
        logger.info(f"[RESPONSE] {response_text[:200]}...")
        logger.info("=" * 80)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("APPROVAL ERROR")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý approval: {str(e)}")

    return ApprovalResponse(
        response=response_text,
        completed=completed,
        requires_approval=requires_approval,
        approval_data=approval_data,
    )


