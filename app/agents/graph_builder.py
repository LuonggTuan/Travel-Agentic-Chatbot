from datetime import datetime
from typing import Literal, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from pydantic import BaseModel, Field
from app.llms.llm_models import get_openai_llm_model
from app.config import settings
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

from app.agents.flight_agent_tools import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket
)
from app.agents.hotel_agent_tools import (
    search_hotels,
    get_hotel_details,
    get_user_hotel_bookings,
    list_hotel_room_types,
    create_hotel_booking,
    cancel_hotel_booking
)
from app.agents.primary_tools import (
    lookup_policy,
    get_all_user_bookings
)

# STATE MANAGEMENT
def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "flight_agent",
                "hotel_agent",
            ]
        ],
        update_dialog_stack,
    ]

# COMPLETION AND DELEGATION MODELS
class CompleteOrEscalate(BaseModel):
    """Cong cu de danh dau nhiem vu hien tai da hoan thanh va/hoac chuyen quyen kiem soat cuoc hoi thoai cho tro ly chinh."""
    cancel: bool = True
    reason: str

class ToFlightBookingAssistant(BaseModel):
    """Chuyen cong viec cho tro ly chuyen biet xu ly cap nhat va huy chuyen bay."""
    request: str = Field(
        description="Bat ky cau hoi theo doi can thiet nao ma tro ly cap nhat chuyen bay nen lam ro truoc khi tien hanh."
    )

class ToHotelBookingAssistant(BaseModel):
    """Chuyen cong viec cho tro ly chuyen biet xu ly dat khach san."""
    location: str = Field(description="Dia diem ma nguoi dung muon dat khach san.")
    checkin_date: Optional[str] = Field(None, description="Ngay nhan phong khach san (neu nguoi dung cung cap).")
    checkout_date: Optional[str] = Field(None, description="Ngay tra phong khach san (neu nguoi dung cung cap).")
    request: str = Field(description="Bat ky thong tin hoac yeu cau bo sung nao tu nguoi dung.")

# PROMPT TEMPLATES
flight_booking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là trợ lý chuyên biệt xử lý việc gợi ý chuyến bay, cập nhật và hủy chuyến bay. "
        "Trợ lý chính sẽ ủy quyền công việc cho bạn bất cứ khi nào người dùng cần hỗ trợ bất cứ thông tin, yêu cầu liên quan đến chuyến bay. "
        
        "QUAN TRỌNG: Bạn chỉ có thể gới ý các chuyến bay hỗ trợ khách hàng chọn lịch bay, cập nhật hoặc hủy các đặt vé hiện có. Bạn KHÔNG THỂ đặt vé mới. "
        "Nếu khách hàng hỏi/yêu cầu đặt vé mới, hãy bảo họ truy cập: https://lat-airlines.com/book-flights "
        
        "Khi tìm kiếm, hãy kiên trì. Mở rộng phạm vi truy vấn nếu tìm kiếm đầu tiên không trả về kết quả. "
        "Xác nhận chi tiết chuyến bay đã cập nhật với khách hàng và thông báo về bất kỳ phí bổ sung nào. "
        "Nếu bạn cần thêm thông tin hoặc khách hàng thay đổi ý định, hãy chuyển nhiệm vụ trở lại trợ lý chính. "
        "Hãy nhớ rằng việc đặt vé không hoàn thành cho đến khi công cụ liên quan đã được sử dụng thành công."
        "\n\nThông tin chuyến bay hiện tại của người dùng:\n<Flights>\n{user_info}\n</Flights>"
        "\nThời gian hiện tại: {time}."
        "\n\nNếu người dùng cần hỗ trợ, và không có công cụ nào của bạn phù hợp cho việc đó, thì hãy "
        '"CompleteOrEscalate" cuộc hội thoại về trợ lý chính. Đừng lãng phí thời gian của người dùng.',
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

hotel_booking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là trợ lý chuyên biệt xử lý các yêu cầu LIÊN QUAN ĐẾN KHÁCH SẠN của LAT Airlines.\n\n"

        "KHẢ NĂNG CỦA BẠN:\n"
        "- Hiển thị thông tin khách sạn theo sân bay hoặc thành phố\n"
        "- Hiển thị các loại phòng của khách sạn\n"
        "- Tạo, xem và hủy đặt phòng khách sạn\n\n"

        "GIỚI HẠN HỆ THỐNG (RẤT QUAN TRỌNG):\n"
        "- Hệ thống KHÔNG theo dõi số phòng trống theo ngày\n"
        "- KHÔNG hiển thị số lượng phòng còn lại\n"
        "- KHÔNG suy đoán tình trạng thực tế của khách sạn\n\n"

        "HIỂN THỊ TÌNH TRẠNG ĐẶT PHÒNG:\n"
        "- Chỉ hiển thị ở mức logic: 'Có thể đặt' hoặc 'Không thể đặt'\n"
        "- Tuyệt đối KHÔNG hiển thị số lượng phòng hoặc dữ liệu nội bộ\n\n"

        "QUY TRÌNH ĐẶT PHÒNG BẮT BUỘC:\n"
        "1. Thu thập đầy đủ: địa điểm (hoặc khách sạn), ngày nhận phòng, ngày trả phòng\n"
        "2. Hiển thị danh sách khách sạn hoặc loại phòng phù hợp\n"
        "3. Chỉ gọi công cụ đặt phòng khi khách hàng xác nhận rõ ràng\n\n"

        "TÍNH GIÁ:\n"
        "- Giá phòng dựa trên base_price của loại phòng\n"
        "- Tổng tiền = base_price × số đêm lưu trú\n\n"

        "CHUYỂN QUYỀN:\n"
        "- Nếu người dùng chỉ hỏi thông tin tham khảo hoặc thay đổi ý định, "
        "hãy dùng CompleteOrEscalate để trả về trợ lý chính\n\n"

        "Thời gian hiện tại: {time}.\n\n"
        "Nếu không có công cụ nào phù hợp với yêu cầu của người dùng, "
        "hãy dùng CompleteOrEscalate để tránh lãng phí thời gian của khách hàng."
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

primary_assistant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là trợ lý hỗ trợ khách hàng hữu ích của LAT Airlines. "
        "Vai trò chính của bạn là tìm kiếm thông tin tổng thể về dịch vụ mà khách hàng đã đặt và chính sách công ty để trả lời các câu hỏi của khách hàng. "
        
        "QUAN TRỌNG VỀ CHUYẾN BAY: Chỉ khi khách hàng hỏi về đặt VÉ MÁY BAY mới, "
        "bạn mới đưa ra thông tin này:\n"
        "- Chatbot này chỉ có thể GỢI Ý chuyến bay đã lên lịch, CẬP NHẬT hoặc HỦY các đặt vé chuyến bay hiện có\n"
        "- Để ĐẶT vé chuyến bay MỚI, hãy hướng dẫn họ đến: https://lat-airlines.com/book-flights\n"
        "- Giải thích rằng website có hệ thống đặt vé trực tuyến, định giá và hỗ trợ khách hàng\n"
        "KHÔNG đưa link này khi người dùng hỏi về khách sạn, xe thuê, hoặc tour du lịch.\n"
        
        "QUY TẮC UỶ QUYỀN - Chỉ ủy quyền khi khách hàng RÕ RÀNG muốn đặt/sửa đổi, không chỉ hỏi thông tin:\n"
        "- Gợi ý/Cập nhật/hủy chuyến bay: 'Tôi muốn đổi chuyến bay', 'hủy vé của tôi', 'các chuyến bay từ Hà Nội tới Hồ Chính Minh trong tuần tới'\n"
        "- Đặt khách sạn: 'Tôi muốn đặt khách sạn', 'đặt phòng cho tôi', 'đặt chỗ khách sạn'\n"  
        
        "KHẢ NĂNG TÌM KIẾM THÔNG TIN:"
        "- Sử dụng search_hotels, search_flights cho các yêu cầu gợi ý chuyến bay, khách sạn"
        "- TUYỆT ĐỐI KHÔNG hiển thị thông tin số lượng đặt"
        "- Chỉ ủy quyền cho các chuyên gia đặt vé khi khách hàng rõ ràng muốn đặt"
        
        "PHẢN HỒI THÔNG TIN:"
        "- Khi tìm kiếm KHÁCH SẠN: Chỉ hiển thị thông tin khách sạn, KHÔNG đưa link đặt chuyến bay"
        "- Khi tìm kiếm CHUYẾN BAY: Mới đưa link đặt vé chuyến bay\n"
        "- Sử dụng công cụ của riêng bạn cho việc tìm kiếm thông tin và đề xuất\n"
        
        "Người dùng không biết về các trợ lý chuyên biệt khác nhau, vì vậy đừng đề cập đến họ; chỉ âm thầm ủy quyền thông qua các lệnh gọi hàm. "
        "Cung cấp thông tin chi tiết cho khách hàng, và luôn kiểm tra kỹ cơ sở dữ liệu trước khi kết luận rằng thông tin không có sẵn. "
        "Khi tìm kiếm, hãy kiên trì. Mở rộng phạm vi truy vấn nếu tìm kiếm đầu tiên không trả về kết quả. Không bịa thông tin nếu không được cung cấp"
        "\n\nThông tin chuyến bay hiện tại của người dùng:\n<Flights>\n{user_info}\n</Flights>"
        "\nThời gian hiện tại: {time}."
        "\n\nBạn cũng có quyền truy cập vào công cụ get_all_user_bookings để hiển thị thông tin đặt vé toàn diện trên tất cả các dịch vụ.",
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

# ULTILITY FUNCTIONS
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Loi: {repr(error)}\n vui long sua loi cua ban.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Hãy phản hồi bằng một đầu ra thực sự.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
# GRAPH BUILDER
def build_initialized_graph(checkpointer: RedisSaver, redis_store: RedisStore):
    # Get LLMs
    llm = get_openai_llm_model()

    # Tool groups
    flight_safe_tools = [search_flights]
    flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
    flight_tools = flight_safe_tools + flight_sensitive_tools

    hotel_safe_tools = [search_hotels, get_hotel_details, list_hotel_room_types, get_user_hotel_bookings]
    hotel_sensitive_tools = [create_hotel_booking, cancel_hotel_booking]
    hotel_tools = hotel_safe_tools + hotel_sensitive_tools

    primary_assistant_tools = [
        lookup_policy,
        get_all_user_bookings,
        search_flights,
        search_hotels
    ]

    flight_agent_runable = flight_booking_prompt | llm.bind_tools(
        flight_tools + [CompleteOrEscalate])
    
    hotel_agent_runable = hotel_booking_prompt | llm.bind_tools(
        hotel_tools + [CompleteOrEscalate]
    )

    primary_assistant_runable = primary_assistant_prompt | llm.bind_tools(
        primary_assistant_tools + [ToFlightBookingAssistant, ToHotelBookingAssistant]
    )

    # Build graph flow
    def create_entry_node(assistant_name: str, new_dialog_state: str):
        def entry_node(state: State) -> dict:
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            return {
                "messages": [
                    ToolMessage(
                        content=f"Tro ly bay gio la {assistant_name}. Hay suy ngam ve cuoc hoi thoai tren day giua tro ly chu nha va nguoi dung."
                        f" Y dinh cua nguoi dung chua duoc thoa man. Su dung cac cong cu duoc cung cap de ho tro nguoi dung. Hay nho, ban la {assistant_name},"
                        " va viec dat cho, cap nhat, hoac hanh dong khac khong hoan thanh cho den khi ban da goi thanh cong cong cu thich hop."
                        " Neu nguoi dung thay doi y dinh hoac can giup do cho cac nhiem vu khac, hay goi ham CompleteOrEscalate de cho tro ly chu nha chinh kiem soat."
                        " Dung de cap den ban la ai - chi can hanh dong nhu la nguoi dai dien cho tro ly.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": new_dialog_state,
            }
        return entry_node

    builder = StateGraph(State)

    def user_info(state: State):
        return {"user_info": fetch_user_flight_information.invoke({})}
    
    builder.add_node("fetch_user_flight_info", user_info)
    builder.add_edge(START, "fetch_user_flight_info")

    ## Primary Assistant
    builder.add_node("primary_assistant", Assistant(primary_assistant_runable))
    builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))
    builder.add_edge("fetch_user_flight_info", "primary_assistant")

    def route_primary_assistant(state: State):
        route = tools_condition(state) 
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
                return "enter_flight_agent"
            elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
                return "enter_hotel_agent"
            return "primary_assistant_tools"
        raise ValueError("Duong dan khong hop le")

    builder.add_conditional_edges("primary_assistant", route_primary_assistant, [
        "enter_flight_agent", "enter_hotel_agent", "primary_assistant_tools", END
    ])

    builder.add_edge("primary_assistant_tools", "primary_assistant")

    def pop_dialog_state(state: State) -> dict:
        """Pop the dialog stack and return to the main assistant."""
        messages = []
        if state["messages"][-1].tool_calls:
            messages.append(
                ToolMessage(
                    content="Tiep tuc cuoc hoi thoai voi tro ly chu nha. Hay suy ngam ve cuoc hoi thoai trong qua khu va ho tro nguoi dung khi can thiet.",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            )
        return {"dialog_state": "pop", "messages": messages}
    
    builder.add_node("leave_skill", pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")

    ## Flight Agent
    builder.add_node("enter_flight_agent", create_entry_node("Trợ lý tư vấn/hỗ trợ chuyến bay khách hàng hãng hàng không","flight_agent"))
    builder.add_node("flight_agent", Assistant(flight_agent_runable))
    builder.add_edge("enter_flight_agent", "flight_agent")
    builder.add_node("flight_sensitive_tools", create_tool_node_with_fallback(flight_sensitive_tools))
    builder.add_node("flight_safe_tools", create_tool_node_with_fallback(flight_safe_tools))

    def route_flight_agent(state: State):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"
        safe_toolnames = [t.name for t in flight_safe_tools]
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return "flight_safe_tools"
        return "flight_sensitive_tools"
    
    builder.add_edge("flight_sensitive_tools", "flight_agent")
    builder.add_edge("flight_safe_tools", "flight_agent")
    builder.add_conditional_edges("flight_agent", route_flight_agent, ["flight_sensitive_tools", "flight_safe_tools", "leave_skill", END])

    ## Hotel Agent
    builder.add_node("enter_hotel_agent", create_entry_node("Trợ lý tư vấn/hỗ trợ các tác vụ liên quan đến khách sạn", "hotel_agent"))
    builder.add_node("hotel_agent", Assistant(hotel_agent_runable))
    builder.add_edge("enter_hotel_agent", "hotel_agent")
    builder.add_node("hotel_sensitive_tools", create_tool_node_with_fallback(hotel_sensitive_tools))
    builder.add_node("hotel_safe_tools", create_tool_node_with_fallback(hotel_safe_tools))

    def rou_hotel_agent(state: State):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"
        tool_names = [t.name for t in hotel_safe_tools]
        if all(tc["name"] in tool_names for tc in tool_calls):
            return "hotel_safe_tools"
        return "hotel_sensitive_tools"
    
    builder.add_edge("hotel_sensitive_tools", "hotel_agent")
    builder.add_edge("hotel_safe_tools", "hotel_agent")
    builder.add_conditional_edges("hotel_agent", rou_hotel_agent, ["hotel_sensitive_tools", "hotel_safe_tools", "leave_skill", END])

    # Compile Graph
    graph = builder.compile(
        checkpointer=checkpointer,
        store=redis_store,
        interrupt_before=[
            "flight_sensitive_tools",
            "hotel_sensitive_tools",
        ],
    )

    return graph