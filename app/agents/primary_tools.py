from app.config import settings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.services.milvus_service import query_milvus
from app.agents.flight_agent_tools import fetch_user_flight_information
from app.agents.hotel_agent_tools import get_user_hotel_bookings
from langchain_core.tools import tool

@tool
def lookup_policy(query: str) -> str:
    """truy vấn milvus các chính sách, các câu hỏi thường gặp hãng hàng không"""
    results = query_milvus(
        collection_name=settings.COLLECTION_NAME, 
        query=query
    )
    results = sorted(results, key=lambda x: x['id'])
    if not results:
        return "No relevant policy found."
    return "\n\n".join([f"{res['content']}" for res in results])

@tool
def get_all_user_bookings(*, config: RunnableConfig) -> dict:
    """Lay tat ca dat phong khach san va ve may bay cua nguoi dung."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None) 
    if not user_id:
        raise ValueError("Khong co ID hanh khach duoc cau hinh.")

    flights = fetch_user_flight_information.invoke({}, config)
    hotels = get_user_hotel_bookings.invoke({}, config)

    return {
        "flights": flights,
        "hotels": hotels
    }