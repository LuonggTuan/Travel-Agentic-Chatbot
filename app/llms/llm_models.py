from app.config import settings
from langchain_openai import ChatOpenAI

def get_openai_llm_model() -> ChatOpenAI:
    """Tra ve doi tuong mo hinh ngon ngu LLM duoc cau hinh."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    return llm