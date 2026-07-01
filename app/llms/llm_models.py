from app.config import settings
from langchain_openai import ChatOpenAI


def get_openai_llm_model() -> ChatOpenAI:
    """Returns the configured OpenAI LLM model."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    return llm

def get_ollama_llm_model() -> ChatOpenAI:
    """Returns the configured Ollama LLM model."""
    llm = ChatOpenAI(
        model=settings.LLAMA_CPP_MODEL,
        base_url=settings.LLAMA_CPP_BASE_URL,
        api_key="not-needed",
        temperature=settings.TEMPERATURE,
    )
    return llm