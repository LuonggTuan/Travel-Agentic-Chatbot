from langchain_openai import OpenAIEmbeddings
from app.config import settings

OPEN_AI_API_KEY = settings.OPENAI_API_KEY
EMBEDDING_MODEL = settings.EMBEDDING_MODEL

def get_embedding_model():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPEN_AI_API_KEY)