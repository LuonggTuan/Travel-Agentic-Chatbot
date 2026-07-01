from langchain_huggingface import HuggingFaceEmbeddings
import os

# Function to get the multilingual embedding model
def get_multilingual_embedding_model():
    MODEL_NAME = "intfloat/multilingual-e5-base"
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
    DEVICE = "cpu"  # Change to "cuda" if you want to use GPU

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        cache_folder=CACHE_DIR,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )