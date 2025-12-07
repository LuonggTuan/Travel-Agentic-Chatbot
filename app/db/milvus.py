
from pymilvus import DataType
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from app.utils import logger
from app.config import settings

MILVUS_URI = settings.MILVUS_URI
MILVUS_DB_NAME = settings.MILVUS_DB_NAME

def connect_milvus():
    try:
        connections.connect(alias="default", uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
        logger.info("Connected to Milvus successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")

def check_collection_milvus(collection_name: str):
    try:
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=40000),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
            ]

            schema = CollectionSchema(fields = fields, description="Document chunks collection")
            collection = Collection(name=collection_name, schema=schema)

            index_params = {
                "index_type": "FLAT",
                "metric_type": "COSINE"
            }

            collection.create_index(field_name="vector", index_params=index_params)

            logger.info(f"Collection {collection_name} created successfully.")
    except Exception as e:
        logger.error(f"Failed to check or create collection {collection_name}: {e}")
    
