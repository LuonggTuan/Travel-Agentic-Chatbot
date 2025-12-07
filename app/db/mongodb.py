# from motor.motor_asyncio import AsyncIOMotorClient
# import os
# import dotenv

# dotenv.load_dotenv()

# MONGO_URI = os.getenv("MONGO_URI", "")
# print("Connecting to MongoDB at:", MONGO_URI)
# client = AsyncIOMotorClient(MONGO_URI)
# db = client.airline_chatbot_mongodb     # database
# users_collection = db.users     # collection

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

MONGO_URI = settings.MONGO_URI
DB_NAME = settings.DB_NAME

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

async def get_user_collection():
    return db["users"]