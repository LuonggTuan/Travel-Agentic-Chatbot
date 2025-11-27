import asyncio
import uuid
from pwdlib import PasswordHash
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_URI)
db = client.airline_chatbot_mongodb
users_collection = db.users

password_hash = PasswordHash.recommended()

async def create_admin():
    username = "admin"

    # Kiểm tra trùng username
    if await users_collection.find_one({"username": username}):
        print("⚠ Username already exists!")
        return

    # Sinh user_id duy nhất
    while True:
        user_id = str(uuid.uuid4())
        if not await users_collection.find_one({"user_id": user_id}):
            break

    hashed_password = password_hash.hash("admin123")

    admin_doc = {
        "user_id": user_id,
        "username": username,
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": hashed_password,
        "role": "admin",
        "disabled": False,
    }

    result = await users_collection.insert_one(admin_doc)
    print(f"✅ Admin created! user_id: {user_id}")

if __name__ == "__main__":
    asyncio.run(create_admin())
