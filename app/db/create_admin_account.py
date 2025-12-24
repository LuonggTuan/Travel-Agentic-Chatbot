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

    list_user_id = [
        "ed8d0cba-6743-4d85-b5d5-5b7613ba9823",
        "48dc56e4-ff2f-4048-8a2d-843b356e9e7c",
        "106d99ad-d535-42b6-ba96-e81328a349ad",
        "746c4579-abcc-4518-9300-4be1ea86a7c0",
        "04aa42a4-a973-4b49-9b3a-7c371301e27d"
    ]

    for i, users_id in enumerate(list_user_id):
        hashed_password = password_hash.hash(f"user{i+1}")
        username = f"user{i+1}"
        user_doc = {
            "user_id": users_id,
            "username": username,
            "full_name": username,
            "email": f"user{i+1}@gmail.com",
            "hashed_password": hashed_password,
            "role": "user",
            "disabled": False,
        }
        result = await users_collection.insert_one(user_doc)
        print(f"✅ User created! user_id: {username}")

if __name__ == "__main__":
    asyncio.run(create_admin())
