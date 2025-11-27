from app.models.auth_models import UserInDB
from app.db.mongodb import get_user_collection
from pwdlib import PasswordHash
import uuid

password_hash = PasswordHash.recommended()

def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password):
    return password_hash.hash(password)

async def get_user_by_user_id(user_id: str):
    users = await get_user_collection()
    user_dict = await users.find_one({"user_id": user_id})
    if user_dict:
        return UserInDB(**user_dict)

async def create_user(username: str, password: str, email: str | None = None, full_name: str | None = None, role: str = "user"):
    users = await get_user_collection()
    while True:
        user_id = str(uuid.uuid4())
        existing = await get_user_by_user_id(user_id)
        if not existing:
            break

    hashed_password = password_hash.hash(password)
    user_dict = {
        "user_id": user_id,
        "username": username,
        "hashed_password": hashed_password,
        "email": email,
        "full_name": full_name,
        "disabled": False,
        "role": role
    }
    await users.insert_one(user_dict)
    return UserInDB(**user_dict)