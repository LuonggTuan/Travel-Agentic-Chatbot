from app.models.auth_models import UserInDB
from app.db.mongodb import get_user_collection
from pwdlib import PasswordHash
import uuid

password_hash = PasswordHash.recommended()

def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password):
    return password_hash.hash(password)


async def get_user_by_email(email: str):
    users = await get_user_collection()
    user_dict = await users.find_one({"email": email})
    if user_dict:
        return UserInDB(**user_dict)

async def get_user_by_username(username: str):
    users = await get_user_collection()
    user_dict = await users.find_one({"username": username})
    if user_dict:
        return UserInDB(**user_dict)

async def get_user_by_user_id(user_id: str):
    users = await get_user_collection()
    user_dict = await users.find_one({"user_id": user_id})
    if user_dict:
        return UserInDB(**user_dict)

async def create_user(username: str, email: str, password: str, full_name: str | None = None, role: str = "user"):
    users = await get_user_collection()

    # check email trùng
    if await users.find_one({"email": email}):
        raise ValueError("Email already exists")

    # Generate unique user_id
    user_id = str(uuid.uuid4())

    # check user_id trùng (hiếm nhưng vẫn check)
    while await users.find_one({"user_id": user_id}):
        user_id = str(uuid.uuid4())

    hashed_password = password_hash.hash(password)

    user_dict = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "full_name": full_name,
        "disabled": False,
        "role": role
    }
    await users.insert_one(user_dict)
    return UserInDB(**user_dict)