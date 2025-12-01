from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import encode, decode
from jwt.exceptions import InvalidTokenError
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
from app.models.auth_models import User, TokenData
from app.services.mongodb_crud import get_user_by_email, verify_password
import os
import dotenv

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 2))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ------------------- AUTH LOGIC -------------------
async def authenticate_user(email: str, password: str):
    user = await get_user_by_email(email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": "access"})
    return encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(username: str):
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": username, "exp": expire, "type": "refresh"}
    return encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_refresh_token(token: str):
    try:
        payload = decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        username = payload.get("sub")
        return username
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

# ------------------- DEPENDENCIES -------------------
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise credentials_exception
        email = payload.get("sub")
        role = payload.get("role", "user")
        if email is None:
            raise credentials_exception
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token expired")
    except InvalidTokenError:
        raise credentials_exception

    user = await get_user_by_email(email)
    if user is None:
        raise credentials_exception

    user.role = role
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user