from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str | None = None

class TokenData(BaseModel):
    email: str | None = None
    role: str | None = None

class User(BaseModel):
    user_id: str
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None
    role: str = "user"

class UserInDB(User):
    hashed_password: str