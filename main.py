from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.models.auth_models import Token, User, UserInDB
from app.core.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
    get_current_active_user
)
from app.core.memory import get_redis_saver
from app.services.mongodb_crud import create_user, get_user_by_email
from app.api.routers.milvus_upload import router as milvus_router
from app.api.routers.chat import router as chat_router
from app.utils import logger
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.milvus import connect_milvus
from app.agents.graph_builder import build_initialized_graph
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Starting up the Airline Chatbot API...")
    try:
        # Connect to Milvus
        connect_milvus()
        logger.info("Connect to milvus...done!!")
        # Setup redis
        checkpointer, redis_store = get_redis_saver()
        logger.info("Create saver for agent...done!!!")
        # Build graph
        app.state.graph = build_initialized_graph(
            checkpointer=checkpointer,
            redis_store=redis_store
        )
        logger.info("Build graph...done!!!")
    except Exception as e:
        logger.info("Startup failed")
        raise RuntimeError(str(e))

    yield

app = FastAPI(
    title="Airline Chatbot API",
    description="API for Airline Chatbot",
    version="0.0.1",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(milvus_router)
app.include_router(chat_router)

# REGISTER
@app.post("/register", response_model=UserInDB, tags=["Authentication"])
async def register_user(
    username: str = Body(...),
    email: str = Body(...),
    password: str = Body(...),
    full_name: str | None = Body(None)
):
    existing_user = await get_user_by_email(email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")
    user = await create_user(username=username, password=password, email=email, full_name=full_name)
    return user

# LOGIN
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # OAuth2 form: username = email
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role},
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(user.email)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token
    }

# REFRESH
@app.post("/refresh", response_model=Token, tags=["Authentication"])
async def refresh_access_token(refresh_token: str = Body(...)):
    email = verify_refresh_token(refresh_token)
    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email, "role": user.role},
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token
    }

# USER INFO
@app.get("/users/me/", response_model=User, tags=["Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user