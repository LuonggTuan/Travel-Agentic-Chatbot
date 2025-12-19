from app.config import settings
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

def get_redis_saver():
    with RedisSaver.from_conn_string(
    "redis://localhost:6379"
    ) as redis_saver, RedisStore.from_conn_string(
        "redis://localhost:6379"
    ) as redis_store:
        redis_saver.setup()
        redis_store.setup()

    return redis_saver, redis_store