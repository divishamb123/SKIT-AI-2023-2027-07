import redis.asyncio as redis
from app.core.config import settings

# Create a global Redis connection pool
redis_client = redis.from_url(
    settings.REDIS_URI, encoding="utf8", decode_responses=True
)


async def store_refresh_token(user_id: str, token: str):
    # Store token with user_id as value, expires matching refresh token lifespan
    await redis_client.setex(
        f"refresh_token:{token}",
        settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        user_id,
    )


async def verify_and_delete_refresh_token(token: str) -> str | None:
    # Get user_id if valid
    user_id = await redis_client.get(f"refresh_token:{token}")
    if user_id:
        # Single use refresh token logic: delete upon use
        await redis_client.delete(f"refresh_token:{token}")
    return user_id


async def delete_refresh_token(token: str):
    await redis_client.delete(f"refresh_token:{token}")
