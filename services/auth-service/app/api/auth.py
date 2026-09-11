from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import uuid

from shared.db.session import get_async_session
from shared.db.models.user import User
from shared.schemas.auth import UserCreate, UserLogin, Token, RefreshTokenRequest
from app.core import security
from app.core import redis
from app.core.limiter import limiter

router = APIRouter(tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def register(
    request: Request, user_in: UserCreate, db: AsyncSession = Depends(get_async_session)
):
    # Check if user exists
    stmt = select(User).where(User.email == user_in.email)
    result = await db.execute(stmt)
    if result.scalars().first() is not None:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = security.get_password_hash(user_in.password)
    new_user = User(
        id=uuid.uuid4(),
        name=user_in.name,
        email=user_in.email,
        password_hash=hashed_password,
        role="user",
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}


@router.post("/login")
@limiter.limit("20/minute")
async def login(
    request: Request, response: Response, user_in: UserLogin, db: AsyncSession = Depends(get_async_session)
):
    from app.core.config import settings
    stmt = select(User).where(User.email == user_in.email)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if not user or not security.verify_password(user_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Generate tokens
    token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
    access_token = security.create_access_token(data=token_data)
    refresh_token = security.create_refresh_token(data={"sub": str(user.id)})

    # Store refresh token in redis
    await redis.store_refresh_token(str(user.id), refresh_token)

    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/api/v1/auth"
    )

    return {"message": "Successfully logged in", "token_type": "cookie"}


@router.post("/refresh")
async def refresh_token(
    request: Request, response: Response, db: AsyncSession = Depends(get_async_session)
):
    from app.core.config import settings
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    # Verify signature
    decoded = security.verify_token(token)
    if not decoded or decoded.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Check if exists in redis
    user_id = await redis.verify_and_delete_refresh_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked or expired",
        )

    # Get user
    stmt = select(User).where(User.id == uuid.UUID(user_id))
    result = await db.execute(stmt)
    user = result.scalars().first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Generate new tokens
    token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
    new_access_token = security.create_access_token(data=token_data)
    new_refresh_token = security.create_refresh_token(data={"sub": str(user.id)})

    await redis.store_refresh_token(str(user.id), new_refresh_token)

    response.set_cookie(
        key="access_token",
        value=new_access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/api/v1/auth"
    )

    return {"message": "Successfully refreshed", "token_type": "cookie"}


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if token:
        await redis.delete_refresh_token(token)
        
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/api/v1/auth")
    
    return {"message": "Successfully logged out"}
