from fastapi import APIRouter, Depends, HTTPException, Request, Response
import httpx
from typing import Dict, Any

from ..core.config import settings
from .deps import get_current_user

router = APIRouter(tags=["auth"])


async def proxy_request(method: str, path: str, request: Request, json_data: dict = None):
    url = f"{settings.AUTH_SERVICE_URL}{path}"
    async with httpx.AsyncClient() as client:
        try:
            req = client.build_request(method, url, json=json_data, cookies=request.cookies)
            resp = await client.send(req)
            
            response = Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
            
            # Forward Set-Cookie headers
            for cookie in resp.headers.get_list("set-cookie"):
                response.headers.append("Set-Cookie", cookie)
                
            return response
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503, detail=f"Auth service unavailable: {exc}"
            )


@router.post("/register")
async def proxy_register(request: Request):
    data = await request.json()
    return await proxy_request("POST", "/register", request, json_data=data)


@router.post("/login")
async def proxy_login(request: Request):
    data = await request.json()
    return await proxy_request("POST", "/login", request, json_data=data)


@router.post("/refresh")
async def proxy_refresh(request: Request):
    return await proxy_request("POST", "/refresh", request)


@router.post("/logout")
async def proxy_logout(request: Request):
    return await proxy_request("POST", "/logout", request)


@router.get("/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    # Simply return the JWT decoded payload
    return {"user": current_user}
