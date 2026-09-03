from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from datetime import timedelta


def test_password_hashing():
    password = "supersecretpassword"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False


def test_access_token():
    data = {"sub": "user_id_123", "email": "test@test.com", "role": "user"}
    token = create_access_token(data)
    decoded = verify_token(token)
    assert decoded is not None
    assert decoded["sub"] == data["sub"]
    assert decoded["email"] == data["email"]
    assert decoded["role"] == data["role"]
    assert "exp" in decoded


def test_refresh_token():
    data = {"sub": "user_id_123"}
    token = create_refresh_token(data)
    decoded = verify_token(token)
    assert decoded is not None
    assert decoded["sub"] == data["sub"]
    assert decoded["type"] == "refresh"
    assert "exp" in decoded


def test_expired_token():
    data = {"sub": "user_id_123"}
    token = create_access_token(data, expires_delta=timedelta(seconds=-1))
    decoded = verify_token(token)
    assert decoded is None
