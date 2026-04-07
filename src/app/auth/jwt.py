"""JWT creation and verification for RewardSense auth."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

_ALGORITHM = "HS256"


def _secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable is not set. "
            "Set it to a long random string before starting the server."
        )
    return secret


def _expiry_minutes() -> int:
    try:
        return int(os.getenv("JWT_EXPIRY_MINUTES", "60"))
    except ValueError:
        return 60


def create_access_token(user_id: int) -> str:
    """Return a signed JWT encoding the given user_id."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=_expiry_minutes())
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    """Decode and verify a JWT. Returns user_id or None if invalid/expired."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            return None
        return int(sub)
    except (JWTError, ValueError):
        return None
