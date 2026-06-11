"""Auth endpoints: signup, login, logout, email verification."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from src.app.auth import service
from src.app.auth.jwt import create_access_token, decode_token
from src.app.db.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

_bearer = HTTPBearer()


def _current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> int:
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    return user_id


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    display_name: str
    is_verified: bool = False


class VerifyEmailRequest(BaseModel):
    otp: str = Field(min_length=6, max_length=6)


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user, _ = service.signup(
        db,
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
    )
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
        is_verified=user.is_verified,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = service.authenticate(db, email=payload.email, password=payload.password)
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
        is_verified=user.is_verified,
    )


@router.post("/verify-email", response_model=TokenResponse)
def verify_email(
    payload: VerifyEmailRequest,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> TokenResponse:
    user_id = _current_user_id(credentials)
    user = service.verify_otp(db, user_id=user_id, otp=payload.otp)
    return TokenResponse(
        access_token=credentials.credentials,
        user_id=user.id,
        display_name=user.display_name,
        is_verified=user.is_verified,
    )


@router.post("/resend-otp", status_code=204)
def resend_otp(
    user_id: int = Depends(_current_user_id),
    db: Session = Depends(get_db),
) -> None:
    service.resend_otp(db, user_id=user_id)


@router.post("/logout")
def logout() -> dict:
    return {"ok": True}
