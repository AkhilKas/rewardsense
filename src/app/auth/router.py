"""Auth endpoints: signup, login, logout."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from src.app.auth import service
from src.app.auth.jwt import create_access_token
from src.app.db.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


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


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = service.signup(
        db,
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
    )
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = service.authenticate(db, email=payload.email, password=payload.password)
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )


@router.post("/logout")
def logout() -> dict:
    # JWT is stateless; the client discards the token.
    return {"ok": True}
