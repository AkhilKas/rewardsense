"""Auth business logic: signup, login, OTP verification."""

from __future__ import annotations

import hashlib
import logging
import os
import random
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.app.db.models import AuthCredential, User, UserSettings

logger = logging.getLogger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

OTP_EXPIRY_MINUTES = 10


def _hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def _generate_otp() -> str:
    return f"{random.randint(0, 999999):06d}"


def _hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def _send_verification_email(email: str, otp: str) -> None:
    api_key = os.getenv("RESEND_API_KEY", "")
    from_address = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

    if not api_key:
        logger.warning("RESEND_API_KEY not set — OTP for %s: %s", email, otp)
        return

    try:
        import resend
        resend.api_key = api_key
        resend.Emails.send({
            "from": f"RewardSense <{from_address}>",
            "to": [email],
            "subject": "Your RewardSense verification code",
            "html": f"""
                <div style="font-family:sans-serif;max-width:480px;margin:0 auto">
                    <h2 style="color:#c2651a">Verify your email</h2>
                    <p>Enter this code to verify your RewardSense account:</p>
                    <div style="font-size:36px;font-weight:bold;letter-spacing:8px;
                                padding:16px;background:#f5f5f5;border-radius:8px;
                                text-align:center;margin:16px 0">
                        {otp}
                    </div>
                    <p style="color:#666;font-size:14px">
                        This code expires in {OTP_EXPIRY_MINUTES} minutes.
                        If you didn't create an account, you can ignore this email.
                    </p>
                </div>
            """,
        })
    except Exception as exc:
        logger.error("Failed to send verification email to %s: %s", email, exc)


def signup(db: Session, email: str, password: str, display_name: str) -> tuple[User, str]:
    """Create a new user, generate OTP, send verification email.

    Returns (user, otp).
    """
    existing = db.query(User).filter(User.email == email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    otp = _generate_otp()
    user = User(
        email=email.lower(),
        display_name=display_name,
        is_verified=False,
        email_otp_hash=_hash_otp(otp),
        otp_expires_at=datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES),
    )
    db.add(user)
    db.flush()

    credential = AuthCredential(user_id=user.id, password_hash=_hash_password(password))
    db.add(credential)

    settings = UserSettings(user_id=user.id)
    db.add(settings)

    db.commit()
    db.refresh(user)

    _send_verification_email(email.lower(), otp)
    return user, otp


def authenticate(db: Session, email: str, password: str) -> User:
    """Validate credentials. Raises 401 on any failure."""
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user or not user.credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not _verify_password(password, user.credential.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    return user


def verify_otp(db: Session, user_id: int, otp: str) -> User:
    """Verify OTP and mark the user as verified."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if user.is_verified:
        return user

    if not user.email_otp_hash or not user.otp_expires_at:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No pending verification.")

    expires_at = user.otp_expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Verification code has expired.")

    if _hash_otp(otp) != user.email_otp_hash:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification code.")

    user.is_verified = True
    user.email_otp_hash = None
    user.otp_expires_at = None
    db.commit()
    db.refresh(user)
    return user


def resend_otp(db: Session, user_id: int) -> None:
    """Generate a fresh OTP and resend the verification email."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if user.is_verified:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already verified.")

    otp = _generate_otp()
    user.email_otp_hash = _hash_otp(otp)
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)
    db.commit()

    _send_verification_email(user.email, otp)
