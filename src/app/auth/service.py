"""Auth business logic: signup and login."""

from __future__ import annotations

from fastapi import HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.app.db.models import AuthCredential, User, UserSettings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def signup(db: Session, email: str, password: str, display_name: str) -> User:
    """Create a new user. Raises 409 if email already registered."""
    existing = db.query(User).filter(User.email == email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(email=email.lower(), display_name=display_name)
    db.add(user)
    db.flush()  # get user.id without committing

    credential = AuthCredential(
        user_id=user.id,
        password_hash=_hash_password(password),
    )
    db.add(credential)

    settings = UserSettings(user_id=user.id)
    db.add(settings)

    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    """Validate credentials. Raises 401 on any failure (no info leak)."""
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
