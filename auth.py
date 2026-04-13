import os
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

load_project_env()

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

try:
    from .database import (
        User,
        clean_auth_provider,
        clean_email_value,
        clean_username_value,
        normalize_username_key,
    )
except ImportError:
    from database import (
        User,
        clean_auth_provider,
        clean_email_value,
        clean_username_value,
        normalize_username_key,
    )

DEFAULT_SECRET_KEY = "dev-secret-key-change-me"

SECRET_KEY = os.environ.get("SECRET_KEY", DEFAULT_SECRET_KEY)
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7)
)

if SECRET_KEY == DEFAULT_SECRET_KEY:
    if os.environ.get("PORT"):
        raise RuntimeError("SECRET_KEY environment variable must be set in production.")
    print("WARNING: SECRET_KEY is not set. Using insecure development fallback.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DuplicateUserError(ValueError):
    def __init__(self, field: str | None = None):
        self.field = field
        super().__init__("A user with this identity already exists.")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_email(db: Session, email: str):
    normalized_email = clean_email_value(email)
    if not normalized_email:
        return None
    return db.query(User).filter(User.email_normalized == normalized_email).first()

def get_user_by_username(db: Session, username: str):
    normalized_username = normalize_username_key(username)
    if not normalized_username:
        return None
    return db.query(User).filter(User.username_normalized == normalized_username).first()

def create_user(db: Session, email: str, username: str, password: str, auth_provider: str = "local"):
    cleaned_email = clean_email_value(email)
    cleaned_username = clean_username_value(username)
    normalized_username = normalize_username_key(cleaned_username)
    hashed = get_password_hash(password)
    user = User(
        email=cleaned_email,
        email_normalized=cleaned_email,
        username=cleaned_username,
        username_normalized=normalized_username,
        hashed_password=hashed,
        auth_provider=clean_auth_provider(auth_provider),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        error_text = str(getattr(exc, "orig", exc)).lower()
        if (
            "username_normalized" in error_text
            or "unique constraint failed: users.username" in error_text
            or "duplicate key value violates unique constraint" in error_text and "username" in error_text
        ):
            raise DuplicateUserError("username") from exc
        if (
            "email_normalized" in error_text
            or "unique constraint failed: users.email" in error_text
            or "duplicate key value violates unique constraint" in error_text and "email" in error_text
        ):
            raise DuplicateUserError("email") from exc
        raise DuplicateUserError() from exc
    db.refresh(user)
    return user


def update_auth_provider(db: Session, user: User, auth_provider: str):
    user.auth_provider = clean_auth_provider(auth_provider)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise
    db.refresh(user)
    return user

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not user.is_active or not verify_password(password, user.hashed_password):
        return None
    return user

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
