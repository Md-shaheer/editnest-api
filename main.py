from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel
from PIL import Image, ImageOps, UnidentifiedImageError
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

load_project_env()

import io
import re
import hashlib
import os
import mimetypes
import gc
import asyncio
import json
import time
import base64
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

# Limit thread usage to optimize for the 0.1 CPU constraint
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    from .database import ActivityEvent, AnalyticsSession, SessionPageView, get_db, create_tables
    from .auth import (
        DuplicateUserError, create_user, authenticate_user, create_access_token,
        get_user_by_email, get_user_by_username, decode_token, update_auth_provider, verify_password
    )
except ImportError:
    from database import ActivityEvent, AnalyticsSession, SessionPageView, get_db, create_tables
    from auth import (
        DuplicateUserError, create_user, authenticate_user, create_access_token,
        get_user_by_email, get_user_by_username, decode_token, update_auth_provider, verify_password
    )

app = FastAPI(title="EditNest API", version="1.0.2")

CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = max(1, int(os.environ.get("CACHE_TTL_HOURS", "24"))) * 60 * 60
CACHE_CLEANUP_INTERVAL_SECONDS = max(300, int(os.environ.get("CACHE_CLEANUP_INTERVAL_MINUTES", "60"))) * 60
last_cache_cleanup_at = 0.0

DEFAULT_SECRET_API_KEY = "editnest-automation-key-123"
SECRET_API_KEY = os.environ.get("API_KEY")

if not SECRET_API_KEY and not os.environ.get("PORT"):
    SECRET_API_KEY = DEFAULT_SECRET_API_KEY
elif not SECRET_API_KEY:
    print("WARNING: API_KEY is not set. X-API-Key automation access is disabled.")

create_tables()
# Pre-download the rembg model on startup
from rembg import new_session, remove


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


REMBG_MODEL_NAME = os.environ.get("REMBG_MODEL_NAME", "u2net").strip() or "u2net"
REMBG_POST_PROCESS_MASK = env_flag("REMBG_POST_PROCESS_MASK", True)
REMBG_ALPHA_MATTING = env_flag("REMBG_ALPHA_MATTING", True)
REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD = int(
    os.environ.get("REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD", "240")
)
REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD = int(
    os.environ.get("REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD", "10")
)
REMBG_ALPHA_MATTING_ERODE_SIZE = int(
    os.environ.get("REMBG_ALPHA_MATTING_ERODE_SIZE", "10")
)
REMBG_MAX_PROCESSING_SIDE = int(os.environ.get("REMBG_MAX_PROCESSING_SIDE", "2800"))
REMBG_ALPHA_MATTING_MAX_PIXELS = int(
    os.environ.get("REMBG_ALPHA_MATTING_MAX_PIXELS", "4500000")
)
REMBG_LARGE_UPLOAD_SIZE_MB = int(os.environ.get("REMBG_LARGE_UPLOAD_SIZE_MB", "4"))
REMBG_LARGE_IMAGE_PIXEL_THRESHOLD = int(
    os.environ.get("REMBG_LARGE_IMAGE_PIXEL_THRESHOLD", "4000000")
)
REMBG_LARGE_IMAGE_MAX_SIDE = int(os.environ.get("REMBG_LARGE_IMAGE_MAX_SIDE", "2200"))
RESAMPLING_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

print(f"Pre-loading AI model ({REMBG_MODEL_NAME})...")
model_session = new_session(REMBG_MODEL_NAME)
print("AI model loaded!")


def cleanup_expired_cache(force: bool = False):
    global last_cache_cleanup_at

    now = time.time()
    if not force and (now - last_cache_cleanup_at) < CACHE_CLEANUP_INTERVAL_SECONDS:
        return

    removed_files = 0

    try:
        for entry in os.scandir(CACHE_DIR):
            if not entry.is_file():
                continue

            try:
                file_age = now - entry.stat().st_mtime
                if file_age > CACHE_TTL_SECONDS:
                    os.remove(entry.path)
                    removed_files += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"WARNING: failed to inspect/delete cache file {entry.path}: {exc}")
    except FileNotFoundError:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as exc:
        print(f"WARNING: cache cleanup failed: {exc}")
    finally:
        last_cache_cleanup_at = now

    if removed_files:
        print(f"Cache cleanup removed {removed_files} expired file(s).")


cleanup_expired_cache(force=True)


def get_allowed_origins():
    origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://editnest.in",
        "https://www.editnest.in",
    ]

    frontend_url = os.environ.get("FRONTEND_URL", "").strip()
    if frontend_url:
        origins.append(frontend_url)

    extra_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    origins.extend(origin.strip() for origin in extra_origins.split(",") if origin.strip())

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(origins))


cors_allow_origin_regex = os.environ.get(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_TYPE_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

# Ensure we only process one image at a time to prevent RAM crashes
processing_semaphore = asyncio.Semaphore(1)
thread_pool = ThreadPoolExecutor(max_workers=1)

# --- Pydantic Models ---
class SignupRequest(BaseModel):
    email: str
    username: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str
    username: Optional[str] = None
    provider: str = "google"


FirebaseAuthRequest = GoogleAuthRequest

class GenerateBgRequest(BaseModel):
    prompt: str
    image_base64: str


class ClientEventRequest(BaseModel):
    event: str
    page: Optional[str] = None
    session_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None


ALLOWED_CLIENT_EVENTS = {
    "auth_view",
    "dashboard_view",
    "activity_view",
    "logout",
    "result_view",
    "upload_cancelled",
    "page_view",
    "page_leave",
    "session_start",
    "session_ping",
    "session_end",
}

NOISY_ACTIVITY_EVENTS = {
    "session_ping",
    "page_leave",
}

LOGIN_SUCCESS_EVENTS = {
    "login_success",
    "google_login_success",
    "apple_login_success",
}

SUPPORTED_AUTH_PROVIDERS = {
    "google": "Google",
    "apple": "Apple",
}
AUTH_ACCESS_RESTRICTED_DETAIL = "Access restricted. This email is not approved for this website."


def get_admin_emails():
    raw_admin_emails = os.environ.get("ADMIN_EMAILS", "")
    return {email.strip().lower() for email in raw_admin_emails.split(",") if email.strip()}


def normalize_email_address(email: Optional[str]) -> str:
    return (email or "").strip().lower()


def get_allowed_auth_emails():
    raw_allowed_emails = os.environ.get("AUTH_ALLOWED_EMAILS", "")
    allowed_emails = {
        normalize_email_address(email)
        for email in raw_allowed_emails.split(",")
        if normalize_email_address(email)
    }
    return allowed_emails | get_admin_emails()


def get_allowed_auth_domains():
    raw_allowed_domains = os.environ.get("AUTH_ALLOWED_DOMAINS", "")
    return {
        domain.strip().lower().lstrip("@")
        for domain in raw_allowed_domains.split(",")
        if domain.strip()
    }


AUTH_INVITE_ONLY = env_flag("AUTH_INVITE_ONLY", False)


def is_auth_email_allowed(email: Optional[str]) -> bool:
    normalized_email = normalize_email_address(email)
    if not normalized_email:
        return False

    if not AUTH_INVITE_ONLY:
        return True

    if normalized_email in get_allowed_auth_emails():
        return True

    if "@" not in normalized_email:
        return False

    return normalized_email.rsplit("@", 1)[1] in get_allowed_auth_domains()


def is_admin_email(email: Optional[str]) -> bool:
    return bool(email) and normalize_email_address(email) in get_admin_emails()


def get_request_ip(request: Request) -> Optional[str]:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def get_app_timezone():
    timezone_name = (
        os.environ.get("APP_TIMEZONE")
        or os.environ.get("TZ")
        or "Asia/Kolkata"
    ).strip()

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        fixed_offset_timezones = {
            "asia/kolkata": timezone(timedelta(hours=5, minutes=30)),
            "asia/calcutta": timezone(timedelta(hours=5, minutes=30)),
            "ist": timezone(timedelta(hours=5, minutes=30)),
        }
        fallback_timezone = fixed_offset_timezones.get(timezone_name.lower())
        if fallback_timezone:
            return fallback_timezone
        print(f"WARNING: unknown APP_TIMEZONE '{timezone_name}', falling back to UTC.")
        return timezone.utc


APP_TIMEZONE = get_app_timezone()
APP_TIMEZONE_NAME = getattr(APP_TIMEZONE, "key", str(APP_TIMEZONE))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_timestamp_for_math(value: Optional[datetime]) -> Optional[datetime]:
    if not value:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def serialize_timestamp(value: Optional[datetime]) -> Optional[str]:
    normalized_value = normalize_timestamp_for_math(value)
    if not normalized_value:
        return None
    return normalized_value.astimezone(APP_TIMEZONE).isoformat()


def serialize_timestamp_utc(value: Optional[datetime]) -> Optional[str]:
    normalized_value = normalize_timestamp_for_math(value)
    if not normalized_value:
        return None
    return normalized_value.astimezone(timezone.utc).isoformat()


def clamp_non_negative_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def duration_ms_to_seconds(value: Any) -> Optional[int]:
    normalized_value = clamp_non_negative_int(value)
    if normalized_value is None:
        return None
    return normalized_value // 1000


def compute_elapsed_seconds(started_at: Optional[datetime], now: datetime) -> int:
    normalized_started_at = normalize_timestamp_for_math(started_at)
    normalized_now = normalize_timestamp_for_math(now)
    if not normalized_started_at or not normalized_now:
        return 0
    return max(0, int((normalized_now - normalized_started_at).total_seconds()))


def resolve_upload_content_type(file: UploadFile) -> Optional[str]:
    if file.content_type in ALLOWED_TYPES:
        return file.content_type

    filename = (file.filename or "").strip().lower()
    extension = os.path.splitext(filename)[1]
    if extension in ALLOWED_TYPE_BY_EXTENSION:
        return ALLOWED_TYPE_BY_EXTENSION[extension]

    guessed_type, _ = mimetypes.guess_type(filename)
    if guessed_type in ALLOWED_TYPES:
        return guessed_type

    return None


def encode_details(details: Optional[dict[str, Any]]) -> Optional[str]:
    if not details:
        return None
    return json.dumps(details, ensure_ascii=True)


def decode_details(details: Optional[str]) -> Optional[dict[str, Any]]:
    if not details:
        return None
    try:
        return json.loads(details)
    except json.JSONDecodeError:
        return {"raw": details}


def resolve_user_from_auth(db: Session, authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    email = decode_token(token)
    if not email:
        return None
    if not is_auth_email_allowed(email):
        return None
    return get_user_by_email(db, email)


def require_admin_user(db: Session, authorization: Optional[str]):
    user = resolve_user_from_auth(db, authorization)
    if not user or not is_admin_email(user.email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def safe_log_activity(
    db: Session,
    event: str,
    request: Optional[Request] = None,
    user=None,
    email: Optional[str] = None,
    page: Optional[str] = None,
    session_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
):
    try:
        sync_analytics_state(
            db=db,
            event=event,
            request=request,
            user=user,
            email=email,
            page=page,
            session_id=session_id,
            details=details,
        )

        activity = ActivityEvent(
            user_id=getattr(user, "id", None),
            email=email or getattr(user, "email", None),
            event=event,
            page=page,
            method=request.method if request else None,
            path=str(request.url.path) if request else None,
            ip_address=get_request_ip(request) if request else None,
            user_agent=request.headers.get("user-agent") if request else None,
            session_id=session_id,
            details=encode_details(details),
        )
        db.add(activity)
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"WARNING: failed to write activity log for {event}: {exc}")


def get_or_create_analytics_session(
    db: Session,
    session_id: str,
    request: Optional[Request],
    user,
    email: Optional[str],
    page: Optional[str],
    details: Optional[dict[str, Any]],
    event: str,
    now: datetime,
):
    session = db.query(AnalyticsSession).filter(AnalyticsSession.session_id == session_id).first()
    resolved_email = email or getattr(user, "email", None)
    resolved_user_id = getattr(user, "id", None)
    referrer = None
    if details:
        referrer = details.get("referrer") or details.get("document_referrer")

    if session:
        if resolved_user_id and session.user_id != resolved_user_id:
            session.user_id = resolved_user_id
        if resolved_email and session.email != resolved_email:
            session.email = resolved_email
        if resolved_email:
            session.is_authenticated = True
        if page:
            session.current_page = page
            session.exit_page = page
            if not session.landing_page:
                session.landing_page = page
        if request and not session.ip_address:
            session.ip_address = get_request_ip(request)
        if request and not session.user_agent:
            session.user_agent = request.headers.get("user-agent")
        if referrer and not session.referrer:
            session.referrer = str(referrer)[:500]
        session.last_seen_at = now
        session.is_active = event != "session_end"
        session.last_event = event
        session.total_duration_seconds = max(session.total_duration_seconds or 0, compute_elapsed_seconds(session.started_at, now))
        return session

    session = AnalyticsSession(
        session_id=session_id,
        user_id=resolved_user_id,
        email=resolved_email,
        is_authenticated=bool(resolved_email),
        is_active=event != "session_end",
        landing_page=page,
        current_page=page,
        exit_page=page,
        ip_address=get_request_ip(request) if request else None,
        user_agent=request.headers.get("user-agent") if request else None,
        referrer=str(referrer)[:500] if referrer else None,
        started_at=now,
        last_seen_at=now,
        ended_at=now if event == "session_end" else None,
        total_duration_seconds=0,
        total_events=0,
        total_page_views=0,
        total_uploads=0,
        last_event=event,
    )
    db.add(session)
    return session


def get_active_page_view(db: Session, session_id: str, page: Optional[str] = None):
    query = db.query(SessionPageView).filter(
        SessionPageView.session_id == session_id,
        SessionPageView.is_active.is_(True),
    )
    if page:
        query = query.filter(SessionPageView.page == page)

    page_view = query.order_by(SessionPageView.entered_at.desc(), SessionPageView.id.desc()).first()
    if page_view or page is None:
        return page_view

    return (
        db.query(SessionPageView)
        .filter(
            SessionPageView.session_id == session_id,
            SessionPageView.is_active.is_(True),
        )
        .order_by(SessionPageView.entered_at.desc(), SessionPageView.id.desc())
        .first()
    )


def close_page_view(page_view: SessionPageView, now: datetime, exit_event: str, duration_seconds: Optional[int] = None):
    page_view.is_active = False
    page_view.exit_event = exit_event
    page_view.last_seen_at = now
    page_view.left_at = now
    computed_seconds = duration_seconds
    if computed_seconds is None:
        computed_seconds = compute_elapsed_seconds(page_view.entered_at, now)
    page_view.duration_seconds = max(page_view.duration_seconds or 0, computed_seconds)


def ensure_page_view(
    db: Session,
    analytics_session: AnalyticsSession,
    user,
    page: str,
    now: datetime,
    path: Optional[str],
    event: str,
):
    existing_page_view = get_active_page_view(db, analytics_session.session_id, page)
    if existing_page_view:
        existing_page_view.last_seen_at = now
        existing_page_view.duration_seconds = max(
            existing_page_view.duration_seconds or 0,
            compute_elapsed_seconds(existing_page_view.entered_at, now),
        )
        return existing_page_view

    other_active_views = (
        db.query(SessionPageView)
        .filter(
            SessionPageView.session_id == analytics_session.session_id,
            SessionPageView.is_active.is_(True),
        )
        .all()
    )
    for active_view in other_active_views:
        close_page_view(active_view, now, "page_leave")

    page_view = SessionPageView(
        session_id=analytics_session.session_id,
        user_id=getattr(user, "id", None),
        email=getattr(user, "email", None) or analytics_session.email,
        page=page,
        path=path,
        entry_event=event,
        sequence_number=(analytics_session.total_page_views or 0) + 1,
        entered_at=now,
        last_seen_at=now,
        is_active=True,
        duration_seconds=0,
    )
    analytics_session.total_page_views = (analytics_session.total_page_views or 0) + 1
    analytics_session.current_page = page
    analytics_session.exit_page = page
    db.add(page_view)
    return page_view


def sync_analytics_state(
    db: Session,
    event: str,
    request: Optional[Request],
    user,
    email: Optional[str],
    page: Optional[str],
    session_id: Optional[str],
    details: Optional[dict[str, Any]],
):
    normalized_session_id = (session_id or "").strip()
    if not normalized_session_id:
        return

    details = details or {}
    now = utc_now()
    resolved_page = (page or details.get("page") or "").strip() or None
    path = details.get("path") if isinstance(details, dict) else None

    analytics_session = get_or_create_analytics_session(
        db=db,
        session_id=normalized_session_id,
        request=request,
        user=user,
        email=email,
        page=resolved_page,
        details=details,
        event=event,
        now=now,
    )

    analytics_session.total_events = (analytics_session.total_events or 0) + 1
    analytics_session.last_seen_at = now
    analytics_session.last_event = event
    reported_session_duration_seconds = duration_ms_to_seconds(details.get("session_duration_ms"))
    analytics_session.total_duration_seconds = max(
        analytics_session.total_duration_seconds or 0,
        reported_session_duration_seconds or 0,
        compute_elapsed_seconds(analytics_session.started_at, now),
    )

    if event == "remove_bg_completed":
        analytics_session.total_uploads = (analytics_session.total_uploads or 0) + 1

    if event in {"page_view", "auth_view", "dashboard_view", "activity_view", "result_view"} and resolved_page:
        ensure_page_view(
            db=db,
            analytics_session=analytics_session,
            user=user,
            page=resolved_page,
            now=now,
            path=path,
            event=event,
        )
        return

    if event == "session_ping":
        active_page_view = get_active_page_view(db, normalized_session_id, resolved_page)
        if active_page_view:
            active_page_view.last_seen_at = now
            page_duration_seconds = duration_ms_to_seconds(details.get("page_duration_ms"))
            active_page_view.duration_seconds = max(
                active_page_view.duration_seconds or 0,
                page_duration_seconds if page_duration_seconds is not None else compute_elapsed_seconds(active_page_view.entered_at, now),
            )
        return

    if event == "page_leave":
        active_page_view = get_active_page_view(db, normalized_session_id, resolved_page)
        if active_page_view:
            close_page_view(
                active_page_view,
                now,
                "page_leave",
                duration_ms_to_seconds(details.get("duration_ms") or details.get("page_duration_ms")),
            )
        analytics_session.exit_page = resolved_page or analytics_session.exit_page
        return

    if event == "session_end":
        active_page_view = get_active_page_view(db, normalized_session_id, resolved_page)
        if active_page_view:
            close_page_view(
                active_page_view,
                now,
                "session_end",
                duration_ms_to_seconds(details.get("page_duration_ms")),
            )
        analytics_session.is_active = False
        analytics_session.ended_at = now
        analytics_session.exit_page = resolved_page or analytics_session.current_page or analytics_session.exit_page
        return

    if resolved_page:
        active_page_view = get_active_page_view(db, normalized_session_id, resolved_page)
        if active_page_view:
            active_page_view.last_seen_at = now
            active_page_view.duration_seconds = max(
                active_page_view.duration_seconds or 0,
                compute_elapsed_seconds(active_page_view.entered_at, now),
            )


def enforce_auth_allowlist(
    db: Session,
    request: Request,
    email: Optional[str],
    session_id: Optional[str],
    failed_event: str,
    details: Optional[dict[str, Any]] = None,
):
    if is_auth_email_allowed(email):
        return

    safe_log_activity(
        db,
        failed_event,
        request=request,
        email=email,
        session_id=session_id,
        details={"reason": "access_restricted", **(details or {})},
    )
    raise HTTPException(status_code=403, detail=AUTH_ACCESS_RESTRICTED_DETAIL)


def serialize_activity(activity: ActivityEvent):
    return {
        "id": activity.id,
        "email": activity.email,
        "event": activity.event,
        "page": activity.page,
        "method": activity.method,
        "path": activity.path,
        "ip_address": activity.ip_address,
        "session_id": activity.session_id,
        "details": decode_details(activity.details),
        "created_at": serialize_timestamp(activity.created_at),
        "created_at_utc": serialize_timestamp_utc(activity.created_at),
    }


def get_effective_session_duration(session: AnalyticsSession) -> int:
    if session.is_active and session.started_at:
        return max(session.total_duration_seconds or 0, compute_elapsed_seconds(session.started_at, utc_now()))
    return session.total_duration_seconds or 0


def serialize_analytics_session(session: AnalyticsSession):
    return {
        "session_id": session.session_id,
        "email": session.email,
        "user_id": session.user_id,
        "landing_page": session.landing_page,
        "current_page": session.current_page,
        "exit_page": session.exit_page,
        "is_active": bool(session.is_active),
        "total_events": session.total_events or 0,
        "total_page_views": session.total_page_views or 0,
        "total_uploads": session.total_uploads or 0,
        "duration_seconds": get_effective_session_duration(session),
        "started_at": serialize_timestamp(session.started_at),
        "started_at_utc": serialize_timestamp_utc(session.started_at),
        "last_seen_at": serialize_timestamp(session.last_seen_at),
        "last_seen_at_utc": serialize_timestamp_utc(session.last_seen_at),
        "ended_at": serialize_timestamp(session.ended_at),
        "ended_at_utc": serialize_timestamp_utc(session.ended_at),
        "ip_address": session.ip_address,
    }


def build_unique_username(db: Session, preferred_username: Optional[str], email: str) -> str:
    source_value = preferred_username or email.split("@")[0]
    cleaned_value = re.sub(r"[^A-Za-z0-9_]", "", source_value).lower()
    base_username = cleaned_value[:20] or "editnestuser"

    candidate = base_username
    counter = 1

    while get_user_by_username(db, candidate):
        counter += 1
        suffix = str(counter)
        candidate = f"{base_username[: max(1, 20 - len(suffix))]}{suffix}"

    return candidate


def get_provider_label(auth_provider: Optional[str]) -> str:
    return SUPPORTED_AUTH_PROVIDERS.get(auth_provider or "", "Social")


def decode_image_data_url(data_url: str) -> bytes:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image data")

    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def load_normalized_image(contents: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(contents))
    image.verify()

    image = Image.open(io.BytesIO(contents))
    image = ImageOps.exif_transpose(image)

    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    return image


def encode_processing_image(image: Image.Image) -> bytes:
    buffer = io.BytesIO()

    if image.mode == "RGBA":
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format="JPEG", quality=95, subsampling=0)

    return buffer.getvalue()


def resize_for_processing(image: Image.Image, max_side: int) -> tuple[Image.Image, bool]:
    longest_edge = max(image.size)
    if not max_side or longest_edge <= max_side:
        return image, False

    scale = max_side / longest_edge
    resized_width = max(1, int(image.width * scale))
    resized_height = max(1, int(image.height * scale))
    resized_image = image.resize((resized_width, resized_height), RESAMPLING_LANCZOS)
    return resized_image, True


def normalize_remove_output(output: Any) -> Image.Image:
    if isinstance(output, Image.Image):
        return output.convert("RGBA")

    if isinstance(output, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(output))) as output_image:
            return output_image.convert("RGBA")

    if hasattr(output, "read"):
        current_position = None
        if hasattr(output, "tell"):
            try:
                current_position = output.tell()
            except Exception:
                current_position = None

        if hasattr(output, "seek"):
            try:
                output.seek(0)
            except Exception:
                pass

        stream_bytes = output.read()
        if not stream_bytes and hasattr(output, "getvalue"):
            stream_bytes = output.getvalue()

        if current_position is not None and hasattr(output, "seek"):
            try:
                output.seek(current_position)
            except Exception:
                pass

        if stream_bytes:
            with Image.open(io.BytesIO(stream_bytes)) as output_image:
                return output_image.convert("RGBA")

    if hasattr(output, "getvalue"):
        stream_bytes = output.getvalue()
        if stream_bytes:
            with Image.open(io.BytesIO(stream_bytes)) as output_image:
                return output_image.convert("RGBA")

    if hasattr(output, "shape"):
        return Image.fromarray(output).convert("RGBA")

    raise TypeError(f"Unsupported remove() output type: {type(output).__name__}")

# --- Auth Routes ---
@app.get("/auth/config")
def get_auth_config():
    return {
        "invite_only": AUTH_INVITE_ONLY,
        "message": AUTH_ACCESS_RESTRICTED_DETAIL if AUTH_INVITE_ONLY else None,
        "timezone": APP_TIMEZONE_NAME,
    }


@app.post("/auth/signup")
def signup(
    data: SignupRequest,
    request: Request,
    x_session_id: str = Header(None),
    db: Session = Depends(get_db)
):
    enforce_auth_allowlist(db, request, data.email, x_session_id, "signup_failed")

    if get_user_by_email(db, data.email):
        existing_user = get_user_by_email(db, data.email)
        safe_log_activity(
            db,
            "signup_failed",
            request=request,
            email=data.email,
            session_id=x_session_id,
            details={"reason": "email_exists", "username": data.username},
        )
        if existing_user and existing_user.auth_provider in SUPPORTED_AUTH_PROVIDERS:
            provider_label = get_provider_label(existing_user.auth_provider)
            raise HTTPException(status_code=400, detail=f"Use {provider_label} login for this account")
        raise HTTPException(status_code=400, detail="Email already registered")
    if get_user_by_username(db, data.username):
        safe_log_activity(
            db,
            "signup_failed",
            request=request,
            email=data.email,
            session_id=x_session_id,
            details={"reason": "username_taken", "username": data.username},
        )
        raise HTTPException(status_code=400, detail="Username already taken")
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not re.search(r"[A-Za-z]", data.password) or not re.search(r"[0-9]", data.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one letter and one number")
    try:
        user = create_user(db, data.email, data.username, data.password)
    except DuplicateUserError as exc:
        if exc.field == "email":
            raise HTTPException(status_code=400, detail="Email already registered")
        if exc.field == "username":
            raise HTTPException(status_code=400, detail="Username already taken")
        raise HTTPException(status_code=400, detail="Account already exists")
    safe_log_activity(
        db,
        "signup_success",
        request=request,
        user=user,
        session_id=x_session_id,
        details={"username": user.username},
    )
    token = create_access_token({"sub": user.email})
    return {
        "token": token,
        "username": user.username,
        "email": user.email,
        "is_admin": is_admin_email(user.email),
    }


@app.post("/auth/google")
def google_auth(
    data: FirebaseAuthRequest,
    request: Request,
    x_session_id: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token
    except ImportError:
        raise HTTPException(status_code=500, detail="Social authentication is not configured on the server")

    try:
        token_payload = google_id_token.verify_firebase_token(
            data.id_token,
            google_requests.Request(),
        )
    except Exception:
        safe_log_activity(
            db,
            "google_login_failed",
            request=request,
            session_id=x_session_id,
            details={"reason": "invalid_google_token"},
        )
        raise HTTPException(status_code=401, detail="Google authentication failed")

    provider = (data.provider or "google").strip().lower()
    if provider not in SUPPORTED_AUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported social login provider")

    email = token_payload.get("email")
    email_verified = token_payload.get("email_verified")

    if not email or not email_verified:
        safe_log_activity(
            db,
            "google_login_failed",
            request=request,
            email=email,
            session_id=x_session_id,
            details={"reason": "email_not_verified"},
        )
        raise HTTPException(status_code=401, detail="Google account email is not verified")

    enforce_auth_allowlist(
        db,
        request,
        email,
        x_session_id,
        f"{provider}_login_failed",
        details={"provider": provider},
    )

    user = get_user_by_email(db, email)
    if not user:
        username = build_unique_username(
            db,
            token_payload.get("name") or data.username,
            email,
        )
        generated_password = f"{provider}:{token_payload.get('user_id') or token_payload.get('sub') or email}"
        try:
            user = create_user(db, email, username, generated_password, auth_provider=provider)
        except DuplicateUserError:
            user = get_user_by_email(db, email)
            if not user:
                raise HTTPException(status_code=400, detail="Account already exists")
        safe_log_activity(
            db,
            f"{provider}_signup_success",
            request=request,
            user=user,
            session_id=x_session_id,
            details={"username": user.username},
        )
    elif not user.auth_provider:
        user = update_auth_provider(db, user, provider)

    safe_log_activity(
        db,
        f"{provider}_login_success",
        request=request,
        user=user,
        session_id=x_session_id,
        details={"username": user.username},
    )
    token = create_access_token({"sub": user.email})
    return {
        "token": token,
        "username": user.username,
        "email": user.email,
        "is_admin": is_admin_email(user.email),
    }

@app.post("/auth/login")
def login(
    data: LoginRequest,
    request: Request,
    x_session_id: str = Header(None),
    db: Session = Depends(get_db)
):
    enforce_auth_allowlist(db, request, data.email, x_session_id, "login_failed")

    existing_user = get_user_by_email(db, data.email)
    if not existing_user:
        safe_log_activity(
            db,
            "login_failed",
            request=request,
            email=data.email,
            session_id=x_session_id,
            details={"reason": "invalid_credentials"},
        )
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if existing_user.auth_provider in SUPPORTED_AUTH_PROVIDERS and not verify_password(data.password, existing_user.hashed_password):
        provider_label = get_provider_label(existing_user.auth_provider)
        safe_log_activity(
            db,
            "login_failed",
            request=request,
            email=data.email,
            session_id=x_session_id,
            details={"reason": f"use_{existing_user.auth_provider}_login"},
        )
        raise HTTPException(status_code=401, detail=f"Use {provider_label} login for this account")

    user = authenticate_user(db, data.email, data.password)
    if not user:
        safe_log_activity(
            db,
            "login_failed",
            request=request,
            email=data.email,
            session_id=x_session_id,
            details={"reason": "invalid_credentials"},
        )
        raise HTTPException(status_code=401, detail="Invalid email or password")
    safe_log_activity(
        db,
        "login_success",
        request=request,
        user=user,
        session_id=x_session_id,
        details={"username": user.username},
    )
    token = create_access_token({"sub": user.email})
    return {
        "token": token,
        "username": user.username,
        "email": user.email,
        "is_admin": is_admin_email(user.email),
    }

@app.get("/auth/me")
def get_me(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    if not is_auth_email_allowed(email):
        raise HTTPException(status_code=403, detail=AUTH_ACCESS_RESTRICTED_DETAIL)
    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "username": user.username,
        "email": user.email,
        "is_admin": is_admin_email(user.email),
    }


@app.post("/analytics/track")
def track_client_event(
    data: ClientEventRequest,
    request: Request,
    authorization: str = Header(None),
    x_session_id: str = Header(None),
    db: Session = Depends(get_db)
):
    if data.event not in ALLOWED_CLIENT_EVENTS:
        raise HTTPException(status_code=400, detail="Unsupported analytics event")
    user = resolve_user_from_auth(db, authorization)
    safe_log_activity(
        db,
        data.event,
        request=request,
        user=user,
        page=data.page,
        session_id=data.session_id or x_session_id,
        details=data.details,
    )
    return {"status": "tracked"}


@app.get("/analytics/summary")
def get_analytics_summary(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    require_admin_user(db, authorization)

    total_events = db.query(func.count(ActivityEvent.id)).scalar() or 0
    total_visitors = db.query(func.count(AnalyticsSession.session_id)).scalar() or 0
    logged_in_users = (
        db.query(func.count(func.distinct(AnalyticsSession.email)))
        .filter(AnalyticsSession.email.isnot(None))
        .scalar()
        or 0
    )
    total_uploads = (
        db.query(func.coalesce(func.sum(AnalyticsSession.total_uploads), 0))
        .scalar()
        or 0
    )
    active_cutoff = utc_now() - timedelta(minutes=2)
    active_sessions = (
        db.query(func.count(AnalyticsSession.session_id))
        .filter(
            AnalyticsSession.is_active.is_(True),
            AnalyticsSession.last_seen_at >= active_cutoff,
        )
        .scalar()
        or 0
    )
    average_session_seconds = (
        db.query(func.avg(AnalyticsSession.total_duration_seconds))
        .scalar()
        or 0
    )
    average_page_seconds = (
        db.query(func.avg(SessionPageView.duration_seconds))
        .filter(SessionPageView.duration_seconds.isnot(None))
        .scalar()
        or 0
    )

    action_counts = [
        {"event": event, "count": count}
        for event, count in (
            db.query(ActivityEvent.event, func.count(ActivityEvent.id))
            .filter(~ActivityEvent.event.in_(NOISY_ACTIVITY_EVENTS))
            .group_by(ActivityEvent.event)
            .order_by(func.count(ActivityEvent.id).desc())
            .all()
        )
    ]

    recent_users = [
        {
            "email": email,
            "event_count": event_count,
            "last_seen": serialize_timestamp(last_seen),
        }
        for email, event_count, last_seen in (
            db.query(
                ActivityEvent.email,
                func.count(ActivityEvent.id),
                func.max(ActivityEvent.created_at),
            )
            .filter(ActivityEvent.email.isnot(None))
            .group_by(ActivityEvent.email)
            .order_by(func.max(ActivityEvent.created_at).desc())
            .limit(10)
            .all()
        )
    ]

    recent_sessions = [
        serialize_analytics_session(session)
        for session in (
            db.query(AnalyticsSession)
            .order_by(AnalyticsSession.last_seen_at.desc(), AnalyticsSession.started_at.desc())
            .limit(12)
            .all()
        )
    ]

    recent_logins = [
        serialize_activity(event)
        for event in (
            db.query(ActivityEvent)
            .filter(
                ActivityEvent.email.isnot(None),
                ActivityEvent.event.in_(LOGIN_SUCCESS_EVENTS),
            )
            .order_by(ActivityEvent.created_at.desc(), ActivityEvent.id.desc())
            .limit(12)
            .all()
        )
    ]

    top_pages = [
        {
            "page": page,
            "views": views,
            "total_duration_seconds": int(total_duration_seconds or 0),
            "avg_duration_seconds": int(avg_duration_seconds or 0),
        }
        for page, views, total_duration_seconds, avg_duration_seconds in (
            db.query(
                SessionPageView.page,
                func.count(SessionPageView.id),
                func.coalesce(func.sum(SessionPageView.duration_seconds), 0),
                func.coalesce(func.avg(SessionPageView.duration_seconds), 0),
            )
            .group_by(SessionPageView.page)
            .order_by(func.count(SessionPageView.id).desc(), func.coalesce(func.sum(SessionPageView.duration_seconds), 0).desc())
            .limit(10)
            .all()
        )
    ]

    return {
        "totals": {
            "events": total_events,
            "visitors": total_visitors,
            "logged_in_users": logged_in_users,
            "uploads": total_uploads,
            "active_sessions": active_sessions,
            "avg_session_seconds": int(average_session_seconds or 0),
            "avg_page_seconds": int(average_page_seconds or 0),
        },
        "timezone": APP_TIMEZONE_NAME,
        "action_counts": action_counts,
        "recent_users": recent_users,
        "recent_logins": recent_logins,
        "recent_sessions": recent_sessions,
        "top_pages": top_pages,
    }


@app.get("/analytics/events")
def get_analytics_events(
    limit: int = Query(50, ge=1, le=200),
    include_noise: bool = Query(False),
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    require_admin_user(db, authorization)
    events_query = db.query(ActivityEvent)
    if not include_noise:
        events_query = events_query.filter(~ActivityEvent.event.in_(NOISY_ACTIVITY_EVENTS))
    events = (
        events_query
        .order_by(ActivityEvent.created_at.desc(), ActivityEvent.id.desc())
        .limit(limit)
        .all()
    )
    return {"events": [serialize_activity(event) for event in events]}

# --- Image Routes ---
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/remove-bg")
async def remove_background(
    request: Request,
    file: UploadFile = File(...),
    authorization: str = Header(None),
    x_api_key: str = Header(None),
    x_session_id: str = Header(None),
    db: Session = Depends(get_db)
):
    cleanup_expired_cache()

    # Allow machine-to-machine automation via API Key
    is_machine = bool(SECRET_API_KEY) and x_api_key == SECRET_API_KEY
    user = None

    if not is_machine:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Please login or provide a valid X-API-Key")
        token = authorization.split(" ")[1]
        email = decode_token(token)
        if not email:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        if not is_auth_email_allowed(email):
            raise HTTPException(status_code=403, detail=AUTH_ACCESS_RESTRICTED_DETAIL)
        user = get_user_by_email(db, email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

    resolved_content_type = resolve_upload_content_type(file)
    if not resolved_content_type:
        raise HTTPException(status_code=400, detail=f"Unsupported file type")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    try:
        normalized_image = load_normalized_image(contents)
        original_pixels = normalized_image.width * normalized_image.height
        use_fast_profile = (
            len(contents) >= REMBG_LARGE_UPLOAD_SIZE_MB * 1024 * 1024
            or original_pixels >= REMBG_LARGE_IMAGE_PIXEL_THRESHOLD
        )
        processing_max_side = (
            REMBG_LARGE_IMAGE_MAX_SIDE if use_fast_profile else REMBG_MAX_PROCESSING_SIDE
        )
        processing_image, was_resized = resize_for_processing(normalized_image, processing_max_side)
        processing_bytes = encode_processing_image(processing_image)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read as a valid JPG, PNG, or WebP image.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    processed_width, processed_height = processing_image.size
    processed_pixels = processed_width * processed_height
    use_alpha_matting = (
        REMBG_ALPHA_MATTING
        and not use_fast_profile
        and processed_pixels <= REMBG_ALPHA_MATTING_MAX_PIXELS
    )

    # Include model settings in the cache key so quality-profile changes produce fresh results.
    cache_key = "|".join(
        [
            REMBG_MODEL_NAME,
            str(REMBG_POST_PROCESS_MASK),
            str(use_alpha_matting),
            str(REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD),
            str(REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD),
            str(REMBG_ALPHA_MATTING_ERODE_SIZE),
            str(processing_max_side),
            str(use_fast_profile),
        ]
    ).encode("utf-8") + processing_bytes
    file_hash = hashlib.sha256(cache_key).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}_hq.png")
    
    if os.path.exists(cache_path):
        safe_log_activity(
            db,
            "remove_bg_completed",
            request=request,
            user=user,
            session_id=x_session_id,
            details={
                "file_name": file.filename,
                "file_size": len(contents),
                "content_type": resolved_content_type,
                "source": "cache",
                "is_machine": is_machine,
                "model": REMBG_MODEL_NAME,
                "width": processed_width,
                "height": processed_height,
                "was_resized": was_resized,
                "alpha_matting": use_alpha_matting,
                "fast_profile": use_fast_profile,
                "processing_max_side": processing_max_side,
            },
        )
        with open(cache_path, "rb") as f:
            return Response(
                content=f.read(),
                media_type="image/png",
                headers={"Content-Disposition": "attachment; filename=removed_bg.png"},
            )

    try:
        def process_image_pil():
            return remove(
                processing_image.copy(),
                session=model_session,
                post_process_mask=REMBG_POST_PROCESS_MASK,
                alpha_matting=use_alpha_matting,
                alpha_matting_foreground_threshold=REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD,
                alpha_matting_background_threshold=REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD,
                alpha_matting_erode_size=REMBG_ALPHA_MATTING_ERODE_SIZE,
            )

        def process_image_bytes():
            return remove(
                processing_bytes,
                session=model_session,
                post_process_mask=REMBG_POST_PROCESS_MASK,
                alpha_matting=use_alpha_matting,
                alpha_matting_foreground_threshold=REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD,
                alpha_matting_background_threshold=REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD,
                alpha_matting_erode_size=REMBG_ALPHA_MATTING_ERODE_SIZE,
                force_return_bytes=True,
            )

        loop = asyncio.get_running_loop()
        async with processing_semaphore:
            output_result = await loop.run_in_executor(thread_pool, process_image_pil)

            try:
                output_image = normalize_remove_output(output_result)
            except (UnidentifiedImageError, TypeError, ValueError) as exc:
                print(
                    "WARNING: primary rembg output decode failed "
                    f"(type={type(output_result).__name__}), retrying with bytes fallback: {exc}"
                )
                fallback_result = await loop.run_in_executor(thread_pool, process_image_bytes)
                output_image = normalize_remove_output(fallback_result)

        png_buffer = io.BytesIO()
        output_image.save(png_buffer, format="PNG", optimize=True, compress_level=6)
        png_data = png_buffer.getvalue()
        
        # Save the result to the cache
        with open(cache_path, "wb") as f:
            f.write(png_data)

        safe_log_activity(
            db,
            "remove_bg_completed",
            request=request,
            user=user,
            session_id=x_session_id,
            details={
                "file_name": file.filename,
                "file_size": len(contents),
                "content_type": resolved_content_type,
                "source": "processed",
                "is_machine": is_machine,
                "model": REMBG_MODEL_NAME,
                "post_process_mask": REMBG_POST_PROCESS_MASK,
                "alpha_matting": use_alpha_matting,
                "width": processed_width,
                "height": processed_height,
                "was_resized": was_resized,
                "fast_profile": use_fast_profile,
                "processing_max_side": processing_max_side,
            },
        )
            
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=removed_bg.png"},
        )
    except Exception as e:
        error_message = str(e)
        if isinstance(e, UnidentifiedImageError):
            error_message = "The processed image output could not be decoded."

        safe_log_activity(
            db,
            "remove_bg_failed",
            request=request,
            user=user,
            session_id=x_session_id,
            details={
                "file_name": file.filename,
                "file_size": len(contents),
                "content_type": resolved_content_type,
                "error": str(e),
                "is_machine": is_machine,
            },
        )
        raise HTTPException(status_code=500, detail=f"Processing failed: {error_message}")
    finally:
        # Force garbage collection to keep the memory footprint safely under 512MB
        gc.collect()

@app.post("/generate-bg")
async def generate_background_ai(
    request: Request,
    data: GenerateBgRequest,
    authorization: str = Header(None),
    x_session_id: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please login to use AI features")
    
    token = authorization.split(" ")[1]
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not is_auth_email_allowed(email):
        raise HTTPException(status_code=403, detail=AUTH_ACCESS_RESTRICTED_DETAIL)
    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=501, detail="OpenAI API key not configured in backend.")

    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    image_quality = os.environ.get("OPENAI_IMAGE_QUALITY", "medium")
    image_size = os.environ.get("OPENAI_IMAGE_SIZE", "1024x1024")

    try:
        from openai import OpenAI

        source_image_bytes = decode_image_data_url(data.image_base64)
        client = OpenAI(api_key=openai_api_key)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
            temp_image.write(source_image_bytes)
            temp_image_path = temp_image.name

        try:
            with open(temp_image_path, "rb") as image_file:
                output = client.images.edit(
                    model=image_model,
                    image=image_file,
                    prompt=data.prompt.strip(),
                    size=image_size,
                    quality=image_quality,
                )
        finally:
            try:
                os.remove(temp_image_path)
            except OSError:
                pass

        image_data = None
        if getattr(output, "data", None):
            first_item = output.data[0]
            image_data = getattr(first_item, "b64_json", None) or getattr(first_item, "url", None)

        if not image_data:
            raise RuntimeError("OpenAI image API returned no image output.")

        generated_url = image_data
        if not generated_url.startswith("http"):
            generated_url = f"data:image/png;base64,{generated_url}"

        safe_log_activity(
            db,
            "generate_bg_completed",
            request=request,
            user=user,
            page="result",
            session_id=x_session_id,
            details={
                "prompt": data.prompt[:200],
                "model": image_model,
                "quality": image_quality,
                "size": image_size,
            },
        )
        return {"generated_url": generated_url}
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenAI Python package is not installed. Run 'pip install openai'")
    except Exception as e:
        safe_log_activity(
            db,
            "generate_bg_failed",
            request=request,
            user=user,
            page="result",
            session_id=x_session_id,
            details={
                "prompt": data.prompt[:200],
                "model": image_model,
                "quality": image_quality,
                "size": image_size,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
