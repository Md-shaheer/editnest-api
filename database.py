import os

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

load_project_env()

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./editnest.db"
SQLITE_BUSY_TIMEOUT_MS = max(5000, int(os.environ.get("SQLITE_BUSY_TIMEOUT_MS", "15000")))
DATABASE_POOL_SIZE = max(1, int(os.environ.get("DATABASE_POOL_SIZE", "5")))
DATABASE_MAX_OVERFLOW = max(0, int(os.environ.get("DATABASE_MAX_OVERFLOW", "10")))
DATABASE_POOL_TIMEOUT = max(5, int(os.environ.get("DATABASE_POOL_TIMEOUT", "30")))
DATABASE_POOL_RECYCLE_SECONDS = max(60, int(os.environ.get("DATABASE_POOL_RECYCLE_SECONDS", "1800")))


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


def is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite")


def clean_email_value(email: str | None) -> str:
    return (email or "").strip().lower()


def clean_username_value(username: str | None) -> str:
    return " ".join((username or "").strip().split())


def normalize_username_key(username: str | None) -> str:
    return clean_username_value(username).lower()


def clean_auth_provider(auth_provider: str | None) -> str:
    return (auth_provider or "local").strip().lower() or "local"


SQLALCHEMY_DATABASE_URL = normalize_database_url(
    os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
)

engine_kwargs = {
    "pool_pre_ping": True,
    "future": True,
}
if is_sqlite_url(SQLALCHEMY_DATABASE_URL):
    engine_kwargs["connect_args"] = {
        "check_same_thread": False,
        "timeout": max(5, SQLITE_BUSY_TIMEOUT_MS // 1000),
    }
else:
    engine_kwargs.update(
        {
            "pool_size": DATABASE_POOL_SIZE,
            "max_overflow": DATABASE_MAX_OVERFLOW,
            "pool_timeout": DATABASE_POOL_TIMEOUT,
            "pool_recycle": DATABASE_POOL_RECYCLE_SECONDS,
        }
    )

engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_kwargs)


if is_sqlite_url(SQLALCHEMY_DATABASE_URL):
    @event.listens_for(engine, "connect")
    def configure_sqlite_connection(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        pragma_statements = [
            "PRAGMA foreign_keys = ON",
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
            f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}",
            "PRAGMA temp_store = MEMORY",
        ]
        for statement in pragma_statements:
            try:
                cursor.execute(statement)
            except Exception:
                continue
        cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_auth_provider_active", "auth_provider", "is_active"),
    )

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(320), unique=True, index=True, nullable=False)
    email_normalized = Column(String(320), unique=True, index=True, nullable=False)
    username = Column(String(64), unique=True, index=True, nullable=False)
    username_normalized = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    auth_provider = Column(String(32), nullable=False, default="local", index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ActivityEvent(Base):
    __tablename__ = "activity_events"
    __table_args__ = (
        Index("ix_activity_events_event_created_at", "event", "created_at"),
        Index("ix_activity_events_email_created_at", "email", "created_at"),
        Index("ix_activity_events_session_created_at", "session_id", "created_at"),
        Index("ix_activity_events_user_created_at", "user_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    email = Column(String(320), nullable=True, index=True)
    event = Column(String(100), index=True, nullable=False)
    page = Column(String(100), nullable=True, index=True)
    method = Column(String(16), nullable=True)
    path = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(128), nullable=True, index=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


class AnalyticsSession(Base):
    __tablename__ = "analytics_sessions"
    __table_args__ = (
        Index("ix_analytics_sessions_active_last_seen", "is_active", "last_seen_at"),
        Index("ix_analytics_sessions_email_started_at", "email", "started_at"),
    )

    session_id = Column(String(128), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    email = Column(String(320), nullable=True, index=True)
    is_authenticated = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    landing_page = Column(String(100), nullable=True, index=True)
    current_page = Column(String(100), nullable=True, index=True)
    exit_page = Column(String(100), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(Text, nullable=True)
    referrer = Column(String(500), nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    ended_at = Column(DateTime(timezone=True), nullable=True, index=True)
    total_duration_seconds = Column(Integer, default=0, nullable=False)
    total_events = Column(Integer, default=0, nullable=False)
    total_page_views = Column(Integer, default=0, nullable=False)
    total_uploads = Column(Integer, default=0, nullable=False)
    last_event = Column(String(100), nullable=True)


class SessionPageView(Base):
    __tablename__ = "session_page_views"
    __table_args__ = (
        Index("ix_session_page_views_session_active", "session_id", "is_active"),
        Index("ix_session_page_views_page_entered_at", "page", "entered_at"),
        Index("ix_session_page_views_email_entered_at", "email", "entered_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), ForeignKey("analytics_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    email = Column(String(320), nullable=True, index=True)
    page = Column(String(100), nullable=False, index=True)
    path = Column(String(255), nullable=True)
    entry_event = Column(String(100), nullable=True)
    exit_event = Column(String(100), nullable=True)
    sequence_number = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    entered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    left_at = Column(DateTime(timezone=True), nullable=True, index=True)
    duration_seconds = Column(Integer, default=0, nullable=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
    ensure_user_columns()
    ensure_activity_event_indexes()
    ensure_analytics_session_indexes()
    ensure_session_page_view_indexes()


def add_missing_column(connection, column_names: set[str], column_name: str, sql: str):
    if column_name in column_names:
        return
    connection.execute(text(sql))
    column_names.add(column_name)


def ensure_user_columns():
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    column_names = {column["name"] for column in inspector.get_columns("users")}
    dialect = engine.dialect.name
    created_at_sql = (
        "ALTER TABLE users ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
        if dialect == "postgresql"
        else "ALTER TABLE users ADD COLUMN created_at DATETIME"
    )
    updated_at_sql = (
        "ALTER TABLE users ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
        if dialect == "postgresql"
        else "ALTER TABLE users ADD COLUMN updated_at DATETIME"
    )

    with engine.begin() as connection:
        add_missing_column(connection, column_names, "auth_provider", "ALTER TABLE users ADD COLUMN auth_provider VARCHAR(32)")
        add_missing_column(connection, column_names, "email_normalized", "ALTER TABLE users ADD COLUMN email_normalized VARCHAR(320)")
        add_missing_column(connection, column_names, "username_normalized", "ALTER TABLE users ADD COLUMN username_normalized VARCHAR(64)")
        add_missing_column(connection, column_names, "created_at", created_at_sql)
        add_missing_column(connection, column_names, "updated_at", updated_at_sql)

        users = connection.execute(
            text(
                "SELECT id, email, username, auth_provider, email_normalized, username_normalized "
                "FROM users"
            )
        ).mappings().all()

        for user in users:
            cleaned_email = clean_email_value(user["email"]) or None
            cleaned_username = clean_username_value(user["username"]) or None
            normalized_username = normalize_username_key(user["username"]) or None
            auth_provider = clean_auth_provider(user["auth_provider"])

            connection.execute(
                text(
                    "UPDATE users "
                    "SET email = COALESCE(:email, email), "
                    "email_normalized = :email_normalized, "
                    "username = COALESCE(:username, username), "
                    "username_normalized = :username_normalized, "
                    "auth_provider = :auth_provider, "
                    "created_at = COALESCE(created_at, CURRENT_TIMESTAMP), "
                    "updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP) "
                    "WHERE id = :id"
                ),
                {
                    "id": user["id"],
                    "email": cleaned_email,
                    "email_normalized": cleaned_email,
                    "username": cleaned_username,
                    "username_normalized": normalized_username,
                    "auth_provider": auth_provider,
                },
            )

        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email_normalized "
                "ON users (email_normalized)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username_normalized "
                "ON users (username_normalized)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_users_auth_provider_active "
                "ON users (auth_provider, is_active)"
            )
        )


def ensure_activity_event_indexes():
    inspector = inspect(engine)
    if "activity_events" not in inspector.get_table_names():
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_activity_events_event_created_at "
                "ON activity_events (event, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_activity_events_email_created_at "
                "ON activity_events (email, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_activity_events_session_created_at "
                "ON activity_events (session_id, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_activity_events_user_created_at "
                "ON activity_events (user_id, created_at)"
            )
        )


def ensure_analytics_session_indexes():
    inspector = inspect(engine)
    if "analytics_sessions" not in inspector.get_table_names():
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_analytics_sessions_active_last_seen "
                "ON analytics_sessions (is_active, last_seen_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_analytics_sessions_email_started_at "
                "ON analytics_sessions (email, started_at)"
            )
        )


def ensure_session_page_view_indexes():
    inspector = inspect(engine)
    if "session_page_views" not in inspector.get_table_names():
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_session_page_views_session_active "
                "ON session_page_views (session_id, is_active)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_session_page_views_page_entered_at "
                "ON session_page_views (page, entered_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_session_page_views_email_entered_at "
                "ON session_page_views (email, entered_at)"
            )
        )
