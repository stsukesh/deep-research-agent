"""
Database Connection
===================
WHAT: SQLAlchemy async engine and session factory for the app database.
HOW:  Uses create_async_engine with asyncpg driver. Sessions are created via
      async_sessionmaker and used as context managers for automatic cleanup.
WHY:  The app needs its own database tables (ResearchJob, Report) separate
      from LangGraph's checkpoint tables. They share the same PostgreSQL
      instance but have different schemas and connection libraries.

CONNECTION MANAGEMENT:
  - Engine: One per app lifetime (connection pool manager)
  - Session: One per request/operation (unit of work)
  - Uses async context managers for automatic commit/rollback/close

INTERVIEW Q&A:
  Q: Why use async SQLAlchemy with asyncpg?
  A: FastAPI is async. Using sync database calls would block the event loop,
     killing concurrency. asyncpg is the fastest async PostgreSQL driver.
     SQLAlchemy 2.0+ has native async support via create_async_engine.
     
  Q: Why separate from the LangGraph checkpointer connection?
  A: LangGraph's checkpointer uses psycopg3 (not asyncpg) because that's what
     langgraph-checkpoint-postgres requires internally. Our app uses asyncpg via
     SQLAlchemy for its own tables. Same database, different drivers — this is
     normal in production systems where different libraries have different
     driver requirements.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import get_settings

# Module-level engine and session factory (initialized on first import)
_engine = None
_async_session_factory = None


def get_engine():
    """Get or create the async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.APP_ENV == "development",  # SQL logging in dev
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory():
    """Get or create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
        )
    return _async_session_factory


async def get_db_session() -> AsyncSession:
    """
    FastAPI dependency — yields a database session.
    Usage in routes:
        async def my_endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables on startup."""
    from app.database.models import Base
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close the engine on shutdown."""
    global _engine, _async_session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
    _async_session_factory = None
