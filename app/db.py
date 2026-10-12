"""Prediction log stored with SQLAlchemy 2.0 (async).

SQLite locally, PostgreSQL in docker-compose; the code is identical because
only ``DDA_DATABASE_URL`` changes.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    filename: Mapped[str | None] = mapped_column(String(255))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    backend: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float] = mapped_column(Float)
    num_detections: Mapped[int] = mapped_column(Integer)
    # Full detections as JSON: simple, and works on both SQLite and Postgres.
    detections: Mapped[list] = mapped_column(JSON)


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables if missing. (A larger project would use Alembic migrations.)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
