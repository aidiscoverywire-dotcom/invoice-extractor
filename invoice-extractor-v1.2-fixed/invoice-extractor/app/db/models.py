"""SQLAlchemy models: documents, review queue, append-only correction history."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Timezone-aware UTC (datetime.utcnow() is deprecated)."""
    return datetime.now(timezone.utc)


def _doc_id() -> str:
    return uuid.uuid4().hex[:12]


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_doc_id)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    pages: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="needs_review", index=True)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    fields_json: Mapped[dict] = mapped_column(JSON, default=dict)
    line_items_json: Mapped[list] = mapped_column(JSON, default=list)
    ocr_lines_json: Mapped[list] = mapped_column(JSON, default=list)
    notes_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ReviewItem(Base):
    """One uncertain field waiting for a human. Never auto-deleted."""
    __tablename__ = "review_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    field_name: Mapped[str] = mapped_column(String(64))
    current_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Correction(Base):
    """Append-only history: every human fix is logged, never overwritten."""
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_item_id: Mapped[int] = mapped_column(ForeignKey("review_items.id"), index=True)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str] = mapped_column(Text)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
