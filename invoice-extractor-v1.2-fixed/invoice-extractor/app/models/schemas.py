"""Pydantic models for API requests/responses."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """Axis-aligned bounding box in image pixels: [x_min, y_min, x_max, y_max]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class OcrLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[BBox] = None
    page: int = 1


class FieldValue(BaseModel):
    value: Optional[str | float] = None
    confidence: float = 0.0
    source: str = ""          # e.g. "regex:label_ranked", "arithmetic:validated"


class LineItem(BaseModel):
    description: str = ""
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    confidence: float = 0.0


class InvoiceFields(BaseModel):
    vendor: FieldValue = Field(default_factory=FieldValue)
    invoice_number: FieldValue = Field(default_factory=FieldValue)
    invoice_date: FieldValue = Field(default_factory=FieldValue)   # ISO YYYY-MM-DD
    currency: FieldValue = Field(default_factory=FieldValue)
    subtotal: FieldValue = Field(default_factory=FieldValue)
    tax: FieldValue = Field(default_factory=FieldValue)
    total: FieldValue = Field(default_factory=FieldValue)
    arithmetic_ok: bool = False   # subtotal + tax == total within 1 cent


class ExtractionResult(BaseModel):
    document_id: str
    filename: str
    pages: int
    status: str                      # auto | needs_review | quarantined | verified
    avg_confidence: float
    fields: InvoiceFields
    line_items: list[LineItem] = []
    low_confidence_fields: list[str] = []
    notes: list[str] = []


class DocumentSummary(BaseModel):
    id: str
    filename: str
    pages: int
    status: str
    avg_confidence: float
    total: Optional[float] = None
    created_at: datetime


class ReviewItemOut(BaseModel):
    id: int
    document_id: str
    field_name: str
    current_value: Optional[str] = None
    reason: str
    bbox: Optional[BBox] = None
    status: str
    created_at: datetime


class CorrectionIn(BaseModel):
    corrected_value: str = Field(min_length=1, max_length=500)
    note: str = ""

    @field_validator("corrected_value")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        # min_length=1 alone lets a whitespace-only " " through.
        v = v.strip()
        if not v:
            raise ValueError("corrected_value must not be blank")
        return v
