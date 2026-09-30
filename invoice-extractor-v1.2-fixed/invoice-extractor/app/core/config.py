"""Central settings. Every value can be overridden via environment / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "invoice-extractor"

    # Database: sqlite for zero-setup local dev, postgres in docker-compose.
    database_url: str = "sqlite:///./data/app.db"

    # OCR
    ocr_lang: str = "en"
    ocr_device: str = "cpu"          # PaddleOCR 3.x: "cpu" (GPU only helps batch training scale)
    model_dir: str = "./models"      # PaddleOCR model cache (pre-baked in Docker)

    # Confidence gates — starting points from research; CALIBRATE on your own
    # labeled data. Scores are engine-specific, not ground truth.
    auto_threshold: float = 0.92     # page avg >= this  -> auto-accept
    review_threshold: float = 0.75   # page avg below this -> quarantine
    field_confidence_floor: float = 0.80  # critical fields (total, date) must clear this

    # Safety caps at the API boundary
    max_file_mb: int = 25
    max_pages: int = 20

    # Optional Tesseract fallback pass on low-confidence regions
    enable_tesseract_fallback: bool = False

    # Aggressive binarization for severely degraded photos (off by default —
    # minimal preprocessing wins on ~95% of typical documents).
    preprocess_aggressive: bool = False

    # Ambiguous numeric dates ("05/06/2026"): True = day-first (05 Jun),
    # False = month-first (May 06). Either way an ambiguous date carries a
    # warning note on the document so a human verifies it.
    date_day_first: bool = True

    # API auth: set API_KEY to require an X-API-Key header on /api/v1/*.
    # Unset = open mode (local dev only); a warning is logged at boot.
    api_key: str = ""

    # Auth'd routes are rate-limited per client per 60s window (0 = disabled).
    rate_limit_per_minute: int = 120


settings = Settings()
