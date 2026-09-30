"""Pytest bootstrap for the invoice-extractor test suite.

Sets DATABASE_URL before ANY test module is imported. This matters because
app/db/database.py binds its engine at import time from settings: whichever
test module imports the app first decides which database the whole suite
uses. Relying on filename sort order for that (test_pipeline_smoke.py
happened to import first) broke the moment another DB-touching test file
was added. Setting it here makes the choice explicit and order-independent.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/smoke_test.db")
if os.path.exists("/tmp/smoke_test.db"):
    os.remove("/tmp/smoke_test.db")
