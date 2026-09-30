"""Unit tests for geometric line-item clustering (stdlib only)."""
from app.parser.tables import cluster_rows, extract_line_items


def L(text, conf=0.95, x0=0, y0=0, x1=100, y1=20):
    return {"text": text, "confidence": conf,
            "bbox": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}, "page": 1}


def test_rows_cluster_by_vertical_overlap():
    lines = [
        L("Widget A", y0=100, y1=120, x0=80, x1=400),
        L("2", y0=102, y1=122, x0=640, x1=680),
        L("Widget B", y0=160, y1=180, x0=80, x1=400),
    ]
    rows = cluster_rows(lines)
    assert len(rows) == 2
    assert [l["text"] for l in rows[0]] == ["Widget A", "2"]  # sorted left->right


def test_line_items_parsed_from_table():
    lines = [
        L("Description", y0=500, y1=520, x0=80, x1=300),
        L("Qty", y0=500, y1=520, x0=640, x1=700),
        L("Amount", y0=500, y1=520, x0=1000, x1=1100),
        L("Widget A", y0=570, y1=590, x0=80, x1=300),
        L("2", y0=570, y1=590, x0=640, x1=700),
        L("1000.00", y0=570, y1=590, x0=1000, x1=1100),
        L("Grand Total 2030.00", y0=700, y1=720, x0=780, x1=1100),
    ]
    items = extract_line_items(lines)
    assert len(items) == 1
    assert items[0]["description"] == "Widget A"
    assert items[0]["qty"] == 2
    assert items[0]["amount"] == 1000.00


def test_no_header_no_items():
    lines = [L("Just some text", y0=100, y1=120)]
    assert extract_line_items(lines) == []


def test_rows_never_merge_across_pages():
    # Same y-coordinate on page 1 and page 2 must NOT become one row.
    lines = [
        L("Widget A", y0=100, y1=120, x0=80, x1=400),
        L("Widget B", y0=100, y1=120, x0=80, x1=400),
    ]
    lines[1]["page"] = 2
    rows = cluster_rows(lines)
    assert len(rows) == 2
    assert {r[0]["text"] for r in rows} == {"Widget A", "Widget B"}
