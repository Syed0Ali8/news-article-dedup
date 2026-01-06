from datetime import datetime
from news_article_dedup import normalization, parse_date


def test_normalization_basic():
    text = "META Releases contact lens as the ultimate interface for AI-powered XR computing!!"
    result = normalization(text)
    assert result == "meta releases contact lens as the ultimate interface for ai powered xr computing"

def test_normalization_numbers():
    text = "$250 million investment"
    result = normalization(text)
    assert result == "250m investment"

def test_normalization_empty():
    assert normalization(None) == ""
    assert normalization("") == ""

def test_parse_date_valid():
    item = {"date_published": "2025-12-26"}
    result = parse_date(item)
    assert result.year == 2025 and result.month == 12 and result.day == 26

def test_parse_date_missing():
    item = {}
    result = parse_date(item)
    assert result == datetime.max
