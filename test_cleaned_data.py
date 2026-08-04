import pandas as pd
from clean import clean_news

def test_clean_news_removes_duplicate_rows():
    data = {
        "topic": ["SPORTS", "SPORTS"],
        "link": ["https://example.com/1", "https://example.com/1"],
        "domain": ["example.com", "example.com"],
        "published_date": ["2020-08-15", "2020-08-15"],
        "title": ["First title", "First title"],
        "lang": ["en", "en"]
    }

    df = pd.DataFrame(data)

    cleaned = clean_news(df)

    assert len(cleaned) == 1
    assert cleaned.duplicated().sum() == 0

def test_clean_news_removes_rows_with_na():
    data = {
        "topic": ["SPORTS", None],
        "link": ["https://example.com/1", None],
        "domain": ["example.com", None],
        "published_date": ["2020-08-15", None],
        "title": ["First title", None],
        "lang": ["en", None]
    }
    df = pd.DataFrame(data)
    cleaned = clean_news(df)
    assert len(cleaned) ==1

def test_clean_news_strips_whitespace():
    data = {
        "topic" : [" SPORTS "],
        "link" : ["https://example.com/1"],
        "domain": ["example.com "],
        "published_date": ["2020-08-15 "],
        "title": ["First title "],
        "lang": [ "en"]

    }
    df = pd.DataFrame(data)
    cleaned = clean_news(df)
    assert cleaned.iloc[0]["topic"] == "sports"
    assert cleaned.iloc[0]["domain"] == "example.com"
    assert cleaned.iloc[0]["title"] == "First title"
    assert cleaned.iloc[0]["lang"] == "en"


def test_clean_news_strips_whitespace():
    data = {
        "topic" : [" SPORTS "],
        "link" : ["https://example.com/1"],
        "domain": ["example.com "],
        "published_date": ["2020-08-15 "],
        "title": ["First title "],
        "lang": [ "en"]

    }
    df = pd.DataFrame(data)
    cleaned = clean_news(df)
    assert cleaned.iloc[0]["topic"] == "sports"
    assert cleaned.iloc[0]["domain"] == "example.com"
    assert cleaned.iloc[0]["title"] == "First title"
    assert cleaned.iloc[0]["lang"] == "en"


def test_clean_news_returns_lowercase():
    data = {
        "topic" : ["SPORTS "],
        "link" : ["https://example.com/1"],
        "domain": ["example.com "],
        "published_date": ["2020-08-15"],
        "title": ["First title"],
        "lang": ["EN"]

    }
    df = pd.DataFrame(data)
    cleaned = clean_news(df)
    assert cleaned.iloc[0]["topic"] == "sports"
    assert cleaned.iloc[0]["domain"] == "example.com"
    assert cleaned.iloc[0]["title"] == "First title"
    assert cleaned.iloc[0]["lang"] == "en"


def test_clean_news_removes_empty_entries():
    data = {
        "topic" : ["SPORTS ", " "],
        "link" : ["https://example.com/1", " "],
        "domain": ["example.com ", " "],
        "published_date": ["2020-08-15", " "],
        "title": ["First title", " "],
        "lang": ["EN", " "]

    }
    df = pd.DataFrame(data)
    cleaned = clean_news(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["title"] == "First title"
    assert cleaned.iloc[0]["topic"] == "sports"
    assert cleaned.iloc[0]["lang"] == "en"
    assert cleaned.iloc[0]["published_date"] == pd.Timestamp("2020-08-15")
    assert cleaned.iloc[0]["link"] == "https://example.com/1"
    assert cleaned.iloc[0]["domain"] == "example.com"



def test_clean_news_converts_published_date_to_datetime():
    data = {
        "topic": ["SPORTS"],
        "link": ["https://example.com/1"],
        "domain": ["example.com"],
        "published_date": ["2020-08-15"],
        "title": ["First title"],
        "lang": ["EN"]
    }

    df = pd.DataFrame(data)

    cleaned = clean_news(df)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["published_date"])
    assert cleaned.iloc[0]["published_date"] == pd.Timestamp("2020-08-15")

    

def test_clean_news_keeps_only_english_rows():
    data = {
        "topic": ["SPORTS", "WORLD"],
        "link": ["https://example.com/1", "https://example.com/2"],
        "domain": ["example.com", "example.org"],
        "published_date": ["2020-08-15", "2020-08-16"],
        "title": ["English title", "French title"],
        "lang": ["EN", "FR"]
    }

    df = pd.DataFrame(data)

    cleaned = clean_news(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["lang"] == "en"
    assert cleaned.iloc[0]["title"] == "English title"