#!/usr/bin/env python3
"""
Fetch and filter NewsAPI results.

Ensure you have python-dotenv installed (`pip install python-dotenv`) and that
your `.env` file contains: NEWS_API_KEY="48656fa80ccc4408ba518da13d980406"
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List

import requests
from dotenv import load_dotenv

NEWS_API_URL = "https://newsapi.org/v2/everything"
BASE_PARAMS = {
    "q": "technology",
    "pageSize": 100,
    "sortBy": "publishedAt",
    "sources": "techcrunch",
}


def fetch_news(api_key: str, params: Dict[str, str]) -> Dict:
    """Call NewsAPI and return the raw JSON response."""
    headers = {"Authorization": api_key}
    response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def filter_articles(
    data: Dict,
    predicate: Callable[[Dict], bool]
) -> Dict:
    """Return a copy of the response with articles filtered by predicate."""
    if "articles" not in data:
        return data
    filtered = [article for article in data["articles"] if predicate(article)]
    new_data = dict(data)
    new_data["articles"] = filtered
    new_data["totalResults"] = len(filtered)
    return new_data


def keyword_filter(keyword: str) -> Callable[[Dict], bool]:
    """Build a predicate that keeps articles containing keyword in title/description/content."""
    keyword_lower = keyword.lower()

    def _predicate(article: Dict) -> bool:
        for field in ("title", "description", "content"):
            text = article.get(field) or ""
            if keyword_lower in text.lower():
                return True
        return False

    return _predicate


def main(keyword: str = "") -> None:
    load_dotenv()
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("NEWS_API_KEY is not set in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    params = dict(BASE_PARAMS)
    params["from"] = week_ago.isoformat(timespec="seconds")
    params["to"] = now.isoformat(timespec="seconds")

    try:
        data = fetch_news(api_key, params)
    except requests.HTTPError as exc:
        print(f"NewsAPI request failed: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        sys.exit(1)

    if keyword:
        data = filter_articles(data, keyword_filter(keyword))

    articles = data.get("articles", [])
    if not articles:
        print("No technology headlines found in the past 24 hours.")
        return

    print(
        f"Technology headlines from the past week "
        f"({week_ago.date()} – {now.date()}) ({len(articles)} results):\n"
    )
    for idx, article in enumerate(articles, start=1):
        title = article.get("title", "Untitled")
        source = (article.get("source") or {}).get("name") or "Unknown source"
        url = article.get("url") or ""
        description = article.get("description", "No description")
        # content = article.get("content", "No content")
        print(f"{idx}. {title}")
        print(f"   Source: {source}")
        if url:
            print(f"   URL: {url}")
        print(f"   Description: {description}")
        print()


if __name__ == "__main__":
    # Optional command-line argument: keyword to filter on.
    # Example: python news_fetcher.py inflation
    kw = sys.argv[1] if len(sys.argv) > 1 else ""
    main(kw)