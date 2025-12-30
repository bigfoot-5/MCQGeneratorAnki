#!/usr/bin/env python3
"""
Test script for MCQ Generator addon.
This allows testing without loading Anki.

Usage:
    python3 test_main.py
"""

import os
import csv
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests

# --- Import from new modules ---
try:
    from config import (
        API_URL, AI_MODEL, TEMPERATURE, LLM_PROVIDER, API_KEY,
        NEWS_BASE_PARAMS, NEWS_API_URL
    )
    # Import the Graph
    from graph import create_quiz_graph
except ImportError as e:
    # Fallback if running from a different context where imports might fail
    print(f"[ERROR] Could not import modules: {e}")
    print("Ensure config.py, graph.py are in the same directory and dependencies (langchain, langgraph) are installed.")
    import sys
    sys.exit(1)

# Mock news fetcher extraction (could be moved to a util if reused)
try:
    from news_fetcher import fetch_news as newsapi_fetch_news
except ImportError:
    newsapi_fetch_news = None

if newsapi_fetch_news is None:
    def newsapi_fetch_news(api_key: str, params: Dict[str, str]) -> Dict:
        headers = {"Authorization": api_key}
        response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

def load_news_articles(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Fetch recent news articles to provide context for MCQ generation."""
    api_key = (os.getenv("NEWS_API_KEY") or "").strip()
    if not api_key:
        print("[WARNING] NEWS_API_KEY not configured; proceeding without article context.")
        return []

    params = dict(NEWS_BASE_PARAMS) if NEWS_BASE_PARAMS else {
        "q": "technology",
        "pageSize": 100,
        "sortBy": "publishedAt",
    }

    if NEWS_API_URL.endswith("everything"):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        params.setdefault("from", week_ago.isoformat(timespec="seconds"))
        params.setdefault("to", now.isoformat(timespec="seconds"))

    try:
        data = newsapi_fetch_news(api_key, params)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch news articles: {exc}")
        return []

    articles = data.get("articles", []) if isinstance(data, dict) else []
    if limit is not None:
        articles = articles[:limit]

    print(f"[INFO] Loaded {len(articles)} news articles for context.")
    return articles

def load_csv_data(csv_path: str) -> List[Dict[str, str]]:
    """Load word data from CSV file"""
    words = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Word', '').strip():
                words.append({
                    'Word': row['Word'].strip(),
                    'Back': row.get('Back', '').strip(),
                })
    return words

def generate_quiz_for_words(
    words: List[Dict[str, str]],
    articles: List[Dict[str, str]],
    test_count: int = 3
) -> List[Dict[str, str]]:
    
    if not words:
        print("[ERROR] No words provided")
        return []
    
    # Initialize the Graph
    print("[INFO] Compiling Quiz Graph...")
    app = create_quiz_graph()
    
    test_words = words[:test_count]
    results = []
    
    print(f"\n{'='*60}")
    print(f"Generating Quiz for {len(test_words)} words (LangGraph)")
    print(f"{'='*60}\n")
    
    for index, word_data in enumerate(test_words, 1):
        word = word_data['Word']
        if not word:
            continue
        definition = word_data.get('Back', '')
        
        print(f"\n[{index}/{len(test_words)}] Processing: {word}")
        
        # Choose article context
        article = articles[(index - 1) % len(articles)] if articles else None
        if article:
            print(f"  Article: {article.get('title', 'Untitled article')}")

        # Invoke Graph
        try:
            initial_state = {
                "word": word,
                "definition": definition,
                "article": article,
                "sentence": None,
                "sentence_masked": None,
                "synonyms": [],
                "explanation": None,
                "word_length": len(word),
                "first_letter": word[0] if word else "",
                "sentence_generated": False,
                "error": None
            }
            
            final_state = app.invoke(initial_state)
            
            if not final_state.get("sentence"):
                print(f"[SKIP] Failed to generate quiz data for {word}. Error: {final_state.get('error')}")
                continue

            result = {
                'Word': word,
                'SentenceMasked': final_state.get("sentence_masked"),
                'Synonyms': ", ".join(final_state.get("synonyms") or []),
                'Answer': final_state.get("sentence"),
                'WordLength': str(final_state.get("word_length")),
                'FirstLetter': final_state.get("first_letter"),
                'Explanation': final_state.get("explanation") or "",
                'ArticleTitle': article.get('title') if article else '',
                'ArticleSource': ((article or {}).get('source') or {}).get('name', '') if article else '',
            }
            
            results.append(result)
            
            print(f"  Sentence: {result['SentenceMasked']}")
            print(f"  Synonyms: {result['Synonyms']}")
            
        except Exception as e:
            print(f"[ERROR] Graph execution failed for {word}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def save_results_to_csv(results: List[Dict[str, str]], output_path: str):
    if not results:
        print("[WARNING] No results to save")
        return
    
    fieldnames = [
        'Word',
        'SentenceMasked',
        'Synonyms',
        'Answer',
        'WordLength',
        'FirstLetter',
        'Explanation',
        'ArticleTitle',
        'ArticleSource',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[SUCCESS] Saved {len(results)} results to {output_path}")

def main():
    print("="*60)
    print("MCQ Generator Test Script (Modular)")
    print("="*60)
    
    csv_path = os.path.join(os.path.dirname(__file__), 'word_cards.csv')
    output_path = os.path.join(os.path.dirname(__file__), 'test_results.csv')
    test_count = 3
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    print(f"\nConfiguration:")
    print(f"  CSV file: {csv_path}")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    if LLM_PROVIDER == "openai":
         print(f"  Model: {AI_MODEL}")
    
    words = load_csv_data(csv_path)
    print(f"[SUCCESS] Loaded {len(words)} words from CSV")
    
    if not words:
        return
    
    articles = load_news_articles(limit=len(words))
    
    results = generate_quiz_for_words(words, articles, test_count=test_count)
    
    if results:
        save_results_to_csv(results, output_path)
    else:
        print("\n[WARNING] No Quizzes were generated")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")
