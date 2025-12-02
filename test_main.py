#!/usr/bin/env python3
"""
Test script for MCQ Generator addon.
This allows testing without loading Anki.

Usage:
    python3 test_main.py
"""

import random
import sys
import os
import csv
import json
import time
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

# Mock aqt components
class MockAQT:
    def showInfo(self, msg):
        print(f"[INFO] {msg}")
    
    def showWarning(self, msg):
        print(f"[WARNING] {msg}")
    
    def showError(self, msg):
        print(f"[ERROR] {msg}")

# Set up vendor and user_files directories (same as main.py)
_addon_dir = os.path.dirname(os.path.abspath(__file__))
_vendor_dir = os.path.join(_addon_dir, 'vendor')
_user_files_dir = os.path.join(_addon_dir, 'user_files')
if os.path.exists(_vendor_dir):
    if _vendor_dir not in sys.path:
        sys.path.insert(0, _vendor_dir)

# Mock sys.modules before importing
sys.modules['aqt'] = type(sys)('aqt')
sys.modules['aqt'].showInfo = MockAQT().showInfo
sys.modules['aqt.utils'] = type(sys)('aqt.utils')
sys.modules['aqt.utils'].showInfo = MockAQT().showInfo
sys.modules['aqt.qt'] = type(sys)('aqt.qt')
sys.modules['aqt'].gui_hooks = type(sys)('gui_hooks')

# Import the actual functions from main.py
import requests
from dotenv import load_dotenv

try:
    from news_fetcher import BASE_PARAMS as NEWS_BASE_PARAMS, NEWS_API_URL, fetch_news as newsapi_fetch_news
except ImportError:
    NEWS_BASE_PARAMS = None
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    newsapi_fetch_news = None

if newsapi_fetch_news is None:

    def newsapi_fetch_news(api_key: str, params: Dict[str, str]) -> Dict:
        headers = {"Authorization": api_key}
        response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

# Load environment variables using explicit paths to match main.py behavior
load_dotenv(os.path.join(_addon_dir, ".env"), override=False)
if os.path.isdir(_user_files_dir):
    load_dotenv(os.path.join(_user_files_dir, ".env"), override=False)

# Mirror configuration resolution from main.py
DEFAULT_PROMPT = (
    """
   You are creating a two-sentence news-style output for an English-learner exercise.

Inputs (do not alter):
- Target word or phrase: {word}
- Target word inflection: {inflection}
- Definition to use as guidance: {definition}
- News article title: {article_title}
- News article summary: {article_summary}

Output requirements (strict — follow exactly):
1. Produce exactly **two sentences** and nothing else (no commentary, no bullets, no metadata).
2. Sentence 1 must be the **unchanged** news article excerpt provided in {article_summary}. Do **not** modify it in any way and do not add blanks to it.
3. Sentence 2 must be a short, natural-sounding **continuation or extension** of the article excerpt (an extra sentence that could plausibly follow the excerpt).
4. In Sentence 2, **include the target word or phrase naturally** in the sentence. The word/phrase must appear **only** in Sentence 2 (never in Sentence 1), and the inflection must be exactly as provided in {inflection}. Write the complete sentence with the actual word/phrase, not a blank.
5. Use the provided {definition} to shape the meaning of Sentence 2 so the target word/phrase is contextually constrained and the definition would make it answerable. Do not restate the definition verbatim.
6. Keep both sentences short and clear (each ≤ 30 words). Prefer neutral, factual tone consistent with news.
7. Return only the two sentences (Sentence 1 + Sentence 2). No extra whitespace, no headings, no punctuation outside the sentences.

If you cannot produce a natural continuation that fits the definition, produce a plausible imaginary continuation sentence that follows the excerpt and still conforms to all rules above.

"""
)

DISTRACTOR_PROMPT = (
    """
You are generating vocabulary distractors for an English multiple-choice question.

Target sentence (with blank): {sentence_with_blank}
Correct answer (target word): {word}
Definition/context: {definition}
Number of distractors needed: {num_distractors}

Requirements:
1. Return exactly {num_distractors} unique English words or short phrases that are **plausible but incorrect** answers when inserted into the blank in the target sentence.
2. Each distractor must:
   - Fit grammatically and syntactically in the sentence (so students might think it's correct)
   - Be in the same part of speech as the target word
   - Be at a similar difficulty/vocabulary level as the target word
   - Be semantically wrong or inappropriate for the sentence context (this is what makes it a good distractor)
3. The goal is to "trick" students into thinking the distractor could be correct by making it sound natural in the sentence, but it must still be wrong.
4. Distractors must NOT be:
   - The target word itself
   - Extremely obscure/rare words
   - Simple morphological variants of the target word (e.g., plural, tense change, suffix addition)
5. Output must be valid JSON: a list of strings, e.g. ["option1", "option2", "option3"]. No commentary, explanations, or extra text.
"""
)

EXPLANATION_PROMPT = (
    """
You are generating an explanation for a vocabulary multiple-choice question.

Target sentence (with blank): {sentence_with_blank}
Correct answer: {correct_word}
Distractors: {distractors}
Definition/context: {definition}

Requirements:
1. Write a clear, educational explanation (2-4 sentences) that:
   - Explains why "{correct_word}" is the correct answer in this sentence context
   - Briefly explains why each distractor is incorrect (mention each distractor by name)
   - Uses the definition/context provided to support your explanation
2. The explanation should be helpful for English learners, clear and concise.
3. Format: Write in plain text, no bullet points or numbered lists. Use natural, flowing sentences.
4. Output only the explanation text, no headings, no metadata, no extra formatting.
"""
)

LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

# OpenAI settings
API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
API_URL = (os.getenv("OPENAI_API_URL") or "https://api.openai.com/v1/chat/completions").strip()
AI_MODEL = (os.getenv("OPENAI_MODEL") or "").strip()

# Ollama (local SLM) settings
OLLAMA_URL = (os.getenv("OLLAMA_URL") or "http://localhost:11434/api/chat").strip()
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "gemma3:1b").strip()

# Prompt + sampling settings
PROMPT_TEMPLATE = (
    os.getenv("OPENAI_PROMPT_TEMPLATE")
    or os.getenv("OLLAMA_PROMPT_TEMPLATE")
    or DEFAULT_PROMPT
)
try:
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))
except (TypeError, ValueError):
    TEMPERATURE = 1.5


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
    except requests.HTTPError as exc:
        print(f"[ERROR] Failed to fetch news articles: {exc}")
        if exc.response is not None:
            print(exc.response.text)
        return []
    except Exception as exc:
        print(f"[ERROR] Unexpected error fetching news articles: {exc}")
        return []

    articles = data.get("articles", []) if isinstance(data, dict) else []
    if limit is not None:
        articles = articles[:limit]

    print(f"[INFO] Loaded {len(articles)} news articles for context.")
    return articles


def show_info(msg):
    """Mock showInfo function"""
    print(f"[INFO] {msg}")


def non_blocking_wait(seconds):
    """Mock non-blocking wait - just print"""
    print(f"[WAIT] Waiting {seconds} seconds...")


# Core API Call with Retry Logic (from main.py)
def _build_prompt(
    word: str,
    definition: Optional[str],
    article: Optional[Dict[str, str]],
) -> str:
    article = article or {}
    article_title = (article.get("title") or "No title provided").strip()
    article_summary = (
        article.get("description") or article.get("summary") or "No summary provided"
    ).strip()
    clean_definition = (definition or "No definition provided").strip().replace("\n", " ")
    inflection = "plural"
    return PROMPT_TEMPLATE.format(
        word=word,
        inflection=inflection,
        definition=clean_definition,
        article_title=article_title,
        article_summary=article_summary,
    )


def generate_sentence_for_word(
    word: str,
    definition: Optional[str] = None,
    article: Optional[Dict[str, str]] = None,
    max_retries: int = 5
) -> Optional[str]:
    """
    Call OpenAI API to generate a sentence with a blank for the given word/phrase.
    Implements retry logic on HTTP 429 errors.
    Returns the sentence as plain text.
    """
    import time

    prompt = _build_prompt(word, definition, article)

    if LLM_PROVIDER == "ollama":
        if not OLLAMA_MODEL:
            print("[ERROR] Ollama model is not configured. Set OLLAMA_MODEL or update your .env file.")
            return None

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": TEMPERATURE},
            "stream": False,
        }

        try:
            article_title = (article or {}).get("title") if article else None
            print(
                f"[SLM] Generating sentence for '{word}' using Ollama model '{OLLAMA_MODEL}'"
                + (f" with article '{article_title}'." if article_title else ".")
            )
            res = requests.post(OLLAMA_URL, json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            # Depending on Ollama version, `message` may be nested differently.
            content = ""
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    content = data["message"].get("content", "")
                elif "content" in data:
                    content = data["content"]
            content = (content or "").strip()
            if not content:
                print("[ERROR] Empty response from Ollama.")
                return None
            print(f"[SUCCESS] Generated (Ollama): {content}")
            return content
        except requests.exceptions.RequestException as e:
            show_info(f"HTTP error calling Ollama: {e}")
        except Exception as e:
            show_info(f"Error processing Ollama response: {e}")
        return None

    # Default to OpenAI
    if not API_KEY:
        print("[ERROR] OpenAI API key is not configured. Set it via .env or user_files/api_key.txt.")
        return None
    if not AI_MODEL:
        print("[ERROR] OpenAI model is not configured. Set it via .env or user_files/model.txt.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE
    }

    retries = 0
    while retries <= max_retries:
        try:
            article_title = (article or {}).get("title") if article else None
            context_msg = f" with article '{article_title}'" if article_title else ""
            print(f"[API] Generating sentence for '{word}'{context_msg} (attempt {retries + 1})...")
            res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if res.status_code == 429:
                wait_time = 30
                show_info(f"Rate limit reached. Retrying in {wait_time} seconds...")
                non_blocking_wait(wait_time)
                retries += 1
                continue
            res.raise_for_status()
            data = res.json()
            content = data["choices"][0]["message"]["content"].strip()
            print(f"[SUCCESS] Generated (OpenAI): {content}")
            return content
        except requests.exceptions.RequestException as e:
            show_info(f"HTTP error: {e}")
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(3)
        except Exception as e:
            show_info(f"Error processing response: {e}")
            raise

    show_info("Maximum retries exceeded. Please try again later.")
    return None


def generate_distractors_with_llm(
    word: str,
    sentence_with_blank: str,
    definition: Optional[str] = None,
    num_distractors: int = 3,
    article: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Ask the configured LLM to propose distractor options that are plausible in the sentence context but incorrect.
    Falls back to an empty list on any failure so the caller can handle retries/fallbacks.
    """
    prompt = DISTRACTOR_PROMPT.format(
        word=word,
        sentence_with_blank=sentence_with_blank,
        definition=(definition or "No definition provided").strip().replace("\n", " "),
        num_distractors=num_distractors,
    )

    def _parse_response(content: str) -> List[str]:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                cleaned = []
                for item in data:
                    if isinstance(item, str):
                        candidate = item.strip()
                        if candidate and candidate.lower() != word.lower():
                            cleaned.append(candidate)
                return cleaned[:num_distractors]
        except json.JSONDecodeError:
            pass
        # Fallback: split lines
        parts = [p.strip("- ").strip() for p in content.splitlines()]
        cleaned = [
            p for p in parts if p and p.lower() != word.lower()
        ]
        return cleaned[:num_distractors]

    if LLM_PROVIDER == "ollama":
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": TEMPERATURE},
            "stream": False,
        }
        try:
            print(f"[SLM] Generating distractors for '{word}' using Ollama.")
            res = requests.post(OLLAMA_URL, json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            content = ""
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    content = data["message"].get("content", "")
                elif "content" in data:
                    content = data["content"]
            content = (content or "").strip()
            if not content:
                print("[ERROR] Empty distractor response from Ollama.")
                return []
            return _parse_response(content)
        except Exception as exc:
            print(f"[ERROR] Failed to generate distractors via Ollama: {exc}")
            return []

    if not API_KEY or not AI_MODEL:
        print("[ERROR] Cannot generate distractors: OpenAI credentials incomplete.")
        return []

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    }

    try:
        print(f"[API] Generating distractors for '{word}'...")
        res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"].strip()
        return _parse_response(content)
    except Exception as exc:
        print(f"[ERROR] Failed to generate distractors via API: {exc}")
        return []


def generate_explanation_with_llm(
    word: str,
    sentence_with_blank: str,
    distractors: List[str],
    definition: Optional[str] = None,
) -> Optional[str]:
    """
    Ask the configured LLM to generate an explanation for why the target word is correct
    and why each distractor is wrong.
    Returns None on failure.
    """
    distractors_str = ", ".join(distractors)
    prompt = EXPLANATION_PROMPT.format(
        sentence_with_blank=sentence_with_blank,
        correct_word=word,
        distractors=distractors_str,
        definition=(definition or "No definition provided").strip().replace("\n", " "),
    )

    if LLM_PROVIDER == "ollama":
        if not OLLAMA_MODEL:
            print("[ERROR] Ollama model is not configured for explanation generation.")
            return None
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": TEMPERATURE},
            "stream": False,
        }
        try:
            print(f"[SLM] Generating explanation for '{word}' using Ollama.")
            res = requests.post(OLLAMA_URL, json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            content = ""
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    content = data["message"].get("content", "")
                elif "content" in data:
                    content = data["content"]
            content = (content or "").strip()
            if not content:
                print("[ERROR] Empty explanation response from Ollama.")
                return None
            return content
        except Exception as exc:
            print(f"[ERROR] Failed to generate explanation via Ollama: {exc}")
            return None

    if not API_KEY or not AI_MODEL:
        print("[ERROR] Cannot generate explanation: OpenAI credentials incomplete.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    }

    try:
        print(f"[API] Generating explanation for '{word}'...")
        res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content
    except Exception as exc:
        print(f"[ERROR] Failed to generate explanation via API: {exc}")
        return None


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
                    'SentenceBlank': row.get('SentenceBlank', '').strip(),
                    'OptionA': row.get('OptionA', '').strip(),
                    'OptionB': row.get('OptionB', '').strip(),
                    'OptionC': row.get('OptionC', '').strip(),
                    'OptionD': row.get('OptionD', '').strip(),
                    'Answer': row.get('Answer', '').strip(),
                })
    return words


def generate_mcq_for_words(
    words: List[Dict[str, str]],
    articles: List[Dict[str, str]],
    test_count: int = 3
) -> List[Dict[str, str]]:
    """
    Generate MCQs for given words using local distractors.
    Similar to generate_mcq_for_cards in main.py but works with CSV data.
    """
    if not words:
        print("[ERROR] No words provided")
        return []
    
    if len(words) < 4:
        print("[ERROR] Need at least 4 words for MCQ generation.")
        return []
    
    # Test with first N words
    test_words = words[:test_count]
    results = []
    
    print(f"\n{'='*60}")
    print(f"Generating MCQs for {len(test_words)} words")
    print(f"{'='*60}\n")
    
    for index, word_data in enumerate(test_words, 1):
        word = word_data['Word']
        if not word:
            continue
        definition = word_data.get('Back', '')
        
        print(f"\n[{index}/{len(test_words)}] Processing: {word}")
        
        # Choose article context (cycle if fewer articles than words)
        article = articles[(index - 1) % len(articles)] if articles else None
        if article:
            print(f"  Article: {article.get('title', 'Untitled article')}")

        # Generate sentence first
        try:
            sentence = generate_sentence_for_word(word, definition=definition, article=article)
            if sentence is None:
                print(f"[SKIP] Failed to generate sentence for {word}")
                continue
        except Exception as e:
            print(f"[ERROR] Exception generating sentence for {word}: {e}")
            continue
        
        # Replace the word in the sentence with a blank (case-insensitive, whole word only)
        # Create a regex pattern that matches the word as a whole word (case-insensitive)
        # Escape special regex characters in the word
        escaped_word = re.escape(word)
        # Match whole word boundaries, handling punctuation
        pattern = r'\b' + escaped_word + r'\b'
        sentence_with_blank = re.sub(pattern, '_______', sentence, flags=re.IGNORECASE) if sentence else ""
        
        # If no replacement happened (word not found), try without word boundaries for phrases
        if sentence_with_blank == sentence and word in sentence:
            sentence_with_blank = sentence.replace(word, '_______', 1)
        
        # Generate distractors via LLM using the sentence context, fallback to CSV words if needed
        distractors = generate_distractors_with_llm(
            word, 
            sentence_with_blank=sentence_with_blank,
            definition=definition, 
            num_distractors=3
        )
        if len(distractors) < 3:
            print(f"[WARNING] LLM distractors unavailable for {word}; using CSV fallbacks.")
            fallback_pool = [w['Word'] for w in words if w['Word'] and w['Word'] != word]
            if len(fallback_pool) < 3:
                print(f"[ERROR] Not enough fallback distractors for {word}, skipping")
                continue
            distractors = random.sample(fallback_pool, 3)
        
        # Create options
        options = [word] + distractors
        random.shuffle(options)
        
        # Find which option is the correct answer after shuffling
        correct_option = None
        for i, opt in enumerate(options):
            if opt == word:
                correct_option = chr(65 + i)  # A, B, C, or D
                break

        # Generate explanation
        explanation = generate_explanation_with_llm(
            word=word,
            sentence_with_blank=sentence_with_blank,
            distractors=distractors,
            definition=definition
        )
        if not explanation:
            print(f"[WARNING] Failed to generate explanation for {word}, continuing without it.")
            explanation = ""

        result = {
            'Word': word,
            'SentenceBlank': sentence_with_blank,
            'OptionA': options[0],
            'OptionB': options[1],
            'OptionC': options[2],
            'OptionD': options[3],
            'Answer': sentence or word,
            'CorrectOption': correct_option or '',
            'Explanation': explanation,
            'ArticleTitle': article.get('title') if article else '',
            'ArticleSource': ((article or {}).get('source') or {}).get('name', '') if article else '',
            'ArticleDescription': article.get('description') if article else '',
        }
        
        results.append(result)
        
        # Display result
        print(f"  Sentence: {sentence}")
        print(f"  Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}")
        print(f"  Answer: {word} ({correct_option})")
        if explanation:
            print(f"  Explanation: {explanation[:100]}...")
    
    return results


def save_results_to_csv(results: List[Dict[str, str]], output_path: str):
    """Save generated results to CSV file"""
    if not results:
        print("[WARNING] No results to save")
        return
    
    fieldnames = [
        'Word',
        'SentenceBlank',
        'OptionA',
        'OptionB',
        'OptionC',
        'OptionD',
        'Answer',
        'CorrectOption',
        'Explanation',
        'ArticleTitle',
        'ArticleSource',
        'ArticleDescription',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[SUCCESS] Saved {len(results)} results to {output_path}")


def main():
    """Main test function"""
    print("="*60)
    print("MCQ Generator Test Script")
    print("="*60)
    
    # Configuration
    csv_path = os.path.join(os.path.dirname(__file__), 'word_cards.csv')
    output_path = os.path.join(os.path.dirname(__file__), 'test_results.csv')
    test_count = 1 # Number of words to test
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    # Check configuration
    print(f"\nConfiguration:")
    print(f"  CSV file: {csv_path}")
    print(f"  API URL: {API_URL or 'https://api.openai.com/v1/chat/completions'}")
    print(f"  Model: {AI_MODEL}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  API Key: {'***' + API_KEY[-4:] if API_KEY else 'NOT SET'}")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    
    if LLM_PROVIDER == "openai" and not API_KEY:
        print("\n[ERROR] OPENAI_API_KEY not configured!")
        print("Please set it in your .env file or environment variables.")
        return
    if LLM_PROVIDER == "openai" and not AI_MODEL:
        print("\n[ERROR] OPENAI_MODEL not configured! Set it in your .env file or environment variables.")
        return
    
    # Load words from CSV
    print(f"\nLoading words from {csv_path}...")
    words = load_csv_data(csv_path)
    print(f"[SUCCESS] Loaded {len(words)} words from CSV")
    
    if not words:
        print("[ERROR] No words found in CSV file")
        return
    
    # Load news articles for context
    articles = load_news_articles(limit=len(words))
    if not articles:
        print("[WARNING] Proceeding without news article context.")
    
    # Generate MCQs
    results = generate_mcq_for_words(words, articles, test_count=test_count)
    
    if results:
        # Save results
        save_results_to_csv(results, output_path)
        
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"Total words processed: {len(results)}")
        print(f"Results saved to: {output_path}")
    else:
        print("\n[WARNING] No MCQs were generated")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
