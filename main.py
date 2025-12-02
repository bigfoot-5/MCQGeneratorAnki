import random
import re
import sys
import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# Add vendor directory to Python path for bundled dependencies
# This allows us to ship libraries with the addon
_addon_dir = os.path.dirname(os.path.abspath(__file__))
_vendor_dir = os.path.join(_addon_dir, 'vendor')
_user_files_dir = os.path.join(_addon_dir, 'user_files')
if os.path.exists(_vendor_dir):
    # Insert at index 0 to prioritize vendored packages over system packages
    if _vendor_dir not in sys.path:
        sys.path.insert(0, _vendor_dir)

import requests
from dotenv import load_dotenv
from aqt import mw
from aqt.qt import QAction, QInputDialog, QDialog, \
    QVBoxLayout, QProgressBar, QApplication, \
    QEventLoop, QTimer
from aqt.utils import showInfo
from aqt import gui_hooks

try:
    from news_fetcher import (
        BASE_PARAMS as NEWS_BASE_PARAMS,
        NEWS_API_URL,
        fetch_news as newsapi_fetch_news,
    )
except ImportError:
    NEWS_BASE_PARAMS = None
    NEWS_API_URL = "https://newsapi.org/v2/everything"

    def newsapi_fetch_news(api_key: str, params: Dict[str, str]) -> Dict:
        """Fallback NewsAPI fetcher if news_fetcher module is unavailable."""
        headers = {"Authorization": api_key}
        response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

# Load environment variables from .env file(s) if they exist.
# We explicitly point to the addon directory so Anki's working directory does not matter.
load_dotenv(os.path.join(_addon_dir, ".env"), override=False)
if os.path.isdir(_user_files_dir):
    load_dotenv(os.path.join(_user_files_dir, ".env"), override=False)


# ——— Load Configuration ———
DEFAULT_PROMPT_TEMPLATE = (
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

PROMPT_TEMPLATE = (
    os.getenv("OPENAI_PROMPT_TEMPLATE")
    or os.getenv("OLLAMA_PROMPT_TEMPLATE")
    or DEFAULT_PROMPT_TEMPLATE
)
try:
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))
except (TypeError, ValueError):
    TEMPERATURE = 1.5

def non_blocking_wait(seconds):
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


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
    inflection = "plural" # Defaulting to plural as per test_main.py logic, or could be passed in
    return PROMPT_TEMPLATE.format(
        word=word,
        inflection=inflection,
        definition=clean_definition,
        article_title=article_title,
        article_summary=article_summary,
    )


def load_news_articles(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Fetch recent news articles to provide context for MCQ generation."""
    api_key = (os.getenv("NEWS_API_KEY") or "").strip()
    if not api_key:
        print("[MCQGenerator] NEWS_API_KEY not configured; proceeding without article context.")
        return []

    params = dict(NEWS_BASE_PARAMS or {
        "q": "technology",
        "pageSize": 100,
        "sortBy": "publishedAt",
        "sources": "techcrunch"
    })

    if NEWS_API_URL.endswith("everything"):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        params.setdefault("from", week_ago.isoformat(timespec="seconds"))
        params.setdefault("to", now.isoformat(timespec="seconds"))

    try:
        data = newsapi_fetch_news(api_key, params)
    except requests.HTTPError as exc:
        print(f"[MCQGenerator] Failed to fetch news articles: {exc}")
        if exc.response is not None:
            print(exc.response.text)
        return []
    except Exception as exc:
        print(f"[MCQGenerator] Unexpected error fetching news articles: {exc}")
        return []

    articles = data.get("articles", []) if isinstance(data, dict) else []
    if limit is not None:
        articles = articles[:limit]

    print(f"[MCQGenerator] Loaded {len(articles)} news articles for context.")
    return articles


def _note_field_exists(note, field_name: str) -> bool:
    return field_name in note.keys()


def get_note_field(note, field_name: str) -> str:
    if not _note_field_exists(note, field_name):
        return ""
    return (note[field_name] or "").strip()


def set_note_field(note, field_name: str, value: str):
    if not _note_field_exists(note, field_name):
        return
    note[field_name] = value


def get_word_from_note(note) -> str:
    """Get word from note, trying 'Word' field first, then 'Front' field."""
    word = get_note_field(note, "Word")
    if word:
        return word
    return get_note_field(note, "Front")

# ——— Core API Call with Retry Logic ———
def generate_sentence_for_word(
    word: str,
    definition: Optional[str] = None,
    article: Optional[Dict[str, str]] = None,
    max_retries: int = 5,
):
    """
    Call OpenAI API to generate a sentence with a blank for the given word/phrase.
    Implements retry logic on HTTP 429 errors.
    Returns the sentence as plain text.
    """
    try:
        prompt = _build_prompt(word, definition, article)
    except Exception as e:
        showInfo(f"Prompt template is invalid: {e}")
        return None

    if LLM_PROVIDER == "ollama":
        if not OLLAMA_MODEL:
            showInfo("Ollama model is not configured. Set OLLAMA_MODEL in your .env file.")
            return None
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": TEMPERATURE},
            "stream": False,
        }
        try:
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
                showInfo("Ollama returned an empty response.")
                return None
            return content
        except requests.exceptions.RequestException as e:
            showInfo(f"HTTP error when calling Ollama: {e}")
            return None
        except Exception as e:
            showInfo(f"Error processing Ollama response: {e}")
            return None

    if not API_KEY:
        showInfo("OpenAI API key is not configured. Set it via .env or user_files/api_key.txt.")
        return None
    if not AI_MODEL:
        showInfo("OpenAI model is not configured. Set it via .env or user_files/model.txt.")
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
            res = requests.post(API_URL, headers=headers, json=payload)
            if res.status_code == 429:
                wait_time = 30
                showInfo(f"Rate limit reached. Retrying in {wait_time} seconds...")
                non_blocking_wait(wait_time)
                retries += 1
                continue
            res.raise_for_status()
            data = res.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content
        except requests.exceptions.RequestException as e:
            showInfo(f"HTTP error: {e}")
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(3)
        except Exception as e:
            showInfo(f"Error processing response: {e}")
            raise

    showInfo("Maximum retries exceeded. Please try again later.")
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
        content = content.strip()
        
        # 1. Extract content from markdown code blocks if present
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if code_block_match:
            content = code_block_match.group(1).strip()

        try:
            # 2. Try to find JSON array
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
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

        # 3. Fallback: split lines and clean up
        parts = [p.strip("- ,\"\'[]") for p in content.splitlines()]
        cleaned = [
            p for p in parts if p and p.lower() != word.lower() and len(p) > 1
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
                return []
            return _parse_response(content)
        except Exception:
            return []

    if not API_KEY or not AI_MODEL:
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
        res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"].strip()
        return _parse_response(content)
    except Exception:
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
            return None
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": TEMPERATURE},
            "stream": False,
        }
        try:
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
            return content
        except Exception:
            return None

    if not API_KEY or not AI_MODEL:
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
        res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content
    except Exception:
        return None


# ——— Helpers ———
def get_all_deck_words(did):
    """Collect the 'Word' or 'Front' field from all notes in the given deck."""
    cids = mw.col.decks.cids(did)
    words = []
    for cid in cids:
        note = mw.col.getCard(cid).note()
        w = get_word_from_note(note)
        if w:
            words.append(w)
    return list(set(words))

def create_progress_dialog(total_tasks):
    """Create and display a progress dialog."""
    dialog = QDialog(mw)
    dialog.setWindowTitle("Generating MCQs")
    layout = QVBoxLayout()
    progress_bar = QProgressBar()
    progress_bar.setRange(0, total_tasks)
    progress_bar.setValue(0)
    layout.addWidget(progress_bar)
    dialog.setLayout(layout)
    dialog.setModal(True)
    dialog.show()
    return dialog, progress_bar

# ——— Core Generation ———
def generate_mcq_for_cards(cids):
    """Generate MCQs for given card IDs using local distractors."""
    if not cids:
        return
    first_card = mw.col.getCard(cids[0])
    deck_words = get_all_deck_words(first_card.did)
    
    # We don't strictly need 4 words anymore if we use LLM distractors, 
    # but it's good for fallback.
    
    articles = load_news_articles(limit=len(cids))

    dialog, progress_bar = create_progress_dialog(len(cids))

    for index, cid in enumerate(cids, start=1):
        note = mw.col.getCard(cid).note()
        word = get_word_from_note(note)
        if not word:
            continue
            
        definition = get_note_field(note, "Back")
        article = articles[(index - 1) % len(articles)] if articles else None
        
        # 1. Generate Sentence
        try:
            sentence = generate_sentence_for_word(
                word,
                definition=definition,
                article=article,
            )
            if sentence is None:
                continue
        except Exception as e:
            showInfo(f"Error calling API: {e}")
            dialog.close()
            mw.col.reset()
            return
            
        # 2. Create Blank
        # Replace the word in the sentence with a blank (case-insensitive, whole word only)
        escaped_word = re.escape(word)
        pattern = r'\b' + escaped_word + r'\b'
        sentence_with_blank = re.sub(pattern, '_______', sentence, flags=re.IGNORECASE) if sentence else ""
        
        # If no replacement happened (word not found), try without word boundaries for phrases
        if sentence_with_blank == sentence and word in sentence:
            sentence_with_blank = sentence.replace(word, '_______', 1)
        
        # 3. Generate Distractors
        distractors = generate_distractors_with_llm(
            word,
            sentence_with_blank=sentence_with_blank,
            definition=definition,
            num_distractors=3,
            article=article
        )
        
        # Fallback to deck words if LLM fails
        if len(distractors) < 3:
            others = [w for w in deck_words if w != word]
            if len(others) >= 3:
                distractors = random.sample(others, 3)
            else:
                # If we still don't have enough, we can't make a valid MCQ
                # (Or we could duplicate, but let's just skip or fill with placeholders)
                while len(distractors) < 3:
                    distractors.append("N/A")
        
        # 4. Shuffle Options
        options = [word] + distractors
        random.shuffle(options)
        
        # Find correct option letter
        correct_option_char = ""
        for i, opt in enumerate(options):
            if opt == word:
                correct_option_char = chr(65 + i)  # A, B, C, or D
                break
        
        # 5. Generate Explanation
        explanation = generate_explanation_with_llm(
            word=word,
            sentence_with_blank=sentence_with_blank,
            distractors=distractors,
            definition=definition
        )
        
        # 6. Update Note
        set_note_field(note, 'SentenceBlank', sentence_with_blank)
        set_note_field(note, 'OptionA', options[0])
        set_note_field(note, 'OptionB', options[1])
        set_note_field(note, 'OptionC', options[2])
        set_note_field(note, 'OptionD', options[3])
        set_note_field(note, 'Answer', sentence or word)
        
        # New fields (check if they exist before setting)
        set_note_field(note, 'CorrectOption', correct_option_char)
        if explanation:
            set_note_field(note, 'Explanation', explanation)

        if article:
            title = (article.get("title") or "").strip()
            description = (
                article.get("description")
                or article.get("summary")
                or ""
            ).strip()
            source = article.get("source")
            if isinstance(source, dict):
                source_name = (source.get("name") or "").strip()
            else:
                source_name = ""
            article_parts = [part for part in [title, description, source_name] if part]
            article_value = " — ".join(article_parts)
            set_note_field(note, "Article", article_value)
            
        note.flush()
        progress_bar.setValue(index)
        QApplication.processEvents()  # Update the UI

    dialog.close()
    mw.col.reset()
    showInfo("MCQs generated. Sync to AnkiWeb to review from elsewhere.")

# ——— Menu Actions ———
def on_generate_for_current(browser):
    cids = browser.selectedCards()
    if not cids:
        showInfo("Select at least one card to generate MCQ.")
        return
    generate_mcq_for_cards(cids)

def on_generate_for_deck():
    decks = list(mw.col.decks.all_names())
    deck, ok = QInputDialog.getItem(mw, "Select Deck", "Deck:", decks, 0, False)
    if not ok:
        return
    cids = mw.col.decks.cids(mw.col.decks.id(deck))
    generate_mcq_for_cards(cids)

# ——— Hook into UI ———
def add_menu_1():
    menu = mw.form.menuTools
    menu.addSeparator()
    action = QAction("Generate MCQ (whole deck)", mw)
    action.triggered.connect(lambda: on_generate_for_deck())
    menu.addAction(action)

def add_menu_2(browser):
    menu = browser.form.menuEdit
    menu.addSeparator()
    action = QAction("Generate MCQ (selected notes)", browser)
    action.triggered.connect(lambda: on_generate_for_current(browser))
    menu.addAction(action)

gui_hooks.main_window_did_init.append(add_menu_1)
gui_hooks.browser_menus_did_init.append(add_menu_2)
