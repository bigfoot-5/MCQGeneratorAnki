import random
import sys
import os

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

# Load environment variables from .env file(s) if they exist.
# We explicitly point to the addon directory so Anki's working directory does not matter.
load_dotenv(os.path.join(_addon_dir, ".env"), override=False)
if os.path.isdir(_user_files_dir):
    load_dotenv(os.path.join(_user_files_dir, ".env"), override=False)


# ——— Load Configuration ———
DEFAULT_PROMPT_TEMPLATE = (
    "Generate a normal length English sentence using the word or phrase '{word}', "
    "replacing the target word or phrase with a blank (_____). Difficulty should be "
    "{level} based on CEFR. Return only the sentence."
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

# ——— Core API Call with Retry Logic ———
def generate_sentence_for_word(word, max_retries=5):
    """
    Call OpenAI API to generate a sentence with a blank for the given word/phrase.
    Implements retry logic on HTTP 429 errors.
    Returns the sentence as plain text.
    """
    import time

    level = random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    try:
        prompt = PROMPT_TEMPLATE.format(word=word, level=level)
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
        "model": AI_MODEL,  # Example: "gpt-3.5-turbo" or "gpt-4o-mini"
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

# ——— Helpers ———
def get_all_deck_words(did):
    """Collect the 'Word' field from all notes in the given deck."""
    cids = mw.col.decks.cids(did)
    words = []
    for cid in cids:
        note = mw.col.getCard(cid).note()
        w = note['Word'].strip()
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
    if len(deck_words) < 4:
        showInfo("Need at least 4 notes with 'Word' field in deck for MCQ generation.")
        return

    dialog, progress_bar = create_progress_dialog(len(cids))

    for index, cid in enumerate(cids, start=1):
        note = mw.col.getCard(cid).note()
        word = note['Word'].strip()
        if not word:
            continue
        others = [w for w in deck_words if w != word]
        distractors = random.sample(others, 3)
        try:
            sentence = generate_sentence_for_word(word)
            if sentence is None:
                continue
        except Exception as e:
            showInfo(f"Error calling API: {e}")
            dialog.close()
            mw.col.reset()
            return
        options = [word] + distractors
        random.shuffle(options)
        note['SentenceBlank'] = sentence
        note['OptionA'], note['OptionB'], note['OptionC'], note['OptionD'] = options
        note['Answer'] = word
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
