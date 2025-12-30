import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# Add vendor directory to Python path for bundled dependencies
_addon_dir = os.path.dirname(os.path.abspath(__file__))
_vendor_dir = os.path.join(_addon_dir, 'vendor')
_user_files_dir = os.path.join(_addon_dir, 'user_files')

# Ensure addon dir is in sys.path so we can import 'config', 'graph', etc.
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

if os.path.exists(_vendor_dir):
    if _vendor_dir not in sys.path:
        sys.path.insert(0, _vendor_dir)

# --- Dependency Conflict Fix ---
# Force load typing_extensions from vendor to ensure 'Sentinel' is available for Pydantic.
import importlib.util
try:
    _te_path = os.path.join(_vendor_dir, 'typing_extensions.py')
    if os.path.exists(_te_path):
        import typing_extensions
        # Only reload if missing Sentinel (saving time/risk if already correct)
        if not hasattr(typing_extensions, 'Sentinel'):
            spec = importlib.util.spec_from_file_location("typing_extensions", _te_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["typing_extensions"] = module
                spec.loader.exec_module(module)
                print("[MCQGenerator] Force reloaded typing_extensions from vendor.")
except Exception as e:
    print(f"[MCQGenerator] Failed to force load typing_extensions: {e}")
# -------------------------------

import requests
from dotenv import load_dotenv
from aqt import mw
from aqt.qt import (
    QAction, QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, 
    QProgressBar, QApplication, QEventLoop, QTimer,
    QLabel, QLineEdit, QPushButton, QSpinBox, QFormLayout, Qt
)
from aqt.utils import showInfo
from aqt import gui_hooks

# --- Import from new modules ---
try:
    from config import (
        API_URL, AI_MODEL, TEMPERATURE, LLM_PROVIDER, API_KEY,
        NEWS_BASE_PARAMS, NEWS_API_URL
    )
    from graph import create_quiz_graph, QuizState
except ImportError as e:
    showInfo(f"Error importing modules: {e}")
    # Defaults to prevent total crash
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    NEWS_BASE_PARAMS = None


# Load environment variables
load_dotenv(os.path.join(_addon_dir, ".env"), override=False)
if os.path.isdir(_user_files_dir):
    load_dotenv(os.path.join(_user_files_dir, ".env"), override=False)

# --- News Fetching (Moved from module or inline) ---
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

def load_news_articles(
    query: str = "technology",
    days_back: int = 7,
    limit: int = 20
) -> List[Dict[str, str]]:
    """Fetch recent news articles based on query and time range."""
    api_key = (os.getenv("NEWS_API_KEY") or "").strip()
    if not api_key:
        print("[MCQGenerator] NEWS_API_KEY not configured.")
        return []

    params = dict(NEWS_BASE_PARAMS) if NEWS_BASE_PARAMS else {
        "pageSize": 100,
        "sortBy": "publishedAt",
    }
    # Override query
    params["q"] = query

    if NEWS_API_URL.endswith("everything"):
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)
        params["from"] = start_date.isoformat(timespec="seconds")
        params["to"] = now.isoformat(timespec="seconds")

    try:
        data = newsapi_fetch_news(api_key, params)
    except Exception as exc:
        print(f"[MCQGenerator] Failed to fetch news articles: {exc}")
        return []

    articles = data.get("articles", []) if isinstance(data, dict) else []
    return articles[:limit]


# --- UI Classes ---

class NewsFilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fetch News for Quiz Context")
        self.setMinimumWidth(400)
        self.articles = []
        
        layout = QVBoxLayout()
        
        # Form
        form_layout = QFormLayout()
        self.query_input = QLineEdit("technology")
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 30)
        self.days_input.setValue(7)
        
        form_layout.addRow("Topic:", self.query_input)
        form_layout.addRow("Days Back:", self.days_input)
        layout.addLayout(form_layout)
        
        # Fetch Button
        self.fetch_btn = QPushButton("Fetch Articles")
        self.fetch_btn.clicked.connect(self.on_fetch)
        layout.addWidget(self.fetch_btn)
        
        # Status Label
        self.status_label = QLabel("No articles loaded.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Dialog Buttons
        self.button_box = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.confirm_btn = QPushButton("Start Generation")
        self.confirm_btn.setEnabled(False) # Enable only after fetching
        
        self.cancel_btn.clicked.connect(self.reject)
        self.confirm_btn.clicked.connect(self.accept)
        
        self.button_box.addWidget(self.cancel_btn)
        self.button_box.addWidget(self.confirm_btn)
        layout.addLayout(self.button_box)
        
        self.setLayout(layout)

    def on_fetch(self):
        query = self.query_input.text().strip()
        days = self.days_input.value()
        
        self.status_label.setText("Fetching...")
        self.fetch_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            self.articles = load_news_articles(query=query, days_back=days, limit=50)
            count = len(self.articles)
            self.status_label.setText(f"Loaded {count} articles.")
            if count > 0:
                self.confirm_btn.setEnabled(True)
            else:
                self.confirm_btn.setEnabled(False)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
        finally:
            self.fetch_btn.setEnabled(True)
            
    def get_articles(self):
        return self.articles


def create_progress_dialog(total_tasks):
    dialog = QDialog(mw)
    dialog.setWindowTitle("Generating Quizzes")
    layout = QVBoxLayout()
    progress_bar = QProgressBar()
    progress_bar.setRange(0, total_tasks)
    progress_bar.setValue(0)
    layout.addWidget(progress_bar)
    dialog.setLayout(layout)
    dialog.setModal(True)
    dialog.show()
    return dialog, progress_bar


# --- Anki Helpers ---

def _note_field_exists(note, field_name: str) -> bool:
    return field_name in note.keys()

def set_note_field(note, field_name: str, value: str):
    if _note_field_exists(note, field_name):
        note[field_name] = value

def get_note_field(note, field_name: str) -> str:
    if not _note_field_exists(note, field_name):
        return ""
    return (note[field_name] or "").strip()

def get_word_from_note(note) -> str:
    word = get_note_field(note, "Word")
    if word: return word
    return get_note_field(note, "Front")


# --- Main Logic ---

def generate_mcq_for_cards(cids):
    if not cids:
        return

    # 1. Show News Filter Dialog
    dialog = NewsFilterDialog(mw)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return # User cancelled
        
    articles = dialog.get_articles()
    if not articles:
        showInfo("No articles selected. Cannot proceed.")
        return
        
    # 2. Compile Graph
    try:
        app = create_quiz_graph()
    except Exception as e:
        showInfo(f"Failed to initialize AI Graph: {e}")
        return

    # 3. Process Cards
    pd, progress_bar = create_progress_dialog(len(cids))
    
    for index, cid in enumerate(cids, start=1):
        note = mw.col.getCard(cid).note()
        word = get_word_from_note(note)
        
        if not word:
            continue
            
        definition = get_note_field(note, "Back")
        
        # Pick an article (round-robin)
        article = articles[(index - 1) % len(articles)]
        
        try:
            # Init State
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
            
            # Invoke Graph
            final_state = app.invoke(initial_state)
            
            # Check success
            if not final_state.get("sentence"):
                # Could log error?
                print(f"Failed for {word}: {final_state.get('error')}")
                continue
                
            # Update Note
            set_note_field(note, 'SentenceBlank', final_state.get("sentence_masked") or "")
            
            synonyms = final_state.get("synonyms") or []
            while len(synonyms) < 4: synonyms.append("")
            
            set_note_field(note, 'OptionA', synonyms[0])
            set_note_field(note, 'OptionB', synonyms[1])
            set_note_field(note, 'OptionC', synonyms[2])
            set_note_field(note, 'OptionD', synonyms[3])
            
            set_note_field(note, 'Answer', final_state.get("sentence") or "")
            set_note_field(note, 'Explanation', final_state.get("explanation") or "")
            
            set_note_field(note, 'WordLength', str(final_state.get("word_length")))
            set_note_field(note, 'FirstLetter', final_state.get("first_letter"))
            
            # Article metadata
            title = article.get("title", "")
            source = (article.get("source") or {}).get("name", "")
            set_note_field(note, "Article", f"{title} - {source}" if source else title)
            
            note.flush()
            
        except Exception as e:
            print(f"Error processing card {cid}: {e}")
        
        progress_bar.setValue(index)
        QApplication.processEvents()

    pd.close()
    mw.col.reset()
    showInfo(f"Processed {len(cids)} cards.")


# --- Menu Hooks ---
def on_generate_for_current(browser):
    cids = browser.selectedCards()
    if not cids:
        showInfo("Select at least one card.")
        return
    generate_mcq_for_cards(cids)

def on_generate_for_deck():
    decks = list(mw.col.decks.all_names())
    deck, ok = QInputDialog.getItem(mw, "Select Deck", "Deck:", decks, 0, False)
    if not ok:
        return
    cids = mw.col.decks.cids(mw.col.decks.id(deck))
    generate_mcq_for_cards(cids)

def add_menu_1():
    menu = mw.form.menuTools
    menu.addSeparator()
    # Updated label
    action = QAction("Generate Quiz (News & Synonyms)", mw)
    action.triggered.connect(lambda: on_generate_for_deck())
    menu.addAction(action)

def add_menu_2(browser):
    menu = browser.form.menuEdit
    menu.addSeparator()
    action = QAction("Generate Quiz (News & Synonyms)", browser)
    action.triggered.connect(lambda: on_generate_for_current(browser))
    menu.addAction(action)

gui_hooks.main_window_did_init.append(add_menu_1)
gui_hooks.browser_menus_did_init.append(add_menu_2)
