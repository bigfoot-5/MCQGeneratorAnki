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
from typing import List, Dict, Optional

# Mock Anki components
class MockMW:
    """Mock Anki main window"""
    class MockConfig:
        def __init__(self):
            self.config_file = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(self.config_file, 'r') as f:
                self._config = json.load(f)
        
        def get(self, key, default=None):
            return self._config.get(key, default)
    
    def __init__(self):
        self.addonManager = self.MockConfig()

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
sys.modules['aqt'].mw = MockMW()
sys.modules['aqt'].showInfo = MockAQT().showInfo
sys.modules['aqt.utils'] = type(sys)('aqt.utils')
sys.modules['aqt.utils'].showInfo = MockAQT().showInfo
sys.modules['aqt.qt'] = type(sys)('aqt.qt')
sys.modules['aqt'].gui_hooks = type(sys)('gui_hooks')

# Import the actual functions from main.py
import requests
from dotenv import load_dotenv

# Load environment variables using explicit paths to match main.py behavior
load_dotenv(os.path.join(_addon_dir, ".env"), override=False)
if os.path.isdir(_user_files_dir):
    load_dotenv(os.path.join(_user_files_dir, ".env"), override=False)

# Load config (for any non-secret defaults the user may keep there)
config = MockMW().addonManager

# Mirror configuration resolution from main.py
DEFAULT_PROMPT = (
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


def show_info(msg):
    """Mock showInfo function"""
    print(f"[INFO] {msg}")


def non_blocking_wait(seconds):
    """Mock non-blocking wait - just print"""
    print(f"[WAIT] Waiting {seconds} seconds...")


# Core API Call with Retry Logic (from main.py)
def generate_sentence_for_word(word: str, max_retries: int = 5) -> Optional[str]:
    """
    Call OpenAI API to generate a sentence with a blank for the given word/phrase.
    Implements retry logic on HTTP 429 errors.
    Returns the sentence as plain text.
    """
    import time

    level = random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    prompt = PROMPT_TEMPLATE.format(word=word, level=level)

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
            print(f"[SLM] Generating sentence for '{word}' using Ollama model '{OLLAMA_MODEL}'...")
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
            print(f"[API] Generating sentence for '{word}' (attempt {retries + 1})...")
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


def generate_mcq_for_words(words: List[Dict[str, str]], test_count: int = 3) -> List[Dict[str, str]]:
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
    
    # Get all words for distractors
    all_words = [w['Word'] for w in words if w['Word']]
    
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
        
        print(f"\n[{index}/{len(test_words)}] Processing: {word}")
        
        # Get distractors from other words
        others = [w for w in all_words if w != word]
        if len(others) < 3:
            print(f"[WARNING] Not enough distractors for {word}, skipping")
            continue
        
        distractors = random.sample(others, 3)
        
        # Generate sentence
        try:
            sentence = generate_sentence_for_word(word)
            if sentence is None:
                print(f"[SKIP] Failed to generate sentence for {word}")
                continue
        except Exception as e:
            print(f"[ERROR] Exception generating sentence for {word}: {e}")
            continue
        
        # Create options
        options = [word] + distractors
        random.shuffle(options)
        
        result = {
            'Word': word,
            'SentenceBlank': sentence,
            'OptionA': options[0],
            'OptionB': options[1],
            'OptionC': options[2],
            'OptionD': options[3],
            'Answer': word,
        }
        
        results.append(result)
        
        # Display result
        print(f"  Sentence: {sentence}")
        print(f"  Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}")
        print(f"  Answer: {word}")
    
    return results


def save_results_to_csv(results: List[Dict[str, str]], output_path: str):
    """Save generated results to CSV file"""
    if not results:
        print("[WARNING] No results to save")
        return
    
    fieldnames = ['Word', 'SentenceBlank', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'Answer']
    
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
    
    if not API_KEY:
        print("\n[ERROR] API_KEY not configured!")
        print("Please set it in config.json or .env file")
        return
    
    # Load words from CSV
    print(f"\nLoading words from {csv_path}...")
    words = load_csv_data(csv_path)
    print(f"[SUCCESS] Loaded {len(words)} words from CSV")
    
    if not words:
        print("[ERROR] No words found in CSV file")
        return
    
    # Generate MCQs
    results = generate_mcq_for_words(words, test_count=test_count)
    
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
