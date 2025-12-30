import os
import sys
from dotenv import load_dotenv

# --- Directory Setup ---
_addon_dir = os.path.dirname(os.path.abspath(__file__))
_user_files_dir = os.path.join(_addon_dir, 'user_files')

# --- Load Environment Variables ---
load_dotenv(os.path.join(_addon_dir, ".env"), override=False)
if os.path.isdir(_user_files_dir):
    load_dotenv(os.path.join(_user_files_dir, ".env"), override=False)

# --- LLM Settings ---
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

# OpenAI
API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
API_URL = (os.getenv("OPENAI_API_URL") or "https://api.openai.com/v1/chat/completions").strip()
AI_MODEL = (os.getenv("OPENAI_MODEL") or "").strip()

# Ollama
OLLAMA_URL = (os.getenv("OLLAMA_URL") or "http://localhost:11434/api/chat").strip()
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "gemma3:1b").strip()

# General
try:
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))
except (TypeError, ValueError):
    TEMPERATURE = 1.0

# --- News API Settings ---
try:
    from news_fetcher import (
        BASE_PARAMS as NEWS_BASE_PARAMS,
        NEWS_API_URL
    )
except ImportError:
    NEWS_BASE_PARAMS = None
    NEWS_API_URL = "https://newsapi.org/v2/everything"

# --- Prompt Templates ---
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

SYNONYM_PROMPT = (
    """
You are generating synonyms for a vocabulary exercise.

Target word: {word}
Context sentence: {sentence}
Definition/context: {definition}
Number of synonyms needed: {num_synonyms}

Requirements:
1. Return exactly {num_synonyms} unique English synonyms or short phrases that mean the same as "{word}" in the context of the sentence provided.
2. The synonyms must be suitable as hints for a student trying to guess the target word.
3. Do NOT include the target word itself in the synonyms.
4. Output must be valid JSON: a list of strings, e.g. ["synonym1", "synonym2", "synonym3"]. No commentary, explanations, or extra text.
"""
)

EXPLANATION_PROMPT = (
    """
You are generating an explanation for a vocabulary exercise.

sentence: {sentence}
Correct answer: {correct_word}
Definition/context: {definition}

Requirements:
1. Write a clear, educational explanation (2-4 sentences) that:
   - Explains the meaning of "{correct_word}" in this specific context.
   - Uses the definition provided to support your explanation.
2. The explanation should be helpful for English learners, clear and concise.
3. Format: Write in plain text, no bullet points or numbered lists. Use natural, flowing sentences.
4. Output only the explanation text, no headings, no metadata, no extra formatting.
"""
)

PROMPT_TEMPLATE = (
    os.getenv("OPENAI_PROMPT_TEMPLATE")
    or os.getenv("OLLAMA_PROMPT_TEMPLATE")
    or DEFAULT_PROMPT_TEMPLATE
)
