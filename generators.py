import re
import json
import random
from typing import List, Dict, Optional, Tuple

try:
    from config import (
        PROMPT_TEMPLATE, SYNONYM_PROMPT, EXPLANATION_PROMPT
    )
    from llm_client import LLMClient
except ImportError:
    pass

class QuizGenerator:
    def __init__(self, logger=None):
        self.client = LLMClient(logger=logger)
        self.logger = self.client.log

    def _build_sentence_prompt(
        self,
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
        inflection = "plural" # Could be customizable
        
        return PROMPT_TEMPLATE.format(
            word=word,
            inflection=inflection,
            definition=clean_definition,
            article_title=article_title,
            article_summary=article_summary,
        )

    def generate_sentence_for_word(
        self,
        word: str,
        definition: Optional[str] = None,
        article: Optional[Dict[str, str]] = None,
        max_retries: int = 5
    ) -> Optional[str]:
        """
        Generates a sentence containing the target word, with retry logic to ensure the word is present.
        """
        prompt = self._build_sentence_prompt(word, definition, article)
        
        # We handle the "validation retry" loop here manually because it's specific business logic,
        # distinct from the generic API retry logic in LLMClient.
        
        retries = 0
        while retries <= max_retries:
            content = self.client.generate_completion(prompt, max_retries=2) # Short generic retry inside client
            if not content:
                # If client failed entirely (e.g. API down), we stop.
                return None
            
            # Validation: Check if word is in content
            if word.lower() in content.lower():
                self.logger(f"Generated sentence verified for '{word}'.")
                return content
            
            self.logger(f"[RETRY] Generated sentence did not contain target word '{word}'. Retrying...")
            retries += 1
            
        self.logger(f"Failed to generate valid sentence for '{word}' after validation retries.")
        return None

    def generate_synonyms(
        self,
        word: str,
        sentence: str,
        definition: Optional[str] = None,
        num_synonyms: int = 4,
    ) -> List[str]:
        prompt = SYNONYM_PROMPT.format(
            word=word,
            sentence=sentence,
            definition=(definition or "No definition provided").strip().replace("\n", " "),
            num_synonyms=num_synonyms,
        )
        
        content = self.client.generate_completion(prompt)
        if not content:
            return []
            
        # Parse logic
        try:
            # Try to find JSON array
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                if isinstance(data, list):
                    cleaned = []
                    for item in data:
                        if isinstance(item, str):
                            candidate = item.strip()
                            if candidate and candidate.lower() != word.lower():
                                cleaned.append(candidate)
                    return cleaned[:num_synonyms]
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        parts = []
        if '\n' not in content and ',' in content:
             parts = [p.strip() for p in content.split(',')]
        else:
             parts = [p.strip("- ").strip() for p in content.splitlines()]
             
        cleaned = [
            p for p in parts if p and p.lower() != word.lower() and len(p) > 1
        ]
        return cleaned[:num_synonyms]

    def generate_explanation(
        self,
        word: str,
        sentence: str,
        definition: Optional[str] = None,
    ) -> Optional[str]:
        prompt = EXPLANATION_PROMPT.format(
            sentence=sentence,
            correct_word=word,
            definition=(definition or "No definition provided").strip().replace("\n", " "),
        )
        return self.client.generate_completion(prompt)

    def generate_quiz_data(
        self,
        word: str,
        definition: Optional[str],
        article: Optional[Dict[str, str]] = None
    ) -> Optional[Dict]:
        """
        Orchestrates the generation of a single quiz item (Sentence, Synonyms, Explanation).
        """
        # 1. Generate Sentence
        sentence = self.generate_sentence_for_word(word, definition, article)
        if not sentence:
            return None
        
        # 2. Mask the sentence
        word_len = len(word)
        first_letter = word[0] if word_len > 0 else "?"
        masked_representation = f"{first_letter} ({word_len})"
        
        escaped_word = re.escape(word)
        pattern = r'\b' + escaped_word + r'\b'
        
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        sentence_masked = ""
        if match:
            actual_word = match.group(0)
            mask_with_case = f"{actual_word[0]} ({len(actual_word)})"
            sentence_masked = re.sub(pattern, mask_with_case, sentence, count=1, flags=re.IGNORECASE)
        else:
            # Fallback
            if word.lower() in sentence.lower():
                # Case insensitive replace? 
                # Let's do a compiled case-insensitive replacement if standard string replace fails
                pattern_loose = re.compile(re.escape(word), re.IGNORECASE)
                sentence_masked = pattern_loose.sub(masked_representation, sentence, count=1)
            else:
                sentence_masked = sentence # Should have been caught by retry logic, but safe fallback

        # 3. Generate Synonyms
        synonyms = self.generate_synonyms(word, sentence, definition)
        
        # 4. Generate Explanation
        explanation = self.generate_explanation(word, sentence, definition)
        
        # 5. Return structured data
        return {
            "sentence": sentence,
            "sentence_masked": sentence_masked,
            "synonyms": synonyms,
            "explanation": explanation or "",
            "word_length": str(word_len),
            "first_letter": first_letter,
            "article": article
        }
