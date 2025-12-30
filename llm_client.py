import requests
import json
import time
from typing import Optional, Dict, Any

try:
    from config import (
        LLM_PROVIDER, API_KEY, API_URL, AI_MODEL, 
        OLLAMA_URL, OLLAMA_MODEL, TEMPERATURE
    )
except ImportError:
    # If running from a context where config isn't in path directly (e.g. some tests)
    # But usually it should be.
    pass

class LLMClient:
    def __init__(self, logger=None):
        """
        logger: A function or object with showInfo/print interface to log errors.
                If None, uses print.
        """
        self.logger = logger if logger else self._default_logger

    def _default_logger(self, msg):
        print(f"[LLMClient] {msg}")

    def log(self, msg):
        if hasattr(self.logger, 'showInfo'):
             # Anki style
             # but showInfo is a bit intrusive for debug logs, maybe use print for debug
             print(f"[LLMClient] {msg}")
        elif callable(self.logger):
            self.logger(msg)
        else:
            print(f"[LLMClient] {msg}")
            
    def show_error(self, msg):
         if hasattr(self.logger, 'showInfo'):
             self.logger.showInfo(msg)
         elif hasattr(self.logger, 'showError'):
             self.logger.showError(msg)
         else:
             print(f"[ERROR] {msg}")


    def generate_completion(
        self, 
        prompt: str, 
        temperature: float = TEMPERATURE, 
        max_retries: int = 5
    ) -> Optional[str]:
        """
        Generate completion using the configured provider (Ollama or OpenAI).
        """
        if LLM_PROVIDER == "ollama":
            return self._call_ollama(prompt, temperature, max_retries)
        else:
            return self._call_openai(prompt, temperature, max_retries)

    def _call_ollama(self, prompt: str, temperature: float, max_retries: int) -> Optional[str]:
        if not OLLAMA_MODEL:
            self.show_error("Ollama model is not configured.")
            return None
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature},
            "stream": False,
        }
        
        retries = 0
        while retries <= max_retries:
            try:
                self.log(f"Calling Ollama ({OLLAMA_MODEL})...")
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
                    self.log("Empty response from Ollama")
                    # Could retry?
                    return None
                
                return content
            except requests.exceptions.RequestException as e:
                self.log(f"HTTP error calling Ollama: {e}")
                retries += 1
                time.sleep(2)
            except Exception as e:
                self.log(f"Error processing Ollama response: {e}")
                retries += 1
                time.sleep(2)
        
        self.show_error(f"Failed to call Ollama after {max_retries} retries.")
        return None

    def _call_openai(self, prompt: str, temperature: float, max_retries: int) -> Optional[str]:
        if not API_KEY:
            self.show_error("OpenAI API key is not configured.")
            return None
        if not AI_MODEL:
            self.show_error("OpenAI model is not configured.")
            return None

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }

        retries = 0
        while retries <= max_retries:
            try:
                self.log(f"Calling OpenAI ({AI_MODEL})...")
                res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                if res.status_code == 429:
                    wait_time = 30
                    self.log(f"Rate limit reached. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"].strip()
                return content
                
            except requests.exceptions.RequestException as e:
                self.log(f"HTTP error: {e}")
                retries += 1
                time.sleep(3)
            except Exception as e:
                self.log(f"Error processing response: {e}")
                retries += 1
                time.sleep(3)

        self.show_error("Maximum retries with OpenAI exceeded.")
        return None
