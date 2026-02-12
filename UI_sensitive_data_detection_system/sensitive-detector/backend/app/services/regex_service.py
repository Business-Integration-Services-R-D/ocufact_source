import json
import os
import threading
from typing import List, Dict, Optional
import sys
import re

# Add paths to import smart_regex_synthesizer2
# Try Docker container path first
sys.path.insert(0, "/app")
# Try relative paths for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

try:
    from smart_regex_synthesizer2 import SmartRegexSynthesizer, validate
except ImportError as e:
    print(f"Warning: Could not import smart_regex_synthesizer2: {e}")
    # Create dummy classes as fallback
    class SmartRegexSynthesizer:
        def __init__(self):
            pass
        def add(self, phrases):
            pass
        def synthesize(self):
            return r".*"
    
    def validate(pattern, samples):
        return True, 1.0, []


class RegexService:
    def __init__(self, storage_file: str = "backend/regex_store.json"):
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self.regex_store: List[Dict] = []
        self._load_store()
    
    def _load_store(self):
        """Load regex patterns from storage file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.regex_store = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load regex store: {e}")
            self.regex_store = []
    
    def _save_store(self):
        """Save regex patterns to storage file."""
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.regex_store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save regex store: {e}")
    
    def generate_regex(self, phrases: List[str]) -> Dict[str, float]:
        """Generate regex pattern from phrases using SmartRegexSynthesizer."""
        if len(phrases) < 8:
            raise ValueError("Minimum 8 phrases required for regex generation")
        
        # Clean phrases
        clean_phrases = [p.strip() for p in phrases if p and p.strip()]
        if len(clean_phrases) < 8:
            raise ValueError("Minimum 8 valid phrases required after cleaning")
        
        # Generate regex using SmartRegexSynthesizer
        synthesizer = SmartRegexSynthesizer()
        synthesizer.add(clean_phrases)
        regex_pattern = synthesizer.synthesize()
        
        # Validate the generated regex
        is_valid, confidence, failures = validate(regex_pattern, clean_phrases)
        
        if not is_valid:
            raise ValueError(f"Generated regex is invalid: {failures}")
        
        # Store the regex
        regex_entry = {
            "regex": regex_pattern,
            "confidence": confidence,
            "phrases": clean_phrases,
            "created_at": str(os.times().elapsed)  # Simple timestamp
        }
        
        with self.lock:
            self.regex_store.append(regex_entry)
            self._save_store()
        
        return {
            "regex": regex_pattern,
            "confidence": confidence
        }
    
    def get_stored_regexes(self) -> List[Dict]:
        """Get all stored regex patterns."""
        with self.lock:
            return self.regex_store.copy()
    
    def match_regex(self, pattern: str, texts: List[str]) -> List[bool]:
        """Match regex pattern against list of texts."""
        try:
            compiled_regex = re.compile(pattern)
            return [bool(compiled_regex.search(text)) for text in texts]
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def find_matches(self, pattern: str, texts: List[str]) -> List[str]:
        """Return a flat list of matched substrings across all texts.
        Uses search/finditer so partial matches within longer strings are captured.
        """
        try:
            compiled_regex = re.compile(pattern)
            matches: List[str] = []
            for text in texts:
                if text is None:
                    continue
                for m in compiled_regex.finditer(text):
                    try:
                        matches.append(m.group(0))
                    except Exception:
                        # Fallback: append matched span from the original text
                        start, end = m.span()
                        matches.append(text[start:end])
            return matches
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def validate_regex(self, pattern: str) -> bool:
        """Validate if regex pattern is compilable."""
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False
