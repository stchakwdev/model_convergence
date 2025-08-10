"""
Anthropic model implementation for Claude models.

This module provides wrapper for Anthropic's Claude models (Claude-3 family, etc.)
with response caching and error handling for pattern discovery experiments.
"""

import os
import hashlib
import json
from typing import Optional, List, Dict
import time

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .model_interface import ModelInterface


class AnthropicModel(ModelInterface):
    """
    Anthropic Claude model wrapper with caching and error handling.
    
    Supports Claude-3 family models (Haiku, Sonnet, Opus).
    """
    
    def __init__(self, 
                 model_name: str = "claude-3-haiku-20240307",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 use_cache: bool = True):
        """
        Initialize Anthropic model wrapper.
        
        Args:
            model_name: Claude model name (e.g., "claude-3-haiku-20240307")
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature for reproducibility
            max_tokens: Maximum response length
            use_cache: Whether to cache responses to avoid duplicate API calls
        """
        super().__init__(name=model_name, architecture="transformer")
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        # Initialize Anthropic client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Model configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        
        # Response cache
        self.response_cache: Dict[str, str] = {}
        self.cache_file = f".cache_{model_name.replace('-', '_').replace('.', '_')}.json"
        
        # Load existing cache
        self._load_cache()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
    def generate(self, prompt: str) -> str:
        """
        Generate response to prompt with caching and error handling.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated response text
        """
        # Check cache first
        if self.use_cache:
            cache_key = self._get_cache_key(prompt)
            if cache_key in self.response_cache:
                print(f"  📁 Cache hit for {self.name}")
                return self.response_cache[cache_key]
        
        # Rate limiting
        self._rate_limit()
        
        try:
            print(f"  🌐 API call to {self.name}")
            
            # Make API request
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract response text
            response_text = ""
            if response.content and len(response.content) > 0:
                # Claude returns content as a list of text blocks
                response_text = response.content[0].text
            
            # Cache the response
            if self.use_cache:
                cache_key = self._get_cache_key(prompt)
                self.response_cache[cache_key] = response_text
                self._save_cache()
            
            return response_text
            
        except anthropic.RateLimitError:
            print(f"  ⚠️  Rate limit hit for {self.name}, waiting 60 seconds...")
            time.sleep(60)
            return self.generate(prompt)  # Retry
            
        except anthropic.APIError as e:
            print(f"  ❌ API error for {self.name}: {e}")
            return f"ERROR: API error - {str(e)}"
            
        except Exception as e:
            print(f"  ❌ Unexpected error for {self.name}: {e}")
            return f"ERROR: {str(e)}"
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts with progress tracking.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated responses
        """
        responses = []
        print(f"🔄 Generating {len(prompts)} responses with {self.name}")
        
        for i, prompt in enumerate(prompts):
            print(f"  Progress: {i+1}/{len(prompts)}")
            response = self.generate(prompt)
            responses.append(response)
            
        return responses
    
    def has_weight_access(self) -> bool:
        """Anthropic models are API-only, no weight access."""
        return False
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate deterministic cache key for prompt."""
        key_string = f"{self.model_name}:{self.temperature}:{self.max_tokens}:{prompt}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_cache(self):
        """Load response cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.response_cache = json.load(f)
                print(f"📁 Loaded {len(self.response_cache)} cached responses for {self.name}")
            except Exception as e:
                print(f"⚠️  Could not load cache for {self.name}: {e}")
                self.response_cache = {}
    
    def _save_cache(self):
        """Save response cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.response_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Could not save cache for {self.name}: {e}")
    
    def _rate_limit(self):
        """Simple rate limiting to respect API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_api_cost_estimate(self) -> Dict[str, float]:
        """
        Estimate API costs based on model and usage.
        
        Returns:
            Dictionary with cost estimates
        """
        # Approximate costs per 1K tokens (as of 2024)
        costs_per_1k = {
            "claude-3-haiku-20240307": 0.00025,    # Input cost
            "claude-3-sonnet-20240229": 0.003,     # Input cost  
            "claude-3-opus-20240229": 0.015,       # Input cost
        }
        
        base_cost = costs_per_1k.get(self.model_name, 0.003)
        
        return {
            "cost_per_1k_tokens": base_cost,
            "estimated_tokens_per_prompt": 50,  # Conservative estimate
            "estimated_cost_per_prompt": base_cost * 0.05,
        }