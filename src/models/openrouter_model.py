"""
OpenRouter model implementation for unified LLM access.

This module provides a wrapper for OpenRouter's unified API that gives access
to hundreds of AI models including GLM-4.5, Kimi-K2, Qwen-3, GPT-OSS, and more
through a single endpoint with automatic cost optimization and failover.
"""

import os
import hashlib
import json
from typing import Optional, List, Dict, Any
import time

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .model_interface import ModelInterface


class OpenRouterModel(ModelInterface):
    """
    OpenRouter unified model wrapper with multi-provider support.
    
    Supports access to 300+ models including:
    - GLM-4.5 (Zhipu AI) - Best open-source agentic model
    - Kimi-K2 (Moonshot AI) - 1T param MoE with 256K context
    - Qwen-3 (Alibaba) - Strong coding and multilingual capabilities  
    - GPT-OSS (OpenAI) - Open-source reasoning model
    - And many more through OpenRouter's unified API
    """
    
    def __init__(self, 
                 model_id: str,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 use_cache: bool = True,
                 site_name: str = "Universal Alignment Patterns",
                 site_url: str = "https://github.com/stchakwdev/universal_patterns"):
        """
        Initialize OpenRouter model wrapper.
        
        Args:
            model_id: OpenRouter model ID (e.g., "openai/gpt-oss-120b", "zhipu/glm-4.5")
            api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY env var)
            temperature: Sampling temperature for reproducibility
            max_tokens: Maximum response length
            use_cache: Whether to cache responses to avoid duplicate API calls
            site_name: Your site name for OpenRouter analytics
            site_url: Your site URL for OpenRouter analytics
        """
        # Extract model name for display
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        super().__init__(name=model_name, architecture="transformer")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai>=1.0.0")
        
        # Initialize OpenRouter client
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass api_key parameter")
        
        # OpenRouter uses OpenAI SDK with custom base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Model configuration
        self.model_id = model_id
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        
        # OpenRouter headers for analytics and attribution
        self.default_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name
        }
        
        # Response cache
        self.response_cache: Dict[str, str] = {}
        self.cache_file = f".cache_openrouter_{model_name.replace('-', '_')}.json"
        
        # Load existing cache
        self._load_cache()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # OpenRouter handles rate limiting better
        
        # Model metadata (populated from registry if available)
        self.model_metadata = self._get_model_metadata()
        
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
            print(f"  🌐 OpenRouter API call to {self.name} ({self.model_id})")
            
            # Make API request with OpenRouter headers
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers=self.default_headers
            )
            
            # Extract response text
            response_text = response.choices[0].message.content or ""
            
            # Cache the response
            if self.use_cache:
                cache_key = self._get_cache_key(prompt)
                self.response_cache[cache_key] = response_text
                self._save_cache()
            
            return response_text
            
        except openai.RateLimitError:
            print(f"  ⚠️  Rate limit hit for {self.name}, waiting 30 seconds...")
            time.sleep(30)
            return self.generate(prompt)  # Retry
            
        except openai.APIError as e:
            print(f"  ❌ OpenRouter API error for {self.name}: {e}")
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
        print(f"🔄 Generating {len(prompts)} responses with {self.name} via OpenRouter")
        
        for i, prompt in enumerate(prompts):
            print(f"  Progress: {i+1}/{len(prompts)}")
            response = self.generate(prompt)
            responses.append(response)
            
        return responses
    
    def has_weight_access(self) -> bool:
        """OpenRouter models are API-only, no weight access."""
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information from OpenRouter.
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "provider": "OpenRouter",
            "metadata": self.model_metadata
        }
        return info
    
    def _get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata from registry or defaults."""
        # This will be populated by the model registry
        # For now, return basic metadata based on model ID
        metadata = {"context_length": 8192, "family": "unknown"}
        
        if "gpt-oss" in self.model_id:
            metadata.update({
                "family": "gpt-oss",
                "architecture": "moe-transformer",
                "parameters": "120B (5.1B active)" if "120b" in self.model_id else "20B",
                "context_length": 8192,
                "provider": "OpenAI",
                "capabilities": ["reasoning", "coding", "chain-of-thought"]
            })
        elif "glm" in self.model_id:
            metadata.update({
                "family": "glm",
                "architecture": "moe-transformer", 
                "parameters": "355B (32B active)",
                "context_length": 128000,
                "provider": "Zhipu AI",
                "capabilities": ["agentic", "tool-use", "reasoning", "coding"]
            })
        elif "kimi" in self.model_id:
            metadata.update({
                "family": "kimi",
                "architecture": "moe-transformer",
                "parameters": "1T (32B active)",
                "context_length": 256000,
                "provider": "Moonshot AI",
                "capabilities": ["agentic", "long-context", "coding", "mcp-support"]
            })
        elif "qwen" in self.model_id:
            metadata.update({
                "family": "qwen",
                "architecture": "moe-transformer",
                "parameters": "480B (35B active)" if "coder" in self.model_id else "235B (22B active)",
                "context_length": 256000,
                "provider": "Alibaba",
                "capabilities": ["coding", "multilingual", "long-context", "thinking"]
            })
        
        return metadata
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate deterministic cache key for prompt."""
        key_string = f"{self.model_id}:{self.temperature}:{self.max_tokens}:{prompt}"
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
        """Simple rate limiting to be respectful to OpenRouter."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_cost_info(self) -> Dict[str, Any]:
        """
        Get cost information for this model from OpenRouter.
        
        Returns:
            Dictionary with cost estimates
        """
        # OpenRouter provides dynamic pricing
        # These are approximate values - real costs may vary
        cost_estimates = {
            "provider": "OpenRouter",
            "note": "OpenRouter automatically routes to cheapest available provider",
            "benefits": [
                "Automatic cost optimization",
                "Provider failover",
                "Unified billing"
            ]
        }
        
        # Add model-specific cost hints
        if "gpt-oss" in self.model_id:
            cost_estimates["tier"] = "free/low-cost"
            cost_estimates["note"] += " - GPT-OSS models often have free tiers"
        elif any(x in self.model_id for x in ["glm", "kimi", "qwen"]):
            cost_estimates["tier"] = "competitive"
            cost_estimates["note"] += " - Chinese models typically 10-100x cheaper than Western equivalents"
        
        return cost_estimates