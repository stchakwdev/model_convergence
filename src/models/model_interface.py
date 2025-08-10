"""
Abstract interface for interacting with different AI model types.

This module provides the base ModelInterface class that standardizes 
interactions with various AI models (OpenAI, Anthropic, open-source, etc.)
enabling universal pattern analysis across different architectures.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class ModelInterface(ABC):
    """Abstract interface for interacting with different model types"""
    
    def __init__(self, name: str, architecture: str):
        """
        Initialize model interface.
        
        Args:
            name: Human-readable model name (e.g., "gpt-4", "claude-3-opus")
            architecture: Architecture family (e.g., "transformer", "mamba")
        """
        self.name = name
        self.architecture = architecture
        self.num_layers = 12  # Default, override in subclass
        
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate response to prompt.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def has_weight_access(self) -> bool:
        """
        Whether we have access to internal weights/activations.
        
        Returns:
            True if internal access available, False for API-only models
        """
        pass
    
    def get_gradients(self, prompt: str, target: str) -> Optional[np.ndarray]:
        """
        Get gradients with respect to target output.
        
        Args:
            prompt: Input prompt
            target: Target output for gradient calculation
            
        Returns:
            Gradient array if available, None otherwise
        """
        return None
    
    def get_neuron_activation(self, layer_idx: int, neuron_idx: int, 
                             prompt: str) -> float:
        """
        Get activation of specific neuron for given prompt.
        
        Args:
            layer_idx: Layer index
            neuron_idx: Neuron index within layer
            prompt: Input prompt
            
        Returns:
            Activation value (0.0 if not accessible)
        """
        return 0.0
    
    def layer_size(self, layer_idx: int) -> int:
        """
        Get size of specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Number of neurons in layer
        """
        return 768  # Default transformer hidden size
    
    def batch_generate(self, prompts: list[str]) -> list[str]:
        """
        Generate responses for multiple prompts.
        
        Default implementation calls generate() for each prompt.
        Subclasses can override for more efficient batch processing.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated responses
        """
        return [self.generate(prompt) for prompt in prompts]
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}', architecture='{self.architecture}')"