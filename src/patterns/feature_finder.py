"""
Adaptive system for finding where features are implemented in models.

This module implements self-adapting search that discovers feature locations
without prior knowledge of model architecture, using the "water finding its level" 
analogy for pattern discovery.
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from collections import defaultdict

from .data_structures import UniversalFeature, FeatureLocation

if TYPE_CHECKING:
    from ..models.model_interface import ModelInterface


class AdaptiveFeatureFinder:
    """
    Self-adapting system that discovers where features are implemented
    without prior knowledge of model architecture
    """
    
    def __init__(self):
        self.search_history = []
        self.discovered_mappings = {}
        
    def find_feature(self, model: 'ModelInterface', 
                     feature: UniversalFeature) -> Optional[FeatureLocation]:
        """
        Adaptively search for where a feature is implemented in a model.
        Like water finding its level, this finds where features "settle".
        """
        print(f"    🔎 Searching for '{feature.name}' in {model.name}")
        
        # Start with behavioral probing
        behavioral_signature = self._probe_behavior(model, feature)
        
        if model.has_weight_access():
            # Use internal access for more precise localization
            return self._internal_feature_search(model, feature, behavioral_signature)
        else:
            # Use black-box techniques
            return self._blackbox_feature_search(model, feature, behavioral_signature)
    
    def _probe_behavior(self, model: 'ModelInterface', 
                       feature: UniversalFeature) -> np.ndarray:
        """Create behavioral signature through probing"""
        signatures = []
        
        for prompt, expected in feature.behavioral_signature:
            # Test with variations
            variations = self._generate_prompt_variations(prompt)
            
            responses = []
            for variant in variations:
                response = model.generate(variant)
                responses.append(response)
            
            # Analyze response patterns
            signature = self._analyze_response_pattern(responses, expected)
            signatures.append(signature)
            
        return np.array(signatures)
    
    def _internal_feature_search(self, model: 'ModelInterface', 
                                feature: UniversalFeature,
                                behavioral_signature: np.ndarray) -> Optional[FeatureLocation]:
        """Search for features with internal model access"""
        
        # For models with weight access, use gradient-based localization
        # This is a placeholder implementation
        return FeatureLocation(
            feature_name=feature.name,
            layer_indices=[6, 7, 8],  # Middle layers often contain key features
            neuron_indices=[100, 200, 300],
            activation_strength=0.8,
            confidence=0.7,
            extraction_method="internal_search"
        )
    
    def _blackbox_feature_search(self, model: 'ModelInterface', 
                                feature: UniversalFeature,
                                behavioral_signature: np.ndarray) -> Optional[FeatureLocation]:
        """Search for features using only black-box access"""
        
        # For API-only models, infer from behavioral patterns
        avg_signature_strength = np.mean(behavioral_signature) if len(behavioral_signature) > 0 else 0.0
        
        return FeatureLocation(
            feature_name=feature.name,
            layer_indices=[],  # Cannot determine without internal access
            neuron_indices=[],
            activation_strength=avg_signature_strength,
            confidence=avg_signature_strength * 0.5,  # Lower confidence for black-box
            extraction_method="blackbox_inference"
        )
    
    def _generate_prompt_variations(self, prompt: str) -> List[str]:
        """Generate variations of a prompt for robust testing"""
        variations = [prompt]
        
        # Add simple variations
        variations.append(prompt.upper())
        variations.append(prompt.lower())
        variations.append(f"Please {prompt}")
        variations.append(f"{prompt} Thank you.")
        
        return variations
    
    def _analyze_response_pattern(self, responses: List[str], 
                                 expected: str) -> np.ndarray:
        """Analyze pattern of responses to create signature"""
        pattern_vector = []
        
        for response in responses:
            # Extract features from response
            features = [
                len(response),
                1.0 if expected.lower() in response.lower() else 0.0,
                response.count(" "),
                1.0 if response.startswith(("I", "The", "A")) else 0.0,
            ]
            pattern_vector.extend(features)
            
        return np.array(pattern_vector)