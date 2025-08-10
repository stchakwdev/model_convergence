"""
Core data structures for universal pattern discovery.

This module defines the fundamental data classes used throughout
the pattern discovery system.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class UniversalFeature:
    """Represents a universal feature that should exist across all capable models"""
    name: str
    description: str
    behavioral_signature: List[Tuple[str, str]]  # (prompt, expected_behavior)
    activation_pattern: Optional[np.ndarray] = None
    importance_score: float = 0.0
    convergence_threshold: float = 0.75
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class FeatureLocation:
    """Where a universal feature is located in a specific model"""
    feature_name: str
    layer_indices: List[int]
    neuron_indices: List[int]
    activation_strength: float
    confidence: float
    extraction_method: str


@dataclass
class ModelFingerprint:
    """Complete fingerprint of a model's universal features"""
    model_id: str
    architecture: str
    features: Dict[str, FeatureLocation]
    behavioral_scores: Dict[str, float]
    convergence_score: float = 0.0
    timestamp: str = ""