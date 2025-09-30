"""
Universal Alignment Patterns
============================

A Python package for discovering and analyzing universal patterns in AI model alignment.

This package implements the Universal Alignment Patterns Hypothesis: that sufficiently 
capable AI models converge to functionally equivalent internal representations for 
core capabilities, analogous to water transfer printing where patterns emerge 
consistently across different objects.

Author: Samuel Chakwera
"""

__version__ = "0.1.0"
__author__ = "Samuel Chakwera"
__email__ = "stchakwera@gmail.com"

from .patterns import PatternDiscoveryEngine, ConvergenceAnalyzer, UniversalEvaluator
from .models import ModelInterface

__all__ = [
    "PatternDiscoveryEngine",
    "ConvergenceAnalyzer", 
    "UniversalEvaluator",
    "ModelInterface"
]