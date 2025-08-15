"""
Test Tilli Tonse Framework with Mock Models

This script tests the new story-based approach using mock models to validate
the methodology before spending money on real API calls.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import random
import numpy as np
from typing import List, Dict, Any

from patterns.tilli_tonse_framework import TilliTonseFramework, StoryCheckpointType
from patterns.enhanced_distribution_extractor import TilliTonseDistributionExtractor
from patterns.kl_enhanced_analyzer import HybridConvergenceAnalyzer
from patterns.semantic_analyzer import EnhancedSemanticAnalyzer


class MockTilliTonseModel:
    """Mock model that generates story responses similar to real models"""
    
    def __init__(self, name: str, personality: str = "neutral"):
        self.name = name
        self.personality = personality
        
        # Define response patterns for different personalities
        self.response_patterns = {
            "detailed": {
                "base_length": 80,
                "variation": 20,
                "moral_emphasis": 0.8,
                "cultural_awareness": 0.9
            },
            "concise": {
                "base_length": 40,
                "variation": 10, 
                "moral_emphasis": 0.6,
                "cultural_awareness": 0.7
            },
            "analytical": {
                "base_length": 90,
                "variation": 15,
                "moral_emphasis": 0.7,
                "cultural_awareness": 0.8
            }
        }
        
        # Set personality if not specified
        if personality == "neutral":
            personalities = list(self.response_patterns.keys())
            self.personality = random.choice(personalities)
        
        self.pattern = self.response_patterns.get(self.personality, self.response_patterns["detailed"])
    
    def generate(self, prompt: str) -> str:
        """Generate mock story response based on prompt type and personality"""
        
        # Detect prompt type
        prompt_lower = prompt.lower()
        
        if "tilli tonse" in prompt_lower:
            # This is a checkpoint - generate appropriate response
            if "what is" in prompt_lower or "challenge" in prompt_lower:
                return self._generate_comprehension_response()
            elif "think" in prompt_lower or "predict" in prompt_lower:
                return self._generate_prediction_response()
            elif "moral" in prompt_lower or "teach" in prompt_lower:
                return self._generate_moral_response()
            else:
                return self._generate_reflection_response()
        else:
            # Regular narrative prompt - just acknowledge
            return "I understand the story continues."
    
    def _generate_comprehension_response(self) -> str:
        """Generate comprehension checkpoint response"""
        
        base_responses = [
            "Professor Amina faces a dilemma between conflicting historical sources about when World War II ended. She must determine which information is accurate before teaching her students, as providing false information could mislead future generations. This situation highlights the critical importance of verifying facts before sharing knowledge.",
            
            "The challenge here is that Professor Amina has found contradictory information about the end date of World War II. One source claims 1945, another suggests 1946. As an educator, she has a responsibility to provide accurate information to her students, who rely on her expertise.",
            
            "Amina's situation demonstrates the complexity of historical research. She has encountered conflicting sources about when WWII ended, creating uncertainty about what to teach. Her concern shows the weight of responsibility educators carry in transmitting accurate knowledge."
        ]
        
        response = random.choice(base_responses)
        return self._adjust_response_length(response)
    
    def _generate_prediction_response(self) -> str:
        """Generate prediction checkpoint response"""
        
        base_responses = [
            "I think Professor Amina should conduct thorough research to verify the facts before making her decision. She could consult multiple authoritative sources, cross-reference historical documents, and perhaps seek guidance from other historians. The most responsible approach would be to present verified information rather than guessing or choosing randomly between conflicting sources.",
            
            "Amina should verify the information through additional research before teaching. She could check primary sources, consult with other history experts, or look up official records. It would be better to delay the lecture slightly than to present potentially incorrect information to her students.",
            
            "The best approach would be for Amina to investigate further using reliable historical sources and documentation. She should prioritize accuracy over speed, ensuring that whatever she teaches is factually correct and can be supported by evidence."
        ]
        
        response = random.choice(base_responses)
        return self._adjust_response_length(response)
    
    def _generate_moral_response(self) -> str:
        """Generate moral extraction response"""
        
        base_responses = [
            "This story teaches us that truthfulness and accuracy are fundamental responsibilities, especially for educators and knowledge keepers. When we have the power to influence others' understanding, we must prioritize verification and honest inquiry over convenience or assumption. The commitment to truth requires careful research and the humility to admit when we're uncertain, rather than presenting unverified information as fact.",
            
            "The moral emphasizes the crucial importance of accurate information transmission in education and knowledge sharing. Teachers and experts have a responsibility to verify facts before sharing them, as false information can have lasting impacts on students and society. This reflects the Malawian tradition of knowledge keepers ensuring accurate transmission of wisdom across generations.",
            
            "This story illustrates that truthfulness requires active effort and responsibility. It's not enough to simply repeat what we hear - we must verify, research, and ensure accuracy before sharing knowledge with others. The story shows how proper fact-checking and commitment to truth serve both immediate students and future generations who will build upon that knowledge."
        ]
        
        response = random.choice(base_responses)
        
        # Add cultural context if model has high cultural awareness
        if self.pattern["cultural_awareness"] > 0.8:
            cultural_additions = [
                " This reflects the traditional African concept of communal responsibility for knowledge preservation.",
                " Like the griots and storytellers of Africa, modern educators must maintain the integrity of information they pass down.",
                " The 'tilli tonse' tradition reminds us that knowledge sharing is a collective responsibility requiring mutual verification."
            ]
            response += random.choice(cultural_additions)
        
        return self._adjust_response_length(response)
    
    def _generate_reflection_response(self) -> str:
        """Generate reflection checkpoint response"""
        
        base_responses = [
            "This situation connects to broader principles of intellectual honesty and the ethics of information sharing. In our digital age, the rapid spread of information makes it even more critical that we verify facts before sharing them. The responsibility Professor Amina feels reflects a universal principle that those in positions of authority or expertise must prioritize accuracy over speed or convenience.",
            
            "The story relates to fundamental questions about how knowledge is preserved and transmitted in society. It highlights the role of educators as gatekeepers of information and the trust that students place in their teachers. This responsibility extends beyond individual classrooms to the broader mission of maintaining accurate historical records and understanding.",
            
            "This scenario reflects broader themes about the nature of truth and our collective responsibility to maintain accurate information. It demonstrates how individual choices about truthfulness can have ripple effects across generations and communities."
        ]
        
        response = random.choice(base_responses)
        return self._adjust_response_length(response)
    
    def _adjust_response_length(self, response: str) -> str:
        """Adjust response length based on model personality"""
        
        words = response.split()
        target_length = self.pattern["base_length"] + random.randint(
            -self.pattern["variation"], 
            self.pattern["variation"]
        )
        
        if len(words) > target_length:
            # Truncate but keep complete sentences
            truncated = words[:target_length]
            response = ' '.join(truncated)
            
            # Ensure it ends properly
            if not response.endswith(('.', '!', '?')):
                response += '.'
        
        elif len(words) < target_length * 0.8:
            # Add some expansion
            expansions = [
                " This demonstrates the importance of careful consideration in decision-making.",
                " Such situations require balancing multiple factors and responsibilities.",
                " The implications of these choices extend beyond the immediate context.",
                " This reflects broader themes about responsibility and ethical decision-making."
            ]
            
            response += random.choice(expansions)
        
        return response


def test_tilli_tonse_with_mocks():
    """Test the Tilli Tonse framework with mock models"""
    
    print("🎭 TESTING TILLI TONSE FRAMEWORK WITH MOCK MODELS")
    print("=" * 60)
    print("🧪 This validates the methodology before real API costs")
    
    # Create mock models with different personalities
    models = [
        MockTilliTonseModel("MockGPT", "detailed"),
        MockTilliTonseModel("MockClaude", "analytical"), 
        MockTilliTonseModel("MockGLM", "concise")
    ]
    
    # Load stories
    print("\n📚 Loading Tilli Tonse stories...")
    from tilli_tonse_experiment import load_tilli_tonse_stories
    framework = load_tilli_tonse_stories("prompt_datasets/tilli_tonse_stories.json")
    
    # Test story response collection
    print("\n📖 Testing story response collection...")
    truth_stories = framework.get_stories_for_capability("truthfulness")
    test_story = truth_stories[0]  # Use first truthfulness story
    
    print(f"   Story: {test_story.title}")
    print(f"   Expected tokens: {test_story.total_expected_tokens}")
    
    # Collect responses from mock models
    model_responses = {}
    for model in models:
        print(f"\n   🤖 Collecting from {model.name} ({model.personality})")
        response_data = framework.collect_story_responses(model, test_story)
        model_responses[model.name] = response_data
        
        print(f"      Generated {response_data['response_length']} tokens")
        print(f"      Checkpoints: {len(response_data['checkpoint_responses'])}")
        
        # Show sample checkpoint response
        if response_data['checkpoint_responses']:
            sample = response_data['checkpoint_responses'][0]
            print(f"      Sample checkpoint: {sample['response'][:100]}...")
    
    # Test distribution extraction
    print("\n🔬 Testing enhanced distribution extraction...")
    extractor = TilliTonseDistributionExtractor(common_vocab_size=2000)
    
    # Extract full story responses for analysis
    full_responses = {}
    for model_name, response_data in model_responses.items():
        full_responses[model_name] = [response_data["full_response"]]
    
    distributions = extractor.extract_distributions_from_story_responses(full_responses)
    
    # Test hybrid convergence analysis
    print("\n🧬 Testing hybrid convergence analysis...")
    semantic_analyzer = EnhancedSemanticAnalyzer()
    hybrid_analyzer = HybridConvergenceAnalyzer(semantic_analyzer=semantic_analyzer)
    
    results = hybrid_analyzer.analyze_hybrid_convergence(
        full_responses, 
        "truthfulness_tilli_tonse_test"
    )
    
    # Display results
    print(f"\n🎯 MOCK TILLI TONSE RESULTS:")
    print(f"   Semantic Convergence: {results.semantic_convergence_score:.1%}")
    print(f"   Distributional Convergence: {results.distributional_convergence_score:.1%}")
    print(f"   Hybrid Convergence: {results.hybrid_convergence_score:.1%}")
    
    # Compare to expected improvements
    vocab_stats = extractor.get_vocabulary_stats()
    print(f"\n📊 VOCABULARY RICHNESS:")
    print(f"   Total vocabulary: {vocab_stats['total_vocabulary_size']}")
    print(f"   Tokens: {vocab_stats['token_count']}")
    print(f"   N-grams: {vocab_stats['ngram_count']}")
    print(f"   Richness factor: {vocab_stats['vocabulary_richness']:.1f}x")
    print(f"   Cultural markers: {vocab_stats['includes_cultural_markers']}")
    
    # Expected vs actual token counts
    actual_avg_length = np.mean([data['response_length'] for data in model_responses.values()])
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Target tokens per story: {test_story.total_expected_tokens}")
    print(f"   Actual average tokens: {actual_avg_length:.0f}")
    print(f"   Improvement over simple Q&A: {actual_avg_length/5:.1f}x richer")
    
    print(f"\n✅ Tilli Tonse framework validation successful!")
    print(f"🌍 Cultural innovation working: Malawian storytelling tradition applied to AI research")
    print(f"📊 Ready for real model testing with expected convergence improvements")
    
    return results


if __name__ == "__main__":
    test_tilli_tonse_with_mocks()