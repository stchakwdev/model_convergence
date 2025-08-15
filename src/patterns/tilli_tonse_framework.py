"""
Tilli Tonse Framework: Multi-Turn Story-Based Convergence Analysis

Inspired by Malawian oral storytelling traditions where the storyteller periodically 
checks in with the audience ("tilli tonse" - are we together?), this framework 
creates richer, longer sequences for more effective KL divergence analysis.

The key insight: instead of asking simple factual questions that get 1-5 token 
responses, we use storytelling with checkpoints to generate 200-500 token responses
that provide meaningful probability distributions for convergence analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from enum import Enum


class StoryCheckpointType(Enum):
    """Types of checkpoints in story sequences"""
    COMPREHENSION = "comprehension"  # "Are you following so far?"
    PREDICTION = "prediction"        # "What do you think will happen?"
    MORAL_EXTRACTION = "moral"       # "What's the lesson/moral?"
    REFLECTION = "reflection"        # "How does this relate to...?"


@dataclass
class StorySegment:
    """A segment of a story with optional checkpoint"""
    content: str
    is_checkpoint: bool = False
    checkpoint_type: Optional[StoryCheckpointType] = None
    checkpoint_prompt: Optional[str] = None
    expected_tokens: int = 50


@dataclass 
class TilliTonseStory:
    """Complete multi-turn story with checkpoints for alignment testing"""
    story_id: str
    capability: str  # truthfulness, safety, etc.
    title: str
    segments: List[StorySegment]
    cultural_context: str
    expected_convergence_points: List[str]
    total_expected_tokens: int


class TilliTonseFramework:
    """Framework for story-based multi-turn convergence analysis"""
    
    def __init__(self):
        self.stories: Dict[str, List[TilliTonseStory]] = {}
        self.checkpoint_responses: Dict[str, Dict[str, List[str]]] = {}
        
    def create_story_sequence(self, 
                            story_id: str,
                            capability: str,
                            title: str,
                            narrative_parts: List[str],
                            checkpoints: List[Tuple[int, StoryCheckpointType, str]],
                            cultural_context: str = "") -> TilliTonseStory:
        """
        Create a multi-turn story with embedded checkpoints.
        
        Args:
            story_id: Unique identifier for the story
            capability: Which alignment capability this tests
            title: Story title
            narrative_parts: List of story segments (non-checkpoint)
            checkpoints: List of (segment_index, type, prompt) tuples
            cultural_context: Cultural background/inspiration
        """
        segments = []
        checkpoint_map = {idx: (ctype, prompt) for idx, ctype, prompt in checkpoints}
        
        for i, part in enumerate(narrative_parts):
            # Add narrative segment
            segments.append(StorySegment(
                content=part,
                is_checkpoint=False,
                expected_tokens=len(part.split()) * 2  # Rough estimate
            ))
            
            # Add checkpoint if one exists for this position
            if i in checkpoint_map:
                ctype, prompt = checkpoint_map[i]
                segments.append(StorySegment(
                    content="",
                    is_checkpoint=True,
                    checkpoint_type=ctype,
                    checkpoint_prompt=prompt,
                    expected_tokens=50  # Checkpoints expect substantive responses
                ))
        
        # Calculate total expected tokens
        total_tokens = sum(seg.expected_tokens for seg in segments)
        
        story = TilliTonseStory(
            story_id=story_id,
            capability=capability,
            title=title,
            segments=segments,
            cultural_context=cultural_context,
            expected_convergence_points=[cp[2] for cp in checkpoints],
            total_expected_tokens=total_tokens
        )
        
        # Store story by capability
        if capability not in self.stories:
            self.stories[capability] = []
        self.stories[capability].append(story)
        
        return story
    
    def collect_story_responses(self, 
                              model,
                              story: TilliTonseStory) -> Dict[str, Any]:
        """
        Collect complete multi-turn response from a model for a story.
        
        This is the core innovation: instead of single Q&A, we get rich
        narrative responses with multiple checkpoints.
        """
        responses = []
        checkpoint_responses = []
        full_context = ""
        
        for i, segment in enumerate(story.segments):
            if segment.is_checkpoint:
                # This is a "tilli tonse" checkpoint
                checkpoint_prompt = self._format_checkpoint_prompt(
                    full_context, 
                    segment.checkpoint_type, 
                    segment.checkpoint_prompt
                )
                
                try:
                    response = model.generate(checkpoint_prompt)
                    checkpoint_responses.append({
                        "type": segment.checkpoint_type.value,
                        "prompt": segment.checkpoint_prompt,
                        "response": response,
                        "position": i
                    })
                    responses.append(response)
                    full_context += f"\n\nTilli tonse? {segment.checkpoint_prompt}\n{response}"
                    
                except Exception as e:
                    print(f"Error at checkpoint {i}: {e}")
                    responses.append("[ERROR: No response]")
                    
            else:
                # Regular narrative segment
                responses.append(segment.content)
                full_context += f"\n{segment.content}"
        
        # Combine all responses into rich sequence
        full_response = " ".join(responses)
        
        return {
            "story_id": story.story_id,
            "capability": story.capability,
            "full_response": full_response,
            "checkpoint_responses": checkpoint_responses,
            "response_length": len(full_response.split()),
            "context_used": full_context,
            "segments": len(story.segments)
        }
    
    def _format_checkpoint_prompt(self, 
                                context: str, 
                                checkpoint_type: StoryCheckpointType,
                                specific_prompt: str) -> str:
        """Format a checkpoint prompt with cultural context"""
        
        # Traditional Malawian checkpoint framing
        tilli_tonse_intro = "Tilli tonse? (Are we together? Are you following the story?)"
        
        if checkpoint_type == StoryCheckpointType.COMPREHENSION:
            return f"{context}\n\n{tilli_tonse_intro}\n\n{specific_prompt}\n\nPlease provide a thoughtful summary showing your understanding."
            
        elif checkpoint_type == StoryCheckpointType.PREDICTION:
            return f"{context}\n\n{tilli_tonse_intro}\n\n{specific_prompt}\n\nPlease share your predictions and reasoning."
            
        elif checkpoint_type == StoryCheckpointType.MORAL_EXTRACTION:
            return f"{context}\n\n{tilli_tonse_intro}\n\n{specific_prompt}\n\nPlease reflect on the deeper meaning and lessons."
            
        elif checkpoint_type == StoryCheckpointType.REFLECTION:
            return f"{context}\n\n{tilli_tonse_intro}\n\n{specific_prompt}\n\nPlease connect this to broader principles and ideas."
            
        else:
            return f"{context}\n\n{tilli_tonse_intro}\n\n{specific_prompt}"
    
    def analyze_story_convergence(self, 
                                model_responses: Dict[str, Dict[str, Any]],
                                story: TilliTonseStory) -> Dict[str, Any]:
        """
        Analyze convergence across models for a specific story.
        
        This provides multiple convergence measurements:
        1. Overall narrative convergence
        2. Checkpoint-specific convergence
        3. Moral/lesson convergence
        """
        
        # Extract full responses for overall convergence
        full_responses = {}
        for model_name, response_data in model_responses.items():
            full_responses[model_name] = [response_data["full_response"]]
        
        # Extract checkpoint responses for granular analysis
        checkpoint_convergence = {}
        for checkpoint_type in StoryCheckpointType:
            type_responses = {}
            for model_name, response_data in model_responses.items():
                type_specific = []
                for checkpoint in response_data["checkpoint_responses"]:
                    if checkpoint["type"] == checkpoint_type.value:
                        type_specific.append(checkpoint["response"])
                if type_specific:
                    type_responses[model_name] = type_specific
            
            if len(type_responses) >= 2:  # Need at least 2 models to compare
                checkpoint_convergence[checkpoint_type.value] = type_responses
        
        return {
            "story_id": story.story_id,
            "capability": story.capability,
            "full_narrative_responses": full_responses,
            "checkpoint_convergence": checkpoint_convergence,
            "response_stats": {
                model: {
                    "length": data["response_length"],
                    "segments": data["segments"],
                    "checkpoints": len(data["checkpoint_responses"])
                }
                for model, data in model_responses.items()
            }
        }
    
    def get_stories_for_capability(self, capability: str) -> List[TilliTonseStory]:
        """Get all stories for a specific capability"""
        return self.stories.get(capability, [])
    
    def get_all_capabilities(self) -> List[str]:
        """Get all capabilities with stories"""
        return list(self.stories.keys())
    
    def save_stories_to_file(self, filepath: str):
        """Save story collection to JSON file"""
        story_data = {}
        for capability, stories in self.stories.items():
            story_data[capability] = []
            for story in stories:
                story_dict = {
                    "story_id": story.story_id,
                    "capability": story.capability,
                    "title": story.title,
                    "cultural_context": story.cultural_context,
                    "total_expected_tokens": story.total_expected_tokens,
                    "segments": []
                }
                
                for segment in story.segments:
                    seg_dict = {
                        "content": segment.content,
                        "is_checkpoint": segment.is_checkpoint,
                        "expected_tokens": segment.expected_tokens
                    }
                    if segment.is_checkpoint:
                        seg_dict["checkpoint_type"] = segment.checkpoint_type.value
                        seg_dict["checkpoint_prompt"] = segment.checkpoint_prompt
                    story_dict["segments"].append(seg_dict)
                
                story_data[capability].append(story_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)
        
        print(f"📚 Saved {sum(len(stories) for stories in self.stories.values())} stories to {filepath}")


class TilliTonseAnalyzer:
    """Analyzer specifically designed for multi-turn story convergence"""
    
    def __init__(self, hybrid_analyzer=None):
        self.hybrid_analyzer = hybrid_analyzer
        self.framework = TilliTonseFramework()
    
    def run_story_experiment(self, 
                           models: List[Any],
                           capability: str,
                           max_stories: int = 10) -> Dict[str, Any]:
        """
        Run complete story-based experiment for a capability.
        
        This is the enhanced version of our convergence analysis using
        rich story sequences instead of simple Q&A.
        """
        
        stories = self.framework.get_stories_for_capability(capability)
        if not stories:
            raise ValueError(f"No stories found for capability: {capability}")
        
        # Limit stories if requested
        if max_stories:
            stories = stories[:max_stories]
        
        all_story_results = []
        
        for story in stories:
            print(f"\n📖 Testing story: {story.title}")
            print(f"   Expected tokens: {story.total_expected_tokens}")
            
            # Collect responses from all models
            model_responses = {}
            for model in models:
                print(f"   🤖 Collecting from {model.name}")
                response_data = self.framework.collect_story_responses(model, story)
                model_responses[model.name] = response_data
                print(f"      Generated {response_data['response_length']} tokens")
            
            # Analyze convergence for this story
            story_convergence = self.framework.analyze_story_convergence(
                model_responses, story
            )
            
            # Apply hybrid analysis if available
            if self.hybrid_analyzer:
                # Full narrative convergence
                full_responses = story_convergence["full_narrative_responses"]
                hybrid_results = self.hybrid_analyzer.analyze_hybrid_convergence(
                    full_responses, f"{capability}_story_{story.story_id}"
                )
                story_convergence["hybrid_analysis"] = hybrid_results
                
                # Checkpoint-specific convergence
                checkpoint_analyses = {}
                for checkpoint_type, responses in story_convergence["checkpoint_convergence"].items():
                    checkpoint_hybrid = self.hybrid_analyzer.analyze_hybrid_convergence(
                        responses, f"{capability}_{checkpoint_type}"
                    )
                    checkpoint_analyses[checkpoint_type] = checkpoint_hybrid
                story_convergence["checkpoint_hybrid_analyses"] = checkpoint_analyses
            
            all_story_results.append(story_convergence)
        
        # Aggregate results across all stories
        return self._aggregate_story_results(all_story_results, capability)
    
    def _aggregate_story_results(self, 
                               story_results: List[Dict[str, Any]], 
                               capability: str) -> Dict[str, Any]:
        """Aggregate convergence results across multiple stories"""
        
        if not story_results:
            return {"error": "No story results to aggregate"}
        
        # Aggregate hybrid convergence scores
        if "hybrid_analysis" in story_results[0]:
            hybrid_scores = []
            semantic_scores = []
            distributional_scores = []
            
            for result in story_results:
                if "hybrid_analysis" in result:
                    hybrid_scores.append(result["hybrid_analysis"]["hybrid_convergence_score"])
                    semantic_scores.append(result["hybrid_analysis"]["semantic_convergence_score"])
                    distributional_scores.append(result["hybrid_analysis"]["distributional_convergence_score"])
            
            aggregate_hybrid = {
                "mean_hybrid_convergence": np.mean(hybrid_scores),
                "std_hybrid_convergence": np.std(hybrid_scores),
                "mean_semantic_convergence": np.mean(semantic_scores),
                "mean_distributional_convergence": np.mean(distributional_scores),
                "story_count": len(hybrid_scores),
                "individual_scores": hybrid_scores
            }
        else:
            aggregate_hybrid = {"error": "No hybrid analysis available"}
        
        # Aggregate response statistics
        total_tokens = 0
        total_checkpoints = 0
        story_count = len(story_results)
        
        for result in story_results:
            for model_stats in result["response_stats"].values():
                total_tokens += model_stats["length"]
                total_checkpoints += model_stats["checkpoints"]
        
        avg_tokens_per_response = total_tokens / (story_count * len(story_results[0]["response_stats"]))
        avg_checkpoints_per_story = total_checkpoints / (story_count * len(story_results[0]["response_stats"]))
        
        return {
            "capability": capability,
            "story_experiment_results": {
                "story_count": story_count,
                "average_tokens_per_response": avg_tokens_per_response,
                "average_checkpoints_per_story": avg_checkpoints_per_story,
                "total_tokens_analyzed": total_tokens,
                "aggregate_hybrid_convergence": aggregate_hybrid,
                "individual_story_results": story_results
            },
            "methodology": "Tilli Tonse Framework - Multi-turn story-based convergence analysis",
            "cultural_inspiration": "Malawian oral storytelling traditions with audience check-ins"
        }