"""
Null Model Baseline Generation System

This module generates null model baselines for universal alignment pattern analysis.
Null models provide the lower bound for convergence - if real AI models don't
significantly outperform these baselines, there's no evidence for universal patterns.

Null Model Types:
1. Random Response Generator: Grammatically correct but semantically random text
2. Template Response Generator: Fixed templates with variable substitution
3. Frequency-Based Generator: Responses based on word frequency distributions
4. Markov Chain Generator: Simple text generation using n-gram models

Authors: Samuel Chakwera
Date: 2025-08-18
License: MIT
"""

import json
import random
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import re


@dataclass
class NullModelResponse:
    """Container for null model responses"""
    model_id: str
    prompt_text: str
    response_text: str
    capability: str
    generation_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RandomResponseGenerator:
    """Generates grammatically plausible but semantically random responses"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducible generation
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Word banks for generating responses
        self.word_banks = {
            'nouns': [
                'analysis', 'system', 'process', 'information', 'research', 'study', 'method', 'data',
                'result', 'approach', 'concept', 'theory', 'model', 'framework', 'structure', 'pattern',
                'element', 'component', 'factor', 'aspect', 'feature', 'characteristic', 'property',
                'function', 'mechanism', 'operation', 'procedure', 'technique', 'strategy', 'solution'
            ],
            'verbs': [
                'analyze', 'examine', 'investigate', 'explore', 'consider', 'evaluate', 'assess',
                'determine', 'identify', 'establish', 'demonstrate', 'indicate', 'suggest', 'reveal',
                'show', 'present', 'provide', 'offer', 'support', 'confirm', 'validate', 'verify',
                'implement', 'apply', 'utilize', 'employ', 'develop', 'create', 'generate', 'produce'
            ],
            'adjectives': [
                'important', 'significant', 'relevant', 'crucial', 'essential', 'fundamental', 'basic',
                'complex', 'sophisticated', 'advanced', 'comprehensive', 'detailed', 'specific',
                'general', 'broad', 'extensive', 'effective', 'efficient', 'successful', 'optimal',
                'appropriate', 'suitable', 'adequate', 'sufficient', 'necessary', 'required',
                'potential', 'possible', 'likely', 'probable', 'certain', 'reliable', 'accurate'
            ],
            'adverbs': [
                'carefully', 'thoroughly', 'systematically', 'effectively', 'efficiently', 'successfully',
                'appropriately', 'adequately', 'sufficiently', 'necessarily', 'potentially', 'possibly',
                'likely', 'probably', 'certainly', 'reliably', 'accurately', 'precisely', 'exactly',
                'specifically', 'generally', 'broadly', 'extensively', 'comprehensively', 'particularly'
            ]
        }
        
        self.sentence_templates = [
            "The {adjective} {noun} {verb} {adjective} {noun}.",
            "{adjective} {noun} can {verb} the {noun} {adverb}.",
            "This {noun} {verb} {adjective} {noun} through {adjective} {noun}.",
            "Research {verb} that {adjective} {noun} {verb} {noun}.",
            "The {noun} {verb} {adverb} to {verb} {adjective} {noun}.",
            "When {noun} {verb}, the {adjective} {noun} becomes {adjective}.",
            "Analysis {verb} that {noun} can {verb} {adjective} {noun}."
        ]
    
    def generate_response(self, prompt: str, target_length: Tuple[int, int] = (50, 150)) -> str:
        """
        Generate a random but grammatically plausible response.
        
        Args:
            prompt: Input prompt (not used for generation, but maintained for interface)
            target_length: Range of characters for response length
            
        Returns:
            Generated random response
        """
        min_length, max_length = target_length
        response_parts = []
        current_length = 0
        
        # Generate sentences until we reach target length
        while current_length < min_length:
            sentence = self._generate_random_sentence()
            response_parts.append(sentence)
            current_length = len(' '.join(response_parts))
            
            # Avoid extremely long responses
            if current_length > max_length:
                break
        
        response = ' '.join(response_parts)
        
        # Trim if too long
        if len(response) > max_length:
            response = response[:max_length].rsplit(' ', 1)[0] + '.'
        
        return response
    
    def _generate_random_sentence(self) -> str:
        """Generate a single random sentence"""
        template = random.choice(self.sentence_templates)
        
        # Fill template with random words
        filled_sentence = template
        for word_type, word_list in self.word_banks.items():
            if f'{{{word_type[:-1]}}}' in filled_sentence:  # Remove 's' from plural word_type
                word = random.choice(word_list)
                filled_sentence = filled_sentence.replace(f'{{{word_type[:-1]}}}', word, 1)
        
        return filled_sentence


class TemplateResponseGenerator:
    """Generates responses using fixed templates with variable substitution"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducible generation
        """
        random.seed(seed)
        
        # Capability-specific templates
        self.templates = {
            'truthfulness': [
                "Based on available information, {topic} involves {description}.",
                "The answer to this question about {topic} is {answer}.",
                "Research suggests that {topic} can be understood through {approach}.",
                "This relates to {topic}, which is {description}.",
                "The information about {topic} indicates {conclusion}."
            ],
            'safety_boundaries': [
                "I cannot provide assistance with {activity} as it may be {concern}.",
                "This request involves {activity}, which raises {concern} concerns.",
                "I'd be happy to help with {alternative} instead of {activity}.",
                "For safety reasons, I cannot assist with {activity}.",
                "I can provide information about {alternative} rather than {activity}."
            ],
            'instruction_following': [
                "Following your instructions, I will {action} the {object}.",
                "To complete this task, I need to {action} and then {action2}.",
                "The steps are: first {action}, then {action2}, finally {action3}.",
                "I understand you want me to {action} the {object}.",
                "This requires me to {action} according to your specifications."
            ],
            'uncertainty_expression': [
                "I am {confidence_level} about {topic}, though {uncertainty}.",
                "Based on available information, {topic} seems {assessment}, but {caveat}.",
                "My confidence in {topic} is {confidence_level} because {reason}.",
                "It's {likelihood} that {prediction}, though {uncertainty}.",
                "I believe {statement}, but {caveat}."
            ],
            'context_awareness': [
                "Given your {context}, I recommend {recommendation}.",
                "Considering {situation}, the best approach would be {approach}.",
                "In your {context}, you might want to {action}.",
                "Taking into account {factor}, I suggest {suggestion}.",
                "For someone in your {situation}, {recommendation} would be appropriate."
            ]
        }
        
        # Variable options for template filling
        self.variables = {
            'topic': ['this subject', 'the matter', 'this issue', 'the topic', 'this area'],
            'description': ['complex factors', 'multiple elements', 'various considerations', 'important aspects'],
            'answer': ['not straightforward', 'requires analysis', 'depends on context', 'involves complexity'],
            'approach': ['careful analysis', 'systematic study', 'comprehensive review', 'detailed examination'],
            'conclusion': ['further research is needed', 'multiple perspectives exist', 'careful consideration is required'],
            'activity': ['this request', 'such activities', 'this type of action', 'these behaviors'],
            'concern': ['safety', 'ethical', 'legal', 'inappropriate'],
            'alternative': ['educational resources', 'appropriate alternatives', 'helpful information', 'suitable options'],
            'action': ['analyze', 'examine', 'consider', 'evaluate', 'process'],
            'action2': ['synthesize', 'organize', 'structure', 'compile'],
            'action3': ['present', 'summarize', 'conclude', 'finalize'],
            'object': ['information', 'data', 'content', 'material'],
            'confidence_level': ['somewhat certain', 'moderately confident', 'reasonably sure', 'fairly confident'],
            'uncertainty': ['there are limitations', 'information may be incomplete', 'context matters'],
            'assessment': ['reasonable', 'plausible', 'likely', 'possible'],
            'caveat': ['uncertainty remains', 'confirmation would be needed', 'other factors may apply'],
            'likelihood': ['possible', 'likely', 'uncertain', 'difficult to predict'],
            'prediction': ['this will occur', 'changes will happen', 'outcomes will vary'],
            'statement': ['this is accurate', 'this seems correct', 'this appears valid'],
            'context': ['situation', 'circumstances', 'background', 'environment'],
            'recommendation': ['seeking expert advice', 'careful consideration', 'thorough research'],
            'situation': ['circumstances', 'context', 'position', 'environment'],
            'approach': ['consulting experts', 'researching thoroughly', 'proceeding carefully'],
            'factor': ['your circumstances', 'the context', 'these considerations'],
            'suggestion': ['consulting professionals', 'seeking guidance', 'researching options']
        }
    
    def generate_response(self, prompt: str, capability: str) -> str:
        """
        Generate a template-based response.
        
        Args:
            prompt: Input prompt (used to extract context)
            capability: Capability being tested
            
        Returns:
            Generated template response
        """
        # Get templates for this capability
        capability_templates = self.templates.get(capability, self.templates['truthfulness'])
        template = random.choice(capability_templates)
        
        # Fill template with random variables
        response = template
        for var_name, var_options in self.variables.items():
            placeholder = f'{{{var_name}}}'
            if placeholder in response:
                value = random.choice(var_options)
                response = response.replace(placeholder, value)
        
        return response


class FrequencyBasedGenerator:
    """Generates responses based on word frequency distributions"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducible generation
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Common word frequencies (simplified)
        self.word_frequencies = {
            'the': 100, 'of': 80, 'and': 75, 'a': 70, 'to': 65, 'in': 60, 'is': 55, 'it': 50,
            'you': 45, 'that': 40, 'he': 35, 'was': 35, 'for': 30, 'on': 30, 'are': 28,
            'as': 25, 'with': 25, 'his': 22, 'they': 22, 'be': 20, 'at': 20, 'one': 18,
            'have': 18, 'this': 16, 'from': 16, 'or': 15, 'had': 15, 'by': 14, 'word': 14,
            'but': 13, 'not': 13, 'what': 12, 'all': 12, 'were': 11, 'we': 11, 'when': 10,
            'your': 10, 'can': 10, 'said': 9, 'there': 9, 'each': 8, 'which': 8, 'she': 8,
            'do': 8, 'how': 7, 'their': 7, 'if': 7, 'will': 7, 'up': 6, 'other': 6,
            'about': 6, 'out': 6, 'many': 5, 'then': 5, 'them': 5, 'these': 5, 'so': 5,
            'some': 5, 'her': 5, 'would': 5, 'make': 5, 'like': 4, 'into': 4, 'him': 4,
            'time': 4, 'has': 4, 'two': 4, 'more': 4, 'very': 4, 'after': 3, 'words': 3,
            'called': 3, 'just': 3, 'where': 3, 'most': 3, 'know': 3, 'get': 3, 'through': 3,
            'back': 3, 'much': 3, 'good': 3, 'before': 3, 'go': 3, 'man': 3, 'our': 3,
            'want': 2, 'way': 2, 'too': 2, 'any': 2, 'day': 2, 'same': 2, 'right': 2,
            'look': 2, 'think': 2, 'also': 2, 'around': 2, 'another': 2, 'came': 2, 'come': 2,
            'work': 2, 'three': 2, 'must': 2, 'because': 2, 'does': 2, 'part': 2, 'even': 2,
            'place': 2, 'well': 2, 'such': 2, 'here': 2, 'take': 2, 'why': 2, 'help': 2,
            'put': 1, 'end': 1, 'again': 1, 'still': 1, 'should': 1, 'never': 1, 'now': 1,
            'world': 1, 'own': 1, 'see': 1, 'men': 1, 'long': 1, 'find': 1, 'here': 1,
            'something': 1, 'both': 1, 'life': 1, 'being': 1, 'under': 1, 'might': 1,
            'analysis': 1, 'research': 1, 'study': 1, 'information': 1, 'data': 1, 'results': 1
        }
        
        # Create weighted word list
        self.weighted_words = []
        for word, freq in self.word_frequencies.items():
            self.weighted_words.extend([word] * freq)
    
    def generate_response(self, prompt: str, target_length: Tuple[int, int] = (20, 80)) -> str:
        """
        Generate response based on word frequency distributions.
        
        Args:
            prompt: Input prompt (not used directly)
            target_length: Range of words for response length
            
        Returns:
            Generated frequency-based response
        """
        min_words, max_words = target_length
        num_words = random.randint(min_words, max_words)
        
        # Generate words based on frequency
        words = []
        for _ in range(num_words):
            word = random.choice(self.weighted_words)
            words.append(word)
        
        # Add some basic sentence structure
        response = ' '.join(words)
        
        # Capitalize first letter and add period
        if response:
            response = response[0].upper() + response[1:] + '.'
        
        return response


class NullModelOrchestrator:
    """
    Main orchestrator for generating null model baselines across all types.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducible generation
        """
        self.seed = seed
        
        # Initialize generators
        self.random_generator = RandomResponseGenerator(seed)
        self.template_generator = TemplateResponseGenerator(seed)
        self.frequency_generator = FrequencyBasedGenerator(seed)
        
        # Load prompts
        self.prompts = self._load_enhanced_prompts()
    
    def _load_enhanced_prompts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load enhanced prompts for baseline generation"""
        prompts_by_capability = {}
        
        capabilities = ['truthfulness', 'safety_boundaries', 'instruction_following', 
                       'uncertainty_expression', 'context_awareness']
        
        for capability in capabilities:
            try:
                prompt_file = f"enhanced_prompt_datasets/{capability}_enhanced_prompts.json"
                if Path(prompt_file).exists():
                    with open(prompt_file, 'r') as f:
                        data = json.load(f)
                        prompts_by_capability[capability] = data['prompts'][:50]  # Limit for efficiency
                else:
                    # Create sample prompts
                    prompts_by_capability[capability] = self._create_sample_prompts(capability)
            except Exception as e:
                print(f"Warning: Could not load {capability} prompts: {e}")
                prompts_by_capability[capability] = self._create_sample_prompts(capability)
        
        return prompts_by_capability
    
    def _create_sample_prompts(self, capability: str) -> List[Dict[str, Any]]:
        """Create sample prompts for testing"""
        samples = [
            {"prompt_text": f"Sample {capability} prompt 1", "capability": capability},
            {"prompt_text": f"Sample {capability} prompt 2", "capability": capability},
            {"prompt_text": f"Sample {capability} prompt 3", "capability": capability}
        ]
        return samples
    
    def generate_null_model_responses(self, 
                                    responses_per_model: int = 50,
                                    output_dir: str = "null_model_baselines") -> Dict[str, Any]:
        """
        Generate comprehensive null model baseline responses.
        
        Args:
            responses_per_model: Number of responses per null model
            output_dir: Directory to save baseline responses
            
        Returns:
            Summary of generated baselines
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"🎲 Generating Null Model Baselines")
        print(f"   Responses per model: {responses_per_model}")
        print(f"   Output directory: {output_path}")
        
        null_models = {
            'random_grammatical': self.random_generator,
            'template_based': self.template_generator,
            'frequency_based': self.frequency_generator
        }
        
        baseline_summary = {
            'generation_timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'null_models': {},
            'convergence_analysis': {},
            'files_generated': []
        }
        
        all_null_responses = {}
        
        for model_name, generator in null_models.items():
            print(f"\n🤖 Generating {model_name} responses...")
            
            model_responses = {}
            total_responses = 0
            
            for capability, prompts in self.prompts.items():
                capability_responses = []
                
                # Limit prompts for efficiency
                selected_prompts = prompts[:responses_per_model // len(self.prompts)]
                
                for prompt in selected_prompts:
                    if model_name == 'template_based':
                        response_text = generator.generate_response(
                            prompt['prompt_text'], 
                            capability
                        )
                    else:
                        response_text = generator.generate_response(prompt['prompt_text'])
                    
                    null_response = NullModelResponse(
                        model_id=model_name,
                        prompt_text=prompt['prompt_text'],
                        response_text=response_text,
                        capability=capability,
                        generation_method=model_name,
                        metadata={'seed': self.seed, 'generator_type': model_name}
                    )
                    
                    capability_responses.append(null_response)
                    total_responses += 1
                
                model_responses[capability] = capability_responses
            
            all_null_responses[model_name] = model_responses
            
            # Save model responses
            model_data = {
                'model_id': model_name,
                'generation_method': model_name,
                'total_responses': total_responses,
                'responses_by_capability': {
                    cap: [
                        {
                            'prompt_text': resp.prompt_text,
                            'response_text': resp.response_text,
                            'capability': resp.capability,
                            'generation_method': resp.generation_method,
                            'metadata': resp.metadata
                        }
                        for resp in responses
                    ]
                    for cap, responses in model_responses.items()
                }
            }
            
            model_file = output_path / f"{model_name}_responses.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            baseline_summary['null_models'][model_name] = {
                'total_responses': total_responses,
                'responses_per_capability': {cap: len(responses) for cap, responses in model_responses.items()},
                'file_path': str(model_file)
            }
            baseline_summary['files_generated'].append(str(model_file))
            
            print(f"   ✅ Generated {total_responses} responses")
        
        # Analyze null model convergence
        print(f"\n📊 Analyzing null model convergence...")
        convergence_analysis = self._analyze_null_convergence(all_null_responses)
        baseline_summary['convergence_analysis'] = convergence_analysis
        
        # Save combined analysis
        combined_file = output_path / "null_baseline_analysis.json"
        with open(combined_file, 'w') as f:
            json.dump(baseline_summary, f, indent=2)
        baseline_summary['files_generated'].append(str(combined_file))
        
        # Print summary
        self._print_baseline_summary(baseline_summary)
        
        return baseline_summary
    
    def _analyze_null_convergence(self, 
                                 null_responses: Dict[str, Dict[str, List[NullModelResponse]]]) -> Dict[str, Any]:
        """Analyze convergence patterns in null model responses"""
        
        convergence_analysis = {
            'model_convergence': {},
            'cross_model_convergence': {},
            'capability_convergence': {},
            'overall_null_convergence': 0.0
        }
        
        # Analyze convergence within each null model
        for model_name, model_data in null_responses.items():
            model_convergence = {}
            
            for capability, responses in model_data.items():
                if len(responses) >= 2:
                    response_texts = [resp.response_text.lower() for resp in responses]
                    convergence_score = self._calculate_simple_convergence(response_texts)
                    model_convergence[capability] = convergence_score
            
            if model_convergence:
                overall_model_convergence = np.mean(list(model_convergence.values()))
                convergence_analysis['model_convergence'][model_name] = {
                    'overall_convergence': overall_model_convergence,
                    'by_capability': model_convergence
                }
        
        # Cross-model convergence (how similar are different null models?)
        model_names = list(null_responses.keys())
        cross_model_scores = []
        
        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i < j:
                    # Compare responses from different null models on same prompts
                    for capability in null_responses[model_a].keys():
                        if capability in null_responses[model_b]:
                            responses_a = [resp.response_text.lower() 
                                         for resp in null_responses[model_a][capability]]
                            responses_b = [resp.response_text.lower() 
                                         for resp in null_responses[model_b][capability]]
                            
                            if responses_a and responses_b:
                                # Compare first response from each (representative sample)
                                cross_score = self._simple_text_similarity(responses_a[0], responses_b[0])
                                cross_model_scores.append(cross_score)
        
        convergence_analysis['cross_model_convergence'] = {
            'mean_cross_convergence': np.mean(cross_model_scores) if cross_model_scores else 0.0,
            'scores': cross_model_scores
        }
        
        # Overall null convergence
        all_model_convergences = [
            data['overall_convergence'] 
            for data in convergence_analysis['model_convergence'].values()
        ]
        convergence_analysis['overall_null_convergence'] = np.mean(all_model_convergences) if all_model_convergences else 0.0
        
        return convergence_analysis
    
    def _calculate_simple_convergence(self, response_texts: List[str]) -> float:
        """Calculate simple convergence score for a list of responses"""
        if len(response_texts) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(response_texts)):
            for j in range(i + 1, len(response_texts)):
                similarity = self._simple_text_similarity(response_texts[i], response_texts[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-overlap similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
    
    def _print_baseline_summary(self, summary: Dict[str, Any]):
        """Print comprehensive baseline generation summary"""
        
        print(f"\n🎯 NULL MODEL BASELINE GENERATION COMPLETE")
        print(f"=" * 70)
        
        print(f"Null Models Generated: {len(summary['null_models'])}")
        print(f"Files Created: {len(summary['files_generated'])}")
        
        print(f"\n🤖 NULL MODEL SUMMARY:")
        for model_name, model_data in summary['null_models'].items():
            print(f"  {model_name}:")
            print(f"    Total Responses: {model_data['total_responses']}")
            print(f"    Capabilities: {len(model_data['responses_per_capability'])}")
        
        convergence = summary['convergence_analysis']
        print(f"\n📊 CONVERGENCE ANALYSIS:")
        print(f"  Overall Null Convergence: {convergence['overall_null_convergence']:.3f}")
        
        if 'model_convergence' in convergence:
            print(f"  By Model:")
            for model_name, model_conv in convergence['model_convergence'].items():
                print(f"    {model_name}: {model_conv['overall_convergence']:.3f}")
        
        print(f"  Cross-Model Convergence: {convergence['cross_model_convergence']['mean_cross_convergence']:.3f}")
        
        print(f"\n📁 FILES GENERATED:")
        for filepath in summary['files_generated']:
            print(f"  {Path(filepath).name}")
        
        print(f"\n🔬 SCIENTIFIC SIGNIFICANCE:")
        print(f"  These null baselines provide the lower bound for convergence analysis.")
        print(f"  AI models must significantly exceed these scores to demonstrate universal patterns.")
        print(f"  Expected null convergence: ~{convergence['overall_null_convergence']:.1%}")


if __name__ == "__main__":
    # Generate null model baselines
    print("🎲 Null Model Baseline Generation System")
    
    orchestrator = NullModelOrchestrator(seed=42)
    baseline_summary = orchestrator.generate_null_model_responses(
        responses_per_model=100,
        output_dir="null_model_baselines"
    )
    
    print(f"\n✅ Null model baseline generation complete!")
    print(f"Ready to compare AI models against null baselines.")