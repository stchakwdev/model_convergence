"""
Advanced Cost Monitoring and Budget Control System

This module provides comprehensive cost tracking, budget management, and spending
controls for the Universal Alignment Patterns research project. Designed to ensure
we stay within the $50 fellowship application budget with real-time monitoring.

Author: Samuel Tchakwera
Purpose: Rigorous cost control for research experiments
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings


@dataclass
class APICall:
    """Record of a single API call with cost information"""
    timestamp: datetime
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    prompt_text: str
    response_text: str
    capability: str
    prompt_id: str
    cached: bool = False


@dataclass
class BudgetAlert:
    """Budget alert with threshold and action"""
    threshold_percentage: float
    alert_message: str
    action: str  # "warn", "pause", "stop"
    triggered: bool = False
    triggered_at: Optional[datetime] = None


@dataclass
class CostReport:
    """Comprehensive cost analysis report"""
    total_cost_usd: float
    total_calls: int
    total_tokens: int
    cost_by_model: Dict[str, float]
    cost_by_capability: Dict[str, float]
    average_cost_per_call: float
    average_cost_per_token: float
    budget_remaining: float
    budget_utilization: float
    projected_full_experiment_cost: float
    estimated_completion_date: Optional[datetime]
    cost_efficiency_metrics: Dict[str, Any]


class CostMonitor:
    """
    Real-time cost monitoring and budget control system.
    
    Features:
    - Real-time cost tracking with token-level precision
    - Multi-level budget alerts and automatic shutoff
    - Cost optimization recommendations
    - Detailed spending analytics and projections
    - Emergency budget controls
    """
    
    def __init__(self, 
                 budget_limit_usd: float = 50.0,
                 cost_log_file: str = "results/cost_tracking.json",
                 enable_emergency_stop: bool = True):
        """
        Initialize the cost monitoring system.
        
        Args:
            budget_limit_usd: Maximum budget in USD for the entire experiment
            cost_log_file: File to store detailed cost logs
            enable_emergency_stop: Whether to enable automatic experiment stopping
        """
        self.budget_limit_usd = budget_limit_usd
        self.cost_log_file = Path(cost_log_file)
        self.enable_emergency_stop = enable_emergency_stop
        
        # Ensure log directory exists
        self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.api_calls: List[APICall] = []
        self.total_cost_usd = 0.0
        self.experiment_start_time = datetime.now()
        self.last_alert_time = None
        
        # Load existing cost data
        self._load_cost_history()
        
        # Configure budget alerts
        self.budget_alerts = [
            BudgetAlert(50.0, "📊 Budget Alert: 50% of budget used", "warn"),
            BudgetAlert(75.0, "⚠️  Budget Warning: 75% of budget used", "warn"),
            BudgetAlert(90.0, "🚨 Budget Critical: 90% of budget used", "pause"),
            BudgetAlert(95.0, "🛑 Budget Emergency: 95% of budget used", "stop"),
            BudgetAlert(100.0, "❌ Budget Exceeded: Stopping all experiments", "stop")
        ]
        
        # OpenRouter pricing (approximate - will update with real data)
        self.model_pricing = {
            # Free tier models
            "openai/gpt-oss-120b": {"input": 0.0, "output": 0.0},
            "deepseek/deepseek-chat": {"input": 0.0, "output": 0.0},
            "meta-llama/llama-3.1-70b:free": {"input": 0.0, "output": 0.0},
            
            # Low cost models
            "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},  # per 1M tokens
            "alibaba/qwen-2.5-72b": {"input": 0.8, "output": 2.4},
            "zhipu/glm-4.5": {"input": 1.0, "output": 3.0},
            
            # Default fallback
            "default": {"input": 1.0, "output": 3.0}
        }
        
        print(f"💰 Cost Monitor initialized with ${budget_limit_usd} budget")
        print(f"📊 Current spending: ${self.total_cost_usd:.4f}")
        print(f"🎯 Budget remaining: ${self.budget_remaining:.2f}")
    
    def record_api_call(self, 
                       model_id: str,
                       prompt_text: str, 
                       response_text: str,
                       capability: str,
                       prompt_id: str,
                       prompt_tokens: Optional[int] = None,
                       completion_tokens: Optional[int] = None,
                       actual_cost: Optional[float] = None,
                       cached: bool = False) -> float:
        """
        Record an API call and calculate its cost.
        
        Args:
            model_id: OpenRouter model identifier
            prompt_text: Input prompt text
            response_text: Generated response text
            capability: Capability being tested
            prompt_id: Unique prompt identifier
            prompt_tokens: Actual prompt tokens (if available)
            completion_tokens: Actual completion tokens (if available)
            actual_cost: Actual cost from API (if available)
            cached: Whether this was a cached response
            
        Returns:
            Cost of this API call in USD
        """
        
        # If cached, no cost
        if cached:
            call_cost = 0.0
        elif actual_cost is not None:
            call_cost = actual_cost
        else:
            # Estimate tokens if not provided
            if prompt_tokens is None:
                prompt_tokens = self._estimate_tokens(prompt_text)
            if completion_tokens is None:
                completion_tokens = self._estimate_tokens(response_text)
            
            # Calculate cost using model pricing
            call_cost = self._calculate_cost(model_id, prompt_tokens, completion_tokens)
        
        # Create API call record
        api_call = APICall(
            timestamp=datetime.now(),
            model_id=model_id,
            prompt_tokens=prompt_tokens or self._estimate_tokens(prompt_text),
            completion_tokens=completion_tokens or self._estimate_tokens(response_text),
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
            cost_usd=call_cost,
            prompt_text=prompt_text,
            response_text=response_text,
            capability=capability,
            prompt_id=prompt_id,
            cached=cached
        )
        
        # Add to records
        self.api_calls.append(api_call)
        self.total_cost_usd += call_cost
        
        # Save to disk
        self._save_cost_history()
        
        # Check budget alerts
        self._check_budget_alerts()
        
        # Print cost update
        if not cached:
            print(f"  💰 Cost: ${call_cost:.4f} | Total: ${self.total_cost_usd:.4f} | Remaining: ${self.budget_remaining:.2f}")
        
        return call_cost
    
    def estimate_experiment_cost(self, 
                                n_models: int,
                                n_capabilities: int, 
                                prompts_per_capability: int,
                                avg_prompt_length: int = 50,
                                avg_response_length: int = 200) -> Dict[str, Any]:
        """
        Estimate total cost for a full experiment.
        
        Args:
            n_models: Number of models to test
            n_capabilities: Number of capabilities to test
            prompts_per_capability: Number of prompts per capability
            avg_prompt_length: Average prompt length in tokens
            avg_response_length: Average response length in tokens
            
        Returns:
            Detailed cost estimation
        """
        
        total_calls = n_models * n_capabilities * prompts_per_capability
        
        # Calculate cost for each model type
        model_costs = {}
        total_estimated_cost = 0.0
        
        for model_id, pricing in self.model_pricing.items():
            if model_id == "default":
                continue
                
            calls_per_model = n_capabilities * prompts_per_capability
            prompt_cost = (avg_prompt_length * calls_per_model * pricing["input"]) / 1_000_000
            completion_cost = (avg_response_length * calls_per_model * pricing["output"]) / 1_000_000
            model_total = prompt_cost + completion_cost
            
            model_costs[model_id] = {
                "calls": calls_per_model,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": model_total,
                "cost_per_call": model_total / calls_per_model if calls_per_model > 0 else 0
            }
            
            total_estimated_cost += model_total
        
        # Safety margin
        safety_margin = 1.2  # 20% buffer
        total_with_buffer = total_estimated_cost * safety_margin
        
        return {
            "total_api_calls": total_calls,
            "estimated_cost_usd": total_estimated_cost,
            "cost_with_buffer": total_with_buffer,
            "within_budget": total_with_buffer <= self.budget_limit_usd,
            "budget_utilization": (total_with_buffer / self.budget_limit_usd) * 100,
            "model_breakdown": model_costs,
            "safety_margin": safety_margin,
            "recommendation": self._get_cost_recommendation(total_with_buffer)
        }
    
    def generate_cost_report(self) -> CostReport:
        """Generate comprehensive cost analysis report."""
        
        if not self.api_calls:
            return CostReport(
                total_cost_usd=0.0,
                total_calls=0,
                total_tokens=0,
                cost_by_model={},
                cost_by_capability={},
                average_cost_per_call=0.0,
                average_cost_per_token=0.0,
                budget_remaining=self.budget_limit_usd,
                budget_utilization=0.0,
                projected_full_experiment_cost=0.0,
                estimated_completion_date=None,
                cost_efficiency_metrics={}
            )
        
        # Calculate metrics
        cost_by_model = {}
        cost_by_capability = {}
        total_tokens = 0
        
        for call in self.api_calls:
            # By model
            if call.model_id not in cost_by_model:
                cost_by_model[call.model_id] = 0.0
            cost_by_model[call.model_id] += call.cost_usd
            
            # By capability  
            if call.capability not in cost_by_capability:
                cost_by_capability[call.capability] = 0.0
            cost_by_capability[call.capability] += call.cost_usd
            
            total_tokens += call.total_tokens
        
        # Calculate averages
        avg_cost_per_call = self.total_cost_usd / len(self.api_calls)
        avg_cost_per_token = self.total_cost_usd / total_tokens if total_tokens > 0 else 0.0
        
        # Project completion cost
        experiment_duration = datetime.now() - self.experiment_start_time
        if experiment_duration.total_seconds() > 0:
            cost_rate = self.total_cost_usd / (experiment_duration.total_seconds() / 3600)  # USD per hour
            projected_cost = cost_rate * 2  # Assume 2 more hours needed
        else:
            projected_cost = 0.0
        
        # Efficiency metrics
        cached_calls = sum(1 for call in self.api_calls if call.cached)
        cache_rate = cached_calls / len(self.api_calls) * 100
        
        efficiency_metrics = {
            "cache_hit_rate": cache_rate,
            "average_tokens_per_call": total_tokens / len(self.api_calls),
            "cost_per_thousand_tokens": (self.total_cost_usd / total_tokens) * 1000 if total_tokens > 0 else 0,
            "free_tier_utilization": sum(1 for call in self.api_calls if call.cost_usd == 0.0) / len(self.api_calls) * 100
        }
        
        return CostReport(
            total_cost_usd=self.total_cost_usd,
            total_calls=len(self.api_calls),
            total_tokens=total_tokens,
            cost_by_model=cost_by_model,
            cost_by_capability=cost_by_capability,
            average_cost_per_call=avg_cost_per_call,
            average_cost_per_token=avg_cost_per_token,
            budget_remaining=self.budget_remaining,
            budget_utilization=(self.total_cost_usd / self.budget_limit_usd) * 100,
            projected_full_experiment_cost=projected_cost,
            estimated_completion_date=None,
            cost_efficiency_metrics=efficiency_metrics
        )
    
    def check_budget_status(self) -> Dict[str, Any]:
        """Check current budget status and return summary."""
        
        budget_utilization = (self.total_cost_usd / self.budget_limit_usd) * 100
        
        if budget_utilization < 50:
            status = "GOOD"
            message = "Budget utilization is healthy"
        elif budget_utilization < 75:
            status = "CAUTION"
            message = "Approaching budget limits - monitor closely"
        elif budget_utilization < 90:
            status = "WARNING"
            message = "High budget utilization - consider cost optimization"
        elif budget_utilization < 100:
            status = "CRITICAL"
            message = "Budget nearly exhausted - immediate action required"
        else:
            status = "EXCEEDED"
            message = "Budget exceeded - stop all experiments"
        
        return {
            "status": status,
            "message": message,
            "budget_limit": self.budget_limit_usd,
            "current_spending": self.total_cost_usd,
            "budget_remaining": self.budget_remaining,
            "utilization_percentage": budget_utilization,
            "total_api_calls": len(self.api_calls),
            "can_continue": budget_utilization < 95
        }
    
    @property
    def budget_remaining(self) -> float:
        """Calculate remaining budget."""
        return max(0.0, self.budget_limit_usd - self.total_cost_usd)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple estimation: ~4 characters per token for English
        return max(1, len(text) // 4)
    
    def _calculate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for API call based on token usage."""
        
        pricing = self.model_pricing.get(model_id, self.model_pricing["default"])
        
        prompt_cost = (prompt_tokens * pricing["input"]) / 1_000_000
        completion_cost = (completion_tokens * pricing["output"]) / 1_000_000
        
        return prompt_cost + completion_cost
    
    def _check_budget_alerts(self):
        """Check if any budget thresholds have been crossed."""
        
        current_utilization = (self.total_cost_usd / self.budget_limit_usd) * 100
        
        for alert in self.budget_alerts:
            if current_utilization >= alert.threshold_percentage and not alert.triggered:
                alert.triggered = True
                alert.triggered_at = datetime.now()
                
                print(f"\\n{alert.alert_message}")
                print(f"Budget utilization: {current_utilization:.1f}%")
                
                if alert.action == "stop" and self.enable_emergency_stop:
                    print("🛑 EMERGENCY STOP: Budget limit reached - halting all experiments")
                    raise RuntimeError(f"Budget emergency stop triggered at {current_utilization:.1f}% utilization")
                elif alert.action == "pause":
                    print("⏸️  Experiment paused for budget review")
                    input("Press Enter to continue or Ctrl+C to stop...")
    
    def _get_cost_recommendation(self, estimated_cost: float) -> str:
        """Get recommendation based on estimated cost."""
        
        if estimated_cost <= self.budget_limit_usd * 0.5:
            return "PROCEED: Well within budget"
        elif estimated_cost <= self.budget_limit_usd * 0.8:
            return "CAUTION: Close to budget limit - monitor carefully"
        elif estimated_cost <= self.budget_limit_usd:
            return "WARNING: Very close to budget - consider reducing scope"
        else:
            return "STOP: Exceeds budget - reduce scope or increase budget"
    
    def _load_cost_history(self):
        """Load existing cost tracking data."""
        
        if self.cost_log_file.exists():
            try:
                with open(self.cost_log_file, 'r') as f:
                    data = json.load(f)
                
                # Load API calls
                for call_data in data.get("api_calls", []):
                    call_data["timestamp"] = datetime.fromisoformat(call_data["timestamp"])
                    if "triggered_at" in call_data and call_data["triggered_at"]:
                        call_data["triggered_at"] = datetime.fromisoformat(call_data["triggered_at"])
                    self.api_calls.append(APICall(**call_data))
                
                # Update total cost
                self.total_cost_usd = sum(call.cost_usd for call in self.api_calls)
                
                print(f"📂 Loaded {len(self.api_calls)} previous API calls")
                
            except Exception as e:
                print(f"⚠️  Could not load cost history: {e}")
    
    def _save_cost_history(self):
        """Save cost tracking data to disk."""
        
        try:
            data = {
                "budget_limit_usd": self.budget_limit_usd,
                "total_cost_usd": self.total_cost_usd,
                "experiment_start_time": self.experiment_start_time.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "api_calls": []
            }
            
            # Convert API calls to serializable format
            for call in self.api_calls:
                call_dict = asdict(call)
                call_dict["timestamp"] = call.timestamp.isoformat()
                if call_dict.get("triggered_at"):
                    call_dict["triggered_at"] = call_dict["triggered_at"].isoformat()
                data["api_calls"].append(call_dict)
            
            with open(self.cost_log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not save cost history: {e}")
    
    def print_cost_summary(self):
        """Print a formatted cost summary."""
        
        report = self.generate_cost_report()
        
        print(f"\\n{'='*60}")
        print(f"💰 COST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Spending: ${report.total_cost_usd:.4f}")
        print(f"Budget Remaining: ${report.budget_remaining:.2f}")
        print(f"Budget Utilization: {report.budget_utilization:.1f}%")
        print(f"Total API Calls: {report.total_calls}")
        print(f"Total Tokens: {report.total_tokens:,}")
        print(f"Avg Cost/Call: ${report.average_cost_per_call:.4f}")
        print(f"Cache Hit Rate: {report.cost_efficiency_metrics.get('cache_hit_rate', 0):.1f}%")
        
        if report.cost_by_model:
            print(f"\\n📊 Cost by Model:")
            for model, cost in sorted(report.cost_by_model.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model}: ${cost:.4f}")
        
        if report.cost_by_capability:
            print(f"\\n🎯 Cost by Capability:")
            for capability, cost in sorted(report.cost_by_capability.items(), key=lambda x: x[1], reverse=True):
                print(f"  {capability}: ${cost:.4f}")
        
        print(f"{'='*60}\\n")