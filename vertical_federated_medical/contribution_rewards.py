"""
Federated Contribution Reward System
Rewards SuperNodes based on their contribution to the global model
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import json
import time
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContributionMetric(Enum):
    """Different metrics for measuring contribution"""
    DATA_QUALITY = "data_quality"
    DATA_QUANTITY = "data_quantity"
    MODEL_PERFORMANCE = "model_performance"
    PRIVACY_COMPLIANCE = "privacy_compliance"
    PARTICIPATION_RATE = "participation_rate"
    DIVERSITY = "diversity"
    INNOVATION = "innovation"

@dataclass
class ContributionScore:
    """Individual contribution score for a metric"""
    metric: ContributionMetric
    score: float
    weight: float
    explanation: str

@dataclass
class OrganizationContribution:
    """Overall contribution assessment for an organization"""
    org_id: str
    total_score: float
    individual_scores: Dict[ContributionMetric, ContributionScore]
    rank: int
    reward_amount: float
    timestamp: float

class ContributionRewardSystem:
    """System for calculating and distributing rewards based on contributions"""
    
    def __init__(self, 
                 total_reward_pool: float = 1000.0,
                 reward_distribution_method: str = "proportional"):
        self.total_reward_pool = total_reward_pool
        self.reward_distribution_method = reward_distribution_method
        self.contribution_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        self.metric_weights = self._initialize_metric_weights()
        
    def _initialize_metric_weights(self) -> Dict[ContributionMetric, float]:
        """Initialize weights for different contribution metrics"""
        return {
            ContributionMetric.DATA_QUALITY: 0.25,
            ContributionMetric.DATA_QUANTITY: 0.15,
            ContributionMetric.MODEL_PERFORMANCE: 0.30,
            ContributionMetric.PRIVACY_COMPLIANCE: 0.15,
            ContributionMetric.PARTICIPATION_RATE: 0.10,
            ContributionMetric.DIVERSITY: 0.05
        }
    
    def calculate_contribution_scores(self, 
                                    round_results: Dict[str, Any],
                                    organization_data: Dict[str, Any]) -> Dict[str, OrganizationContribution]:
        """Calculate contribution scores for all organizations"""
        contributions = {}
        
        for org_id in organization_data.keys():
            # Calculate individual metric scores
            individual_scores = {}
            
            # Data Quality Score
            data_quality_score = self._calculate_data_quality_score(org_id, organization_data[org_id])
            individual_scores[ContributionMetric.DATA_QUALITY] = ContributionScore(
                metric=ContributionMetric.DATA_QUALITY,
                score=data_quality_score,
                weight=self.metric_weights[ContributionMetric.DATA_QUALITY],
                explanation=f"Data quality score: {data_quality_score:.3f}"
            )
            
            # Data Quantity Score
            data_quantity_score = self._calculate_data_quantity_score(org_id, organization_data[org_id])
            individual_scores[ContributionMetric.DATA_QUANTITY] = ContributionScore(
                metric=ContributionMetric.DATA_QUANTITY,
                score=data_quantity_score,
                weight=self.metric_weights[ContributionMetric.DATA_QUANTITY],
                explanation=f"Data quantity score: {data_quantity_score:.3f}"
            )
            
            # Model Performance Score
            model_performance_score = self._calculate_model_performance_score(org_id, round_results)
            individual_scores[ContributionMetric.MODEL_PERFORMANCE] = ContributionScore(
                metric=ContributionMetric.MODEL_PERFORMANCE,
                score=model_performance_score,
                weight=self.metric_weights[ContributionMetric.MODEL_PERFORMANCE],
                explanation=f"Model performance contribution: {model_performance_score:.3f}"
            )
            
            # Privacy Compliance Score
            privacy_score = self._calculate_privacy_compliance_score(org_id, organization_data[org_id])
            individual_scores[ContributionMetric.PRIVACY_COMPLIANCE] = ContributionScore(
                metric=ContributionMetric.PRIVACY_COMPLIANCE,
                score=privacy_score,
                weight=self.metric_weights[ContributionMetric.PRIVACY_COMPLIANCE],
                explanation=f"Privacy compliance score: {privacy_score:.3f}"
            )
            
            # Participation Rate Score
            participation_score = self._calculate_participation_rate_score(org_id, round_results)
            individual_scores[ContributionMetric.PARTICIPATION_RATE] = ContributionScore(
                metric=ContributionMetric.PARTICIPATION_RATE,
                score=participation_score,
                weight=self.metric_weights[ContributionMetric.PARTICIPATION_RATE],
                explanation=f"Participation rate: {participation_score:.3f}"
            )
            
            # Diversity Score
            diversity_score = self._calculate_diversity_score(org_id, organization_data[org_id])
            individual_scores[ContributionMetric.DIVERSITY] = ContributionScore(
                metric=ContributionMetric.DIVERSITY,
                score=diversity_score,
                weight=self.metric_weights[ContributionMetric.DIVERSITY],
                explanation=f"Data diversity score: {diversity_score:.3f}"
            )
            
            # Calculate total weighted score
            total_score = sum(
                score.score * score.weight 
                for score in individual_scores.values()
            )
            
            contribution = OrganizationContribution(
                org_id=org_id,
                total_score=total_score,
                individual_scores=individual_scores,
                rank=0,  # Will be set after ranking
                reward_amount=0.0,  # Will be calculated
                timestamp=time.time()
            )
            
            contributions[org_id] = contribution
        
        # Rank organizations
        ranked_contributions = self._rank_organizations(contributions)
        
        # Calculate rewards
        final_contributions = self._calculate_rewards(ranked_contributions)
        
        # Store in history
        for org_id, contribution in final_contributions.items():
            self.contribution_history[org_id].append(contribution)
            self.reward_history[org_id].append(contribution.reward_amount)
        
        return final_contributions
    
    def _calculate_data_quality_score(self, org_id: str, org_data: Dict[str, Any]) -> float:
        """Calculate data quality score based on various factors"""
        # Completeness
        completeness = org_data.get('completeness', 0.8)
        
        # Consistency
        consistency = org_data.get('consistency', 0.8)
        
        # Accuracy
        accuracy = org_data.get('accuracy', 0.8)
        
        # Timeliness
        timeliness = org_data.get('timeliness', 0.8)
        
        # Validity
        validity = org_data.get('validity', 0.8)
        
        # Weighted average
        quality_score = (
            completeness * 0.25 +
            consistency * 0.25 +
            accuracy * 0.25 +
            timeliness * 0.15 +
            validity * 0.10
        )
        
        return min(quality_score, 1.0)
    
    def _calculate_data_quantity_score(self, org_id: str, org_data: Dict[str, Any]) -> float:
        """Calculate data quantity score"""
        num_samples = org_data.get('num_samples', 1000)
        num_features = org_data.get('num_features', 10)
        
        # Normalize by expected values
        sample_score = min(num_samples / 10000, 1.0)  # Max at 10k samples
        feature_score = min(num_features / 100, 1.0)  # Max at 100 features
        
        # Combined score
        quantity_score = (sample_score * 0.7 + feature_score * 0.3)
        
        return quantity_score
    
    def _calculate_model_performance_score(self, org_id: str, round_results: Dict[str, Any]) -> float:
        """Calculate model performance contribution score"""
        # Get organization-specific metrics
        accuracy = round_results.get(f"{org_id}_accuracy", 0.5)
        loss = round_results.get(f"{org_id}_loss", 1.0)
        
        # Convert loss to score (lower loss = higher score)
        loss_score = max(0, 1.0 - loss)
        
        # Combine accuracy and loss
        performance_score = (accuracy * 0.7 + loss_score * 0.3)
        
        return performance_score
    
    def _calculate_privacy_compliance_score(self, org_id: str, org_data: Dict[str, Any]) -> float:
        """Calculate privacy compliance score"""
        # Privacy budget usage
        privacy_budget_usage = org_data.get('privacy_budget_usage', 0.1)
        privacy_score = max(0, 1.0 - privacy_budget_usage)
        
        # Differential privacy implementation
        dp_implementation = org_data.get('differential_privacy', True)
        dp_score = 1.0 if dp_implementation else 0.5
        
        # Data anonymization
        anonymization_score = org_data.get('anonymization_quality', 0.8)
        
        # Combined privacy score
        compliance_score = (
            privacy_score * 0.4 +
            dp_score * 0.3 +
            anonymization_score * 0.3
        )
        
        return compliance_score
    
    def _calculate_participation_rate_score(self, org_id: str, round_results: Dict[str, Any]) -> float:
        """Calculate participation rate score"""
        total_rounds = round_results.get('total_rounds', 10)
        participated_rounds = round_results.get(f"{org_id}_participated_rounds", 8)
        
        participation_rate = participated_rounds / total_rounds if total_rounds > 0 else 0.0
        
        return participation_rate
    
    def _calculate_diversity_score(self, org_id: str, org_data: Dict[str, Any]) -> float:
        """Calculate data diversity score"""
        # Data modality diversity
        num_modalities = org_data.get('num_modalities', 1)
        modality_diversity = min(num_modalities / 5, 1.0)  # Max 5 modalities
        
        # Demographic diversity
        demographic_diversity = org_data.get('demographic_diversity', 0.5)
        
        # Geographic diversity
        geographic_diversity = org_data.get('geographic_diversity', 0.5)
        
        # Temporal diversity
        temporal_diversity = org_data.get('temporal_diversity', 0.5)
        
        # Combined diversity score
        diversity_score = (
            modality_diversity * 0.3 +
            demographic_diversity * 0.3 +
            geographic_diversity * 0.2 +
            temporal_diversity * 0.2
        )
        
        return diversity_score
    
    def _rank_organizations(self, contributions: Dict[str, OrganizationContribution]) -> Dict[str, OrganizationContribution]:
        """Rank organizations by their contribution scores"""
        sorted_orgs = sorted(
            contributions.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )
        
        # Assign ranks
        for rank, (org_id, contribution) in enumerate(sorted_orgs, 1):
            contribution.rank = rank
        
        return dict(sorted_orgs)
    
    def _calculate_rewards(self, contributions: Dict[str, OrganizationContribution]) -> Dict[str, OrganizationContribution]:
        """Calculate reward amounts for each organization"""
        if self.reward_distribution_method == "proportional":
            return self._calculate_proportional_rewards(contributions)
        elif self.reward_distribution_method == "ranked":
            return self._calculate_ranked_rewards(contributions)
        elif self.reward_distribution_method == "hybrid":
            return self._calculate_hybrid_rewards(contributions)
        else:
            raise ValueError(f"Unknown reward distribution method: {self.reward_distribution_method}")
    
    def _calculate_proportional_rewards(self, contributions: Dict[str, OrganizationContribution]) -> Dict[str, OrganizationContribution]:
        """Calculate proportional rewards based on contribution scores"""
        total_score = sum(contribution.total_score for contribution in contributions.values())
        
        if total_score == 0:
            # Equal distribution if no scores
            equal_reward = self.total_reward_pool / len(contributions)
            for contribution in contributions.values():
                contribution.reward_amount = equal_reward
        else:
            # Proportional distribution
            for contribution in contributions.values():
                proportion = contribution.total_score / total_score
                contribution.reward_amount = proportion * self.total_reward_pool
        
        return contributions
    
    def _calculate_ranked_rewards(self, contributions: Dict[str, OrganizationContribution]) -> Dict[str, OrganizationContribution]:
        """Calculate ranked rewards with exponential decay"""
        num_orgs = len(contributions)
        if num_orgs == 0:
            return contributions
        
        # Exponential decay weights
        weights = [math.exp(-0.5 * rank) for rank in range(1, num_orgs + 1)]
        total_weight = sum(weights)
        
        for contribution in contributions.values():
            weight = weights[contribution.rank - 1]
            proportion = weight / total_weight
            contribution.reward_amount = proportion * self.total_reward_pool
        
        return contributions
    
    def _calculate_hybrid_rewards(self, contributions: Dict[str, OrganizationContribution]) -> Dict[str, OrganizationContribution]:
        """Calculate hybrid rewards combining proportional and ranked methods"""
        # 70% proportional, 30% ranked
        proportional_contributions = self._calculate_proportional_rewards(contributions.copy())
        ranked_contributions = self._calculate_ranked_rewards(contributions.copy())
        
        for org_id in contributions:
            proportional_reward = proportional_contributions[org_id].reward_amount
            ranked_reward = ranked_contributions[org_id].reward_amount
            
            hybrid_reward = proportional_reward * 0.7 + ranked_reward * 0.3
            contributions[org_id].reward_amount = hybrid_reward
        
        return contributions
    
    def get_contribution_summary(self, org_id: str, num_rounds: int = 10) -> Dict[str, Any]:
        """Get contribution summary for an organization"""
        if org_id not in self.contribution_history:
            return {"error": "Organization not found"}
        
        recent_contributions = self.contribution_history[org_id][-num_rounds:]
        
        if not recent_contributions:
            return {"error": "No contribution history"}
        
        # Calculate averages
        avg_total_score = np.mean([c.total_score for c in recent_contributions])
        avg_reward = np.mean([c.reward_amount for c in recent_contributions])
        
        # Calculate trend
        if len(recent_contributions) >= 2:
            score_trend = recent_contributions[-1].total_score - recent_contributions[0].total_score
            reward_trend = recent_contributions[-1].reward_amount - recent_contributions[0].reward_amount
        else:
            score_trend = 0.0
            reward_trend = 0.0
        
        # Get latest individual scores
        latest_contribution = recent_contributions[-1]
        individual_scores = {
            metric.value: {
                "score": score.score,
                "weight": score.weight,
                "explanation": score.explanation
            }
            for metric, score in latest_contribution.individual_scores.items()
        }
        
        return {
            "org_id": org_id,
            "avg_total_score": avg_total_score,
            "avg_reward": avg_reward,
            "score_trend": score_trend,
            "reward_trend": reward_trend,
            "current_rank": latest_contribution.rank,
            "individual_scores": individual_scores,
            "total_rounds": len(recent_contributions)
        }
    
    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get leaderboard of top contributing organizations"""
        # Get latest contributions for all organizations
        latest_contributions = {}
        for org_id, history in self.contribution_history.items():
            if history:
                latest_contributions[org_id] = history[-1]
        
        # Sort by total score
        sorted_contributions = sorted(
            latest_contributions.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )
        
        # Create leaderboard
        leaderboard = []
        for rank, (org_id, contribution) in enumerate(sorted_contributions[:top_n], 1):
            leaderboard.append({
                "rank": rank,
                "org_id": org_id,
                "total_score": contribution.total_score,
                "reward_amount": contribution.reward_amount,
                "timestamp": contribution.timestamp
            })
        
        return leaderboard
    
    def export_reward_report(self, filename: str = "reward_report.json") -> str:
        """Export comprehensive reward report"""
        report = {
            "summary": {
                "total_organizations": len(self.contribution_history),
                "total_reward_pool": self.total_reward_pool,
                "distribution_method": self.reward_distribution_method
            },
            "leaderboard": self.get_leaderboard(),
            "organization_summaries": {
                org_id: self.get_contribution_summary(org_id)
                for org_id in self.contribution_history.keys()
            },
            "metric_weights": {metric.value: weight for metric, weight in self.metric_weights.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename

# Example usage
if __name__ == "__main__":
    # Initialize reward system
    reward_system = ContributionRewardSystem(
        total_reward_pool=1000.0,
        reward_distribution_method="hybrid"
    )
    
    # Sample organization data
    organization_data = {
        "hospital_1": {
            "completeness": 0.95,
            "consistency": 0.90,
            "accuracy": 0.88,
            "timeliness": 0.92,
            "validity": 0.94,
            "num_samples": 5000,
            "num_features": 50,
            "num_modalities": 3,
            "demographic_diversity": 0.8,
            "geographic_diversity": 0.7,
            "temporal_diversity": 0.9,
            "privacy_budget_usage": 0.1,
            "differential_privacy": True,
            "anonymization_quality": 0.9
        },
        "clinic_1": {
            "completeness": 0.85,
            "consistency": 0.80,
            "accuracy": 0.82,
            "timeliness": 0.88,
            "validity": 0.86,
            "num_samples": 2000,
            "num_features": 30,
            "num_modalities": 2,
            "demographic_diversity": 0.6,
            "geographic_diversity": 0.5,
            "temporal_diversity": 0.7,
            "privacy_budget_usage": 0.15,
            "differential_privacy": True,
            "anonymization_quality": 0.8
        },
        "research_1": {
            "completeness": 0.98,
            "consistency": 0.95,
            "accuracy": 0.96,
            "timeliness": 0.94,
            "validity": 0.97,
            "num_samples": 10000,
            "num_features": 100,
            "num_modalities": 4,
            "demographic_diversity": 0.9,
            "geographic_diversity": 0.8,
            "temporal_diversity": 0.95,
            "privacy_budget_usage": 0.05,
            "differential_privacy": True,
            "anonymization_quality": 0.95
        }
    }
    
    # Sample round results
    round_results = {
        "hospital_1_accuracy": 0.85,
        "hospital_1_loss": 0.3,
        "hospital_1_participated_rounds": 8,
        "clinic_1_accuracy": 0.82,
        "clinic_1_loss": 0.35,
        "clinic_1_participated_rounds": 9,
        "research_1_accuracy": 0.90,
        "research_1_loss": 0.25,
        "research_1_participated_rounds": 7,
        "total_rounds": 10
    }
    
    # Calculate contributions
    contributions = reward_system.calculate_contribution_scores(round_results, organization_data)
    
    # Print results
    print("Contribution Rewards Calculation Results:")
    print("=" * 50)
    
    for org_id, contribution in contributions.items():
        print(f"\nOrganization: {org_id}")
        print(f"Total Score: {contribution.total_score:.3f}")
        print(f"Rank: {contribution.rank}")
        print(f"Reward Amount: ${contribution.reward_amount:.2f}")
        print("\nIndividual Scores:")
        for metric, score in contribution.individual_scores.items():
            print(f"  {metric.value}: {score.score:.3f} ({score.explanation})")
    
    # Get leaderboard
    leaderboard = reward_system.get_leaderboard()
    print(f"\nLeaderboard:")
    print("=" * 30)
    for entry in leaderboard:
        print(f"{entry['rank']}. {entry['org_id']}: {entry['total_score']:.3f} (${entry['reward_amount']:.2f})")
    
    # Export report
    report_file = reward_system.export_reward_report()
    print(f"\nReward report exported to: {report_file}")
    
    print("\nContribution reward system demonstration completed!")

