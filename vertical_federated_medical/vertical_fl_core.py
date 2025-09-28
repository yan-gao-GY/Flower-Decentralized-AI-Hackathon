"""
Vertical Federated Learning Core Implementation
Enables different organizations to contribute different data modalities
while preserving privacy and enabling collaborative model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataModality(Enum):
    """Different types of medical data modalities"""
    IMAGES = "images"           # X-rays, MRIs, CT scans
    CLINICAL = "clinical"      # Lab results, vital signs
    DEMOGRAPHICS = "demographics"  # Age, gender, insurance
    GENOMICS = "genomics"      # Genetic markers, family history
    MEDICATIONS = "medications"  # Drug prescriptions, dosages

@dataclass
class PatientRecord:
    """Represents a patient record across multiple organizations"""
    patient_id: str
    modalities: Dict[DataModality, torch.Tensor]
    metadata: Dict[str, Any]
    privacy_level: int  # 1-5, higher = more sensitive

@dataclass
class OrganizationProfile:
    """Profile of an organization in the vertical FL system"""
    org_id: str
    name: str
    data_modalities: List[DataModality]
    data_quality_score: float
    privacy_compliance: float
    participation_rate: float
    contribution_rewards: float

class VerticalFLOrchestrator:
    """Orchestrates vertical federated learning across multiple organizations"""
    
    def __init__(self, num_organizations: int = 4):
        self.num_organizations = num_organizations
        self.organizations = {}
        self.global_model = None
        self.contribution_rewards = {}
        self.privacy_budget = 1.0  # Differential privacy budget
        
    def register_organization(self, org_profile: OrganizationProfile):
        """Register a new organization in the federation"""
        self.organizations[org_profile.org_id] = org_profile
        self.contribution_rewards[org_profile.org_id] = 0.0
        logger.info(f"Registered organization: {org_profile.name}")
    
    def align_patient_data(self, patient_records: List[PatientRecord]) -> Dict[str, torch.Tensor]:
        """Align patient data across different organizations"""
        aligned_data = {}
        
        for record in patient_records:
            patient_id = record.patient_id
            
            # Combine features from different modalities
            combined_features = []
            modality_weights = []
            
            for modality, data in record.modalities.items():
                if data is not None:
                    # Normalize features based on modality
                    normalized_data = self._normalize_modality_data(data, modality)
                    combined_features.append(normalized_data)
                    
                    # Weight based on data quality and privacy level
                    weight = self._calculate_modality_weight(modality, record.privacy_level)
                    modality_weights.append(weight)
            
            if combined_features:
                # Weighted combination of features
                weights = torch.tensor(modality_weights, dtype=torch.float32)
                weights = F.softmax(weights, dim=0)
                
                # Concatenate and weight features
                combined = torch.cat(combined_features, dim=-1)
                weighted_features = combined * weights.unsqueeze(0)
                
                aligned_data[patient_id] = weighted_features
        
        return aligned_data
    
    def _normalize_modality_data(self, data: torch.Tensor, modality: DataModality) -> torch.Tensor:
        """Normalize data based on modality type"""
        if modality == DataModality.IMAGES:
            # Normalize image data to [0, 1]
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
        elif modality == DataModality.CLINICAL:
            # Standardize clinical data
            return (data - data.mean()) / (data.std() + 1e-8)
        elif modality == DataModality.DEMOGRAPHICS:
            # One-hot encode categorical data
            return F.one_hot(data.long(), num_classes=data.max().int() + 1).float()
        else:
            # Default normalization
            return F.normalize(data, p=2, dim=-1)
    
    def _calculate_modality_weight(self, modality: DataModality, privacy_level: int) -> float:
        """Calculate weight for modality based on importance and privacy"""
        base_weights = {
            DataModality.IMAGES: 0.4,
            DataModality.CLINICAL: 0.3,
            DataModality.DEMOGRAPHICS: 0.1,
            DataModality.GENOMICS: 0.2,
            DataModality.MEDICATIONS: 0.1
        }
        
        # Adjust weight based on privacy level (higher privacy = lower weight)
        privacy_factor = 1.0 - (privacy_level - 1) * 0.1
        return base_weights.get(modality, 0.1) * privacy_factor
    
    def calculate_contribution_rewards(self, round_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards for each organization based on their contribution"""
        rewards = {}
        
        for org_id, org_profile in self.organizations.items():
            # Data Quality Score (0-100)
            data_quality_score = org_profile.data_quality_score * 100
            # Model Performance Contribution (0-100)
            performance_score = self._calculate_performance_contribution(org_id, round_results)
            # Privacy Compliance Score (0-100)
            privacy_score = org_profile.privacy_compliance * 100
            # Participation Score (0-100)
            participation_score = org_profile.participation_rate * 100
            # Weighted reward calculation
            total_reward = (
                data_quality_score * 0.3 +
                performance_score * 0.4 +
                privacy_score * 0.2 +
                participation_score * 0.1
            )
            rewards[org_id] = total_reward
            self.contribution_rewards[org_id] = total_reward
            logger.info(f"Organization {org_id} reward: {total_reward:.2f}")
        return rewards
    
    def _calculate_performance_contribution(self, org_id: str, round_results: Dict[str, Any]) -> float:
        """Calculate how much an organization contributed to model performance"""
        # This would be calculated based on:
        # 1. Improvement in model accuracy when their data is included
        # 2. Quality of gradients they provided
        # 3. Diversity of their data compared to others
        
        # Simplified calculation for demonstration
        org_data_size = round_results.get(f"{org_id}_data_size", 1000)
        org_accuracy = round_results.get(f"{org_id}_accuracy", 0.8)
        
        # Normalize to 0-100 scale
        size_score = min(org_data_size / 10000, 1.0) * 50  # Max 50 points for data size
        accuracy_score = org_accuracy * 50  # Max 50 points for accuracy
        
        return size_score + accuracy_score
    
    def aggregate_models(self, local_models: Dict[str, torch.nn.Module]) -> torch.nn.Module:
        """Aggregate models from different organizations using vertical FL"""
        if not local_models:
            return self.global_model
        
        # Get the first model as base
        aggregated_model = list(local_models.values())[0]
        
        # For vertical FL, we need to combine different parts of models
        # that were trained on different modalities
        with torch.no_grad():
            for param_name, param in aggregated_model.named_parameters():
                if param_name in ['weight', 'bias']:  # Only aggregate certain layers
                    # Weighted average based on contribution rewards
                    total_weight = 0.0
                    weighted_param = torch.zeros_like(param)
                    
                    for org_id, model in local_models.items():
                        weight = self.contribution_rewards.get(org_id, 1.0)
                        if hasattr(model, param_name):
                            weighted_param += weight * getattr(model, param_name)
                            total_weight += weight
                    
                    if total_weight > 0:
                        param.data = weighted_param / total_weight
        
        return aggregated_model

class MultiModalMedicalModel(nn.Module):
    """Multi-modal model for vertical federated learning"""
    
    def __init__(self, 
                 image_features: int = 512,
                 clinical_features: int = 128,
                 demographic_features: int = 64,
                 genomic_features: int = 256,
                 num_classes: int = 10):
        super().__init__()
        
        # Modality-specific encoders
        self.image_encoder = nn.Sequential(
            nn.Linear(image_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        self.demographic_encoder = nn.Sequential(
            nn.Linear(demographic_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )
        
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Fusion layer
        total_features = 128 + 32 + 16 + 64  # Sum of encoder outputs
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: Dict[DataModality, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-modal model"""
        encoded_features = []
        
        # Encode each modality
        if DataModality.IMAGES in x:
            img_features = self.image_encoder(x[DataModality.IMAGES])
            encoded_features.append(img_features)
        
        if DataModality.CLINICAL in x:
            clinical_features = self.clinical_encoder(x[DataModality.CLINICAL])
            encoded_features.append(clinical_features)
        
        if DataModality.DEMOGRAPHICS in x:
            demo_features = self.demographic_encoder(x[DataModality.DEMOGRAPHICS])
            encoded_features.append(demo_features)
        
        if DataModality.GENOMICS in x:
            genomic_features = self.genomic_encoder(x[DataModality.GENOMICS])
            encoded_features.append(genomic_features)
        
        # Fuse all modalities
        if encoded_features:
            fused_features = torch.cat(encoded_features, dim=-1)
            output = self.fusion_layer(fused_features)
            return output
        else:
            # Return zeros if no features available
            batch_size = next(iter(x.values())).size(0)
            return torch.zeros(batch_size, self.fusion_layer[-1].out_features)

class DataHarmonizer:
    """Automatically harmonizes data across different organizations"""
    
    def __init__(self):
        self.schema_mappings = {}
        self.quality_metrics = {}
    
    def detect_schema_differences(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Detect differences in data schemas across organizations"""
        schema_analysis = {}
        
        for org_id, dataset in datasets.items():
            schema_analysis[org_id] = {
                'columns': list(dataset.columns) if hasattr(dataset, 'columns') else [],
                'dtypes': dataset.dtypes.to_dict() if hasattr(dataset, 'dtypes') else {},
                'shape': dataset.shape if hasattr(dataset, 'shape') else (0, 0),
                'missing_values': dataset.isnull().sum().to_dict() if hasattr(dataset, 'isnull') else {}
            }
        
        return schema_analysis
    
    def create_schema_mapping(self, source_schema: Dict, target_schema: Dict) -> Dict[str, str]:
        """Create mapping between different schemas"""
        mapping = {}
        
        # Simple string similarity-based mapping
        for source_col in source_schema.get('columns', []):
            best_match = None
            best_similarity = 0.0
            
            for target_col in target_schema.get('columns', []):
                similarity = self._calculate_string_similarity(source_col, target_col)
                if similarity > best_similarity and similarity > 0.7:  # Threshold
                    best_similarity = similarity
                    best_match = target_col
            
            if best_match:
                mapping[source_col] = best_match
        
        return mapping
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple metrics"""
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        # Exact match
        if str1_lower == str2_lower:
            return 1.0
        
        # Substring match
        if str1_lower in str2_lower or str2_lower in str1_lower:
            return 0.8
        
        # Word overlap
        words1 = set(str1_lower.split())
        words2 = set(str2_lower.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) > 0:
            return len(intersection) / len(union)
        
        return 0.0
    
    def harmonize_data(self, data: Any, mapping: Dict[str, str]) -> Any:
        """Apply schema mapping to harmonize data"""
        if hasattr(data, 'rename'):
            return data.rename(columns=mapping)
        return data

# Example usage and testing
if __name__ == "__main__":
    # Initialize vertical FL orchestrator
    orchestrator = VerticalFLOrchestrator(num_organizations=4)
    
    # Register organizations
    hospital = OrganizationProfile(
        org_id="hospital_1",
        name="General Hospital",
        data_modalities=[DataModality.IMAGES, DataModality.CLINICAL],
        data_quality_score=0.9,
        privacy_compliance=0.95,
        participation_rate=0.8,
        contribution_rewards=0.0
    )
    
    clinic = OrganizationProfile(
        org_id="clinic_1", 
        name="Family Clinic",
        data_modalities=[DataModality.CLINICAL, DataModality.MEDICATIONS],
        data_quality_score=0.8,
        privacy_compliance=0.9,
        participation_rate=0.9,
        contribution_rewards=0.0
    )
    
    insurance = OrganizationProfile(
        org_id="insurance_1",
        name="Health Insurance Co",
        data_modalities=[DataModality.DEMOGRAPHICS],
        data_quality_score=0.7,
        privacy_compliance=0.85,
        participation_rate=0.7,
        contribution_rewards=0.0
    )
    
    research = OrganizationProfile(
        org_id="research_1",
        name="Medical Research Center",
        data_modalities=[DataModality.GENOMICS],
        data_quality_score=0.95,
        privacy_compliance=0.98,
        participation_rate=0.6,
        contribution_rewards=0.0
    )
    
    # Register all organizations
    for org in [hospital, clinic, insurance, research]:
        orchestrator.register_organization(org)
    
    # Create sample patient records
    patient_records = [
        PatientRecord(
            patient_id="P001",
            modalities={
                DataModality.IMAGES: torch.randn(1, 512),  # Hospital data
                DataModality.CLINICAL: torch.randn(1, 128),  # Clinic data
                DataModality.DEMOGRAPHICS: torch.randn(1, 64),  # Insurance data
                DataModality.GENOMICS: torch.randn(1, 256)  # Research data
            },
            metadata={"age": 45, "gender": "M"},
            privacy_level=3
        )
    ]
    
    # Align patient data
    aligned_data = orchestrator.align_patient_data(patient_records)
    print(f"Aligned data for {len(aligned_data)} patients")
    
    # Calculate contribution rewards
    round_results = {
        "hospital_1_accuracy": 0.85,
        "clinic_1_accuracy": 0.82,
        "insurance_1_accuracy": 0.78,
        "research_1_accuracy": 0.88
    }
    
    rewards = orchestrator.calculate_contribution_rewards(round_results)
    print(f"Contribution rewards: {rewards}")
    
    print("Vertical Federated Learning system initialized successfully!")
