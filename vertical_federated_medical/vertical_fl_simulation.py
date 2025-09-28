"""
Vertical Federated Learning Simulation using Flower
Simulates large-scale vertical FL across multiple healthcare organizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Any
import flwr as fl
from flwr.common import Metrics
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
import logging
from dataclasses import dataclass
import json
import time

# Import our vertical FL components
from vertical_fl_core import (
    VerticalFLOrchestrator, 
    MultiModalMedicalModel, 
    DataModality, 
    PatientRecord,
    OrganizationProfile
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerticalFLConfig:
    """Configuration for vertical FL simulation"""
    num_organizations: int = 8
    num_rounds: int = 20
    num_clients_per_round: int = 4
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 3
    privacy_budget: float = 1.0
    contribution_reward_weight: float = 0.5

class VerticalFLClient(fl.client.NumPyClient):
    """Flower client for vertical federated learning"""
    
    def __init__(self, 
                 org_id: str,
                 data_modalities: List[DataModality],
                 model: nn.Module,
                 train_data: Dict[DataModality, torch.Tensor],
                 test_data: Dict[DataModality, torch.Tensor],
                 labels: torch.Tensor):
        self.org_id = org_id
        self.data_modalities = data_modalities
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = self._create_data_loader(train_data, labels, shuffle=True)
        self.test_loader = self._create_data_loader(test_data, labels, shuffle=False)
        
        # Training metrics
        self.contribution_score = 0.0
        self.data_quality_score = 0.0
        self.privacy_compliance_score = 0.0
    
    def _create_data_loader(self, data: Dict[DataModality, torch.Tensor], 
                          labels: torch.Tensor, shuffle: bool = True) -> DataLoader:
        """Create data loader for multi-modal data"""
        # Combine all available modalities
        combined_features = []
        for modality in self.data_modalities:
            if modality in data:
                combined_features.append(data[modality])
        
        if combined_features:
            # Concatenate features from different modalities
            X = torch.cat(combined_features, dim=-1)
            dataset = TensorDataset(X, labels)
            return DataLoader(dataset, batch_size=32, shuffle=shuffle)
        else:
            # Return empty dataset if no data available
            empty_X = torch.zeros(1, 10)
            empty_y = torch.zeros(1, dtype=torch.long)
            dataset = TensorDataset(empty_X, empty_y)
            return DataLoader(dataset, batch_size=1, shuffle=False)
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model on local data"""
        self.set_parameters(parameters)
        
        # Training configuration
        epochs = config.get("local_epochs", 3)
        lr = config.get("learning_rate", 0.001)
        
        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - need to restructure data for multi-modal model
                # For simplicity, we'll use the combined features directly
                outputs = self.model.fusion_layer(batch_X)  # Direct fusion layer
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            total_loss += epoch_loss
        
        # Calculate contribution metrics
        self._calculate_contribution_metrics()
        
        # Return updated parameters and metrics
        metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "contribution_score": self.contribution_score,
            "data_quality_score": self.data_quality_score,
            "privacy_compliance_score": self.privacy_compliance_score,
            "num_samples": len(self.train_loader.dataset),
            "org_id": self.org_id
        }
        
        return self.get_parameters({}), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on local test data"""
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model.fusion_layer(batch_X)
                loss = nn.CrossEntropyLoss()(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "loss": loss,
            "num_samples": total,
            "org_id": self.org_id
        }
        
        return loss, total, metrics
    
    def _calculate_contribution_metrics(self):
        """Calculate contribution metrics for reward system"""
        # Data quality score (based on data diversity and completeness)
        data_diversity = len(self.data_modalities) / 5.0  # Normalize by max modalities
        data_completeness = sum(1 for modality in self.data_modalities 
                              if modality in self.train_data) / len(self.data_modalities)
        self.data_quality_score = (data_diversity + data_completeness) / 2.0
        
        # Privacy compliance score (simplified)
        self.privacy_compliance_score = 0.9  # Assume good compliance
        
        # Overall contribution score
        self.contribution_score = (
            self.data_quality_score * 0.4 +
            self.privacy_compliance_score * 0.3 +
            np.random.uniform(0.7, 0.95) * 0.3  # Simulate other factors
        )

class VerticalFLStrategy(FedAvg):
    """Custom strategy for vertical federated learning with contribution rewards"""
    
    def __init__(self, orchestrator: VerticalFLOrchestrator, **kwargs):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self.contribution_rewards = {}
    
    def aggregate_fit(self, server_round: int, results: List[Tuple], failures: List) -> Tuple:
        """Aggregate fit results with contribution-based weighting"""
        if not results:
            return None, {}
        
        # Extract metrics for contribution calculation
        round_metrics = {}
        for _, metrics in results:
            org_id = metrics.get("org_id", "unknown")
            round_metrics[f"{org_id}_accuracy"] = metrics.get("accuracy", 0.0)
            round_metrics[f"{org_id}_data_size"] = metrics.get("num_samples", 0)
        
        # Calculate contribution rewards
        rewards = self.orchestrator.calculate_contribution_rewards(round_metrics)
        self.contribution_rewards.update(rewards)
        
        # Weighted aggregation based on contribution rewards
        weighted_results = []
        for parameters, num_samples, metrics in results:
            org_id = metrics.get("org_id", "unknown")
            weight = rewards.get(org_id, 1.0)
            weighted_results.append((parameters, int(num_samples * weight), metrics))
        
        # Perform standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, weighted_results, failures
        )
        
        # Add contribution rewards to metrics
        aggregated_metrics.update({
            "contribution_rewards": rewards,
            "total_rewards": sum(rewards.values())
        })
        
        return aggregated_parameters, aggregated_metrics

def create_vertical_fl_clients(config: VerticalFLConfig) -> List[VerticalFLClient]:
    """Create clients for vertical FL simulation"""
    clients = []
    
    # Define organization profiles
    organizations = [
        {
            "id": "hospital_1",
            "modalities": [DataModality.IMAGES, DataModality.CLINICAL],
            "data_quality": 0.9
        },
        {
            "id": "hospital_2", 
            "modalities": [DataModality.IMAGES, DataModality.CLINICAL],
            "data_quality": 0.85
        },
        {
            "id": "clinic_1",
            "modalities": [DataModality.CLINICAL, DataModality.MEDICATIONS],
            "data_quality": 0.8
        },
        {
            "id": "clinic_2",
            "modalities": [DataModality.CLINICAL, DataModality.MEDICATIONS],
            "data_quality": 0.75
        },
        {
            "id": "insurance_1",
            "modalities": [DataModality.DEMOGRAPHICS],
            "data_quality": 0.7
        },
        {
            "id": "insurance_2",
            "modalities": [DataModality.DEMOGRAPHICS],
            "data_quality": 0.65
        },
        {
            "id": "research_1",
            "modalities": [DataModality.GENOMICS],
            "data_quality": 0.95
        },
        {
            "id": "research_2",
            "modalities": [DataModality.GENOMICS],
            "data_quality": 0.9
        }
    ]
    
    # Generate synthetic data for each organization
    num_samples = 1000
    num_classes = 10
    
    for org in organizations:
        # Create synthetic data based on organization type
        train_data = {}
        test_data = {}
        
        for modality in org["modalities"]:
            if modality == DataModality.IMAGES:
                # Image features (e.g., from CNN)
                train_data[modality] = torch.randn(num_samples, 512)
                test_data[modality] = torch.randn(200, 512)
            elif modality == DataModality.CLINICAL:
                # Clinical features (lab results, vitals)
                train_data[modality] = torch.randn(num_samples, 128)
                test_data[modality] = torch.randn(200, 128)
            elif modality == DataModality.DEMOGRAPHICS:
                # Demographic features
                train_data[modality] = torch.randn(num_samples, 64)
                test_data[modality] = torch.randn(200, 64)
            elif modality == DataModality.GENOMICS:
                # Genomic features
                train_data[modality] = torch.randn(num_samples, 256)
                test_data[modality] = torch.randn(200, 256)
            elif modality == DataModality.MEDICATIONS:
                # Medication features
                train_data[modality] = torch.randn(num_samples, 32)
                test_data[modality] = torch.randn(200, 32)
        
        # Generate labels (same for all organizations)
        train_labels = torch.randint(0, num_classes, (num_samples,))
        test_labels = torch.randint(0, num_classes, (200,))
        
        # Create model
        model = MultiModalMedicalModel(
            image_features=512,
            clinical_features=128,
            demographic_features=64,
            genomic_features=256,
            num_classes=num_classes
        )
        
        # Create client
        client = VerticalFLClient(
            org_id=org["id"],
            data_modalities=org["modalities"],
            model=model,
            train_data=train_data,
            test_data=test_data,
            labels=train_labels
        )
        
        clients.append(client)
    
    return clients

def run_vertical_fl_simulation(config: VerticalFLConfig):
    """Run vertical federated learning simulation"""
    logger.info("Starting Vertical Federated Learning Simulation")
    
    # Initialize orchestrator
    orchestrator = VerticalFLOrchestrator(num_organizations=config.num_organizations)
    
    # Register organizations
    for i in range(config.num_organizations):
        org_profile = OrganizationProfile(
            org_id=f"org_{i}",
            name=f"Organization {i}",
            data_modalities=[DataModality.IMAGES, DataModality.CLINICAL],
            data_quality_score=0.8,
            privacy_compliance=0.9,
            participation_rate=0.8,
            contribution_rewards=0.0
        )
        orchestrator.register_organization(org_profile)
    
    # Create clients
    clients = create_vertical_fl_clients(config)
    
    # Create strategy
    strategy = VerticalFLStrategy(
        orchestrator=orchestrator,
        fraction_fit=config.num_clients_per_round / config.num_organizations,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients_per_round,
        min_evaluate_clients=config.num_organizations,
        min_available_clients=config.num_organizations
    )
    
    # Define client function
    def client_fn(cid: str) -> VerticalFLClient:
        return clients[int(cid)]
    
    # Start simulation
    logger.info(f"Starting simulation with {config.num_organizations} organizations")
    logger.info(f"Running for {config.num_rounds} rounds")
    
    start_time = time.time()
    
    # Run simulation
    results = start_simulation(
        client_fn=client_fn,
        num_clients=config.num_organizations,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )
    
    end_time = time.time()
    
    # Print results
    logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Final contribution rewards: {orchestrator.contribution_rewards}")
    
    # Save results
    results_dict = {
        "config": {
            "num_organizations": config.num_organizations,
            "num_rounds": config.num_rounds,
            "num_clients_per_round": config.num_clients_per_round
        },
        "contribution_rewards": orchestrator.contribution_rewards,
        "simulation_time": end_time - start_time
    }
    
    with open("vertical_fl_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info("Results saved to vertical_fl_results.json")
    
    return results

if __name__ == "__main__":
    # Configuration
    config = VerticalFLConfig(
        num_organizations=8,
        num_rounds=20,
        num_clients_per_round=4,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=3
    )
    
    # Run simulation
    results = run_vertical_fl_simulation(config)
    print("Vertical FL simulation completed successfully!")
