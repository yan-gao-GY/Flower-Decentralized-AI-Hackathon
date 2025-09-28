"""
Example: Using W&B Mod and Strategy Wrapper with Flower
This example demonstrates how to use both the W&B Mod and Strategy Wrapper
for comprehensive experiment tracking in federated learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Any

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.common.message import Message

# Import our W&B implementations
from wandb_mod import create_wandb_mod, create_comprehensive_wandb_mod
from wandb_strategy_wrapper import WandBStrategyWrapper, wrap_strategy_with_wandb
from flwr.serverapp.strategy import FedAvg

# Simple CNN model for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, dataloader, epochs=1, lr=0.01):
    """Train model for specified epochs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def evaluate_model(model, dataloader):
    """Evaluate model and return loss and accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

# Create ClientApp with W&B Mod
app = ClientApp()

# Register W&B mod for comprehensive logging
wandb_mod = create_comprehensive_wandb_mod(
    project_name="flower-wandb-demo",
    entity=None,  # Add your W&B entity if you have one
    tags=["demo", "hackathon", "comprehensive"]
)

@app.train(mods=[wandb_mod])
def train(msg: Message, context: Context) -> Message:
    """Training function with W&B logging via mod."""
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    
    # Load model weights from message
    if "arrays" in msg.content:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # Generate synthetic training data for demo
    np.random.seed(context.node_id if hasattr(context, 'node_id') else 42)
    X = torch.randn(100, 1, 28, 28)  # 100 samples, 1 channel, 28x28
    y = torch.randint(0, 10, (100,))  # 10 classes
    
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Training configuration from message
    config = msg.content.get("config", {})
    epochs = config.get("local_epochs", 1)
    lr = config.get("learning_rate", 0.01)
    
    # Train the model
    train_loss = train_model(model, train_loader, epochs, lr)
    
    # Prepare response
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num_examples": len(train_dataset),
        "epochs": epochs,
        "learning_rate": lr
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)

@app.evaluate(mods=[wandb_mod])
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluation function with W&B logging via mod."""
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    
    # Load model weights from message
    if "arrays" in msg.content:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # Generate synthetic test data for demo
    np.random.seed((context.node_id if hasattr(context, 'node_id') else 42) + 1000)
    X = torch.randn(50, 1, 28, 28)  # 50 test samples
    y = torch.randint(0, 10, (50,))
    
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate the model
    eval_loss, eval_accuracy = evaluate_model(model, test_loader)
    
    # Prepare response
    metrics = {
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "num_examples": len(test_dataset)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)

# Create ServerApp with W&B Strategy Wrapper
server_app = ServerApp()

def get_global_evaluate_fn():
    """Create a global evaluation function for centralized evaluation."""
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate global model on centralized test set."""
        
        # Initialize model
        model = SimpleCNN(num_classes=10)
        model.load_state_dict(arrays.to_torch_state_dict())
        
        # Generate centralized test data
        np.random.seed(9999)  # Fixed seed for consistent centralized data
        X = torch.randn(200, 1, 28, 28)  # 200 centralized test samples
        y = torch.randint(0, 10, (200,))
        
        test_dataset = TensorDataset(X, y)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        loss, accuracy = evaluate_model(model, test_loader)
        
        return MetricRecord({
            "centralized_loss": loss,
            "centralized_accuracy": accuracy,
            "centralized_samples": len(test_dataset)
        })
    
    return global_evaluate

@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """Main server function with W&B strategy wrapper."""
    
    # Read configuration
    num_rounds = context.run_config.get("num_rounds", 5)
    fraction_train = context.run_config.get("fraction_train", 0.3)
    min_available_clients = context.run_config.get("min_available_clients", 2)
    
    # Create base strategy
    base_strategy = FedAvg(
        fraction_train=fraction_train,
        min_available_clients=min_available_clients,
        min_fit_clients=min_available_clients,
        min_evaluate_clients=min_available_clients,
    )
    
    # Wrap with W&B logging
    strategy = wrap_strategy_with_wandb(
        strategy=base_strategy,
        project_name="flower-wandb-strategy-demo",
        tags=["strategy", "fedavg", "demo", "hackathon"],
        config={
            "fraction_train": fraction_train,
            "min_available_clients": min_available_clients,
            "demo_type": "comprehensive_wandb"
        },
        log_config=True,
        log_timing=True,
        log_system_metrics=True
    )
    
    # Initialize global model
    global_model = SimpleCNN(num_classes=10)
    initial_arrays = ArrayRecord(global_model.state_dict())
    
    # Configuration for training
    train_config = ConfigRecord({
        "local_epochs": 1,
        "learning_rate": 0.01
    })
    
    # Configuration for evaluation
    evaluate_config = ConfigRecord({})
    
    # Global evaluation function
    evaluate_fn = get_global_evaluate_fn()
    
    # Start federated learning with W&B logging
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=train_config,
        evaluate_config=evaluate_config,
        evaluate_fn=evaluate_fn
    )
    
    # Save final model
    try:
        output_dir = context.node_config.get("output_dir", "./")
        final_model = SimpleCNN(num_classes=10)
        final_model.load_state_dict(result.to_torch_state_dict())
        torch.save(final_model.state_dict(), f"{output_dir}/final_wandb_demo_model.pt")
        print(f"Final model saved to {output_dir}/final_wandb_demo_model.pt")
    except Exception as e:
        print(f"Error saving final model: {e}")

# Configuration for the demo
def create_demo_config():
    """Create configuration for the W&B demo."""
    return {
        "num_rounds": 5,
        "fraction_train": 0.5,
        "min_available_clients": 2,
        "local_epochs": 1,
        "learning_rate": 0.01,
        "project_name": "flower-wandb-comprehensive-demo"
    }

# Example of how to run the demo
if __name__ == "__main__":
    print("=" * 60)
    print("Flower W&B Integration Demo")
    print("=" * 60)
    print()
    print("This demo showcases:")
    print("1. W&B Mod for client-side metric logging")
    print("2. W&B Strategy Wrapper for server-side metric logging")
    print("3. Comprehensive experiment tracking")
    print("4. Error handling and graceful fallbacks")
    print()
    print("Features demonstrated:")
    print("✓ Automatic metric streaming from ClientApp")
    print("✓ Strategy-level metric logging")
    print("✓ Centralized evaluation logging")
    print("✓ Configuration and timing metrics")
    print("✓ System performance monitoring")
    print("✓ Custom tags and project organization")
    print()
    print("To run this demo:")
    print("1. Install dependencies: pip install wandb")
    print("2. Login to W&B: wandb login")
    print("3. Run with Flower: flwr run --stream")
    print()
    print("Expected W&B logs:")
    print("• Training metrics from each client")
    print("• Evaluation metrics from each client")  
    print("• Federated aggregation metrics")
    print("• Centralized evaluation metrics")
    print("• System performance metrics")
    print("• Configuration and timing information")
    print("=" * 60)
