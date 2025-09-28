# Hackathon: Weights & Biases Integration for Flower

This contribution implements comprehensive Weights & Biases (W&B) integration for the Flower federated learning framework, providing both **client-side** and **server-side** experiment tracking capabilities.

## 🎯 Hackathon Contributions

### 1. **Weights & Biases Mod** (`wandb_mod.py`)

A self-contained mod that streams metrics in outgoing messages to W&B from the ClientApp.

### 2. **Strategy Wrapper for W&B** (`wandb_strategy_wrapper.py`)

A Strategy wrapper that automatically streams all metrics to W&B from server-side operations.

### 3. **Comprehensive Example** (`wandb_example.py`)

A complete demonstration showing how to use both components together.

## 🚀 Features

### W&B Mod Features

- ✅ **Automatic Metric Streaming**: Logs training and evaluation metrics from ClientApp
- ✅ **Performance Monitoring**: Tracks communication time and system metrics
- ✅ **Error Resilience**: Graceful fallbacks when W&B is unavailable
- ✅ **Configurable Logging**: Customizable logging options and levels
- ✅ **Simulation Support**: Proper handling of multiple clients in simulation
- ✅ **Model Size Tracking**: Logs parameter counts and message sizes

### Strategy Wrapper Features

- ✅ **Comprehensive Logging**: Logs all strategy-level metrics automatically
- ✅ **Centralized Evaluation**: Logs metrics from evaluate_fn
- ✅ **Configuration Tracking**: Logs strategy configuration and hyperparameters
- ✅ **Timing Metrics**: Tracks experiment duration and round timing
- ✅ **System Metrics**: Optional system performance monitoring
- ✅ **Strategy Compatibility**: Works with any Flower strategy (FedAvg, FedProx, etc.)

## 📦 Installation

```bash
# Install Flower (if not already installed)
pip install flwr

# Install Weights & Biases
pip install wandb

# Login to W&B (first time only)
wandb login
```

## 🎯 Quick Start

### Using the W&B Mod

```python
from wandb_mod import create_wandb_mod
from flwr.clientapp import ClientApp

# Create ClientApp with W&B mod
app = ClientApp()

# Create W&B mod
wandb_mod = create_wandb_mod(
    project_name="my-flower-experiment",
    tags=["federated-learning", "demo"]
)

@app.train(mods=[wandb_mod])
def train(msg, context):
    # Your training logic here
    # Metrics will be automatically logged to W&B
    pass
```

### Using the Strategy Wrapper

```python
from wandb_strategy_wrapper import wrap_strategy_with_wandb
from flwr.serverapp.strategy import FedAvg

# Create base strategy
base_strategy = FedAvg(fraction_train=0.3)

# Wrap with W&B logging
strategy = wrap_strategy_with_wandb(
    strategy=base_strategy,
    project_name="my-flower-strategy",
    tags=["fedavg", "experiment"]
)

# Use in ServerApp
# All metrics will be automatically logged to W&B
```

## 🔧 Configuration Options

### W&B Mod Configuration

```python
wandb_mod = create_wandb_mod(
    project_name="flower-experiment",     # W&B project name
    entity="my-team",                     # W&B entity (optional)
    tags=["fl", "demo"],                  # Tags for organization
    config={"model": "cnn"},              # Custom configuration
    log_model_size=True,                  # Log parameter counts
    log_communication_time=True,          # Log timing metrics
    log_system_metrics=True,              # Log system performance
    offline=False,                        # Run in offline mode
    silent=False                          # Suppress W&B output
)
```

### Strategy Wrapper Configuration

```python
strategy = WandBStrategyWrapper(
    strategy=base_strategy,
    project_name="flower-strategy",       # W&B project name
    entity="my-team",                     # W&B entity (optional)
    tags=["strategy", "fedavg"],          # Tags for organization
    config={"rounds": 10},                # Custom configuration
    log_config=True,                      # Log strategy configuration
    log_timing=True,                      # Log timing metrics
    log_system_metrics=True,              # Log system performance
    offline=False,                        # Run in offline mode
    silent=False                          # Suppress W&B output
)
```

## 📊 Logged Metrics

### Client-Side Metrics (Mod)

- **Training Metrics**: loss, accuracy, custom metrics from ClientApp
- **Evaluation Metrics**: loss, accuracy, validation metrics
- **Communication Metrics**: message sizes, processing times
- **Model Metrics**: parameter counts, model sizes
- **System Metrics**: CPU, memory, GPU utilization (optional)

### Server-Side Metrics (Strategy)

- **Aggregation Metrics**: federated training and evaluation results
- **Centralized Evaluation**: metrics from global evaluation function
- **Configuration**: strategy parameters and hyperparameters
- **Timing**: round duration, total experiment time
- **Participation**: number of clients per round

## 🎨 Convenience Functions

### Simple Use Cases

```python
# Simple W&B mod with defaults
simple_mod = create_simple_wandb_mod("my-project")

# Comprehensive logging with all features
comprehensive_mod = create_comprehensive_wandb_mod(
    "my-project",
    entity="my-team"
)

# Lightweight mod with minimal overhead
lightweight_mod = create_lightweight_wandb_mod("my-project")
```

### Strategy Shortcuts

```python
# Create FedAvg with W&B
fedavg_wandb = create_fedavg_with_wandb(
    "my-fedavg-project",
    fraction_train=0.3
)

# Create FedProx with W&B
fedprox_wandb = create_fedprox_with_wandb(
    "my-fedprox-project",
    proximal_mu=0.01
)
```

## 🧪 Complete Example

See `wandb_example.py` for a comprehensive example that demonstrates:

- ClientApp with W&B mod
- ServerApp with W&B strategy wrapper
- Synthetic data generation for testing
- Complete federated learning workflow
- All logging features in action

```bash
# Run the example
python wandb_example.py
```

## 🛡️ Error Handling

Both components include robust error handling:

- **Graceful Fallbacks**: Continue training even if W&B fails
- **Import Safety**: Handle missing W&B installation gracefully
- **Network Resilience**: Handle W&B service interruptions
- **Logging**: Detailed error logging for debugging

## 🔍 Integration with Existing Projects

### Adding to Existing ClientApp

```python
# Add to existing ClientApp
from wandb_mod import create_wandb_mod

# Create mod
wandb_mod = create_wandb_mod("existing-project")

# Add to existing train/evaluate functions
@app.train(mods=[wandb_mod])
def existing_train_function(msg, context):
    # Your existing training code
    # No changes needed - metrics logged automatically
    pass
```

### Adding to Existing Strategy

```python
# Wrap existing strategy
from wandb_strategy_wrapper import wrap_strategy_with_wandb

# Your existing strategy
existing_strategy = MyCustomStrategy()

# Add W&B logging
wandb_strategy = wrap_strategy_with_wandb(
    existing_strategy,
    "existing-project"
)

# Use in place of original strategy
```

## 📈 Benefits

### For Researchers

- **Experiment Tracking**: Comprehensive logging of all FL metrics
- **Visualization**: Rich charts and graphs in W&B dashboard
- **Comparison**: Easy comparison between different runs
- **Collaboration**: Share experiments with team members

### For Practitioners

- **Monitoring**: Real-time monitoring of federated training
- **Debugging**: Detailed logs for troubleshooting
- **Optimization**: Performance metrics for optimization
- **Reproducibility**: Complete experiment tracking

### For the Flower Community

- **Standardization**: Consistent experiment tracking across projects
- **Sharing**: Easy sharing of federated learning experiments
- **Documentation**: Automatic documentation of experimental settings
- **Integration**: Seamless integration with existing Flower projects

## 🧑‍💻 Development

### Testing

```bash
# Test W&B mod
python -c "from wandb_mod import create_wandb_mod; print('Mod working!')"

# Test strategy wrapper
python -c "from wandb_strategy_wrapper import wrap_strategy_with_wandb; print('Wrapper working!')"

# Run comprehensive example
python wandb_example.py
```

### Dependencies

- `flwr>=1.22.0`: Flower federated learning framework
- `wandb>=0.17.8`: Weights & Biases for experiment tracking
- `torch>=2.0.0`: PyTorch for the example (optional)
- `psutil>=5.8.0`: System metrics (optional)

## 🤝 Contributing

This contribution is designed to be:

- **Self-contained**: No modifications to Flower core needed
- **Backward compatible**: Works with existing Flower apps
- **Well-tested**: Includes comprehensive examples and error handling
- **Documented**: Detailed documentation and examples
- **Community-friendly**: Easy to use and extend

## 📄 License

This contribution follows the same license as the Flower project (Apache 2.0).

## 🙏 Acknowledgments

- Flower team for the excellent federated learning framework
- Weights & Biases for the powerful experiment tracking platform
- The open-source community for inspiration and feedback

---

**Ready to submit as Hackathon contribution to Flower! 🚀**
