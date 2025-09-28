"""
Hackathon: Weights & Biases Mod
A self-contained mod that streams metrics in outgoing messages to Weights & Biases (W&B) from the ClientApp.

This mod provides enhanced W&B integration with:
- Automatic metric streaming
- Performance monitoring
- Error handling and resilience
- Configurable logging options
- Support for both training and evaluation metrics
"""

import time
import warnings
from typing import Any, Dict, Optional, cast
import logging

from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message
from flwr.common.record import MetricRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional W&B import with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    warnings.warn(
        "Weights & Biases (wandb) is not installed. "
        "Install it with: pip install wandb"
    )


def create_wandb_mod(
    project_name: str = "flower-federated-learning",
    entity: Optional[str] = None,
    tags: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
    log_model_size: bool = True,
    log_communication_time: bool = True,
    log_system_metrics: bool = False,
    reinit: bool = True,
    offline: bool = False,
    silent: bool = False
) -> Mod:
    """
    Create a Weights & Biases mod for Flower ClientApp.
    
    Args:
        project_name: W&B project name
        entity: W&B entity (user/team)
        tags: List of tags for the experiment
        config: Dictionary of configuration parameters
        log_model_size: Whether to log model parameter sizes
        log_communication_time: Whether to log communication timing
        log_system_metrics: Whether to log system performance metrics
        reinit: Whether to reinitialize W&B for each client
        offline: Whether to run W&B in offline mode
        silent: Whether to suppress W&B output
    
    Returns:
        Flower Mod function
    """
    
    def wandb_mod(msg: Message, context: Context, call_next: ClientAppCallable) -> Message:
        """
        Flower Mod that streams metrics to Weights & Biases.
        
        This mod:
        1. Initializes W&B on first training message
        2. Times message processing
        3. Logs training/evaluation metrics to W&B
        4. Handles errors gracefully
        5. Logs additional performance metrics if enabled
        """
        
        if not WANDB_AVAILABLE:
            # Gracefully fallback if W&B not available
            return call_next(msg, context)
        
        try:
            # Extract server round and message type
            server_round = _extract_server_round(msg, context)
            message_type = msg.metadata.message_type if hasattr(msg, 'metadata') else None
            
            # Initialize W&B on first training message
            if server_round == 1 and message_type == MessageType.TRAIN:
                _initialize_wandb(
                    msg, context, project_name, entity, tags, 
                    config, reinit, offline, silent
                )
            
            # Record start time for performance metrics
            start_time = time.time()
            
            # Log incoming message metrics if enabled
            if log_communication_time or log_model_size:
                _log_incoming_message_metrics(
                    msg, context, server_round, log_model_size, log_communication_time
                )
            
            # Log system metrics if enabled
            if log_system_metrics:
                _log_system_metrics(context, server_round)
            
            # Call the next mod or ClientApp
            reply = call_next(msg, context)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log outgoing metrics based on message type
            if reply.has_content():
                _log_reply_metrics(
                    reply, context, server_round, processing_time,
                    log_communication_time, log_model_size
                )
            
            return reply
            
        except Exception as e:
            # Log error but don't break the training process
            logger.error(f"W&B Mod error: {e}")
            
            # Continue with normal processing
            return call_next(msg, context)
    
    return wandb_mod


def _extract_server_round(msg: Message, context: Context) -> int:
    """Extract server round from message or context."""
    try:
        # Try to get from message config
        if hasattr(msg, 'content') and msg.content:
            config = msg.content.get('config', {})
            if isinstance(config, dict):
                return config.get('server-round', config.get('server_round', 1))
            elif hasattr(config, 'get'):
                return config.get('server-round', 1)
        
        # Try to get from context state
        if hasattr(context, 'state'):
            return context.state.get_value('current_round', 1)
        
        # Default to 1
        return 1
        
    except Exception as e:
        logger.warning(f"Could not extract server round: {e}")
        return 1


def _initialize_wandb(
    msg: Message, 
    context: Context, 
    project_name: str,
    entity: Optional[str],
    tags: Optional[list],
    config: Optional[Dict[str, Any]],
    reinit: bool,
    offline: bool,
    silent: bool
) -> None:
    """Initialize Weights & Biases for the client."""
    try:
        # Generate unique identifiers
        run_id = getattr(msg.metadata, 'run_id', 'unknown') if hasattr(msg, 'metadata') else 'unknown'
        node_id = str(getattr(context, 'node_id', 'unknown'))
        
        # Create descriptive names
        group_name = f"Run-{run_id}" if run_id != 'unknown' else "Flower-FL"
        run_name = f"Client-{node_id}"
        unique_id = f"{run_id}_{node_id}" if run_id != 'unknown' else f"client_{node_id}"
        
        # Prepare W&B configuration
        wandb_config = {
            'node_id': node_id,
            'run_id': run_id,
            'framework': 'flower',
            'client_type': 'federated_client'
        }
        
        # Add user-provided config
        if config:
            wandb_config.update(config)
        
        # Initialize W&B
        wandb.init(
            project=project_name,
            entity=entity,
            group=group_name,
            name=run_name,
            id=unique_id,
            tags=tags or ['flower', 'federated-learning'],
            config=wandb_config,
            resume="allow",
            reinit=reinit,
            mode="offline" if offline else "online",
            settings=wandb.Settings(silent=silent)
        )
        
        # Define custom step metric
        wandb.define_metric("server_round")
        wandb.define_metric("*", step_metric="server_round")
        
        logger.info(f"W&B initialized for client {node_id}")
        
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")


def _log_incoming_message_metrics(
    msg: Message,
    context: Context, 
    server_round: int,
    log_model_size: bool,
    log_communication_time: bool
) -> None:
    """Log metrics about incoming messages."""
    try:
        metrics = {"server_round": server_round}
        
        if log_communication_time:
            metrics["message_received_time"] = time.time()
        
        if log_model_size and hasattr(msg, 'content') and msg.content:
            # Estimate message size
            message_size = len(str(msg))
            metrics["incoming_message_size_bytes"] = message_size
            
            # Count parameters if arrays are present
            arrays = msg.content.get('arrays')
            if arrays:
                try:
                    # Convert to torch tensors to count parameters
                    torch_dict = arrays.to_torch_state_dict()
                    param_count = sum(p.numel() for p in torch_dict.values())
                    metrics["incoming_parameters_count"] = param_count
                except Exception as e:
                    logger.debug(f"Could not count parameters: {e}")
        
        if metrics:
            wandb.log(metrics, commit=False)
            
    except Exception as e:
        logger.debug(f"Error logging incoming message metrics: {e}")


def _log_system_metrics(context: Context, server_round: int) -> None:
    """Log system performance metrics."""
    try:
        import psutil
        import torch
        
        metrics = {
            "server_round": server_round,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            metrics["gpu_utilization"] = torch.cuda.utilization()
        
        wandb.log(metrics, commit=False)
        
    except ImportError:
        logger.debug("psutil not available for system metrics")
    except Exception as e:
        logger.debug(f"Error logging system metrics: {e}")


def _log_reply_metrics(
    reply: Message,
    context: Context,
    server_round: int,
    processing_time: float,
    log_communication_time: bool,
    log_model_size: bool
) -> None:
    """Log metrics from the reply message."""
    try:
        message_type = reply.metadata.message_type if hasattr(reply, 'metadata') else None
        
        # Base metrics
        metrics = {
            "server_round": server_round,
        }
        
        # Add processing time
        if log_communication_time:
            if message_type == MessageType.TRAIN:
                metrics["train_processing_time"] = processing_time
            elif message_type == MessageType.EVALUATE:
                metrics["eval_processing_time"] = processing_time
            else:
                metrics["processing_time"] = processing_time
        
        # Extract metrics from reply content
        if hasattr(reply, 'content') and reply.content:
            # Get metric records
            metric_records = reply.content.get('metric_records', {})
            if isinstance(metric_records, dict):
                metrics_record = metric_records.get('metrics')
                if metrics_record and hasattr(metrics_record, '__iter__'):
                    for key, value in dict(metrics_record).items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value
            
            # Log model size if enabled
            if log_model_size:
                arrays = reply.content.get('arrays')
                if arrays:
                    try:
                        torch_dict = arrays.to_torch_state_dict()
                        param_count = sum(p.numel() for p in torch_dict.values())
                        metrics["outgoing_parameters_count"] = param_count
                    except Exception as e:
                        logger.debug(f"Could not count outgoing parameters: {e}")
                
                # Estimate reply message size
                reply_size = len(str(reply))
                metrics["outgoing_message_size_bytes"] = reply_size
        
        # Log to W&B
        if len(metrics) > 1:  # More than just server_round
            wandb.log(metrics, commit=True)
            
    except Exception as e:
        logger.debug(f"Error logging reply metrics: {e}")


# Convenience functions for common use cases
def create_simple_wandb_mod(project_name: str = "flower-experiment") -> Mod:
    """Create a simple W&B mod with default settings."""
    return create_wandb_mod(
        project_name=project_name,
        log_model_size=True,
        log_communication_time=True,
        log_system_metrics=False
    )


def create_comprehensive_wandb_mod(
    project_name: str = "flower-experiment",
    entity: Optional[str] = None,
    tags: Optional[list] = None
) -> Mod:
    """Create a comprehensive W&B mod with all metrics enabled."""
    return create_wandb_mod(
        project_name=project_name,
        entity=entity,
        tags=tags or ['flower', 'federated-learning', 'comprehensive'],
        log_model_size=True,
        log_communication_time=True,
        log_system_metrics=True
    )


def create_lightweight_wandb_mod(project_name: str = "flower-experiment") -> Mod:
    """Create a lightweight W&B mod with minimal overhead."""
    return create_wandb_mod(
        project_name=project_name,
        log_model_size=False,
        log_communication_time=False,
        log_system_metrics=False
    )


# Example usage
if __name__ == "__main__":
    # Example of how to use the W&B mod
    
    # Simple usage
    simple_mod = create_simple_wandb_mod("my-flower-project")
    
    # Comprehensive usage
    comprehensive_mod = create_comprehensive_wandb_mod(
        project_name="medical-ai-federated",
        entity="my-team",
        tags=["medical", "federated-learning", "hackathon"]
    )
    
    # Custom configuration
    custom_mod = create_wandb_mod(
        project_name="vertical-fl-medical",
        tags=["vertical-fl", "medical", "multi-modal"],
        config={
            "model_type": "multi_modal_cnn",
            "datasets": ["pathmnist", "dermamnist", "retinamnist"],
            "strategy": "fedprox"
        },
        log_model_size=True,
        log_communication_time=True,
        log_system_metrics=True
    )
    
    print("W&B Mods created successfully!")
    print("Usage examples:")
    print("1. Simple mod: logs basic training/eval metrics")
    print("2. Comprehensive mod: logs all metrics including system performance")
    print("3. Custom mod: logs with custom configuration and tags")

