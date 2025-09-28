"""
Hackathon: Strategy Wrapper for W&B
A Strategy wrapper that automatically streams all metrics to Weights & Biases.

This wrapper:
- Logs MetricRecord from aggregate_train() return
- Logs MetricRecord from aggregate_evaluate() return  
- Logs MetricRecord from evaluate_fn passed to start() method
- Provides comprehensive experiment tracking
- Handles errors gracefully
- Supports custom configurations
"""

import time
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple, Callable
import logging
from flwr.common.record import ArrayRecord, ConfigRecord, MetricRecord
from flwr.common.message import Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy

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


class WandBStrategyWrapper(Strategy):
    """
    Strategy wrapper that automatically logs all metrics to Weights & Biases.
    
    This wrapper delegates all strategy operations to an underlying strategy
    while logging comprehensive metrics to W&B for experiment tracking.
    """


    def train_model(model, dataloader, epochs=1, lr=0.01):
        device = torch.device("cuda")
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        total_loss = 0.0
        num_batch = 0.0
        for epoch in range(epochs):
            cls,
            resource_names: str,
            data: Dict[str, Any],
            timeout: int,
            headers: Optional[Dict[str, str]] = None,
            Iterator[Dict[str, Any]]:
            base_path = get_base_path()
            api_key = get_api_key()
            response = requests.post(
                urljoin(base_path, resource_name),
                json = data,
                timeout = timeout,
                headers = {"x-api-key": api_key, **(headers or {})},
                auth=(api_key, "")
                stream=True
            )
            response.raise_for_status()
            for line in response.iter_lines():  
                if line: 
                    try:
                        msg = json.loads(line)
                        yield msg
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {line}")
                        continue

    def evaluate_model(model, dataloader):
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_X, batch_y = batch
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def get_global_evaluate_fn():
        def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            return evaluate_model(model, arrays)
        return global_evaluate

    def main(grid: Grid, context: Context) -> None:
        maybe_init_wandb(use_wandb, wandbtoken)
        server_app = ServerApp(
            grid=grid,
            context=context,
            evaluate_fn=get_global_evaluate_fn(),
            server_resources=server_resources,
            server_timeouts=server_timeouts,
            server_address=server_address,
            server_port=server_port,
            server_auth=server_auth,
            server_cert=server_cert,
            server_key=server_key,
            server_ca=server_ca,
            server_verify=server_verify,
            server_timeout=server_timeout,
            server_max_workers=server_max_workers,
            server_num_workers=server_num_workers,
            server_num_threads=server_num_threads,
            server_num_gpus=server_num_gpus,
            server_num_cpus=server_num_cpus,
            server_num_mem=server_num_mem,
            server_num_disk=server_num_disk,
            server_num_network=server_num_network,
        )
    
    def __init__(
        self,
        strategy: Strategy,
        project_name: str = "flower-strategy",
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        log_config: bool = True,
        log_timing: bool = True,
        log_system_metrics: bool = False,
        offline: bool = False,
        silent: bool = False
    ):
        """
        Initialize the W&B Strategy Wrapper.
        
        Args:
            strategy: The underlying Flower strategy to wrap
            project_name: W&B project name
            entity: W&B entity (user/team)
            tags: List of tags for the experiment
            config: Dictionary of configuration parameters
            log_config: Whether to log strategy configuration
            log_timing: Whether to log timing metrics
            log_system_metrics: Whether to log system performance
            offline: Whether to run W&B in offline mode
            silent: Whether to suppress W&B output
        """
        self.strategy = strategy
        self.project_name = project_name
        self.entity = entity
        self.tags = tags or ['flower', 'strategy', 'federated-learning']
        self.config = config or {}
        self.log_config = log_config
        self.log_timing = log_timing
        self.log_system_metrics = log_system_metrics
        self.offline = offline
        self.silent = silent
        
        # Internal state
        self._wandb_initialized = False
        self._current_round = 0
        self._start_time = None
        
    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases for strategy logging."""
        if not WANDB_AVAILABLE or self._wandb_initialized:
            return
            
        try:
            # Prepare configuration
            wandb_config = {
                'strategy_type': type(self.strategy).__name__,
                'framework': 'flower',
                'role': 'server_strategy'
            }
            
            # Add strategy-specific configuration
            if hasattr(self.strategy, '__dict__'):
                strategy_attrs = {
                    k: v for k, v in self.strategy.__dict__.items()
                    if isinstance(v, (int, float, str, bool, list, dict))
                }
                wandb_config.update({f'strategy_{k}': v for k, v in strategy_attrs.items()})
            
            # Add user-provided config
            wandb_config.update(self.config)
            
            # Initialize W&B
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=f"Strategy-{type(self.strategy).__name__}",
                tags=self.tags,
                config=wandb_config,
                reinit=True,
                mode="offline" if self.offline else "online",
                settings=wandb.Settings(silent=self.silent)
            )
            
            # Define custom step metric
            wandb.define_metric("server_round")
            wandb.define_metric("*", step_metric="server_round")
            
            self._wandb_initialized = True
            logger.info(f"W&B initialized for strategy: {type(self.strategy).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B for strategy: {e}")
    
    def _log_metrics(
        self, 
        metrics: Optional[MetricRecord], 
        prefix: str = "",
        round_num: Optional[int] = None
    ) -> None:
        """Log MetricRecord to W&B."""
        if not WANDB_AVAILABLE or not self._wandb_initialized or not metrics:
            return
            
        try:
            # Convert MetricRecord to dictionary
            metrics_dict = dict(metrics) if metrics else {}
            
            # Add prefix to metric names
            if prefix:
                metrics_dict = {f"{prefix}_{k}": v for k, v in metrics_dict.items()}
            
            # Add round information
            if round_num is not None:
                metrics_dict["server_round"] = round_num
            elif self._current_round > 0:
                metrics_dict["server_round"] = self._current_round
            
            # Add timing information if enabled
            if self.log_timing and self._start_time:
                elapsed_time = time.time() - self._start_time
                metrics_dict[f"{prefix}_elapsed_time" if prefix else "elapsed_time"] = elapsed_time
            
            # Log system metrics if enabled
            if self.log_system_metrics:
                self._add_system_metrics(metrics_dict)
            
            # Log to W&B
            if metrics_dict:
                wandb.log(metrics_dict, commit=True)
                
        except Exception as e:
            logger.debug(f"Error logging metrics to W&B: {e}")
    
    def _add_system_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Add system performance metrics."""
        try:
            import psutil
            
            metrics_dict.update({
                "system_cpu_percent": psutil.cpu_percent(),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_disk_usage_percent": psutil.disk_usage('/').percent
            })
            
        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.debug(f"Error adding system metrics: {e}")
    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int,
        timeout: Optional[float] = None,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None
    ) -> ArrayRecord:
        """
        Start the federated learning process with W&B logging.
        
        Wraps the underlying strategy's start method while logging metrics
        from the optional evaluate_fn.
        """
        # Initialize W&B
        self._initialize_wandb()
        self._start_time = time.time()
        
        # Log strategy configuration if enabled
        if self.log_config and self._wandb_initialized:
            try:
                config_metrics = {
                    "server_round": 0,
                    "total_rounds": num_rounds,
                    "timeout": timeout or -1,
                    "strategy_type": type(self.strategy).__name__
                }
                
                if train_config:
                    config_metrics.update({f"train_config_{k}": v for k, v in dict(train_config).items()})
                
                if evaluate_config:
                    config_metrics.update({f"eval_config_{k}": v for k, v in dict(evaluate_config).items()})
                
                wandb.log(config_metrics, commit=True)
            except Exception as e:
                logger.debug(f"Error logging configuration: {e}")
        
        # Wrap evaluate_fn to log its results
        wrapped_evaluate_fn = None
        if evaluate_fn:
            def logging_evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
                try:
                    self._current_round = server_round
                    result = evaluate_fn(server_round, arrays)
                    
                    # Log evaluate_fn results
                    self._log_metrics(result, "centralized_eval", server_round)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in wrapped evaluate_fn: {e}")
                    return None
            
            wrapped_evaluate_fn = logging_evaluate_fn
        
        # Call the underlying strategy's start method
        try:
            return self.strategy.start(
                grid=grid,
                initial_arrays=initial_arrays,
                num_rounds=num_rounds,
                timeout=timeout,
                train_config=train_config,
                evaluate_config=evaluate_config,
                evaluate_fn=wrapped_evaluate_fn
            )
        except Exception as e:
            logger.error(f"Error in strategy start method: {e}")
            raise
    
    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """Configure training round (delegates to underlying strategy)."""
        self._current_round = server_round
        return self.strategy.configure_train(server_round, arrays, config, grid)
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """
        Aggregate training results and log metrics to W&B.
        
        This method logs the MetricRecord returned by the underlying strategy.
        """
        try:
            # Call underlying strategy
            arrays, metrics = self.strategy.aggregate_train(server_round, replies)
            
            # Log metrics to W&B
            self._log_metrics(metrics, "federated_train", server_round)
            
            # Log additional training round metrics
            if self._wandb_initialized:
                try:
                    num_replies = len(list(replies)) if replies else 0
                    wandb.log({
                        "server_round": server_round,
                        "train_num_clients": num_replies
                    }, commit=False)
                except Exception as e:
                    logger.debug(f"Error logging training round metrics: {e}")
            
            return arrays, metrics
            
        except Exception as e:
            logger.error(f"Error in aggregate_train: {e}")
            raise
    
    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """Configure evaluation round (delegates to underlying strategy)."""
        return self.strategy.configure_evaluate(server_round, arrays, config, grid)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message]
    ) -> Optional[MetricRecord]:
        """
        Aggregate evaluation results and log metrics to W&B.
        
        This method logs the MetricRecord returned by the underlying strategy.
        """
        try:
            # Call underlying strategy
            metrics = self.strategy.aggregate_evaluate(server_round, replies)
            
            # Log metrics to W&B
            self._log_metrics(metrics, "federated_eval", server_round)
            
            # Log additional evaluation round metrics
            if self._wandb_initialized:
                try:
                    num_replies = len(list(replies)) if replies else 0
                    wandb.log({
                        "server_round": server_round,
                        "eval_num_clients": num_replies
                    }, commit=False)
                except Exception as e:
                    logger.debug(f"Error logging evaluation round metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in aggregate_evaluate: {e}")
            raise
    
    def summary(self) -> None:
        """Log strategy summary (delegates to underlying strategy)."""
        try:
            # Call underlying strategy summary
            self.strategy.summary()
            
            # Log final summary to W&B
            if self._wandb_initialized:
                try:
                    final_metrics = {
                        "experiment_completed": True,
                        "total_rounds_completed": self._current_round
                    }
                    
                    if self._start_time:
                        total_time = time.time() - self._start_time
                        final_metrics["total_experiment_time"] = total_time
                    
                    wandb.log(final_metrics, commit=True)
                    wandb.finish()
                    
                except Exception as e:
                    logger.debug(f"Error logging final summary: {e}")
                    
        except Exception as e:
            logger.error(f"Error in strategy summary: {e}")
            raise
    
    def __getattr__(self, name):
        """Delegate any missing attributes to the underlying strategy."""
        return getattr(self.strategy, name)


# Convenience functions for common use cases
def wrap_strategy_with_wandb(
    strategy: Strategy,
    project_name: str = "flower-strategy",
    **kwargs
) -> WandBStrategyWrapper:
    """Wrap any Flower strategy with W&B logging."""
    return WandBStrategyWrapper(
        strategy=strategy,
        project_name=project_name,
        **kwargs
    )

def train_model(model, dataloader, epochs=1, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    num_batches = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_X, batch_y = batch
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        total_loss += epoch_loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_X, batch_y = batch
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy



def create_fedavg_with_wandb(
    project_name: str = "flower-fedavg",
    **fedavg_kwargs
) -> WandBStrategyWrapper:
    """Create FedAvg strategy with W&B logging."""
    from flwr.serverapp.strategy import FedAvg
    
    base_strategy = FedAvg(**fedavg_kwargs)
    return wrap_strategy_with_wandb(base_strategy, project_name)


def create_fedprox_with_wandb(
    project_name: str = "flower-fedprox",
    proximal_mu: float = 0.01,
    **fedprox_kwargs
) -> WandBStrategyWrapper:
    """Create FedProx strategy with W&B logging."""
    from flwr.serverapp.strategy import FedProx
    
    base_strategy = FedProx(proximal_mu=proximal_mu, **fedprox_kwargs)
    return wrap_strategy_with_wandb(base_strategy, project_name)


# Example usage
if __name__ == "__main__":
    # Example of how to use the W&B strategy wrapper
    from flwr.serverapp.strategy import FedAvg, FedProx
    
    # Example 1: Wrap existing strategy
    base_strategy = FedAvg(fraction_train=0.3, min_available_clients=2)
    wandb_strategy = wrap_strategy_with_wandb(
        strategy=base_strategy,
        project_name="medical-federated-learning",
        tags=["medical", "fedavg", "hackathon"],
        config={"experiment": "baseline", "model": "cnn"}
    )
    
    # Example 2: Create FedAvg with W&B
    fedavg_wandb = create_fedavg_with_wandb(
        project_name="flower-fedavg-experiment",
        fraction_train=0.5,
        min_available_clients=3
    )
    
    # Example 3: Create FedProx with W&B
    fedprox_wandb = create_fedprox_with_wandb(
        project_name="flower-fedprox-experiment",
        proximal_mu=0.01,
        fraction_train=0.3
    )
    
    print("W&B Strategy Wrappers created successfully!")
    print("Features:")
    print("1. Automatic logging of aggregate_train() metrics")
    print("2. Automatic logging of aggregate_evaluate() metrics")
    print("3. Automatic logging of evaluate_fn metrics")
    print("4. Configuration and timing metrics")
    print("5. System performance metrics (optional)")
    print("6. Error handling and graceful fallbacks")
