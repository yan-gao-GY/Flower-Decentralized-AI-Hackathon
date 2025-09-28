"""
Test suite for W&B Integration with Flower
Tests both the W&B Mod and Strategy Wrapper implementations
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from typing import Dict, Any

# Import the modules to test
from wandb_mod import create_wandb_mod, create_simple_wandb_mod, create_comprehensive_wandb_mod
from wandb_strategy_wrapper import WandBStrategyWrapper, wrap_strategy_with_wandb

# Mock Flower imports for testing
class MockMessage:
    def __init__(self, content=None, metadata=None):
        self.content = content or {}
        self.metadata = metadata or Mock()
        self.metadata.message_type = Mock()
    
    def has_content(self):
        return bool(self.content)

class MockContext:
    def __init__(self, node_id=1):
        self.node_id = node_id
        self.state = Mock()
        self.state.get_value = Mock(return_value=1)

class MockArrayRecord:
    def __init__(self, data=None):
        self.data = data or {}
    
    def to_torch_state_dict(self):
        return {"weight": torch.randn(10, 10), "bias": torch.randn(10)}

class MockMetricRecord:
    def __init__(self, metrics=None):
        self.metrics = metrics or {"accuracy": 0.85, "loss": 0.15}
    
    def __iter__(self):
        return iter(self.metrics.items())
    
    def items(self):
        return self.metrics.items()
    
    def __bool__(self):
        return bool(self.metrics)

class MockStrategy:
    """Mock strategy for testing wrapper."""
    
    def __init__(self):
        self.name = "MockStrategy"
    
    def start(self, **kwargs):
        return MockArrayRecord()
    
    def configure_train(self, server_round, arrays, config, grid):
        return [MockMessage()]
    
    def aggregate_train(self, server_round, replies):
        metrics = MockMetricRecord({"train_loss": 0.5, "train_accuracy": 0.8})
        return MockArrayRecord(), metrics
    
    def configure_evaluate(self, server_round, arrays, config, grid):
        return [MockMessage()]
    
    def aggregate_evaluate(self, server_round, replies):
        return MockMetricRecord({"eval_loss": 0.3, "eval_accuracy": 0.9})
    
    def summary(self):
        pass

class TestWandBMod(unittest.TestCase):
    """Test cases for W&B Mod functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_context = MockContext()
        self.mock_message = MockMessage(
            content={
                "config": {"server-round": 1},
                "arrays": MockArrayRecord()
            }
        )
    
    @patch('wandb_mod.WANDB_AVAILABLE', False)
    def test_mod_without_wandb(self):
        """Test mod gracefully handles missing W&B."""
        mod = create_simple_wandb_mod("test-project")
        
        def mock_app(msg, ctx):
            return MockMessage()
        
        # Should not raise exception even without W&B
        result = mod(self.mock_message, self.mock_context, mock_app)
        self.assertIsInstance(result, MockMessage)
    
    @patch('wandb_mod.wandb')
    @patch('wandb_mod.WANDB_AVAILABLE', True)
    def test_mod_with_wandb(self, mock_wandb):
        """Test mod with W&B available."""
        mock_wandb.init = Mock()
        mock_wandb.log = Mock()
        mock_wandb.define_metric = Mock()
        
        mod = create_simple_wandb_mod("test-project")
        
        def mock_app(msg, ctx):
            reply = MockMessage()
            reply.content = {
                "metric_records": {
                    "metrics": MockMetricRecord({"accuracy": 0.85})
                }
            }
            return reply
        
        result = mod(self.mock_message, self.mock_context, mock_app)
        
        # Should initialize W&B on first call
        mock_wandb.init.assert_called()
        self.assertIsInstance(result, MockMessage)
    
    def test_create_simple_mod(self):
        """Test creation of simple mod."""
        mod = create_simple_wandb_mod("test-project")
        self.assertTrue(callable(mod))
    
    def test_create_comprehensive_mod(self):
        """Test creation of comprehensive mod."""
        mod = create_comprehensive_wandb_mod(
            "test-project",
            entity="test-entity",
            tags=["test", "mod"]
        )
        self.assertTrue(callable(mod))
    
    def test_create_custom_mod(self):
        """Test creation of mod with custom configuration."""
        mod = create_wandb_mod(
            project_name="custom-test",
            tags=["custom"],
            config={"model": "test"},
            log_model_size=True,
            log_communication_time=True,
            log_system_metrics=False
        )
        self.assertTrue(callable(mod))

class TestWandBStrategyWrapper(unittest.TestCase):
    """Test cases for W&B Strategy Wrapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_strategy = MockStrategy()
        self.mock_grid = Mock()
        self.mock_arrays = MockArrayRecord()
    
    @patch('wandb_strategy_wrapper.WANDB_AVAILABLE', False)
    def test_wrapper_without_wandb(self):
        """Test wrapper gracefully handles missing W&B."""
        wrapper = WandBStrategyWrapper(
            strategy=self.mock_strategy,
            project_name="test-project"
        )
        
        # Should not raise exception even without W&B
        result = wrapper.aggregate_train(1, [MockMessage()])
        self.assertIsNotNone(result)
    
    @patch('wandb_strategy_wrapper.wandb')
    @patch('wandb_strategy_wrapper.WANDB_AVAILABLE', True)
    def test_wrapper_with_wandb(self, mock_wandb):
        """Test wrapper with W&B available."""
        mock_wandb.init = Mock()
        mock_wandb.log = Mock()
        mock_wandb.define_metric = Mock()
        
        wrapper = WandBStrategyWrapper(
            strategy=self.mock_strategy,
            project_name="test-project"
        )
        
        # Test start method
        wrapper.start(
            grid=self.mock_grid,
            initial_arrays=self.mock_arrays,
            num_rounds=5
        )
        
        # Should initialize W&B
        mock_wandb.init.assert_called()
    
    def test_wrapper_delegation(self):
        """Test that wrapper properly delegates to underlying strategy."""
        wrapper = WandBStrategyWrapper(
            strategy=self.mock_strategy,
            project_name="test-project"
        )
        
        # Test delegation of methods
        result = wrapper.configure_train(1, self.mock_arrays, {}, self.mock_grid)
        self.assertIsNotNone(result)
        
        result = wrapper.aggregate_train(1, [MockMessage()])
        self.assertIsNotNone(result)
        
        result = wrapper.configure_evaluate(1, self.mock_arrays, {}, self.mock_grid)
        self.assertIsNotNone(result)
        
        result = wrapper.aggregate_evaluate(1, [MockMessage()])
        self.assertIsNotNone(result)
        
        # Should not raise exception
        wrapper.summary()
    
    def test_wrap_strategy_function(self):
        """Test convenience function for wrapping strategies."""
        wrapper = wrap_strategy_with_wandb(
            strategy=self.mock_strategy,
            project_name="test-wrap"
        )
        
        self.assertIsInstance(wrapper, WandBStrategyWrapper)
        self.assertEqual(wrapper.strategy, self.mock_strategy)
    
    def test_wrapper_configuration(self):
        """Test wrapper configuration options."""
        wrapper = WandBStrategyWrapper(
            strategy=self.mock_strategy,
            project_name="test-config",
            entity="test-entity",
            tags=["test", "config"],
            config={"test": "value"},
            log_config=True,
            log_timing=True,
            log_system_metrics=False
        )
        
        self.assertEqual(wrapper.project_name, "test-config")
        self.assertEqual(wrapper.entity, "test-entity")
        self.assertIn("test", wrapper.tags)
        self.assertIn("config", wrapper.tags)
        self.assertTrue(wrapper.log_config)
        self.assertTrue(wrapper.log_timing)
        self.assertFalse(wrapper.log_system_metrics)

class TestIntegration(unittest.TestCase):
    """Integration tests for both mod and wrapper."""
    
    @patch('wandb_mod.wandb')
    @patch('wandb_strategy_wrapper.wandb')
    @patch('wandb_mod.WANDB_AVAILABLE', True)
    @patch('wandb_strategy_wrapper.WANDB_AVAILABLE', True)
    def test_mod_and_wrapper_together(self, mock_wandb_strategy, mock_wandb_mod):
        """Test using both mod and wrapper together."""
        # Setup mocks
        for mock_wandb in [mock_wandb_mod, mock_wandb_strategy]:
            mock_wandb.init = Mock()
            mock_wandb.log = Mock()
            mock_wandb.define_metric = Mock()
        
        # Create mod
        mod = create_simple_wandb_mod("integration-test")
        
        # Create wrapper
        strategy = MockStrategy()
        wrapper = wrap_strategy_with_wandb(
            strategy=strategy,
            project_name="integration-test"
        )
        
        # Test that both can be created without errors
        self.assertTrue(callable(mod))
        self.assertIsInstance(wrapper, WandBStrategyWrapper)
        
        # Test mod functionality
        def mock_app(msg, ctx):
            return MockMessage(content={"metric_records": {"metrics": MockMetricRecord()}})
        
        mock_message = MockMessage(content={"config": {"server-round": 1}})
        mock_context = MockContext()
        
        result = mod(mock_message, mock_context, mock_app)
        self.assertIsInstance(result, MockMessage)
        
        # Test wrapper functionality
        wrapper.start(
            grid=Mock(),
            initial_arrays=MockArrayRecord(),
            num_rounds=3
        )
        
        train_result = wrapper.aggregate_train(1, [MockMessage()])
        self.assertIsNotNone(train_result)
        
        eval_result = wrapper.aggregate_evaluate(1, [MockMessage()])
        self.assertIsNotNone(eval_result)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and resilience."""
    
    @patch('wandb_mod.wandb')
    @patch('wandb_mod.WANDB_AVAILABLE', True)
    def test_mod_error_handling(self, mock_wandb):
        """Test mod handles W&B errors gracefully."""
        # Setup W&B to raise exception
        mock_wandb.init.side_effect = Exception("W&B Error")
        mock_wandb.log.side_effect = Exception("W&B Error")
        
        mod = create_simple_wandb_mod("error-test")
        
        def mock_app(msg, ctx):
            return MockMessage()
        
        # Should not raise exception even when W&B fails
        result = mod(MockMessage(), MockContext(), mock_app)
        self.assertIsInstance(result, MockMessage)
    
    @patch('wandb_strategy_wrapper.wandb')
    @patch('wandb_strategy_wrapper.WANDB_AVAILABLE', True)
    def test_wrapper_error_handling(self, mock_wandb):
        """Test wrapper handles W&B errors gracefully."""
        # Setup W&B to raise exception
        mock_wandb.init.side_effect = Exception("W&B Error")
        mock_wandb.log.side_effect = Exception("W&B Error")
        
        wrapper = WandBStrategyWrapper(
            strategy=MockStrategy(),
            project_name="error-test"
        )
        
        # Should not raise exception even when W&B fails
        wrapper.start(
            grid=Mock(),
            initial_arrays=MockArrayRecord(),
            num_rounds=3
        )
        
        result = wrapper.aggregate_train(1, [MockMessage()])
        self.assertIsNotNone(result)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestWandBMod,
        TestWandBStrategyWrapper,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing W&B Integration for Flower")
    print("=" * 60)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed! W&B integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("=" * 60)

