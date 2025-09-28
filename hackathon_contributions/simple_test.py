"""
Simple test to verify W&B integration modules can be imported and basic functionality works
"""

def test_imports():
    """Test that modules can be imported successfully."""
    try:
        from wandb_mod import create_wandb_mod, create_simple_wandb_mod
        print("✅ wandb_mod imports successfully")
        
        from wandb_strategy_wrapper import WandBStrategyWrapper, wrap_strategy_with_wandb
        print("✅ wandb_strategy_wrapper imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_mod_creation():
    """Test that mods can be created."""
    try:
        from wandb_mod import create_simple_wandb_mod, create_comprehensive_wandb_mod
        
        # Test simple mod creation
        simple_mod = create_simple_wandb_mod("test-project")
        assert callable(simple_mod), "Simple mod should be callable"
        print("✅ Simple mod creation works")
        
        # Test comprehensive mod creation
        comprehensive_mod = create_comprehensive_wandb_mod("test-project")
        assert callable(comprehensive_mod), "Comprehensive mod should be callable"
        print("✅ Comprehensive mod creation works")
        
        return True
    except Exception as e:
        print(f"❌ Mod creation error: {e}")
        return False

def test_wrapper_creation():
    """Test that strategy wrapper can be created."""
    try:
        from wandb_strategy_wrapper import WandBStrategyWrapper
        
        # Mock strategy
        class MockStrategy:
            def start(self, **kwargs): pass
            def configure_train(self, *args): return []
            def aggregate_train(self, *args): return None, None
            def configure_evaluate(self, *args): return []
            def aggregate_evaluate(self, *args): return None
            def summary(self): pass
        
        mock_strategy = MockStrategy()
        
        # Test wrapper creation
        wrapper = WandBStrategyWrapper(
            strategy=mock_strategy,
            project_name="test-project"
        )
        
        assert wrapper.strategy == mock_strategy, "Wrapper should contain the strategy"
        assert wrapper.project_name == "test-project", "Project name should be set"
        print("✅ Strategy wrapper creation works")
        
        return True
    except Exception as e:
        print(f"❌ Wrapper creation error: {e}")
        return False

def test_error_handling():
    """Test error handling when wandb is not available."""
    try:
        # Test mod with WANDB_AVAILABLE = False
        import wandb_mod
        original_available = wandb_mod.WANDB_AVAILABLE
        wandb_mod.WANDB_AVAILABLE = False
        
        mod = wandb_mod.create_simple_wandb_mod("test")
        assert callable(mod), "Mod should still be callable even without wandb"
        
        # Restore original value
        wandb_mod.WANDB_AVAILABLE = original_available
        print("✅ Error handling works for mod")
        
        # Test wrapper with WANDB_AVAILABLE = False
        import wandb_strategy_wrapper
        original_available = wandb_strategy_wrapper.WANDB_AVAILABLE
        wandb_strategy_wrapper.WANDB_AVAILABLE = False
        
        class MockStrategy:
            def __init__(self): pass
        
        wrapper = wandb_strategy_wrapper.WandBStrategyWrapper(
            strategy=MockStrategy(),
            project_name="test"
        )
        assert wrapper is not None, "Wrapper should be created even without wandb"
        
        # Restore original value
        wandb_strategy_wrapper.WANDB_AVAILABLE = original_available
        print("✅ Error handling works for wrapper")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing W&B Integration for Flower - Simple Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Mod Creation", test_mod_creation), 
        ("Wrapper Creation", test_wrapper_creation),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! W&B integration is ready for submission.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

