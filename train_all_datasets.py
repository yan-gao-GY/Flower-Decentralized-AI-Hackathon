#!/usr/bin/env python3
"""
Training script for all MedMNIST datasets
This script trains federated learning models for all 5 medical datasets
"""

import os
import subprocess
import sys
from pathlib import Path

# Dataset configurations
DATASETS = {
    'pathmnist': {
        'num_classes': 9,
        'description': 'Pathological lung tissue images'
    },
    'dermamnist': {
        'num_classes': 7,
        'description': 'Dermatological skin lesion images'
    },
    'retinamnist': {
        'num_classes': 5,
        'description': 'Retinal fundus images'
    },
    'bloodmnist': {
        'num_classes': 8,
        'description': 'Blood cell images'
    },
    'organamnist': {
        'num_classes': 11,
        'description': 'Organ images from CT scans'
    }
}

def run_federated_training(dataset, num_classes, rounds=10):
    """Run federated training for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Training {dataset.upper()} - {DATASETS[dataset]['description']}")
    print(f"Classes: {num_classes}")
    print(f"{'='*60}")
    
    # Build the command
    cmd = [
        "flwr", "run", "--stream",
        "--run-config", f"dataset='{dataset}' num-classes={num_classes} num-server-rounds={rounds}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the federated training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Successfully completed training for {dataset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train {dataset}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted for {dataset}")
        return False

def main():
    """Main training function"""
    print("🏥 Medical AI Federated Learning Training")
    print("=" * 50)
    print("This script will train models for all 5 MedMNIST datasets")
    print("Each training session will run for 10 rounds with enhanced parameters")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if flwr is installed
    try:
        subprocess.run(["flwr", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Flower (flwr) is not installed or not in PATH")
        print("Please install it with: pip install flwr")
        sys.exit(1)
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with training all datasets? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Create results directory
    results_dir = Path("training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Track results
    results = {}
    
    # Train each dataset
    for dataset, config in DATASETS.items():
        print(f"\n🚀 Starting training for {dataset}...")
        
        success = run_federated_training(
            dataset=dataset,
            num_classes=config['num_classes'],
            rounds=10
        )
        
        results[dataset] = {
            'success': success,
            'num_classes': config['num_classes'],
            'description': config['description']
        }
        
        if success:
            print(f"✅ {dataset} training completed successfully!")
        else:
            print(f"❌ {dataset} training failed!")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for dataset, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{dataset:12} | {status} | {result['num_classes']:2d} classes | {result['description']}")
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total: {len(results)} datasets")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\n🎉 Successfully trained {successful} models!")
        print("You can now use these models in the Medical AI Service.")
    
    if failed > 0:
        print(f"\n⚠️ {failed} models failed to train.")
        print("Check the logs above for error details.")
    
    print("\nNext steps:")
    print("1. Check the training logs for any issues")
    print("2. Use the trained models in the Medical AI Service")
    print("3. Deploy the service for real-world use")

if __name__ == "__main__":
    main()
