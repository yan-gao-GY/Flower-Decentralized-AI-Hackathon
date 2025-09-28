#!/usr/bin/env python3
"""
Vertical Federated Medical AI Revolution - Main Integration Script
Demonstrates the complete Track 2 solution with all three innovations:
1. Vertical Federated Learning
2. Automated Data Harmonization  
3. Contribution-Based Reward System
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
import json
import time
import argparse
from pathlib import Path

# Import our custom modules
from vertical_fl_core import (
    VerticalFLOrchestrator, 
    MultiModalMedicalModel, 
    DataModality, 
    PatientRecord,
    OrganizationProfile
)
from data_harmonization import MedicalDataHarmonizer
from contribution_rewards import ContributionRewardSystem
from vertical_fl_simulation import run_vertical_fl_simulation, VerticalFLConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VerticalMedicalAIRevolution:
    """Main class orchestrating the vertical federated medical AI revolution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = VerticalFLOrchestrator(
            num_organizations=config.get('num_organizations', 8)
        )
        self.harmonizer = MedicalDataHarmonizer()
        self.reward_system = ContributionRewardSystem(
            total_reward_pool=config.get('total_reward_pool', 1000.0),
            reward_distribution_method=config.get('reward_method', 'hybrid')
        )
        self.results = {}
        
    def setup_organizations(self) -> Dict[str, OrganizationProfile]:
        """Setup healthcare organizations for vertical FL"""
        organizations = {}
        
        # Hospital A - Medical Images + Clinical Records
        hospital_a = OrganizationProfile(
            org_id="hospital_a",
            name="General Hospital A",
            data_modalities=[DataModality.IMAGES, DataModality.CLINICAL],
            data_quality_score=0.92,
            privacy_compliance=0.95,
            participation_rate=0.85,
            contribution_rewards=0.0
        )
        organizations["hospital_a"] = hospital_a
        
        # Hospital B - Medical Images + Clinical Records (different quality)
        hospital_b = OrganizationProfile(
            org_id="hospital_b", 
            name="Specialty Hospital B",
            data_modalities=[DataModality.IMAGES, DataModality.CLINICAL],
            data_quality_score=0.88,
            privacy_compliance=0.90,
            participation_rate=0.80,
            contribution_rewards=0.0
        )
        organizations["hospital_b"] = hospital_b
        
        # Clinic A - Clinical Records + Medications
        clinic_a = OrganizationProfile(
            org_id="clinic_a",
            name="Family Clinic A", 
            data_modalities=[DataModality.CLINICAL, DataModality.MEDICATIONS],
            data_quality_score=0.85,
            privacy_compliance=0.88,
            participation_rate=0.90,
            contribution_rewards=0.0
        )
        organizations["clinic_a"] = clinic_a
        
        # Clinic B - Clinical Records + Medications
        clinic_b = OrganizationProfile(
            org_id="clinic_b",
            name="Community Clinic B",
            data_modalities=[DataModality.CLINICAL, DataModality.MEDICATIONS],
            data_quality_score=0.82,
            privacy_compliance=0.85,
            participation_rate=0.75,
            contribution_rewards=0.0
        )
        organizations["clinic_b"] = clinic_b
        
        # Insurance A - Demographics
        insurance_a = OrganizationProfile(
            org_id="insurance_a",
            name="Health Insurance A",
            data_modalities=[DataModality.DEMOGRAPHICS],
            data_quality_score=0.78,
            privacy_compliance=0.82,
            participation_rate=0.70,
            contribution_rewards=0.0
        )
        organizations["insurance_a"] = insurance_a
        
        # Insurance B - Demographics
        insurance_b = OrganizationProfile(
            org_id="insurance_b",
            name="Health Insurance B", 
            data_modalities=[DataModality.DEMOGRAPHICS],
            data_quality_score=0.75,
            privacy_compliance=0.80,
            participation_rate=0.65,
            contribution_rewards=0.0
        )
        organizations["insurance_b"] = insurance_b
        
        # Research Center A - Genomics
        research_a = OrganizationProfile(
            org_id="research_a",
            name="Medical Research Center A",
            data_modalities=[DataModality.GENOMICS],
            data_quality_score=0.96,
            privacy_compliance=0.98,
            participation_rate=0.60,
            contribution_rewards=0.0
        )
        organizations["research_a"] = research_a
        
        # Research Center B - Genomics
        research_b = OrganizationProfile(
            org_id="research_b",
            name="Genomics Research Center B",
            data_modalities=[DataModality.GENOMICS],
            data_quality_score=0.94,
            privacy_compliance=0.96,
            participation_rate=0.55,
            contribution_rewards=0.0
        )
        organizations["research_b"] = research_b
        
        # Register all organizations
        for org in organizations.values():
            self.orchestrator.register_organization(org)
            logger.info(f"Registered organization: {org.name}")
        
        return organizations
    
    def generate_synthetic_data(self, organizations: Dict[str, OrganizationProfile]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic medical data for demonstration"""
        np.random.seed(42)
        datasets = {}
        
        for org_id, org_profile in organizations.items():
            # Generate patient IDs
            num_patients = np.random.randint(1000, 5000)
            patient_ids = [f"P{org_id}_{i:06d}" for i in range(num_patients)]
            
            # Base dataframe
            df = pd.DataFrame({'patient_id': patient_ids})
            
            # Add data based on organization type
            if DataModality.IMAGES in org_profile.data_modalities:
                df['xray_quality'] = np.random.normal(0.8, 0.1, num_patients)
                df['mri_quality'] = np.random.normal(0.85, 0.1, num_patients)
                df['ct_quality'] = np.random.normal(0.82, 0.1, num_patients)
            
            if DataModality.CLINICAL in org_profile.data_modalities:
                df['blood_pressure'] = np.random.normal(120, 20, num_patients)
                df['heart_rate'] = np.random.normal(72, 10, num_patients)
                df['temperature'] = np.random.normal(98.6, 1.0, num_patients)
                df['glucose_level'] = np.random.normal(100, 30, num_patients)
                df['cholesterol'] = np.random.normal(200, 40, num_patients)
            
            if DataModality.DEMOGRAPHICS in org_profile.data_modalities:
                df['age'] = np.random.randint(18, 80, num_patients)
                df['gender'] = np.random.choice(['M', 'F'], num_patients)
                df['insurance_type'] = np.random.choice(['private', 'medicare', 'medicaid'], num_patients)
                df['income_level'] = np.random.choice(['low', 'medium', 'high'], num_patients)
            
            if DataModality.MEDICATIONS in org_profile.data_modalities:
                df['medication_count'] = np.random.poisson(3, num_patients)
                df['prescription_cost'] = np.random.exponential(100, num_patients)
                df['adherence_rate'] = np.random.beta(2, 1, num_patients)
            
            if DataModality.GENOMICS in org_profile.data_modalities:
                df['genetic_risk_score'] = np.random.beta(2, 5, num_patients)
                df['family_history_score'] = np.random.beta(1, 3, num_patients)
                df['genetic_variants'] = np.random.poisson(5, num_patients)
            
            # Add some missing values based on data quality
            missing_rate = 1.0 - org_profile.data_quality_score
            for col in df.columns:
                if col != 'patient_id':
                    mask = np.random.random(num_patients) < missing_rate
                    df.loc[mask, col] = np.nan
            
            datasets[org_id] = df
            logger.info(f"Generated {len(df)} records for {org_id}")
        
        return datasets
    
    def run_data_harmonization(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run automated data harmonization"""
        logger.info("Starting automated data harmonization...")
        
        # Detect data modalities
        modalities = self.harmonizer.detect_data_modalities(datasets)
        logger.info(f"Detected modalities: {modalities}")
        
        # Analyze schema differences
        schema_analysis = self.harmonizer.analyze_schema_differences(datasets)
        logger.info(f"Analyzed schemas for {len(schema_analysis)} organizations")
        
        # Create schema mappings
        mappings = {}
        reference_org = list(datasets.keys())[0]  # Use first org as reference
        
        for org_id in datasets.keys():
            if org_id != reference_org:
                org_mappings = self.harmonizer.create_schema_mappings(
                    org_id, reference_org, schema_analysis
                )
                mappings[org_id] = org_mappings
                logger.info(f"Created {len(org_mappings)} mappings for {org_id}")
        
        # Assess data quality
        quality_metrics = self.harmonizer.assess_data_quality(datasets)
        logger.info(f"Assessed quality for {len(quality_metrics)} organizations")
        
        # Generate harmonization report
        harmonization_report = self.harmonizer.generate_harmonization_report(
            schema_analysis, mappings, quality_metrics
        )
        
        # Save report
        with open('harmonization_report.json', 'w') as f:
            json.dump(harmonization_report, f, indent=2, default=str)
        
        logger.info("Data harmonization completed successfully!")
        return harmonization_report
    
    def run_vertical_fl_simulation(self) -> Dict[str, Any]:
        """Run vertical federated learning simulation"""
        logger.info("Starting vertical federated learning simulation...")
        
        # Create simulation config
        sim_config = VerticalFLConfig(
            num_organizations=self.config.get('num_organizations', 8),
            num_rounds=self.config.get('num_rounds', 20),
            num_clients_per_round=self.config.get('num_clients_per_round', 4),
            learning_rate=self.config.get('learning_rate', 0.001),
            batch_size=self.config.get('batch_size', 32),
            num_epochs=self.config.get('num_epochs', 3)
        )
        
        # Run simulation
        results = run_vertical_fl_simulation(sim_config)
        
        logger.info("Vertical FL simulation completed successfully!")
        return results
    
    def run_contribution_rewards(self, round_results: Dict[str, Any], 
                                organization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run contribution-based reward system"""
        logger.info("Calculating contribution-based rewards...")
        
        # Calculate contribution scores and rewards
        contributions = self.reward_system.calculate_contribution_scores(
            round_results, organization_data
        )
        
        # Get leaderboard
        leaderboard = self.reward_system.get_leaderboard()
        
        # Export reward report
        report_file = self.reward_system.export_reward_report()
        
        logger.info(f"Contribution rewards calculated. Report saved to: {report_file}")
        
        return {
            'contributions': contributions,
            'leaderboard': leaderboard,
            'report_file': report_file
        }
    
    def run_complete_demonstration(self) -> Dict[str, Any]:
        """Run the complete vertical federated medical AI revolution demonstration"""
        logger.info("🚀 Starting Vertical Federated Medical AI Revolution")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        # Step 1: Setup organizations
        logger.info("Step 1: Setting up healthcare organizations...")
        organizations = self.setup_organizations()
        results['organizations'] = {org_id: {
            'name': org.name,
            'modalities': [m.value for m in org.data_modalities],
            'quality_score': org.data_quality_score
        } for org_id, org in organizations.items()}
        
        # Step 2: Generate synthetic data
        logger.info("Step 2: Generating synthetic medical data...")
        datasets = self.generate_synthetic_data(organizations)
        results['data_generation'] = {
            'total_organizations': len(datasets),
            'total_records': sum(len(df) for df in datasets.values())
        }
        
        # Step 3: Data harmonization
        logger.info("Step 3: Running automated data harmonization...")
        harmonization_results = self.run_data_harmonization(datasets)
        results['harmonization'] = harmonization_results
        
        # Step 4: Vertical FL simulation
        logger.info("Step 4: Running vertical federated learning simulation...")
        fl_results = self.run_vertical_fl_simulation()
        results['federated_learning'] = fl_results
        
        # Step 5: Contribution rewards
        logger.info("Step 5: Calculating contribution-based rewards...")
        
        # Create sample round results and organization data
        round_results = {
            f"{org_id}_accuracy": np.random.uniform(0.7, 0.95) for org_id in organizations.keys()
        }
        round_results.update({
            f"{org_id}_loss": np.random.uniform(0.2, 0.5) for org_id in organizations.keys()
        })
        round_results.update({
            f"{org_id}_participated_rounds": np.random.randint(15, 20) for org_id in organizations.keys()
        })
        round_results['total_rounds'] = 20
        
        organization_data = {
            org_id: {
                'completeness': org.data_quality_score,
                'consistency': org.data_quality_score * 0.9,
                'accuracy': org.data_quality_score * 0.95,
                'timeliness': org.data_quality_score * 0.85,
                'validity': org.data_quality_score * 0.9,
                'num_samples': len(datasets[org_id]),
                'num_features': len(datasets[org_id].columns) - 1,  # Exclude patient_id
                'num_modalities': len(org.data_modalities),
                'demographic_diversity': np.random.uniform(0.6, 0.9),
                'geographic_diversity': np.random.uniform(0.5, 0.8),
                'temporal_diversity': np.random.uniform(0.7, 0.95),
                'privacy_budget_usage': np.random.uniform(0.05, 0.2),
                'differential_privacy': True,
                'anonymization_quality': np.random.uniform(0.8, 0.95)
            }
            for org_id, org in organizations.items()
        }
        
        reward_results = self.run_contribution_rewards(round_results, organization_data)
        results['contribution_rewards'] = reward_results
        
        # Calculate total time
        total_time = time.time() - start_time
        results['execution_time'] = total_time
        
        # Print summary
        self._print_demonstration_summary(results)
        
        # Save complete results
        with open('vertical_medical_ai_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("🎉 Vertical Federated Medical AI Revolution completed successfully!")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info("Results saved to: vertical_medical_ai_results.json")
        
        return results
    
    def _print_demonstration_summary(self, results: Dict[str, Any]):
        """Print demonstration summary"""
        print("\n" + "=" * 80)
        print("🏥 VERTICAL FEDERATED MEDICAL AI REVOLUTION - DEMONSTRATION SUMMARY")
        print("=" * 80)
        
        # Organizations
        print(f"\n📊 Organizations: {len(results['organizations'])}")
        for org_id, org_info in results['organizations'].items():
            print(f"  • {org_info['name']}: {org_info['modalities']} (Quality: {org_info['quality_score']:.2f})")
        
        # Data generation
        data_info = results['data_generation']
        print(f"\n💾 Data Generation:")
        print(f"  • Total Organizations: {data_info['total_organizations']}")
        print(f"  • Total Records: {data_info['total_records']:,}")
        
        # Harmonization
        harm_info = results['harmonization']['summary']
        print(f"\n🔧 Data Harmonization:")
        print(f"  • Total Mappings: {harm_info['total_mappings']}")
        print(f"  • Average Quality Score: {harm_info['average_quality_score']:.3f}")
        
        # Contribution rewards
        if 'contribution_rewards' in results:
            leaderboard = results['contribution_rewards']['leaderboard']
            print(f"\n🏆 Contribution Rewards Leaderboard:")
            for entry in leaderboard[:5]:  # Top 5
                print(f"  {entry['rank']}. {entry['org_id']}: {entry['total_score']:.3f} (${entry['reward_amount']:.2f})")
        
        # Execution time
        print(f"\n⏱️  Execution Time: {results['execution_time']:.2f} seconds")
        print("=" * 80)

def main():
    """Main function to run the vertical federated medical AI revolution"""
    parser = argparse.ArgumentParser(description='Vertical Federated Medical AI Revolution')
    parser.add_argument('--num-organizations', type=int, default=8, help='Number of organizations')
    parser.add_argument('--num-rounds', type=int, default=20, help='Number of FL rounds')
    parser.add_argument('--reward-pool', type=float, default=1000.0, help='Total reward pool')
    parser.add_argument('--reward-method', choices=['proportional', 'ranked', 'hybrid'], 
                       default='hybrid', help='Reward distribution method')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'num_organizations': args.num_organizations,
        'num_rounds': args.num_rounds,
        'total_reward_pool': args.reward_pool,
        'reward_method': args.reward_method,
        'output_dir': args.output_dir
    }
    
    # Initialize and run demonstration
    revolution = VerticalMedicalAIRevolution(config)
    results = revolution.run_complete_demonstration()
    
    print(f"\n🎉 Demonstration completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}/")
    print(f"📊 Check 'vertical_medical_ai_results.json' for detailed results")

if __name__ == "__main__":
    main()

