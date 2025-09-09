#!/usr/bin/env python3
"""
Main script for generating simulated experimental data.

This script simulates realistic behavioral data:
1. Run individual learning for N participants (phases 0-1)
2. Randomly assign each participant to one of 3 conditions for phases 2-3:
   - asocial: Continue solo learning
   - 2d: Learn with 2D partner
   - other-1d: Learn with other 1D partner
3. Save datasets in JSON format compatible with preprocessing pipeline

Usage:
    python generate_simulation_data.py

Output saved to: outputs/
"""

import json
from pathlib import Path
from data_generation import generate_simulation_dataset, save_simulation_metadata


def main():
    """Generate simulation data and save to JSON files."""
    
    # Simulation parameters
    n_participants = 1000       # Number of participants to generate and distribute across conditions
    base_seed = 42             # Base random seed for reproducibility  
    relevant_features = [0, 1]  # Features 0 and 1 are relevant
    dangerous_combination = [1, 1]  # Both features = 1 is dangerous
    
    print("=" * 60)
    print("ALCOVE-RL Simulation Data Generation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Total participants: {n_participants}")
    print(f"  - Random seed: {base_seed}")
    print(f"  - Relevant features: {relevant_features}")
    print(f"  - Dangerous combination: {dangerous_combination}")
    print(f"  - Environment: 4 features, 16 stimuli")
    print(f"  - Model: ALCOVE-RL (c=6, phi=30, λw=0.15, λα=0.15)")
    print()
    
    # Generate the simulation dataset
    print("Starting simulation...")
    results = generate_simulation_dataset(
        n_participants=n_participants,
        base_seed=base_seed,
        relevant_features=relevant_features,
        dangerous_combination=dangerous_combination
    )
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving datasets to {output_dir}/")
    
    # Combine all participants into one dataset
    all_participants = []
    for condition, participants in results.items():
        all_participants.extend(participants)
        print(f"  - {condition}: {len(participants)} participants")
    
    # Save combined dataset to single JSON file
    output_file = output_dir / "simulated_data.json"
    print(f"\nSaving combined dataset ({len(all_participants)} participants) -> {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(all_participants, f, indent=2)
    
    # Save metadata
    save_simulation_metadata(str(output_dir), results, base_seed, 
                            relevant_features, dangerous_combination)
    
    print(f"\n✓ Simulation completed successfully!")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  - Total participants: {len(all_participants)}")
    print(f"  - Conditions: {len(results)}")
    for condition, participants in results.items():
        print(f"    * {condition}: {len(participants)} participants")
    
    if all_participants:
        first_participant = all_participants[0]
        total_phases = len(first_participant['data']['task_data'])
        total_trials = sum(len(phase_data) for phase_data in first_participant['data']['task_data'].values())
        
        print(f"  - Phases per participant: {total_phases}")
        print(f"  - Trials per participant: {total_trials}")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Main output file: {output_file.name}")
    print(f"Metadata file: simulation_metadata.json")


if __name__ == "__main__":
    main()