"""
Data generation functions for creating simulation datasets that match the experimental data format.

This module orchestrates the simulation process:
1. Run individual learning for all participants 
2. Identify trapped learners (1D rule learners)
3. Run trapped learners through different conditions (asocial, 2D partner, other-1D partner)
4. Generate JSON data compatible with the preprocessing pipeline
"""

import numpy as np
import json
from typing import List, Dict, Any
from pathlib import Path
from alcove_simulation import (
    BeeEnvironment, ALCOVE_RL, set_random_seed, calculate_scores,
    run_individual_experiment, run_social_experiment
)


def is_trapped_learner(phase1_trials: List[Dict[str, Any]], env: BeeEnvironment, threshold: float = 15/16) -> bool:
    """
    Determine if participant learned 1D rule in phase 1 (test phase).
    
    Args:
        phase1_trials: List of trials from phase 1 (test phase)
        env: Environment to check rule performance
        threshold: Performance threshold for rule classification
        
    Returns:
        True if participant is trapped (learned 1D rule)
    """
    # Extract stimuli and responses from phase 1
    stimuli = []
    responses = []
    
    for trial in phase1_trials:
        stimulus = np.array([
            trial["stimulusParams"][0],
            trial["stimulusParams"][1], 
            trial["stimulusParams"][2],
            trial["stimulusParams"][3]
        ])
        action = 1 if trial["response"] == "approach" else 0
        stimuli.append(stimulus)
        responses.append(action)
    
    # Calculate 1D and 2D scores
    score_1d, score_2d = calculate_scores(stimuli, responses, env)
    
    # Trapped if high 1D performance but low 2D performance
    return score_1d >= threshold and score_2d < threshold


def generate_full_participant_data(participant_id: str, condition: str,
                                  relevant_features: List[int] = [0, 1],
                                  dangerous_combination: List[int] = [1, 1],
                                  model_params: Dict[str, Any] = None,
                                  phase0_trials: List[Dict[str, Any]] = None,
                                  phase1_trials: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate complete participant data for all 4 phases (0-3) in one condition.
    
    Args:
        participant_id: Unique participant identifier (should include condition suffix)
        condition: "asocial", "2d", "1d_a", or "1d_b"
        relevant_features: List of relevant feature indices
        dangerous_combination: List of dangerous values
        model_params: Pre-trained model parameters (required for phases 2-3)
        phase0_trials: Pre-generated phase 0 trials (from individual learning)
        phase1_trials: Pre-generated phase 1 trials (from individual learning)
        
    Returns:
        Complete participant data dictionary with all 4 phases
    """
    # Initialize environment
    env = BeeEnvironment(
        num_features=4,
        num_relevant_features=2,
        relevant_features=relevant_features,
        dangerous_combination=dangerous_combination
    )
    
    # Initialize model and load pre-trained parameters
    model = ALCOVE_RL(num_features=4)
    if model_params:
        model.load(model_params)
    
    # Use pre-generated phase 0 and 1 trials if provided, otherwise generate them
    if phase0_trials is None:
        phase0_trials = run_individual_experiment(env, model, num_trials=64, test=False)
    
    if phase1_trials is None:
        phase1_trials = run_individual_experiment(env, model, num_trials=32, test=True)
    
    # Phases 2-3: Condition-specific learning
    if condition == "asocial":
        # Phase 2: Individual training (64 trials)
        phase2_trials = run_individual_experiment(env, model, num_trials=64, test=False)
        
        # Phase 3: Individual test (32 trials)
        phase3_trials = run_individual_experiment(env, model, num_trials=32, test=True)
        
        game_type = "solo"
        partner_rule = None
        
    elif condition == "2d":
        # Phase 2: Social training with 2D partner (64 trials)
        phase2_trials = run_social_experiment(env, model, partner_rule="2d", num_trials=64, test=False)
        
        # Phase 3: Social test with 2D partner (32 trials)
        phase3_trials = run_social_experiment(env, model, partner_rule="2d", num_trials=32, test=True)
        
        game_type = "duo" 
        partner_rule = "2d"
        
    elif condition == "1d_a":
        # Phase 2: Social training with 1d_a partner (64 trials)
        phase2_trials = run_social_experiment(env, model, partner_rule="1d_a", num_trials=64, test=False)
        
        # Phase 3: Social test with 1d_a partner (32 trials)
        phase3_trials = run_social_experiment(env, model, partner_rule="1d_a", num_trials=32, test=True)
        
        game_type = "duo"
        partner_rule = "1d_a"
        
    elif condition == "1d_b":
        # Phase 2: Social training with 1d_b partner (64 trials)
        phase2_trials = run_social_experiment(env, model, partner_rule="1d_b", num_trials=64, test=False)
        
        # Phase 3: Social test with 1d_b partner (32 trials)
        phase3_trials = run_social_experiment(env, model, partner_rule="1d_b", num_trials=32, test=True)
        
        game_type = "duo"
        partner_rule = "1d_b"
    
    # Organize all trials into task_data structure
    task_data = {
        "0": {str(i): trial for i, trial in enumerate(phase0_trials)},
        "1": {str(i): trial for i, trial in enumerate(phase1_trials)},
        "2": {str(i): trial for i, trial in enumerate(phase2_trials)},
        "3": {str(i): trial for i, trial in enumerate(phase3_trials)}
    }
    
    # Calculate total points
    all_trials = []
    for phase_trials in task_data.values():
        all_trials.extend(phase_trials.values())
    total_points = all_trials[-1]["pointsPostTrial"] if all_trials else 0
    
    # Create participant data structure
    participant_data = {
        "id": participant_id,
        "data": {
            "id": participant_id,
            "recruitment_service": "sim",
            "done": True,
            "points": total_points,
            "conditions": {
                "gameType": game_type
            },
            "task_data": task_data,
            "relevantFeatures": relevant_features,
            "relevantFeaturesBadValues": dangerous_combination
        }
    }
    
    # Add partner rule for duo games
    if partner_rule:
        participant_data["data"]["conditions"]["partnerRule"] = partner_rule
        
    return participant_data


def generate_simulation_dataset(n_participants: int = 100, base_seed: int = 42,
                               relevant_features: List[int] = [0, 1], 
                               dangerous_combination: List[int] = [1, 1]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate complete simulation dataset like real behavioral data.
    
    1. Run individual learning for all participants (phases 0-1)
    2. Randomly assign each participant to one of three conditions for phases 2-3:
       - asocial: Continue solo learning
       - 2d: Learn with 2D partner
       - other1d: Learn with other 1D partner
    
    Args:
        n_participants: Number of participants to generate
        base_seed: Base random seed for reproducibility
        relevant_features: List of relevant feature indices
        dangerous_combination: List of dangerous values
        
    Returns:
        Dictionary with datasets for each condition
    """
    print(f"Starting simulation with {n_participants} participants...")
    
    # Set global seed once for entire simulation
    set_random_seed(base_seed)
    
    # Phase 1: Individual learning for all participants
    print("\\nPhase 1: Individual learning for all participants...")
    all_participants = []
    
    for participant_idx in range(n_participants):
        
        participant_id = f"sim-{participant_idx+1:03d}"
        
        # Initialize environment and model
        env = BeeEnvironment(
            num_features=4,
            num_relevant_features=2,
            relevant_features=relevant_features,
            dangerous_combination=dangerous_combination
        )
        model = ALCOVE_RL(num_features=4)
        
        # Phase 0: Training (64 trials)
        phase0_trials = run_individual_experiment(env, model, num_trials=64, test=False)
        
        # Phase 1: Test (32 trials)
        phase1_trials = run_individual_experiment(env, model, num_trials=32, test=True)
        
        # Store participant data and model state
        all_participants.append({
            'participant_idx': participant_idx,
            'participant_id': participant_id,
            'phase0_trials': phase0_trials,
            'phase1_trials': phase1_trials,
            'model_params': model.save()
        })
    
    print(f"✓ Completed individual learning for {len(all_participants)} participants")
    
    # Phase 2: Randomly assign participants to conditions
    print("\\nPhase 2: Assigning participants to conditions...")
    conditions = ['asocial', '2d', '1d_a', '1d_b']
    results = {}
    
    # Shuffle participants and distribute evenly across conditions
    import random
    shuffled_participants = all_participants.copy()
    random.shuffle(shuffled_participants)  # Uses the sequential RNG state
    
    participants_per_condition = n_participants // len(conditions)
    remainder = n_participants % len(conditions)
    
    start_idx = 0
    for i, condition in enumerate(conditions):
        # Handle remainder by giving extra participants to first conditions
        end_idx = start_idx + participants_per_condition + (1 if i < remainder else 0)
        condition_participants = shuffled_participants[start_idx:end_idx]
        
        print(f"  {condition}: {len(condition_participants)} participants")
        
        condition_data = []
        for participant_data in condition_participants:
            # Keep original participant ID (no condition suffix)
            participant_id = participant_data['participant_id']
            
            # Generate full participant data with condition-specific phases 2-3
            full_participant_data = generate_full_participant_data(
                participant_id=participant_id,
                condition=condition,
                relevant_features=relevant_features,
                dangerous_combination=dangerous_combination,
                model_params=participant_data['model_params'],
                phase0_trials=participant_data['phase0_trials'],
                phase1_trials=participant_data['phase1_trials']
            )
            
            condition_data.append(full_participant_data)
        
        results[condition] = condition_data
        start_idx = end_idx
    
    print(f"\\n✓ Total participants generated: {sum(len(data) for data in results.values())}")
    return results


def save_simulation_metadata(output_dir: str, results: Dict[str, List], base_seed: int,
                           relevant_features: List[int], dangerous_combination: List[int]) -> None:
    """Save metadata about the simulation for reproducibility."""
    from datetime import datetime
    
    metadata = {
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "base_seed": base_seed,
            "relevant_features": relevant_features,
            "dangerous_combination": dangerous_combination,
            "random_seed_info": "Single global seed used for entire simulation - all randomness proceeds sequentially"
        },
        "dataset_summary": {
            condition: len(participants) for condition, participants in results.items()
        },
        "experimental_structure": {
            "phase_0": "Training (64 trials) - individual learning for all",
            "phase_1": "Test (32 trials) - individual learning for all", 
            "phase_2": "Training (64 trials) - condition-specific learning",
            "phase_3": "Test (32 trials) - condition-specific learning"
        },
        "conditions": {
            "asocial": "Solo learning continuation (phases 0-3)",
            "2d": "Social learning with 2D partner (phases 0-3)",
            "other-1d": "Social learning with other 1D partner (phases 0-3)"
        },
        "model_parameters": {
            "c": 6,
            "phi": 30,
            "lambda_w": 0.15,
            "lambda_alpha": 0.15
        },
        "environment_parameters": {
            "num_features": 4,
            "num_relevant_features": 2,
            "reward_structure": {
                "approach_safe": 1,
                "approach_dangerous": -4,
                "avoid": 0
            }
        }
    }
    
    Path(output_dir).mkdir(exist_ok=True)
    metadata_file = f"{output_dir}/simulation_metadata.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_file}")