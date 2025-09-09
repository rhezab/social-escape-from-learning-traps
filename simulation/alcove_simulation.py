"""
Core ALCOVE-RL simulation module for dyadic learning trap experiments.

This module contains the core classes and functions for running ALCOVE-RL simulations,
including the BeeEnvironment, ALCOVE_RL model, and experiment execution functions.
"""

import numpy as np
import random
import copy
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
from datetime import datetime


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all random number generators."""
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class TrialData:
    """Data structure for individual trial results."""
    stimulus: np.ndarray
    stimulus_idx: int
    response: str  # 'approach' or 'avoid'
    partner_response: Optional[str]
    points_pre_trial: int
    points_post_trial: int
    point_change: int
    trial: int
    phase: int
    block: int
    decision_time: float
    decision_duration: float
    timeout: bool
    outcome: str
    approach_outcome: int  # -1 for punishing stimuli, +1 for rewarding stimuli


class BeeEnvironment:
    """
    Environment for the bee approach/avoidance task.
    
    Manages stimulus generation, reward calculation, and block structure
    for the dyadic learning trap experiment.
    """
    
    def __init__(self, num_features: int = 4, num_relevant_features: int = 2, 
                 relevant_features: Optional[List[int]] = None,
                 dangerous_combination: Optional[List[int]] = None):
        self.num_features = num_features
        self.num_relevant_features = num_relevant_features
        self.num_stimuli = 2**num_features
        
        # Set relevant features - either provided or randomly selected
        if relevant_features is not None:
            self.relevant_features = np.array(relevant_features)
        else:
            self.relevant_features = np.random.choice(num_features, num_relevant_features, replace=False)
            self.relevant_features = np.sort(self.relevant_features)
        
        # Set dangerous combination - either provided or randomly selected
        if dangerous_combination is not None:
            self.dangerous_combination = np.array(dangerous_combination)
        else:
            self.dangerous_combination = np.random.choice(2, num_relevant_features)
        
        # Generate all possible stimuli
        self.stimuli = np.array([list(np.binary_repr(i, width=num_features)) 
                                for i in range(self.num_stimuli)], dtype=int)
    
    def is_dangerous(self, stimulus: np.ndarray) -> bool:
        """Check if a stimulus is dangerous based on relevant features."""
        return np.all(stimulus[self.relevant_features] == self.dangerous_combination)
    
    def get_reward(self, stimulus: np.ndarray, action: int) -> int:
        """Get reward for an action on a stimulus."""
        if action == 1:  # Approach
            return -4 if self.is_dangerous(stimulus) else 1
        else:  # Avoid
            return 0
    
    def get_random_stimulus(self) -> np.ndarray:
        """Get a random stimulus from all possible stimuli."""
        return self.stimuli[np.random.randint(self.num_stimuli)]

    def generate_block(self) -> np.ndarray:
        """
        Generate a balanced block of stimuli.
        
        Each block contains all possible stimuli, with equal numbers of dangerous
        stimuli in each half-block.
        """
        # Calculate the number of dangerous stimuli
        num_dangerous = 2**(self.num_features - self.num_relevant_features)
        num_friendly = self.num_stimuli - num_dangerous

        # Calculate the number of dangerous stimuli per half-block
        dangerous_per_half = num_dangerous // 2
        friendly_per_half = self.num_stimuli // 2 - dangerous_per_half

        # Create a list of all stimuli indices
        all_indices = list(range(self.num_stimuli))

        # Separate dangerous and friendly stimuli
        dangerous_indices = [i for i in all_indices if self.is_dangerous(self.stimuli[i])]
        friendly_indices = [i for i in all_indices if not self.is_dangerous(self.stimuli[i])]

        # Shuffle the indices
        np.random.shuffle(dangerous_indices)
        np.random.shuffle(friendly_indices)

        # Create the two halves of the block
        first_half = dangerous_indices[:dangerous_per_half] + friendly_indices[:friendly_per_half]
        second_half = dangerous_indices[dangerous_per_half:] + friendly_indices[friendly_per_half:]

        # Shuffle each half
        np.random.shuffle(first_half)
        np.random.shuffle(second_half)

        # Combine the halves and return the block
        block = np.concatenate([self.stimuli[first_half], self.stimuli[second_half]])
        return block


class ALCOVE_RL:
    """
    ALCOVE-RL model for reinforcement learning with attention.
    
    Implements the ALCOVE (Attention Learning COVEring map) model adapted
    for reinforcement learning tasks with exemplar-based generalization
    and attentional learning.
    """
    
    def __init__(self, num_features: int, num_actions: int = 2, c: float = 6, 
                 phi: float = 30, lambda_w: float = 0.15, lambda_alpha: float = 0.15):
        self.num_features = num_features
        self.num_exemplars = 2**num_features
        self.num_actions = num_actions
        
        # Model parameters
        self.c = c  # Specificity constant
        self.phi = phi  # Temperature parameter for action selection
        self.lambda_w = lambda_w  # Learning rate for association weights
        self.lambda_alpha = lambda_alpha  # Learning rate for attention weights
        
        # Model state
        self.alpha = np.ones(num_features) / num_features  # Attention weights
        self.hidden_exemplars = np.array([list(np.binary_repr(i, width=num_features)) 
                                         for i in range(self.num_exemplars)], dtype=int)
        self.w = np.zeros((num_actions, self.num_exemplars))  # Association weights

    def save(self) -> Dict[str, Any]:
        """Save model parameters to a dictionary."""
        return {
            'alpha': np.copy(self.alpha),
            'w': np.copy(self.w),
            'c': float(self.c),
            'phi': float(self.phi),
            'lambda_w': float(self.lambda_w),
            'lambda_alpha': float(self.lambda_alpha)
        }
        
    def load(self, params: Dict[str, Any]) -> None:
        """Load model parameters from a dictionary."""
        self.alpha = np.copy(params['alpha'])
        self.w = np.copy(params['w'])
        self.c = float(params['c'])
        self.phi = float(params['phi'])
        self.lambda_w = float(params['lambda_w'])
        self.lambda_alpha = float(params['lambda_alpha'])
    
    def hidden_activation(self, stimulus: np.ndarray) -> np.ndarray:
        """Compute activation of hidden exemplar nodes."""
        distances = np.abs(self.hidden_exemplars - stimulus)
        weighted_distances = np.sum(self.alpha * distances, axis=1)
        return np.exp(-self.c * weighted_distances)
    
    def output_activation(self, hidden_activations: np.ndarray) -> np.ndarray:
        """Compute activation of output nodes."""
        raw_output = np.dot(self.w, hidden_activations)
        return raw_output / np.sum(hidden_activations)
    
    def choose_action(self, output_activations: np.ndarray) -> int:
        """Select an action probabilistically based on output activations."""
        probabilities = np.exp(self.phi * output_activations)
        probabilities /= np.sum(probabilities)
        if np.any(np.isnan(probabilities)):
            print("Warning: NaN detected in probabilities")
            print("Probabilities:", probabilities)
            print("Output activations:", output_activations)
        return np.random.choice(self.num_actions, p=probabilities)
    
    def update(self, stimulus: np.ndarray, chosen_action: int, reward: float) -> None:
        """Update association and attention weights based on received reward."""
        hidden_activations = self.hidden_activation(stimulus)
        output_activations = self.output_activation(hidden_activations)
        
        targets = output_activations.copy()
        targets[chosen_action] = reward
        
        error = targets - output_activations
        self.w += self.lambda_w * np.outer(error, hidden_activations)
        
        # Update attention weights
        attention_gradients = np.zeros(self.num_features)
        for j in range(self.num_exemplars):
            for k in range(self.num_actions):
                attention_gradients += error[k] * self.w[k, j] * hidden_activations[j] * \
                                     self.c * np.abs(self.hidden_exemplars[j] - stimulus)
        self.alpha -= self.lambda_alpha * attention_gradients
        self.alpha = np.clip(self.alpha, 1e-1, None)
        self.alpha /= np.sum(self.alpha)
    
    def social_update(self, stimulus: np.ndarray, chosen_action: int, reward: float, 
                     partner_action: int, partner_action_reward: float) -> None:
        """Update weights with social information from partner."""
        hidden_activations = self.hidden_activation(stimulus)
        output_activations = self.output_activation(hidden_activations)
        
        targets = output_activations.copy()
        targets[chosen_action] = reward
        if partner_action == 1 and chosen_action == 0:  # Partner approaches, agent avoids
            targets[partner_action] = partner_action_reward  # Give bonus to approach
        
        error = targets - output_activations
        self.w += self.lambda_w * np.outer(error, hidden_activations)
        
        # Update attention weights
        attention_gradients = np.zeros(self.num_features)
        for j in range(self.num_exemplars):
            for k in range(self.num_actions):
                attention_gradients += error[k] * self.w[k, j] * hidden_activations[j] * \
                                     self.c * np.abs(self.hidden_exemplars[j] - stimulus)
        self.alpha -= self.lambda_alpha * attention_gradients
        self.alpha = np.clip(self.alpha, 1e-1, None)
        self.alpha /= np.sum(self.alpha)

    def act(self, stimulus: np.ndarray) -> int:
        """Process a stimulus and return a chosen action."""
        hidden_activations = self.hidden_activation(stimulus)
        output_activations = self.output_activation(hidden_activations)
        return self.choose_action(output_activations)


def calculate_scores(block_stimuli: List[np.ndarray], block_choices: List[int], 
                    env: BeeEnvironment) -> Tuple[float, float]:
    """
    Calculate 1D and 2D decision rule scores for a block.
    
    Args:
        block_stimuli: List of stimuli presented in the block
        block_choices: List of actions chosen for each stimulus
        env: Environment containing the task structure
    
    Returns:
        Tuple of (1D score, 2D score)
    """
    # Calculate 2D score (optimal strategy)
    correct_2d = sum(action == (1 - env.is_dangerous(s)) 
                    for s, action in zip(block_stimuli, block_choices))
    score_2d = correct_2d / len(block_stimuli)
    
    # Calculate 1D scores for each dimension
    scores_1d = []
    for dim in range(env.num_features):
        correct_1d = sum(action == (1 - s[dim]) 
                        for s, action in zip(block_stimuli, block_choices))
        scores_1d.append(max(correct_1d, len(block_stimuli) - correct_1d) / len(block_stimuli))
    
    # Best 1D score is the maximum over all dimensions
    score_1d = max(scores_1d)
    return score_1d, score_2d


def run_experiment(env: BeeEnvironment, model: ALCOVE_RL, num_trials: int = 64, 
                  verbose: bool = False, test: bool = False, partner_rule: str = None,
                  intrinsic_social_reward: float = 0.5) -> List[Dict[str, Any]]:
    """
    Run an experiment (individual or social) and return trial-by-trial data.
    
    Args:
        env: Environment for the experiment
        model: ALCOVE_RL model to use
        num_trials: Number of trials to run (64 for training, 32 for test)
        verbose: Whether to print progress
        test: Whether this is a test phase (no learning)
        partner_rule: Partner's decision rule ('2d', '1d_a', '1d_b') or None for individual
        intrinsic_social_reward: Reward value for social learning (ignored if partner_rule is None)
    
    Returns:
        List of trial dictionaries. Individual: trial, pointsPreTrial, pointsPostTrial, stimulusParams, response, approachOutcome
        Social: same + partnerResponse
    """
    trials = []
    cumulative_points = 0
    is_social = partner_rule is not None
    
    # Determine the 1D rule if needed
    if is_social and partner_rule in ['1d_a', '1d', '1d_b']:
        if partner_rule == '1d_a' or partner_rule == '1d':
            one_d_feature = env.relevant_features[0]
            one_d_dangerous_value = env.dangerous_combination[0]
        elif partner_rule == '1d_b':
            one_d_feature = env.relevant_features[1]
            one_d_dangerous_value = env.dangerous_combination[1]

    for trial_idx in range(num_trials):
        # Generate stimulus (one per trial, following block structure)
        if trial_idx % 16 == 0:  # New block every 16 trials
            block_stimuli = env.generate_block()
        
        stimulus = block_stimuli[trial_idx % 16]
        
        # Get agent action
        action = model.act(stimulus)
        response = "approach" if action == 1 else "avoid"
        
        # Get reward
        reward = env.get_reward(stimulus, action)
        
        # Determine approach outcome based on stimulus danger
        approach_outcome = -1 if env.is_dangerous(stimulus) else 1
        
        # Create trial data matching README format
        trial_data = {
            "trial": trial_idx,
            "pointsPreTrial": cumulative_points,
            "pointsPostTrial": cumulative_points + reward,
            "stimulusParams": {
                0: int(stimulus[0]),
                1: int(stimulus[1]),
                2: int(stimulus[2]),
                3: int(stimulus[3])
            },
            "response": response,
            "approachOutcome": approach_outcome
        }
        
        # Handle partner if social experiment
        if is_social:
            # Partner's action (based on the rule)
            if partner_rule == '2d':
                partner_action = 1 - env.is_dangerous(stimulus)  # Correct 2D rule
            else:  # 1D rule
                partner_action = 0 if stimulus[one_d_feature] == one_d_dangerous_value else 1
            
            partner_response = "approach" if partner_action == 1 else "avoid"
            trial_data["partnerResponse"] = partner_response
            
            # Update model if not test phase (social update)
            if not test:
                model.social_update(stimulus, action, reward, partner_action, intrinsic_social_reward)
        else:
            # Update model if not test phase (individual update)
            if not test:
                model.update(stimulus, action, reward)
        
        trials.append(trial_data)
        cumulative_points += reward
        
        if verbose and trial_idx % 16 == 15:  # Print every block
            block_num = trial_idx // 16 + 1
            print(f"Block {block_num} completed, Cumulative points: {cumulative_points}")
    
    if verbose:
        print(f"Total Reward: {cumulative_points}")
        print("Final Attention Weights:", model.alpha)
    
    return trials


# Backward compatibility functions
def run_individual_experiment(env: BeeEnvironment, model: ALCOVE_RL, num_trials: int = 64, 
                             verbose: bool = False, test: bool = False) -> List[Dict[str, Any]]:
    """Backward compatibility wrapper for run_experiment with no partner."""
    return run_experiment(env, model, num_trials, verbose, test, partner_rule=None)


def run_social_experiment(env: BeeEnvironment, model: ALCOVE_RL, partner_rule: str,
                         num_trials: int = 64, verbose: bool = False, test: bool = False,
                         intrinsic_social_reward: float = 0.5) -> List[Dict[str, Any]]:
    """Backward compatibility wrapper for run_experiment with partner."""
    return run_experiment(env, model, num_trials, verbose, test, partner_rule, intrinsic_social_reward)



