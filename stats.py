import numpy as np
import pandas as pd
from typing import Tuple


def permutation_test_proportion(group1_successes, group1_total, group2_successes, group2_total, n_permutations=10000):
    """
    Direct permutation test comparing two proportions.
    
    Args:
        group1_successes: Number of successes in group 1
        group1_total: Total number in group 1
        group2_successes: Number of successes in group 2  
        group2_total: Total number in group 2
        n_permutations: Number of permutation iterations
        
    Returns:
        p_value: Two-tailed p-value
    """
    # Calculate observed difference in proportions
    prop1 = group1_successes / group1_total if group1_total > 0 else 0
    prop2 = group2_successes / group2_total if group2_total > 0 else 0
    observed_diff = abs(prop1 - prop2)
    
    # Create combined pool
    combined_successes = group1_successes + group2_successes
    combined_total = group1_total + group2_total
    
    if combined_total == 0:
        return None
    
    # Permutation test
    extreme_count = 0
    
    for _ in range(n_permutations):
        # Randomly assign successes to the two groups
        perm_group1_successes = np.random.hypergeometric(combined_successes, 
                                                         combined_total - combined_successes, 
                                                         group1_total)
        perm_group2_successes = combined_successes - perm_group1_successes
        
        # Calculate permuted difference
        perm_prop1 = perm_group1_successes / group1_total if group1_total > 0 else 0
        perm_prop2 = perm_group2_successes / group2_total if group2_total > 0 else 0
        perm_diff = abs(perm_prop1 - perm_prop2)
        
        # Count if permuted difference is >= observed difference
        if perm_diff >= observed_diff:
            extreme_count += 1
    
    p_value = extreme_count / n_permutations
    return p_value


def bootstrap_proportion_ci(successes, total, confidence_level=0.95, n_bootstraps=10000):
    """
    Calculate bootstrap confidence interval for a proportion.
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstraps: Number of bootstrap iterations
        
    Returns:
        tuple: (lower_ci, upper_ci) or (None, None) if no data
    """
    if total == 0:
        return None, None
    
    # Create original sample array
    sample_array = np.array([1] * successes + [0] * (total - successes))
    
    # Bootstrap resample and calculate proportions
    bootstrap_proportions = []
    
    for _ in range(n_bootstraps):
        boot_sample = np.random.choice(sample_array, size=total, replace=True)
        bootstrap_proportions.append(boot_sample.mean())
    
    # Calculate percentile-based confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_proportions, lower_percentile)
    upper_ci = np.percentile(bootstrap_proportions, upper_percentile)
    
    return lower_ci, upper_ci