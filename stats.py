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


def bootstrap_simultaneous_proportion_cis(counts, confidence_level=0.95, n_bootstraps=10000):
    """
    Calculate Bonferroni-corrected bootstrap confidence intervals for multiple proportions 
    from multinomial data.
    
    Args:
        counts: array-like of category counts (e.g., [count_2d, count_1d, count_neither])
        confidence_level: overall confidence level (default 0.95)
        n_bootstraps: number of bootstrap iterations (default 10000)
        
    Returns:
        list of tuples: [(lower_ci, upper_ci), ...] for each category
    """
    counts = np.array(counts)
    n_total = counts.sum()
    n_categories = len(counts)
    
    if n_total == 0:
        return [(None, None)] * n_categories
    
    # Original proportions
    original_props = counts / n_total
    
    # Bonferroni correction: adjust alpha for multiple comparisons
    # Only count independent proportions (k-1 for k categories due to sum constraint)
    k_independent = n_categories - 1
    alpha_total = 1 - confidence_level
    alpha_individual = alpha_total / k_independent
    
    # Calculate percentile levels for individual CIs
    lower_percentile = (alpha_individual / 2) * 100
    upper_percentile = (1 - alpha_individual / 2) * 100
    
    # Bootstrap resample
    bootstrap_props = []
    for _ in range(n_bootstraps):
        # Multinomial resample maintaining the constraint that proportions sum to 1
        boot_counts = np.random.multinomial(n_total, original_props)
        boot_props = boot_counts / n_total
        bootstrap_props.append(boot_props)
    
    bootstrap_props = np.array(bootstrap_props)
    
    # Calculate simultaneous confidence intervals
    simultaneous_cis = []
    for i in range(n_categories):
        lower_ci = np.percentile(bootstrap_props[:, i], lower_percentile)
        upper_ci = np.percentile(bootstrap_props[:, i], upper_percentile)
        simultaneous_cis.append((lower_ci, upper_ci))
    
    return simultaneous_cis