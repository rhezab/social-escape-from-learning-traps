"""
This script generates statistics for the test drules for each dataset.

Namely, for each dataset:
- It counts how many are following each drule_gen at the first test phase (after indiivdual learning). It computes this both
in absolute terms (as counts) and also as proportions. 
- Next, it selects trapped learners (those displaying 1d rule in the first test). It counts how many are following each 
drule_gen at the second test phase, broken down by the following conditions:
    - Solo (asocial).
    - Duo (social). For each of the following, include a two-sided permutation significance test compared to the counts
    from the solo (asocial) condition.
        - partner_rule_rel 2d.
        - partner_rule_rel is other-1d.
        - partner_rule_rel is same-1d. 


This script saves the results to a dataframe and saves it to outputs.
For significance tests, look in the file `stats.py` at root. You can use `bootstrap_series_test`, `permutation_series_test`,
`bootstrap_difference_test`, and `permutation_difference_test`. You can also write your own function if you think you 
can write it better. 
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import statistical functions from root stats.py
sys.path.append('../../')
from stats import permutation_test_proportion, bootstrap_proportion_ci

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import ALL_DATASETS

# Define which datasets to process - can be easily modified in the future
DATASETS_TO_PROCESS = ALL_DATASETS




def calculate_first_test_stats(players_df, confidence_level=0.95):
    """Calculate decision rule statistics for the first test phase by dataset."""
    first_test_results = []
    
    for dataset in DATASETS_TO_PROCESS:
        dataset_df = players_df[players_df['dataset'] == dataset]
        
        # Count decision rules
        drule_counts = dataset_df['first_test_drule_gen'].value_counts()
        total_participants = len(dataset_df)
        
        # Calculate confidence intervals for each rule
        ci_lower_2d, ci_upper_2d = bootstrap_proportion_ci(drule_counts.get('2d', 0), total_participants, confidence_level)
        ci_lower_1d, ci_upper_1d = bootstrap_proportion_ci(drule_counts.get('1d', 0), total_participants, confidence_level)
        ci_lower_neither, ci_upper_neither = bootstrap_proportion_ci(drule_counts.get('neither', 0), total_participants, confidence_level)
        
        # Add results for each rule type
        for rule in ['2d', '1d', 'neither']:
            count = drule_counts.get(rule, 0)
            proportion = count / total_participants if total_participants > 0 else 0
            
            # Get appropriate CI for this rule
            if rule == '2d':
                ci_lower, ci_upper = ci_lower_2d, ci_upper_2d
            elif rule == '1d':
                ci_lower, ci_upper = ci_lower_1d, ci_upper_1d
            else:  # 'neither'
                ci_lower, ci_upper = ci_lower_neither, ci_upper_neither
            
            first_test_results.append({
                'dataset': dataset,
                'test_phase': 'first',
                'condition': 'all',
                'rule': rule,
                'count': count,
                'proportion': proportion,
                'total_n': total_participants,
                'ci_lower_2d': ci_lower_2d,
                'ci_upper_2d': ci_upper_2d,
                'ci_lower_1d': ci_lower_1d,
                'ci_upper_1d': ci_upper_1d,
                'ci_lower_neither': ci_lower_neither,
                'ci_upper_neither': ci_upper_neither
            })
    
    return first_test_results


def calculate_second_test_stats(players_df, n_randomizations=10000, confidence_level=0.95):
    """Calculate decision rule statistics for trapped learners in second test phase."""
    second_test_results = []
    
    for dataset in DATASETS_TO_PROCESS:
        dataset_df = players_df[players_df['dataset'] == dataset]
        
        # Get trapped learners only (those with 1d rule in first test)
        trapped_learners = dataset_df[dataset_df['first_test_drule_gen'] == '1d']
        
        if len(trapped_learners) == 0:
            continue
            
        # Get solo condition (control group) - renamed to 'asocial'
        solo_trapped = trapped_learners[trapped_learners['game_type'] == 'solo']
        solo_counts = solo_trapped['second_test_drule_gen'].value_counts()
        solo_total = len(solo_trapped)
        
        # Calculate confidence intervals for asocial condition
        ci_lower_2d, ci_upper_2d = bootstrap_proportion_ci(solo_counts.get('2d', 0), solo_total, confidence_level)
        ci_lower_1d, ci_upper_1d = bootstrap_proportion_ci(solo_counts.get('1d', 0), solo_total, confidence_level)
        ci_lower_neither, ci_upper_neither = bootstrap_proportion_ci(solo_counts.get('neither', 0), solo_total, confidence_level)
        
        # Store solo results
        for rule in ['2d', '1d', 'neither']:
            count = solo_counts.get(rule, 0)
            proportion = count / solo_total if solo_total > 0 else 0
            
            # Get appropriate CI for this rule
            if rule == '2d':
                ci_lower, ci_upper = ci_lower_2d, ci_upper_2d
            elif rule == '1d':
                ci_lower, ci_upper = ci_lower_1d, ci_upper_1d
            else:  # 'neither'
                ci_lower, ci_upper = ci_lower_neither, ci_upper_neither
            
            second_test_results.append({
                'dataset': dataset,
                'condition': 'asocial',
                'rule': rule,
                'count': count,
                'proportion': proportion,
                'total_n': solo_total,
                'p_perm_2d': None,  # No test vs self
                'p_perm_1d': None,
                'p_perm_neither': None,
                'ci_lower_2d': ci_lower_2d,
                'ci_upper_2d': ci_upper_2d,
                'ci_lower_1d': ci_lower_1d,
                'ci_upper_1d': ci_upper_1d,
                'ci_lower_neither': ci_lower_neither,
                'ci_upper_neither': ci_upper_neither
            })
        
        # Process each partner rule condition (only 2d and other-1d as specified)
        partner_conditions = ['2d', 'other-1d']
        
        for partner_condition in partner_conditions:
            # Get duo participants with specific partner rule
            duo_trapped = trapped_learners[
                (trapped_learners['game_type'] == 'duo') & 
                (trapped_learners['partner_rule_rel'] == partner_condition)
            ]
            
            if len(duo_trapped) == 0:
                # Still add zeros for completeness
                for rule in ['2d', '1d', 'neither']:
                    second_test_results.append({
                        'dataset': dataset,
                        'condition': partner_condition,
                        'rule': rule,
                        'count': 0,
                        'proportion': 0,
                        'total_n': 0,
                        'p_perm_2d': None,
                        'p_perm_1d': None,
                        'p_perm_neither': None,
                        'ci_lower_2d': None,
                        'ci_upper_2d': None,
                        'ci_lower_1d': None,
                        'ci_upper_1d': None,
                        'ci_lower_neither': None,
                        'ci_upper_neither': None
                    })
                continue
            
            duo_counts = duo_trapped['second_test_drule_gen'].value_counts()
            duo_total = len(duo_trapped)
            
            # Perform permutation tests and calculate confidence intervals
            if solo_total > 0 and duo_total > 0:
                try:
                    # Permutation tests for all rules
                    p_perm_2d = permutation_test_proportion(
                        group1_successes=duo_counts.get('2d', 0),
                        group1_total=duo_total,
                        group2_successes=solo_counts.get('2d', 0),
                        group2_total=solo_total,
                        n_permutations=n_randomizations
                    )
                    
                    p_perm_1d = permutation_test_proportion(
                        group1_successes=duo_counts.get('1d', 0),
                        group1_total=duo_total,
                        group2_successes=solo_counts.get('1d', 0),
                        group2_total=solo_total,
                        n_permutations=n_randomizations
                    )
                    
                    p_perm_neither = permutation_test_proportion(
                        group1_successes=duo_counts.get('neither', 0),
                        group1_total=duo_total,
                        group2_successes=solo_counts.get('neither', 0),
                        group2_total=solo_total,
                        n_permutations=n_randomizations
                    )
                    
                    # Bootstrap confidence intervals for duo condition
                    ci_lower_2d_duo, ci_upper_2d_duo = bootstrap_proportion_ci(duo_counts.get('2d', 0), duo_total, confidence_level)
                    ci_lower_1d_duo, ci_upper_1d_duo = bootstrap_proportion_ci(duo_counts.get('1d', 0), duo_total, confidence_level)
                    ci_lower_neither_duo, ci_upper_neither_duo = bootstrap_proportion_ci(duo_counts.get('neither', 0), duo_total, confidence_level)
                        
                except Exception as e:
                    print(f"Statistical tests failed for dataset {dataset}, condition {partner_condition}: {e}")
                    p_perm_2d = p_perm_1d = p_perm_neither = None
                    ci_lower_2d_duo = ci_upper_2d_duo = ci_lower_1d_duo = ci_upper_1d_duo = ci_lower_neither_duo = ci_upper_neither_duo = None
            else:
                p_perm_2d = p_perm_1d = p_perm_neither = None
                ci_lower_2d_duo = ci_upper_2d_duo = ci_lower_1d_duo = ci_upper_1d_duo = ci_lower_neither_duo = ci_upper_neither_duo = None
            
            # Store duo results with p-values
            for rule in ['2d', '1d', 'neither']:
                count = duo_counts.get(rule, 0)
                proportion = count / duo_total if duo_total > 0 else 0
                
                second_test_results.append({
                    'dataset': dataset,
                    'condition': partner_condition,
                    'rule': rule,
                    'count': count,
                    'proportion': proportion,
                    'total_n': duo_total,
                    'p_perm_2d': p_perm_2d,
                    'p_perm_1d': p_perm_1d,
                    'p_perm_neither': p_perm_neither,
                    'ci_lower_2d': ci_lower_2d_duo,
                    'ci_upper_2d': ci_upper_2d_duo,
                    'ci_lower_1d': ci_lower_1d_duo,
                    'ci_upper_1d': ci_upper_1d_duo,
                    'ci_lower_neither': ci_lower_neither_duo,
                    'ci_upper_neither': ci_upper_neither_duo
                })
    
    return second_test_results


def main(confidence_level=0.95):
    """Main function to calculate test decision rule statistics."""
    print("=" * 60)
    print("TEST DECISION RULE STATISTICS ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Computing permutation p-values and {confidence_level:.0%} bootstrap confidence intervals")
    print()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    try:
        players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
        print(f"Loaded {len(players_df)} participants")
        print(f"Datasets: {sorted(players_df['dataset'].unique())}")
    except FileNotFoundError:
        print("Error: Could not find preprocessed data file")
        print("Please run the preprocessing pipeline first")
        return
    
    # Calculate first test statistics
    print("\nCalculating first test phase statistics...")
    first_test_results = calculate_first_test_stats(players_df, confidence_level)
    print(f"Generated {len(first_test_results)} first test statistics")
    
    # Calculate second test statistics with permutation tests and bootstrap CIs
    print(f"\nCalculating second test phase statistics for trapped learners...")
    print(f"(This may take a moment due to running permutation tests and bootstrap CIs...)")
    second_test_results = calculate_second_test_stats(players_df, n_randomizations=10000, confidence_level=confidence_level)
    print(f"Generated {len(second_test_results)} second test statistics")
    
    # Create separate dataframes
    first_test_df = pd.DataFrame(first_test_results)
    second_test_df = pd.DataFrame(second_test_results)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Save separate dataframes
    first_test_file = 'outputs/first_test_drule_statistics.csv'
    second_test_file = 'outputs/second_test_drule_statistics.csv'
    
    first_test_df.to_csv(first_test_file, index=False)
    second_test_df.to_csv(second_test_file, index=False)
    
    print(f"\nSaved first test results to {first_test_file}")
    print(f"Saved second test results to {second_test_file}")
    
    # Create documentation
    doc_file = 'outputs/test_drule_statistics_info.md'
    with open(doc_file, 'w') as f:
        f.write("# Test Decision Rule Statistics Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Description\n\n")
        f.write("This analysis examines decision rule usage during test phases of the ")
        f.write("dyadic learning trap experiment. Two separate analyses are conducted:\n\n")
        f.write("1. **First Test Phase:** Decision rule counts and proportions for all participants\n")
        f.write("2. **Second Test Phase:** Trapped learner analysis comparing social conditions ")
        f.write("to asocial control using two-sided permutation significance tests\n\n")
        
        f.write("## Output Files\n\n")
        f.write("- `first_test_drule_statistics.csv`: First test phase statistics for all participants\n")
        f.write("- `second_test_drule_statistics.csv`: Second test phase statistics for trapped learners\n\n")
        
        f.write("## First Test Phase Results\n\n")
        f.write("**All participants by dataset and decision rule:**\n\n")
        
        # Create summary table for first test
        ci_label = f"{confidence_level:.0%} CI"
        f.write(f"| Dataset | Total | 2D % | 1D % | Neither % | 2D Count | 1D Count | Neither Count | {ci_label} (2D) | {ci_label} (1D) | {ci_label} (Neither) |\n")
        f.write("|---------|-------|------|------|-----------|----------|----------|---------------|-------------|-------------|---------------|\n")
        
        for dataset in ALL_DATASETS: # sorted(first_test_df['dataset'].unique()):
            dataset_data = first_test_df[first_test_df['dataset'] == dataset]
            
            count_2d = dataset_data[dataset_data['rule'] == '2d']['count'].iloc[0]
            prop_2d = dataset_data[dataset_data['rule'] == '2d']['proportion'].iloc[0]
            count_1d = dataset_data[dataset_data['rule'] == '1d']['count'].iloc[0]
            prop_1d = dataset_data[dataset_data['rule'] == '1d']['proportion'].iloc[0]
            count_neither = dataset_data[dataset_data['rule'] == 'neither']['count'].iloc[0]
            prop_neither = dataset_data[dataset_data['rule'] == 'neither']['proportion'].iloc[0]
            total = dataset_data['total_n'].iloc[0]
            
            # Get confidence intervals for first test
            ci_lower_2d = dataset_data[dataset_data['rule'] == '2d']['ci_lower_2d'].iloc[0]
            ci_upper_2d = dataset_data[dataset_data['rule'] == '2d']['ci_upper_2d'].iloc[0]
            ci_lower_1d = dataset_data[dataset_data['rule'] == '1d']['ci_lower_1d'].iloc[0]
            ci_upper_1d = dataset_data[dataset_data['rule'] == '1d']['ci_upper_1d'].iloc[0]
            ci_lower_neither = dataset_data[dataset_data['rule'] == 'neither']['ci_lower_neither'].iloc[0]
            ci_upper_neither = dataset_data[dataset_data['rule'] == 'neither']['ci_upper_neither'].iloc[0]
            
            # Format confidence intervals
            ci_str_2d = f"[{ci_lower_2d:.3f}, {ci_upper_2d:.3f}]" if ci_lower_2d is not None else "N/A"
            ci_str_1d = f"[{ci_lower_1d:.3f}, {ci_upper_1d:.3f}]" if ci_lower_1d is not None else "N/A"
            ci_str_neither = f"[{ci_lower_neither:.3f}, {ci_upper_neither:.3f}]" if ci_lower_neither is not None else "N/A"
            
            f.write(f"| {dataset} | {total} | {prop_2d:.1%} | {prop_1d:.1%} | {prop_neither:.1%} | {count_2d} | {count_1d} | {count_neither} | {ci_str_2d} | {ci_str_1d} | {ci_str_neither} |\n")
        
        f.write("\n## Second Test Phase Results (Trapped Learners Only)\n\n")
        f.write("**Trapped learners by condition and decision rule:**\n\n")
        
        # Create summary table for second test
        ci_label = f"{confidence_level:.0%} CI"
        f.write(f"| Dataset | Condition | Total | 2D % | 1D % | Neither % | 2D Count | 1D Count | Neither Count | {ci_label} (2D) | {ci_label} (1D) | {ci_label} (Neither) | p-perm (2D) | p-perm (1D) | p-perm (Neither) |\n")
        f.write("|---------|-----------|-------|------|------|-----------|----------|----------|---------------|-------------|-------------|------------------|-------------|-------------|------------------|\n")
        
        for dataset in ALL_DATASETS: # sorted(second_test_df['dataset'].unique()):
            for condition in ['asocial', '2d', 'other-1d']:
                condition_data = second_test_df[
                    (second_test_df['dataset'] == dataset) & 
                    (second_test_df['condition'] == condition)
                ]
                
                if len(condition_data) == 0:
                    continue
                    
                count_2d = condition_data[condition_data['rule'] == '2d']['count'].iloc[0]
                prop_2d = condition_data[condition_data['rule'] == '2d']['proportion'].iloc[0]
                count_1d = condition_data[condition_data['rule'] == '1d']['count'].iloc[0]
                prop_1d = condition_data[condition_data['rule'] == '1d']['proportion'].iloc[0]
                count_neither = condition_data[condition_data['rule'] == 'neither']['count'].iloc[0]
                prop_neither = condition_data[condition_data['rule'] == 'neither']['proportion'].iloc[0]
                total = condition_data['total_n'].iloc[0]
                
                # Get confidence intervals
                ci_lower_2d = condition_data[condition_data['rule'] == '2d']['ci_lower_2d'].iloc[0]
                ci_upper_2d = condition_data[condition_data['rule'] == '2d']['ci_upper_2d'].iloc[0]
                ci_lower_1d = condition_data[condition_data['rule'] == '1d']['ci_lower_1d'].iloc[0]
                ci_upper_1d = condition_data[condition_data['rule'] == '1d']['ci_upper_1d'].iloc[0]
                ci_lower_neither = condition_data[condition_data['rule'] == 'neither']['ci_lower_neither'].iloc[0]
                ci_upper_neither = condition_data[condition_data['rule'] == 'neither']['ci_upper_neither'].iloc[0]
                
                # Format confidence intervals
                ci_str_2d = f"[{ci_lower_2d:.3f}, {ci_upper_2d:.3f}]" if ci_lower_2d is not None else "N/A"
                ci_str_1d = f"[{ci_lower_1d:.3f}, {ci_upper_1d:.3f}]" if ci_lower_1d is not None else "N/A"
                ci_str_neither = f"[{ci_lower_neither:.3f}, {ci_upper_neither:.3f}]" if ci_lower_neither is not None else "N/A"
                
                # Get p-values (only for non-asocial conditions)
                if condition != 'asocial':
                    p_perm_2d = condition_data[condition_data['rule'] == '2d']['p_perm_2d'].iloc[0]
                    p_perm_1d = condition_data[condition_data['rule'] == '1d']['p_perm_1d'].iloc[0]
                    p_perm_neither = condition_data[condition_data['rule'] == 'neither']['p_perm_neither'].iloc[0]
                    
                    p_str_perm_2d = f"{p_perm_2d:.3f}" if p_perm_2d is not None else "N/A"
                    p_str_perm_1d = f"{p_perm_1d:.3f}" if p_perm_1d is not None else "N/A"
                    p_str_perm_neither = f"{p_perm_neither:.3f}" if p_perm_neither is not None else "N/A"
                else:
                    p_str_perm_2d = p_str_perm_1d = p_str_perm_neither = "-"
                
                f.write(f"| {dataset} | {condition} | {total} | {prop_2d:.1%} | {prop_1d:.1%} | {prop_neither:.1%} | {count_2d} | {count_1d} | {count_neither} | {ci_str_2d} | {ci_str_1d} | {ci_str_neither} | {p_str_perm_2d} | {p_str_perm_1d} | {p_str_perm_neither} |\n")
        
        f.write("\n## Methodology\n\n")
        f.write("### Trapped Learners\n")
        f.write("Participants showing 1d rule in first test phase. ")
        f.write("Second test analysis limited to these participants only.\n\n")
        
        f.write("### Conditions (Second Test)\n")
        f.write("- **asocial**: Trapped learners in solo games (control)\n")
        f.write("- **2d**: Trapped learners paired with 2d partner\n")
        f.write("- **other-1d**: Trapped learners paired with different 1d partner\n\n")
        
        f.write("### Statistical Analysis\n")
        f.write("Two complementary statistical approaches (10,000 iterations each):\n\n")
        f.write("- **Permutation tests**: Compare social conditions to asocial control by pooling participants and randomly reassigning to conditions\n")
        f.write(f"- **Bootstrap confidence intervals**: Estimate uncertainty in proportion estimates using {confidence_level:.0%} percentile-based CIs\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **First test rows:** {len(first_test_df)}\n")
        f.write(f"- **Second test rows:** {len(second_test_df)}\n")
        f.write(f"- **Datasets analyzed:** {sorted(players_df['dataset'].unique())}\n\n")
        
        # Count trapped learners by dataset
        f.write("### Trapped Learners by Dataset\n\n")
        f.write("| Dataset | Trapped Learners | Total | Percentage |\n")
        f.write("|---------|------------------|-------|------------|\n")
        for dataset in DATASETS_TO_PROCESS:
            trapped_count = len(players_df[
                (players_df['dataset'] == dataset) & 
                (players_df['first_test_drule_gen'] == '1d')
            ])
            total_count = len(players_df[players_df['dataset'] == dataset])
            f.write(f"| {dataset} | {trapped_count} | {total_count} | {trapped_count/total_count:.1%} |\n")
    
    print(f"Saved documentation to {doc_file}")
    
    # Display summary
    print(f"\nAnalysis complete!")
    print(f"First test statistics: {len(first_test_df)} rows")
    print(f"Second test statistics: {len(second_test_df)} rows")
    print(f"Results saved to outputs/ directory")
    
    return first_test_df, second_test_df


if __name__ == "__main__":
    # You can change confidence_level (e.g., 0.90 for 90% CI, 0.99 for 99% CI)
    first_test_df, second_test_df = main(confidence_level=0.95)