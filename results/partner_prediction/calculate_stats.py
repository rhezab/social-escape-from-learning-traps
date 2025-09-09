import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append('../../')
from constants import PUBLICATION_DATASETS, DATASET_PHASES
from stats import permutation_test_proportion

def get_second_test_results(ids, normalize=True):
    """
    Plot learning outcomes for participants based on their partner prediction performance.

    Parameters:
    - ids: list of participant IDs
    - normalize: if True, normalize the results to proportions instead of counts
    """
    # Just use the given performer_ids, no additional filtering
    selected = players_df[players_df['id'].isin(ids)]

    # Ensure the order is always ['2d', '1d', 'neither']
    categories = ['2d', '1d', 'neither']
    second_test_results = selected['second_test_drule_gen'].value_counts(normalize=normalize)
    second_test_results = second_test_results.reindex(categories, fill_value=0.0)
    return second_test_results

def load_data():
    """Load the preprocessed data files."""
    players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
    blocks_df = pd.read_csv('../../preprocessing/outputs/blocks_all_filtered.csv')
    return players_df, blocks_df


players_df, blocks_df = load_data()
last_prediction_blocks_df = blocks_df[(blocks_df['dataset'] == '5') & (blocks_df['phase'] == 2) & (blocks_df['block'] == 3)]


PERFORMANCE_THRESHOLD = 15
high_performers_ids = last_prediction_blocks_df[last_prediction_blocks_df['partner_predictions_correct'] >= PERFORMANCE_THRESHOLD]['id'].unique().tolist()
low_performers_ids = last_prediction_blocks_df[last_prediction_blocks_df['partner_predictions_correct'] < PERFORMANCE_THRESHOLD]['id'].unique().tolist()
trapped_ids = players_df[players_df['first_test_drule_gen'] == '1d']['id'].unique().tolist()
optimal_partner_ids = players_df[players_df['partner_rule_rel'] == '2d']['id'].unique().tolist()
other_1d_partner_ids = players_df[players_df['partner_rule_rel'] == 'other-1d']['id'].unique().tolist()


trapped_high_performers_ids = set(trapped_ids) & set(high_performers_ids)
trapped_low_performers_ids = set(trapped_ids) & set(low_performers_ids)
informative_partner_ids = set(optimal_partner_ids) | set(other_1d_partner_ids)
trapped_high_performers_informative_partner_ids = set(trapped_high_performers_ids) & set(informative_partner_ids)
trapped_low_performers_informative_partner_ids = set(trapped_low_performers_ids) & set(informative_partner_ids)
trapped_high_performers_optimal_partner_ids = set(trapped_high_performers_ids) & set(optimal_partner_ids)
trapped_high_performers_other_1d_partner_ids = set(trapped_high_performers_ids) & set(other_1d_partner_ids)
trapped_low_performers_optimal_partner_ids = set(trapped_low_performers_ids) & set(optimal_partner_ids)
trapped_low_performers_other_1d_partner_ids = set(trapped_low_performers_ids) & set(other_1d_partner_ids)

# Organize all ID sets into a dictionary
id_sets = {
    'trapped_high_performers': trapped_high_performers_ids,
    'trapped_low_performers': trapped_low_performers_ids,
    'trapped_high_informative_partner': trapped_high_performers_informative_partner_ids,
    'trapped_low_informative_partner': trapped_low_performers_informative_partner_ids,
    'trapped_high_optimal_partner': trapped_high_performers_optimal_partner_ids,
    'trapped_high_other_1d_partner': trapped_high_performers_other_1d_partner_ids,
    'trapped_low_optimal_partner': trapped_low_performers_optimal_partner_ids,
    'trapped_low_other_1d_partner': trapped_low_performers_other_1d_partner_ids,
}

# Compute results for all sets
second_test_counts = {}
second_test_proportions = {}
sample_sizes = {}

for group_name, ids in id_sets.items():
    sample_sizes[group_name] = len(ids)
    second_test_counts[group_name] = get_second_test_results(ids, normalize=False)
    second_test_proportions[group_name] = get_second_test_results(ids, normalize=True)

# Restructure for paired high/low comparisons
condition_groups = {
    'all_trapped': {
        'high': trapped_high_performers_ids,
        'low': trapped_low_performers_ids
    },
    'informative_partner': {
        'high': trapped_high_performers_informative_partner_ids,
        'low': trapped_low_performers_informative_partner_ids
    },
    'optimal_partner': {
        'high': trapped_high_performers_optimal_partner_ids,
        'low': trapped_low_performers_optimal_partner_ids
    },
    'other_1d_partner': {
        'high': trapped_high_performers_other_1d_partner_ids,
        'low': trapped_low_performers_other_1d_partner_ids
    }
}

def test_learning_outcome_differences(high_ids, low_ids):
    """
    Test for significant differences in learning outcomes between high and low performers.
    
    Returns results for all three outcome categories: 2d, 1d, neither
    """
    # Get counts for each group
    high_counts = get_second_test_results(high_ids, normalize=False)
    low_counts = get_second_test_results(low_ids, normalize=False)
    
    high_total = len(high_ids)
    low_total = len(low_ids)
    
    results = {
        'sample_sizes': {'high': high_total, 'low': low_total}
    }
    
    # Test each outcome category
    for outcome in ['2d', '1d', 'neither']:
        high_successes = round(high_counts[outcome])
        low_successes = round(low_counts[outcome])
        
        high_rate = high_successes / high_total if high_total > 0 else 0
        low_rate = low_successes / low_total if low_total > 0 else 0
        difference = high_rate - low_rate
        
        # Perform permutation test
        if high_total > 0 and low_total > 0:
            p_value = permutation_test_proportion(
                high_successes, high_total,
                low_successes, low_total,
                n_permutations=10000
            )
        else:
            p_value = None
        
        results[outcome] = {
            'high_successes': high_successes,
            'low_successes': low_successes,
            'high_total': high_total,
            'low_total': low_total,
            'total_successes': high_successes + low_successes,
            'total_total': high_total + low_total,
            'high_rate': high_rate,
            'low_rate': low_rate,
            'difference': difference,
            'p_value': p_value
        }
    
    return results

# Conduct significance tests for all conditions
significance_results = {}
for condition_name, groups in condition_groups.items():
    significance_results[condition_name] = test_learning_outcome_differences(
        groups['high'], groups['low']
    )

# Generate markdown summary
from datetime import datetime
import os

def format_p_value(p_value):
    """Format p-value with significance stars."""
    if p_value is None:
        return "N/A"
    
    stars = ""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    
    return f"{p_value:.4f}{stars}"

def format_rate(rate):
    """Format rate as percentage."""
    return f"{rate:.1%}"

def format_difference(diff):
    """Format difference with sign."""
    return f"{diff:+.1%}"

# Create markdown summary
os.makedirs('outputs', exist_ok=True)
summary_file = 'outputs/partner_prediction_significance_results.md'

with open(summary_file, 'w') as f:
    f.write("# Partner Prediction Performance and Learning Outcomes: Statistical Analysis\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## Overview\n\n")
    f.write("This analysis examines whether partner rule inference success predicts subsequent learning outcomes ")
    f.write("for trapped learners in **dataset 5** (participants who used 1D rules in the first test phase). ")
    f.write("Dataset 5 is the only dataset with a partner prediction phase, allowing analysis of the relationship ")
    f.write("between partner inference ability and learning outcomes. ")
    f.write(f"Partner prediction performance was categorized using a threshold of ≥{PERFORMANCE_THRESHOLD} correct predictions ")
    f.write("in the final partner prediction block (phase 2, block 3).\n\n")
    
    # Add total sample sizes (dataset 5 only)
    total_high_performers = len(high_performers_ids)
    total_low_performers = len(low_performers_ids) 
    dataset_5_players = players_df[players_df['dataset'] == '5']
    total_dataset_5 = len(dataset_5_players)
    total_trapped_d5 = len(dataset_5_players[dataset_5_players['first_test_drule_gen'] == '1d'])
    
    f.write("### Sample Sizes (Dataset 5 Only)\n\n")
    f.write(f"- **Total participants in dataset 5**: {total_dataset_5}\n")
    f.write(f"- **Trapped learners in dataset 5 (1D first test)**: {total_trapped_d5}\n")
    f.write(f"- **High prediction performers (≥{PERFORMANCE_THRESHOLD} correct)**: {total_high_performers}\n")
    f.write(f"- **Low prediction performers (<{PERFORMANCE_THRESHOLD} correct)**: {total_low_performers}\n")
    f.write(f"- **Trapped high performers**: {len(trapped_high_performers_ids)}\n")
    f.write(f"- **Trapped low performers**: {len(trapped_low_performers_ids)}\n\n")
    
    f.write("## Methods\n\n")
    f.write("- **High performers**: ≥15 correct partner predictions in final block\n")
    f.write("- **Low performers**: <15 correct partner predictions in final block\n")
    f.write("- **Statistical test**: Permutation tests (10,000 iterations)\n")
    f.write("- **Significance levels**: * p<0.05, ** p<0.01, *** p<0.001\n\n")
    
    f.write("## Results by Condition\n\n")
    
    condition_labels = {
        'all_trapped': 'All Trapped Learners',
        'informative_partner': 'Informative Partners (2D + Other-1D)',
        'optimal_partner': 'Optimal Partners (2D)',
        'other_1d_partner': 'Other-1D Partners'
    }
    
    for condition_name, results in significance_results.items():
        condition_label = condition_labels.get(condition_name, condition_name)
        f.write(f"### {condition_label}\n\n")
        
        # Sample sizes
        high_n = results['sample_sizes']['high']
        low_n = results['sample_sizes']['low']
        total_n = high_n + low_n
        f.write(f"**Sample sizes**: High performers N={high_n}, Low performers N={low_n}, Total N={total_n}\n\n")
        
        if high_n == 0 or low_n == 0:
            f.write("*Insufficient data for statistical comparison*\n\n")
            continue
        
        # Results table
        f.write("| Outcome | High Performers | Low Performers | Difference | p-value |\n")
        f.write("|---------|-----------------|----------------|------------|----------|\n")
        
        for outcome in ['2d', '1d', 'neither']:
            outcome_data = results[outcome]
            high_rate_str = format_rate(outcome_data['high_rate'])
            low_rate_str = format_rate(outcome_data['low_rate'])
            diff_str = format_difference(outcome_data['difference'])
            p_str = format_p_value(outcome_data['p_value'])
            
            f.write(f"| {outcome.upper()} Rule | {high_rate_str} ({outcome_data['high_successes']}/{outcome_data['high_total']}) | ")
            f.write(f"{low_rate_str} ({outcome_data['low_successes']}/{outcome_data['low_total']}) | ")
            f.write(f"{diff_str} | {p_str} |\n")
        
        f.write("\n")
    
    f.write("## Key Findings\n\n")
    
    # Identify significant results
    significant_findings = []
    for condition_name, results in significance_results.items():
        condition_label = condition_labels.get(condition_name, condition_name)
        
        if results['sample_sizes']['high'] == 0 or results['sample_sizes']['low'] == 0:
            continue
            
        for outcome in ['2d', '1d', 'neither']:
            p_value = results[outcome]['p_value']
            if p_value is not None and p_value < 0.05:
                difference = results[outcome]['difference']
                high_rate = results[outcome]['high_rate']
                low_rate = results[outcome]['low_rate']
                
                direction = "higher" if difference > 0 else "lower"
                significant_findings.append(
                    f"- **{condition_label} - {outcome.upper()} Rule**: High performers had {direction} rates "
                    f"({format_rate(high_rate)} vs {format_rate(low_rate)}, p={p_value:.4f})"
                )
    
    if significant_findings:
        f.write("### Statistically Significant Differences (p < 0.05)\n\n")
        for finding in significant_findings:
            f.write(f"{finding}\n")
    else:
        f.write("### No statistically significant differences found at p < 0.05 level.\n")
    
    f.write(f"\n## Interpretation\n\n")
    f.write("These results ")
    if significant_findings:
        f.write("provide evidence that partner rule inference ability predicts subsequent learning outcomes. ")
        f.write("Participants who were better at predicting their partner's choices showed different patterns ")
        f.write("of learning in the second test phase.")
    else:
        f.write("suggest that partner rule inference ability may not strongly predict subsequent learning outcomes ")
        f.write("within this sample, or that the effect sizes are smaller than detectable with current sample sizes.")
    
    f.write(f"\n\n*Analysis conducted using permutation tests with 10,000 iterations each.*\n")

print(f"Statistical analysis summary saved to: {summary_file}")

# Save analysis results in easy-to-load formats
import json

# Save detailed results as JSON
results_json_file = 'outputs/partner_prediction_results.json'
with open(results_json_file, 'w') as f:
    # Convert sets to lists for JSON serialization
    json_data = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'performance_threshold': PERFORMANCE_THRESHOLD,
            'total_dataset_5': len(dataset_5_players),
            'total_trapped_d5': total_trapped_d5,
            'total_high_performers': total_high_performers,
            'total_low_performers': total_low_performers
        },
        'condition_groups': {k: {'high': list(v['high']), 'low': list(v['low'])} 
                           for k, v in condition_groups.items()},
        'significance_results': significance_results
    }
    json.dump(json_data, f, indent=2)

# Save results as CSV for easy visualization
results_csv_file = 'outputs/partner_prediction_results.csv'
csv_data = []

for condition_name, results in significance_results.items():
    high_n = results['sample_sizes']['high']
    low_n = results['sample_sizes']['low']
    total_n = high_n + low_n
    
    for outcome in ['2d', '1d', 'neither']:
        outcome_data = results[outcome]
        csv_data.append({
            'condition': condition_name,
            'outcome': outcome,
            'high_n': high_n,
            'low_n': low_n,
            'total_n': total_n,
            'high_successes': outcome_data['high_successes'],
            'low_successes': outcome_data['low_successes'],
            'high_rate': outcome_data['high_rate'],
            'low_rate': outcome_data['low_rate'],
            'difference': outcome_data['difference'],
            'p_value': outcome_data['p_value'],
            'significant': outcome_data['p_value'] < 0.05 if outcome_data['p_value'] is not None else False
        })

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv(results_csv_file, index=False)

print(f"Analysis results saved to:")
print(f"  - JSON format: {results_json_file}")
print(f"  - CSV format: {results_csv_file}")











