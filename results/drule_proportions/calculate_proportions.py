import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import ALL_DATASETS, DATASET_PHASES
from stats import bootstrap_simultaneous_proportion_cis

def drule_proportions_per_dataset(df_all, blocks_all, condition, dataset, calculate_cis=True):
    # Get phases using DATASET_PHASES dictionary
    if condition == 'first':
        # First learning and test phases
        phases = [DATASET_PHASES[dataset]['learning'][0], DATASET_PHASES[dataset]['test'][0]]
    else:
        # Second learning and test phases
        phases = [DATASET_PHASES[dataset]['learning'][1], DATASET_PHASES[dataset]['test'][1]]

    if condition == 'first':
        cond_mask = pd.Series(True, index=df_all.index)
    elif condition == 'asocial':
        cond_mask = ((df_all['game_type'] == 'solo') & 
            (df_all['first_test_drule_gen'] == '1d'))
    elif condition == '2d':
        cond_mask = ((df_all['game_type'] == 'duo') & 
            (df_all['first_test_drule_gen'] == '1d') & 
            (df_all['partner_rule'] == '2d'))
    elif condition == 'other_1d':
        cond_mask = ((df_all['game_type'] == 'duo') & 
            ((df_all['first_test_drule'] == '1d_a') & (df_all['partner_rule'] == '1d_b') |
            (df_all['first_test_drule'] == '1d_b') & (df_all['partner_rule'] == '1d_a')))
    else:
        raise ValueError(f"Invalid condition: {condition}. Must be one of 'first', 'asocial', '2d', or 'other_1d'.")

    # Get IDs
    ids = df_all[((df_all['dataset'] == dataset) & cond_mask)]['id'].tolist()
    
    # Initialize arrays for proportions
    n_blocks_total = sum([4, 2])
    prop_2d = np.zeros(n_blocks_total)
    prop_neither = np.zeros(n_blocks_total)
    prop_1d = np.zeros(n_blocks_total)
    
    # Initialize arrays for confidence intervals if requested
    if calculate_cis:
        ci_2d_lower = np.zeros(n_blocks_total)
        ci_2d_upper = np.zeros(n_blocks_total)
        ci_1d_lower = np.zeros(n_blocks_total)
        ci_1d_upper = np.zeros(n_blocks_total)
        ci_neither_lower = np.zeros(n_blocks_total)
        ci_neither_upper = np.zeros(n_blocks_total)
    
    # Calculate proportions for each block
    for phase_idx, phase in enumerate(phases):
        offset = phase_idx * 4
        for block in range([4, 2][phase_idx]):
            curr_blocks = blocks_all[
                (blocks_all['phase'] == phase) & 
                (blocks_all['block'] == block) &
                (blocks_all['id'].isin(ids))
            ]
            
            # Calculate proportions
            drule_counts = curr_blocks['drule_gen'].value_counts()
            total = len(curr_blocks)
            if total > 0:
                i = block + offset
                count_1d = drule_counts.get('1d', 0)
                count_neither = drule_counts.get('neither', 0)
                count_2d = drule_counts.get('2d', 0)
                
                prop_1d[i] = count_1d / total
                prop_neither[i] = count_neither / total
                prop_2d[i] = count_2d / total
                
                # Calculate confidence intervals if requested
                if calculate_cis:
                    counts = [count_2d, count_1d, count_neither]
                    cis = bootstrap_simultaneous_proportion_cis(
                        counts=counts,
                        confidence_level=0.95,
                        n_bootstraps=10000
                    )
                    
                    # Store CI bounds (cis = [(ci_2d), (ci_1d), (ci_neither)])
                    ci_2d_lower[i] = cis[0][0] if cis[0][0] is not None else 0
                    ci_2d_upper[i] = cis[0][1] if cis[0][1] is not None else 0
                    ci_1d_lower[i] = cis[1][0] if cis[1][0] is not None else 0
                    ci_1d_upper[i] = cis[1][1] if cis[1][1] is not None else 0
                    ci_neither_lower[i] = cis[2][0] if cis[2][0] is not None else 0
                    ci_neither_upper[i] = cis[2][1] if cis[2][1] is not None else 0

    N = len(ids)
    
    if calculate_cis:
        return prop_2d, prop_neither, prop_1d, N, ci_2d_lower, ci_2d_upper, ci_1d_lower, ci_1d_upper, ci_neither_lower, ci_neither_upper
    else:
        return prop_2d, prop_neither, prop_1d, N


def main():
    """
    Main function to calculate decision rule proportions across datasets and conditions.
    Loads preprocessed data and simulation results, calculates proportions, and saves merged results.
    """
    import os
    from datetime import datetime
    
    # Load filtered preprocessed data
    print("Loading preprocessed data...")
    players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
    blocks_df = pd.read_csv('../../preprocessing/outputs/blocks_all_filtered.csv')
    
    print(f"Loaded {len(players_df)} participants and {len(blocks_df)} blocks")
    print(f"Datasets: {sorted(players_df['dataset'].unique())}")
    print(f"Participants per dataset: \n{players_df['dataset'].value_counts().sort_index()}")
    
    # Initialize results dataframe with all columns including CIs
    results_df = pd.DataFrame(columns=[
        'dataset', 'condition', 'block', 'prop_2d', 'prop_neither', 'prop_1d', 'N',
        'ci_2d_lower', 'ci_2d_upper', 'ci_1d_lower', 'ci_1d_upper', 'ci_neither_lower', 'ci_neither_upper'
    ])
    
    print("\nCalculating proportions for each dataset and condition...")
    
    # For each dataset (using ordered processing - sim first, then numbered datasets)
    for dataset in ALL_DATASETS:
        print(f"  Processing dataset {dataset}...")
        # Get proportions for each condition and store in dataframe
        for condition in ['first', '2d', 'other_1d', 'asocial']:
            prop_2d, prop_neither, prop_1d, N, ci_2d_lower, ci_2d_upper, ci_1d_lower, ci_1d_upper, ci_neither_lower, ci_neither_upper = drule_proportions_per_dataset(
                df_all=players_df, 
                blocks_all=blocks_df, 
                condition=condition, 
                dataset=dataset, 
                calculate_cis=True
            )
            
            # Add rows for each block
            for block in range(len(prop_2d)):
                results_df.loc[len(results_df)] = {
                    'dataset': dataset,
                    'condition': condition,
                    'block': block,
                    'prop_2d': prop_2d[block],
                    'prop_neither': prop_neither[block],
                    'prop_1d': prop_1d[block],
                    'N': N,
                    'ci_2d_lower': ci_2d_lower[block],
                    'ci_2d_upper': ci_2d_upper[block],
                    'ci_1d_lower': ci_1d_lower[block],
                    'ci_1d_upper': ci_1d_upper[block],
                    'ci_neither_lower': ci_neither_lower[block],
                    'ci_neither_upper': ci_neither_upper[block]
                }
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Save merged results dataframe
    output_file = 'outputs/decision_rule_proportions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Create documentation file
    doc_file = 'outputs/decision_rule_proportions_info.txt'
    with open(doc_file, 'w') as f:
        f.write("DECISION RULE PROPORTIONS ANALYSIS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DESCRIPTION:\n")
        f.write("-" * 20 + "\n")
        f.write("This file contains proportions of decision rule usage (2d, 1d, neither) \n")
        f.write("across different experimental conditions and datasets for the dyadic \n")
        f.write("learning trap experiment, including simulated data.\n\n")
        
        f.write("DATA STRUCTURE:\n")
        f.write("-" * 20 + "\n")
        f.write("- dataset: Dataset identifier (sim, 1-6)\n")
        f.write("- condition: Experimental condition ('first', '2d', 'other_1d', 'asocial')\n")
        f.write("- block: Block number within learning phases\n")
        f.write("- prop_2d: Proportion using 2D decision rule (optimal)\n")
        f.write("- prop_1d: Proportion using 1D decision rule\n")
        f.write("- prop_neither: Proportion using neither rule consistently\n")
        f.write("- N: Number of participants in this condition/dataset\n")
        f.write("- ci_2d_lower/upper: 95% confidence interval bounds for 2D proportion\n")
        f.write("- ci_1d_lower/upper: 95% confidence interval bounds for 1D proportion\n")
        f.write("- ci_neither_lower/upper: 95% confidence interval bounds for neither proportion\n\n")
        
        f.write("CONDITIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("- first: First test phase (all participants)\n")
        f.write("- asocial: Solo game participants who used 1D rule in first test\n")
        f.write("- 2d: Duo game participants (1D in first test) paired with 2D partner\n")
        f.write("- other_1d: Duo game participants paired with complementary 1D partner\n\n")
        
        f.write("DATA SOURCES:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total data rows: {len(results_df)}\n\n")
        
        f.write("DATA SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Datasets analyzed: {sorted(results_df['dataset'].unique())}\n")
        f.write(f"Conditions: {sorted(results_df['condition'].unique())}\n")
        f.write(f"Blocks per condition: {results_df['block'].max() + 1}\n\n")
        
        f.write("PARTICIPANT COUNTS BY CONDITION:\n")
        f.write("-" * 20 + "\n")
        condition_counts = results_df.groupby(['dataset', 'condition'])['N'].first().unstack(fill_value=0)
        f.write(str(condition_counts))
        f.write("\n\n")
        
        f.write("CONFIDENCE INTERVALS SUMMARY (selected examples):\n")
        f.write("-" * 20 + "\n")
        f.write("Sample confidence intervals from sim dataset, 2d condition:\n")
        sample_data = results_df[(results_df['dataset'] == 'sim') & (results_df['condition'] == '2d')]
        if len(sample_data) > 0:
            for _, row in sample_data.head(4).iterrows():
                block = int(row['block'])
                prop_2d = row['prop_2d']
                ci_2d_lower = row['ci_2d_lower']
                ci_2d_upper = row['ci_2d_upper']
                prop_1d = row['prop_1d']
                ci_1d_lower = row['ci_1d_lower']
                ci_1d_upper = row['ci_1d_upper']
                n = int(row['N'])
                f.write(f"Block {block} (N={n}): 2D={prop_2d:.3f} [{ci_2d_lower:.3f}, {ci_2d_upper:.3f}], ")
                f.write(f"1D={prop_1d:.3f} [{ci_1d_lower:.3f}, {ci_1d_upper:.3f}]\n")
        f.write("\n")
        
        f.write("SOURCE DATA:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Players: {len(players_df)} participants (filtered, no external aid)\n")
        f.write(f"Blocks: {len(blocks_df)} total blocks\n")
        f.write("Input files: ../../preprocessing/outputs/players_df_all_filtered.csv,\n")
        f.write("             ../../preprocessing/outputs/blocks_all_filtered.csv\n")
    
    print(f"Saved documentation to {doc_file}")
    print(f"\nAnalysis complete! Generated {len(results_df)} rows of proportion data.")
    
    return results_df


if __name__ == "__main__":
    results_df = main()