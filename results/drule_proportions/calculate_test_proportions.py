import pandas as pd
import os
import sys
from datetime import datetime

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import ALL_DATASETS
from stats import permutation_test_proportion

def test_drule_proportions_per_dataset(df_all, condition, dataset):
    # Apply condition masks (same as original)
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

    # Get filtered participants
    participants = df_all[(df_all['dataset'] == dataset) & cond_mask]
    N = len(participants)
    
    # Define the 4 test columns to analyze
    test_columns = ['first_test_drule_gen', 'first_test_drule', 'second_test_drule_gen', 'second_test_drule']
    test_phase_types = ['first_gen', 'first_not_gen', 'second_gen', 'second_not_gen']
    
    # Initialize results
    results = []
    
    # Calculate proportions for each test column
    for test_col, phase_type in zip(test_columns, test_phase_types):
        if N > 0:
            # Get value counts for this test column
            counts = participants[test_col].value_counts()
            total = len(participants)
            
            # Calculate proportions based on column type
            if '_gen' in test_col:
                # For generalized columns: 2d, 1d, neither
                prop_2d = counts.get('2d', 0) / total
                prop_1d = counts.get('1d', 0) / total
                prop_neither = counts.get('neither', 0) / total
            else:
                # For non-generalized columns: 1d_a, 1d_b, 2d, neither
                # Combine 1d_a and 1d_b into 1d
                prop_2d = counts.get('2d', 0) / total
                prop_1d = (counts.get('1d_a', 0) + counts.get('1d_b', 0)) / total
                prop_neither = counts.get('neither', 0) / total
        else:
            prop_2d = 0
            prop_1d = 0
            prop_neither = 0
        
        results.append({
            'test_phase_type': phase_type,
            'prop_2d': prop_2d,
            'prop_1d': prop_1d,
            'prop_neither': prop_neither,
            'N': N
        })
    
    return results


def calculate_trapped_learner_comparison(players_df):
    """
    Calculate proportions and p-values comparing trapped learners in asocial vs social conditions.
    
    Parameters:
    -----------
    players_df : pandas.DataFrame
        Filtered players dataframe
        
    Returns:
    --------
    dict : Dictionary with comparison results by dataset
    """
    results = {}
    
    for dataset in ALL_DATASETS:
        # Filter for trapped learners in this dataset
        trapped_learners = players_df[
            (players_df['dataset'] == dataset) & 
            (players_df['first_test_drule_gen'] == '1d')
        ]
        
        # Separate into asocial and social groups
        asocial_group = trapped_learners[trapped_learners['game_type'] == 'solo']
        social_group = trapped_learners[trapped_learners['game_type'] == 'duo']
        
        n_asocial = len(asocial_group)
        n_social = len(social_group)
        
        if n_asocial > 0 and n_social > 0:
            # Calculate proportions for second test phase (where treatment effects occur)
            # Asocial group
            asocial_counts = asocial_group['second_test_drule_gen'].value_counts()
            asocial_2d = asocial_counts.get('2d', 0) / n_asocial
            asocial_1d = asocial_counts.get('1d', 0) / n_asocial
            asocial_neither = asocial_counts.get('neither', 0) / n_asocial
            
            # Social group
            social_counts = social_group['second_test_drule_gen'].value_counts()
            social_2d = social_counts.get('2d', 0) / n_social
            social_1d = social_counts.get('1d', 0) / n_social
            social_neither = social_counts.get('neither', 0) / n_social
            
            # Calculate p-values
            p_2d = permutation_test_proportion(
                round(asocial_2d * n_asocial), n_asocial,
                round(social_2d * n_social), n_social
            )
            p_1d = permutation_test_proportion(
                round(asocial_1d * n_asocial), n_asocial,
                round(social_1d * n_social), n_social
            )
            p_neither = permutation_test_proportion(
                round(asocial_neither * n_asocial), n_asocial,
                round(social_neither * n_social), n_social
            )
            
            results[dataset] = {
                'n_asocial': n_asocial,
                'asocial_2d': asocial_2d,
                'asocial_1d': asocial_1d,
                'asocial_neither': asocial_neither,
                'n_social': n_social,
                'social_2d': social_2d,
                'social_1d': social_1d,
                'social_neither': social_neither,
                'p_2d': p_2d,
                'p_1d': p_1d,
                'p_neither': p_neither
            }
        else:
            # Not enough data for comparison
            results[dataset] = None
    
    return results


def calculate_dataset_comparison_matrices(results_df):
    """
    Calculate p-value matrices comparing 2D rates between datasets for specific conditions.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with test proportions for all datasets and conditions
        
    Returns:
    --------
    dict : Dictionary with '2d' and 'other_1d' matrices
    """
    matrices = {}
    
    for condition in ['2d', 'other_1d']:
        # Get second_gen data for this condition
        condition_data = results_df[
            (results_df['condition'] == condition) & 
            (results_df['test_phase_type'] == 'second_gen')
        ]
        
        # Create matrix for this condition
        matrix = {}
        
        for dataset1 in ALL_DATASETS:
            matrix[dataset1] = {}
            
            for dataset2 in ALL_DATASETS:
                if dataset1 == dataset2:
                    # Can't compare dataset to itself
                    matrix[dataset1][dataset2] = None
                else:
                    # Get data for both datasets
                    data1 = condition_data[condition_data['dataset'] == dataset1]
                    data2 = condition_data[condition_data['dataset'] == dataset2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # Extract values
                        n1 = data1.iloc[0]['N']
                        prop_2d_1 = data1.iloc[0]['prop_2d']
                        n2 = data2.iloc[0]['N']
                        prop_2d_2 = data2.iloc[0]['prop_2d']
                        
                        if n1 > 0 and n2 > 0:
                            # Calculate p-value
                            p_value = permutation_test_proportion(
                                round(prop_2d_1 * n1), n1,
                                round(prop_2d_2 * n2), n2
                            )
                            matrix[dataset1][dataset2] = p_value
                        else:
                            matrix[dataset1][dataset2] = None
                    else:
                        matrix[dataset1][dataset2] = None
        
        matrices[condition] = matrix
    
    return matrices


def main():
    """
    Main function to calculate test decision rule proportions across datasets and conditions.
    Loads preprocessed data and calculates proportions directly from test columns.
    """
    import os
    from datetime import datetime
    
    # Load filtered preprocessed data (only need players_df)
    print("Loading preprocessed data...")
    players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
    
    print(f"Loaded {len(players_df)} participants")
    print(f"Datasets: {sorted(players_df['dataset'].unique())}")
    print(f"Participants per dataset: \n{players_df['dataset'].value_counts().sort_index()}")
    
    # Initialize results dataframe with all columns
    results_df = pd.DataFrame(columns=['dataset', 'condition', 'test_phase_type', 'prop_2d', 'prop_neither', 'prop_1d', 'N'])
    
    print("\nCalculating test proportions for each dataset and condition...")
    
    # For each dataset (using ordered processing - sim first, then numbered datasets)
    for dataset in ALL_DATASETS:
        print(f"  Processing dataset {dataset}...")
        # Get proportions for each condition and store in dataframe
        for condition in ['first', '2d', 'other_1d', 'asocial']:
            results = test_drule_proportions_per_dataset(players_df, condition, dataset)
            
            # Add rows for each test phase type
            for result in results:
                results_df.loc[len(results_df)] = {
                    'dataset': dataset,
                    'condition': condition,
                    'test_phase_type': result['test_phase_type'],
                    'prop_2d': result['prop_2d'],
                    'prop_neither': result['prop_neither'],
                    'prop_1d': result['prop_1d'],
                    'N': result['N']
                }
    
    # Add p-value columns comparing each condition to asocial control
    print("\nCalculating p-values vs asocial control...")
    
    # Initialize p-value columns
    results_df['p_vs_asocial_2d'] = None
    results_df['p_vs_asocial_1d'] = None
    results_df['p_vs_asocial_neither'] = None
    
    # For each row, calculate p-values if condition is not 'asocial'
    for idx, row in results_df.iterrows():
        if row['condition'] == 'asocial':
            # Can't compare asocial to itself, leave as None
            continue
            
        # Find the matching asocial row (same dataset and test_phase_type)
        asocial_row = results_df[
            (results_df['dataset'] == row['dataset']) &
            (results_df['test_phase_type'] == row['test_phase_type']) &
            (results_df['condition'] == 'asocial')
        ]
        
        if len(asocial_row) == 0:
            # No asocial data for this dataset/phase, skip
            continue
            
        asocial_row = asocial_row.iloc[0]
        
        # Calculate p-values for each outcome if both groups have participants
        if row['N'] > 0 and asocial_row['N'] > 0:
            # Calculate counts from proportions
            condition_2d_count = round(row['prop_2d'] * row['N'])
            condition_1d_count = round(row['prop_1d'] * row['N'])
            condition_neither_count = round(row['prop_neither'] * row['N'])
            
            asocial_2d_count = round(asocial_row['prop_2d'] * asocial_row['N'])
            asocial_1d_count = round(asocial_row['prop_1d'] * asocial_row['N'])
            asocial_neither_count = round(asocial_row['prop_neither'] * asocial_row['N'])
            
            # Calculate p-values for each outcome
            p_2d = permutation_test_proportion(
                condition_2d_count, row['N'],
                asocial_2d_count, asocial_row['N']
            )
            p_1d = permutation_test_proportion(
                condition_1d_count, row['N'],
                asocial_1d_count, asocial_row['N']
            )
            p_neither = permutation_test_proportion(
                condition_neither_count, row['N'],
                asocial_neither_count, asocial_row['N']
            )
            
            # Store p-values
            results_df.at[idx, 'p_vs_asocial_2d'] = p_2d
            results_df.at[idx, 'p_vs_asocial_1d'] = p_1d
            results_df.at[idx, 'p_vs_asocial_neither'] = p_neither
    
    # Calculate trapped learner comparison
    print("\nCalculating trapped learner comparison (asocial vs social)...")
    trapped_learner_comparison = calculate_trapped_learner_comparison(players_df)
    
    # Calculate dataset comparison matrices
    print("\nCalculating dataset comparison matrices for 2D rates...")
    dataset_comparison_matrices = calculate_dataset_comparison_matrices(results_df)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Save merged results dataframe
    output_file = 'outputs/test_decision_rule_proportions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Create documentation file as markdown
    doc_file = 'outputs/test_decision_rule_proportions_info.md'
    with open(doc_file, 'w') as f:
        f.write("# Test Decision Rule Proportions Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This analysis calculates proportions of test decision rule usage (2d, 1d, neither) ")
        f.write("directly from participant test phase performance across different experimental ")
        f.write("conditions and datasets for the dyadic learning trap experiment.\n\n")
        
        f.write("## Participant Counts by Dataset and Condition\n\n")
        condition_counts = results_df.groupby(['dataset', 'condition'])['N'].first().unstack(fill_value=0)
        f.write("| Dataset | first | 2d | other_1d | asocial |\n")
        f.write("|---------|-------|----|---------|---------|")  
        for dataset in condition_counts.index:
            first = condition_counts.loc[dataset, 'first']
            td = condition_counts.loc[dataset, '2d']
            other_1d = condition_counts.loc[dataset, 'other_1d']
            asocial = condition_counts.loc[dataset, 'asocial']
            f.write(f"\n| {dataset} | {first} | {td} | {other_1d} | {asocial} |")
        f.write("\n\n")
        
        f.write("## Test Phase Results Summary\n\n")
        
        # Create summary tables for each test phase type (only generalized phases)
        test_phase_types = ['first_gen', 'second_gen']
        test_phase_labels = {
            'first_gen': 'First Test Phase (Generalized)',
            'second_gen': 'Second Test Phase (Generalized)'
        }
        
        # Helper function to format p-values with significance stars
        def format_p_value(p_val):
            if p_val is None or pd.isna(p_val):
                return "N/A"
            stars = ""
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            return f"{p_val:.4f}{stars}"
        
        for phase_type in test_phase_types:
            f.write(f"### {test_phase_labels[phase_type]}\n\n")
            phase_data = results_df[results_df['test_phase_type'] == phase_type]
            
            if len(phase_data) > 0:
                f.write("| Dataset | Condition | 2D | 1D | Neither | N | p(2D) | p(1D) | p(Neither) |\n")
                f.write("|---------|-----------|----|----|---------|----|---------|---------|-----------|\n")
                
                for _, row in phase_data.iterrows():
                    dataset = row['dataset']
                    condition = row['condition']
                    prop_2d = f"{row['prop_2d']:.3f}"
                    prop_1d = f"{row['prop_1d']:.3f}"
                    prop_neither = f"{row['prop_neither']:.3f}"
                    n = row['N']
                    
                    # Format p-values (will show N/A for asocial condition)
                    p_2d_str = format_p_value(row['p_vs_asocial_2d'])
                    p_1d_str = format_p_value(row['p_vs_asocial_1d'])
                    p_neither_str = format_p_value(row['p_vs_asocial_neither'])
                    
                    f.write(f"| {dataset} | {condition} | {prop_2d} | {prop_1d} | {prop_neither} | {n} | {p_2d_str} | {p_1d_str} | {p_neither_str} |\n")
            else:
                f.write("*No data available for this phase type*\n")
            f.write("\n")
        
        # Add trapped learner comparison section
        f.write("### Trapped Learner Comparison: Asocial vs Social Control\n\n")
        f.write("This analysis compares all trapped learners (first_test_drule_gen == '1d') between asocial control ")
        f.write("(solo game) and social control (all duo game participants) conditions. ")
        f.write("The comparison focuses on second test phase performance where treatment effects occur.\n\n")
        
        # Create table header
        f.write("| Dataset | Asocial Control | | | | Social Control | | | | P-values | | |\n")
        f.write("|---------|-----------------|--|--|--|----------------|--|--|--|----------|--|--|\n")
        f.write("| | N | 2D | 1D | Neither | N | 2D | 1D | Neither | p(2D) | p(1D) | p(Neither) |\n")
        
        # Add data rows
        for dataset in ALL_DATASETS:
            if dataset in trapped_learner_comparison and trapped_learner_comparison[dataset] is not None:
                data = trapped_learner_comparison[dataset]
                
                # Format proportions
                asocial_2d = f"{data['asocial_2d']:.3f}"
                asocial_1d = f"{data['asocial_1d']:.3f}"
                asocial_neither = f"{data['asocial_neither']:.3f}"
                social_2d = f"{data['social_2d']:.3f}"
                social_1d = f"{data['social_1d']:.3f}"
                social_neither = f"{data['social_neither']:.3f}"
                
                # Format p-values with stars
                p_2d_str = format_p_value(data['p_2d'])
                p_1d_str = format_p_value(data['p_1d'])
                p_neither_str = format_p_value(data['p_neither'])
                
                f.write(f"| {dataset} | {data['n_asocial']} | {asocial_2d} | {asocial_1d} | {asocial_neither} | ")
                f.write(f"{data['n_social']} | {social_2d} | {social_1d} | {social_neither} | ")
                f.write(f"{p_2d_str} | {p_1d_str} | {p_neither_str} |\n")
            else:
                # No data for this dataset
                f.write(f"| {dataset} | - | - | - | - | - | - | - | - | - | - | - |\n")
        
        f.write("\n**Note**: This comparison combines all social conditions (2D partner and Other-1D partner) ")
        f.write("to test the overall effect of social learning on trapped learners.\n\n")
        
        # Add dataset comparison matrices section
        f.write("### Cross-Dataset Comparisons: 2D Adoption Rates\n\n")
        f.write("These matrices show p-values for pairwise comparisons of 2D adoption rates between datasets ")
        f.write("in the second test phase. Each cell (row, column) contains the p-value for comparing the ")
        f.write("2D rate of the row dataset against the column dataset.\n\n")
        
        # Process each condition
        for condition in ['2d', 'other_1d']:
            condition_label = '2D Partner' if condition == '2d' else 'Other-1D Partner'
            f.write(f"#### {condition_label} Condition\n\n")
            
            # Create matrix table
            f.write("|     |")
            for dataset in ALL_DATASETS:
                f.write(f" {dataset} |")
            f.write("\n")
            
            # Header separator
            f.write("|-----|")
            for _ in ALL_DATASETS:
                f.write("-----|")
            f.write("\n")
            
            # Matrix rows
            matrix = dataset_comparison_matrices[condition]
            for dataset1 in ALL_DATASETS:
                f.write(f"| {dataset1} |")
                for dataset2 in ALL_DATASETS:
                    if dataset1 == dataset2:
                        f.write("  -  |")
                    else:
                        p_value = matrix[dataset1][dataset2]
                        if p_value is not None:
                            # Format p-value with significance stars
                            p_str = format_p_value(p_value)
                            # Limit length for table formatting
                            if len(p_str) > 7:
                                p_str = f"{p_value:.3f}"
                                if p_value < 0.001:
                                    p_str += "***"
                                elif p_value < 0.01:
                                    p_str += "**"
                                elif p_value < 0.05:
                                    p_str += "*"
                            f.write(f" {p_str:^7} |")
                        else:
                            f.write("  N/A  |")
                f.write("\n")
            f.write("\n")
        
        f.write("**Note**: The matrix is symmetric (p-value for row vs column equals column vs row). ")
        f.write("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001.\n\n")
        
        f.write("## Data Structure\n\n")
        f.write("The output CSV file contains the following columns:\n\n")
        f.write("- **dataset**: Dataset identifier (sim, 1-6)\n")
        f.write("- **condition**: Experimental condition ('first', '2d', 'other_1d', 'asocial')\n")
        f.write("- **test_phase_type**: Type of test analysis ('first_gen', 'first_not_gen', 'second_gen', 'second_not_gen')\n")
        f.write("- **prop_2d**: Proportion using 2D decision rule (optimal)\n")
        f.write("- **prop_1d**: Proportion using 1D decision rule\n")
        f.write("- **prop_neither**: Proportion using neither rule consistently\n")
        f.write("- **N**: Number of participants in this condition/dataset\n")
        f.write("- **p_vs_asocial_2d**: P-value comparing 2D proportion to asocial control\n")
        f.write("- **p_vs_asocial_1d**: P-value comparing 1D proportion to asocial control\n")
        f.write("- **p_vs_asocial_neither**: P-value comparing neither proportion to asocial control\n\n")
        
        f.write("### Test Phase Types\n\n")
        f.write("- **first_gen**: First test phase (generalized) from `first_test_drule_gen` column\n")
        f.write("- **first_not_gen**: First test phase (specific) from `first_test_drule` column\n")
        f.write("- **second_gen**: Second test phase (generalized) from `second_test_drule_gen` column\n")
        f.write("- **second_not_gen**: Second test phase (specific) from `second_test_drule` column\n\n")
        f.write("*Note: Only generalized results (first_gen and second_gen) are shown in the tables above ")
        f.write("with integrated p-values. All four test phase types are included in the CSV output file.*\n\n")
        
        f.write("### Conditions\n\n")
        f.write("- **first**: First test phase (all participants)\n")
        f.write("- **asocial**: Solo game participants who used 1D rule in first test\n")
        f.write("- **2d**: Duo game participants (1D in first test) paired with 2D partner\n")
        f.write("- **other_1d**: Duo game participants paired with complementary 1D partner\n\n")
        
        f.write("### Statistical Comparisons\n\n")
        f.write("P-values are calculated using permutation tests (10,000 iterations) comparing each condition ")
        f.write("to the asocial control group. P-values are shown directly in the tables above alongside ")
        f.write("proportions. Statistical significance levels:\n\n")
        f.write("- * p < 0.05\n")
        f.write("- ** p < 0.01\n") 
        f.write("- *** p < 0.001\n")
        f.write("- N/A: No comparison available (e.g., asocial condition compared to itself)\n\n")
        
        f.write("## Analysis Summary\n\n")
        f.write(f"- **Total data rows**: {len(results_df)}\n")
        f.write(f"- **Datasets analyzed**: {', '.join(sorted(results_df['dataset'].unique()))}\n")
        f.write(f"- **Conditions**: {', '.join(sorted(results_df['condition'].unique()))}\n")
        f.write(f"- **Test phase types**: {', '.join(sorted(results_df['test_phase_type'].unique()))}\n\n")
        
        f.write("## Source Data\n\n")
        f.write(f"- **Participants**: {len(players_df)} (filtered, no external aid)\n")
        f.write("- **Input file**: `../../preprocessing/outputs/players_df_all_filtered.csv`\n")
        f.write("- **Note**: Uses participant-level test decision rule columns directly\n")
        f.write("- **Output file**: `test_decision_rule_proportions.csv`\n")
    
    print(f"Saved documentation to {doc_file}")
    print(f"\nAnalysis complete! Generated {len(results_df)} rows of test proportion data.")
    
    return results_df


if __name__ == "__main__":
    results_df = main()