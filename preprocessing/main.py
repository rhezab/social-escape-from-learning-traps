#!/usr/bin/env python3
"""
Create Players and Blocks DataFrames from Dyadic Learning Trap Experiment Data

This script processes raw experiment data from multiple datasets (d1-d6) and creates
two main dataframes:
1. players_df_all: Participant-level data with decision rules for test phases
2. blocks_all: Block-level data with decision matrices and error calculations

The script handles different experimental versions with varying phase structures:
- d1-d4, d6: 4 phases with test phases at 1 and 3
- d5: 5 phases with test phases at 1 and 4 (includes partner prediction phase)

Output files:
- players_df_all.csv: Participant-level data
- blocks_all.csv: Block-level data with decision rules and errors

Author: Generated with Claude Code
Date: 2025-01-17
"""

import pandas as pd
import os
from datetime import datetime
import sys

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import ALL_DATASETS, DATASET_PHASES

# Import local modules (now in same directory)
from dataset import load_data
from players_df import create_players_df, filter_and_add_partner_rules_d1, add_test_drules_to_players_df, add_partner_rule_extensions
from blocks_df import create_blocks_df, merge_blocks_dataframes, analyze_blocks_dataframe, add_generalized_drule_column


def main():
    """
    Main function to process all experimental datasets and create output dataframes.
    """
    print("=" * 60)
    print("DYADIC LEARNING TRAP EXPERIMENT DATA PROCESSING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ============================================================================
    # STEP 1: LOAD RAW DATA
    # ============================================================================
    print("STEP 1: Loading raw experiment data...")
    
    data_files = {
        'sim': '../simulation/outputs/simulated_data.json',
        '1': '../data/exp-1-data-anonymized.json',
        '2': '../data/exp-2-data-anonymized.json', 
        '3': '../data/exp-3-data-anonymized.json',
        '4': '../data/exp-4-data-anonymized.json',
        '5': '../data/exp-5-data-anonymized.json',
        '6': '../data/exp-6-data-anonymized.json'
    }
    
    datasets = {}
    for dataset, filepath in data_files.items():
        print(f"  Loading {dataset}: {filepath}")
        datasets[dataset] = load_data(filepath)
        print(f"    -> {len(datasets[dataset])} participants")
    
    print(f"Total raw participants across all datasets: {sum(len(d) for d in datasets.values())}")
    print()
    
    # ============================================================================
    # STEP 2: CREATE PLAYERS DATAFRAMES
    # ============================================================================
    print("STEP 2: Creating players dataframes...")
    
    players_dfs = {}
    for dataset in ALL_DATASETS:
        # All datasets: only completed participants
        print(f"  Processing {dataset} (completed participants only)...")
        players_dfs[dataset] = create_players_df(datasets[dataset], dataset=dataset)
        print(f"    -> {len(players_dfs[dataset])} participants")
    
    total_players = sum(len(df) for df in players_dfs.values())
    print(f"Total processed participants: {total_players}")
    print()
    
    # ============================================================================
    # STEP 3: CREATE BLOCKS DATAFRAMES
    # ============================================================================
    print("STEP 3: Creating blocks dataframes...")
    
    # Configuration for each dataset
    dataset_configs = {
        'sim': {'n_phases': 4, 'phase_blocks': [4,2,4,2]},
        '1': {'n_phases': 4, 'phase_blocks': [4,2,4,2]},
        '2': {'n_phases': 4, 'phase_blocks': [4,2,4,2]}, 
        '3': {'n_phases': 4, 'phase_blocks': [4,2,4,2]},
        '4': {'n_phases': 4, 'phase_blocks': [4,2,4,2]},
        '5': {'n_phases': 5, 'phase_blocks': [4,2,4,4,2]},  # Special: 5 phases
        '6': {'n_phases': 4, 'phase_blocks': [4,2,4,2]}
    }
    
    epsilon = 1  # Error threshold for decision rule classification (0-16 scale)
    blocks_dfs = {}
    
    for dataset in ALL_DATASETS:
        print(f"  Creating blocks dataframe for {dataset}...")
        
        # Get participant IDs for this dataset
        participant_ids = players_dfs[dataset]['id'].tolist()
        config = dataset_configs[dataset]
        
        # Create blocks dataframe
        # Use sim_data=True for simulated data to bypass Prolific filtering
        sim_data_flag = (dataset == 'sim')
        
        # Set partner_prediction_phase from constants if it exists (only dataset 5)
        partner_pred_phase = None
        if 'partner_prediction' in DATASET_PHASES.get(dataset, {}):
            partner_pred_phase = DATASET_PHASES[dataset]['partner_prediction'][0]
        
        try:
            blocks_dfs[dataset] = create_blocks_df(
                datasets[dataset],
                ids=participant_ids,
                n_phases=config['n_phases'],
                phase_blocks=config['phase_blocks'], 
                epsilon=epsilon,
                sim_data=sim_data_flag,
                partner_prediction_phase=partner_pred_phase
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create blocks dataframe for dataset '{dataset}': {str(e)}") from e
        
        print(f"    -> {len(blocks_dfs[dataset])} blocks")
    
    total_blocks = sum(len(df) for df in blocks_dfs.values())
    print(f"Total blocks across all datasets: {total_blocks}")
    print()
    
    # ============================================================================
    # STEP 4: ADD TEST DECISION RULES TO PLAYERS
    # ============================================================================ 
    print("STEP 4: Adding test decision rules to players dataframes...")
    
    # Test phase configurations (different for dataset 5)
    test_configs = {
        'sim': {'first_test_phase': 1, 'second_test_phase': 3},
        '1': {'first_test_phase': 1, 'second_test_phase': 3},
        '2': {'first_test_phase': 1, 'second_test_phase': 3},
        '3': {'first_test_phase': 1, 'second_test_phase': 3}, 
        '4': {'first_test_phase': 1, 'second_test_phase': 3},
        '5': {'first_test_phase': 1, 'second_test_phase': 4},  # Special: phase 4 for second test
        '6': {'first_test_phase': 1, 'second_test_phase': 3}
    }
    
    epsilon_test = 2  # Error threshold for test phase classification
    
    for dataset in ALL_DATASETS:
        print(f"  Adding test drules for {dataset}...")
        config = test_configs[dataset]
        
        players_dfs[dataset] = add_test_drules_to_players_df(
            players_dfs[dataset], 
            blocks_dfs[dataset],
            epsilon=epsilon_test,
            first_test_phase=config['first_test_phase'],
            second_test_phase=config['second_test_phase']
        )
        
        # Show test drule distribution for this dataset
        first_dist = players_dfs[dataset]['first_test_drule'].value_counts(dropna=False)
        second_dist = players_dfs[dataset]['second_test_drule'].value_counts(dropna=False)
        print(f"    First test: {dict(first_dist)}")
        print(f"    Second test: {dict(second_dist)}")
    
    # ============================================================================
    # STEP 4.5: FILTER D1 AND ADD PARTNER RULES (after test drules are available)
    # ============================================================================
    print("\nSTEP 4.5: Filtering dataset 1 participants and adding partner rules...")
    players_dfs['1'] = filter_and_add_partner_rules_d1(players_dfs['1'])
    print(f"  -> {len(players_dfs['1'])} dataset 1 participants after filtering and adding partner rules")
    
    print()
    
    # ============================================================================
    # STEP 5: MERGE DATAFRAMES
    # ============================================================================
    print("STEP 5: Merging dataframes...")
    
    # Add dataset labels to players dataframes
    for dataset in ALL_DATASETS:
        players_dfs[dataset]['dataset'] = dataset
    
    # Merge players dataframes
    print("  Merging players dataframes...")
    players_df_all = pd.concat(list(players_dfs.values()), axis=0, ignore_index=True)
    players_df_all = players_df_all.where(pd.notnull(players_df_all), None)
    
    # Sort columns for better readability
    cols = ['dataset'] + sorted([col for col in players_df_all.columns if col != 'dataset'])
    players_df_all = players_df_all[cols]
    
    print(f"    -> {len(players_df_all)} total participants")
    
    # Add partner rule extensions (partner_rule_gen and partner_rule_rel)
    print("  Adding partner rule extensions...")
    players_df_all = add_partner_rule_extensions(players_df_all)
    
    # Merge blocks dataframes  
    print("  Merging blocks dataframes...")
    blocks_all = merge_blocks_dataframes(
        list(blocks_dfs.values()),
        dataset_labels=ALL_DATASETS
    )
    
    # Add generalized decision rule column
    blocks_all = add_generalized_drule_column(blocks_all)
    
    print(f"    -> {len(blocks_all)} total blocks")
    
    # Analyze merged blocks dataframe
    print("\\n  Blocks dataframe analysis:")
    analyze_blocks_dataframe(blocks_all)
    print()
    
    # ============================================================================
    # STEP 6: FILTER OUT EXTERNAL AID USERS
    # ============================================================================
    print("STEP 6: Filtering out participants who used external aid...")
    
    # Filter players dataframe to exclude external_aid=True
    external_aid_mask = players_df_all['external_aid'] != True
    players_df_all_filtered = players_df_all[external_aid_mask].copy()
    
    # Get list of participant IDs after filtering
    filtered_participant_ids = set(players_df_all_filtered['id'])
    
    # Filter blocks dataframe to only include participants who didn't use external aid
    blocks_df_mask = blocks_all['id'].isin(filtered_participant_ids)
    blocks_all_filtered = blocks_all[blocks_df_mask].copy()
    
    print(f"  Original participants: {len(players_df_all)}")
    print(f"  Participants with external_aid=True: {len(players_df_all) - len(players_df_all_filtered)}")
    print(f"  Filtered participants: {len(players_df_all_filtered)}")
    print(f"  Original blocks: {len(blocks_all)}")
    print(f"  Filtered blocks: {len(blocks_all_filtered)}")
    
    # Show breakdown by dataset
    print("\\n  External aid usage by dataset:")
    external_aid_by_dataset = players_df_all.groupby('dataset')['external_aid'].value_counts()
    for (dataset, external_aid), count in external_aid_by_dataset.items():
        print(f"    Dataset {dataset}, external_aid={external_aid}: {count} participants")
    
    print()
    
    # ============================================================================
    # STEP 7: SAVE RESULTS  
    # ============================================================================
    print("STEP 7: Saving results...")
    
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original dataframes
    players_output = f"{output_dir}/players_df_all.csv"
    blocks_output = f"{output_dir}/blocks_all.csv"
    
    print(f"  Saving original players dataframe to: {players_output}")
    players_df_all.to_csv(players_output, index=False)
    print(f"    -> {len(players_df_all)} rows, {len(players_df_all.columns)} columns")
    
    print(f"  Saving original blocks dataframe to: {blocks_output}")
    blocks_all.to_csv(blocks_output, index=False) 
    print(f"    -> {len(blocks_all)} rows, {len(blocks_all.columns)} columns")
    
    # Save filtered dataframes
    players_filtered_output = f"{output_dir}/players_df_all_filtered.csv"
    blocks_filtered_output = f"{output_dir}/blocks_all_filtered.csv"
    
    print(f"  Saving filtered players dataframe to: {players_filtered_output}")
    players_df_all_filtered.to_csv(players_filtered_output, index=False)
    print(f"    -> {len(players_df_all_filtered)} rows, {len(players_df_all_filtered.columns)} columns")
    
    print(f"  Saving filtered blocks dataframe to: {blocks_filtered_output}")
    blocks_all_filtered.to_csv(blocks_filtered_output, index=False)
    print(f"    -> {len(blocks_all_filtered)} rows, {len(blocks_all_filtered.columns)} columns")
    
    # Save summary statistics as markdown
    summary_output = f"{output_dir}/processing_summary.md"
    print(f"  Saving processing summary to: {summary_output}")
    
    with open(summary_output, 'w') as f:
        # Header
        f.write("# Dyadic Learning Trap Experiment - Data Processing Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Participant Totals Section
        f.write("## Participant Totals\n\n")
        
        # Raw Data Counts Table
        f.write("### Raw Data Counts\n\n")
        f.write("| Dataset | Participants |\n")
        f.write("|---------|--------------|\n")
        for dataset in ALL_DATASETS:
            f.write(f"| {dataset} | {len(datasets[dataset])} |\n")
        f.write(f"| **Total** | **{sum(len(d) for d in datasets.values())}** |\n")
        f.write("\n")
        
        # Processed Participant Counts Table
        f.write("### Processed Participant Counts\n\n")
        f.write("| Dataset | Participants | Blocks |\n")
        f.write("|---------|--------------|--------|\n")
        for dataset in ALL_DATASETS:
            f.write(f"| {dataset} | {len(players_dfs[dataset])} | {len(blocks_dfs[dataset])} |\n")
        f.write(f"| **Total** | **{len(players_df_all)}** | **{len(blocks_all)}** |\n")
        f.write("\n")
        
        # Final Filtered Counts Table
        f.write("### Final Filtered Counts (Excluding External Aid)\n\n")
        f.write("| Dataset | Participants | Blocks |\n")
        f.write("|---------|--------------|--------|\n")
        # Calculate filtered counts by dataset
        filtered_by_dataset = players_df_all_filtered.groupby('dataset').size()
        filtered_blocks_by_dataset = blocks_all_filtered.groupby('dataset').size()
        for dataset in ALL_DATASETS:
            participants = filtered_by_dataset.get(dataset, 0)
            blocks = filtered_blocks_by_dataset.get(dataset, 0)
            f.write(f"| {dataset} | {participants} | {blocks} |\n")
        f.write(f"| **Total** | **{len(players_df_all_filtered)}** | **{len(blocks_all_filtered)}** |\n")
        f.write(f"\nParticipants excluded due to external aid: {len(players_df_all) - len(players_df_all_filtered)}\n\n")
        
        # External Aid Usage Breakdown
        f.write("## External Aid Usage Breakdown\n\n")
        f.write("| Dataset | Used Aid | No Aid | % Using Aid |\n")
        f.write("|---------|----------|--------|-------------|\n")
        for dataset in ALL_DATASETS:
            dataset_data = players_df_all[players_df_all['dataset'] == dataset]
            used_aid = (dataset_data['external_aid'] == True).sum()
            no_aid = (dataset_data['external_aid'] != True).sum()
            total = len(dataset_data)
            percent_aid = (used_aid / total * 100) if total > 0 else 0
            f.write(f"| {dataset} | {used_aid} | {no_aid} | {percent_aid:.1f}% |\n")
        # Total row
        total_used_aid = (players_df_all['external_aid'] == True).sum()
        total_no_aid = (players_df_all['external_aid'] != True).sum()
        total_all = len(players_df_all)
        total_percent = (total_used_aid / total_all * 100) if total_all > 0 else 0
        f.write(f"| **Total** | **{total_used_aid}** | **{total_no_aid}** | **{total_percent:.1f}%** |\n")
        f.write("\n")
        
        # Decision Rule Analysis Section
        f.write("## Preliminary Results: Decision Rule Analysis\n\n")
        f.write("*Note: Analysis based on filtered data (excluding external aid users)*\n\n")
        
        # First Test Phase Decision Rules Table
        f.write("### First Test Phase Decision Rules\n\n")
        f.write("| Dataset | 2d | 1d | neither | Total |\n")
        f.write("|---------|----|----|---------|---------\n")
        for dataset in ALL_DATASETS:
            dataset_players = players_df_all_filtered[players_df_all_filtered['dataset'] == dataset]
            first_test_counts = dataset_players['first_test_drule_gen'].value_counts()
            total = len(dataset_players)
            
            count_2d = first_test_counts.get('2d', 0)
            count_1d = first_test_counts.get('1d', 0)
            count_neither = first_test_counts.get('neither', 0)
            
            # Format with counts and percentages
            if total > 0:
                f.write(f"| {dataset} | {count_2d} ({count_2d/total*100:.1f}%) | {count_1d} ({count_1d/total*100:.1f}%) | {count_neither} ({count_neither/total*100:.1f}%) | {total} |\n")
            else:
                f.write(f"| {dataset} | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 |\n")
        
        # Total row for first test
        all_first_test = players_df_all_filtered['first_test_drule_gen'].value_counts()
        total_first = len(players_df_all_filtered)
        total_2d = all_first_test.get('2d', 0)
        total_1d = all_first_test.get('1d', 0) 
        total_neither = all_first_test.get('neither', 0)
        f.write(f"| **Total** | **{total_2d} ({total_2d/total_first*100:.1f}%)** | **{total_1d} ({total_1d/total_first*100:.1f}%)** | **{total_neither} ({total_neither/total_first*100:.1f}%)** | **{total_first}** |\n")
        f.write("\n")
        
        # Second Test Phase Decision Rules Table
        f.write("### Second Test Phase Decision Rules\n\n")
        f.write("| Dataset | 2d | 1d | neither | Total |\n")
        f.write("|---------|----|----|---------|---------\n")
        for dataset in ALL_DATASETS:
            dataset_players = players_df_all_filtered[players_df_all_filtered['dataset'] == dataset]
            second_test_counts = dataset_players['second_test_drule_gen'].value_counts()
            total = len(dataset_players)
            
            count_2d = second_test_counts.get('2d', 0)
            count_1d = second_test_counts.get('1d', 0)
            count_neither = second_test_counts.get('neither', 0)
            
            # Format with counts and percentages
            if total > 0:
                f.write(f"| {dataset} | {count_2d} ({count_2d/total*100:.1f}%) | {count_1d} ({count_1d/total*100:.1f}%) | {count_neither} ({count_neither/total*100:.1f}%) | {total} |\n")
            else:
                f.write(f"| {dataset} | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 |\n")
        
        # Total row for second test
        all_second_test = players_df_all_filtered['second_test_drule_gen'].value_counts()
        total_second = len(players_df_all_filtered)
        total_2d = all_second_test.get('2d', 0)
        total_1d = all_second_test.get('1d', 0)
        total_neither = all_second_test.get('neither', 0)
        f.write(f"| **Total** | **{total_2d} ({total_2d/total_second*100:.1f}%)** | **{total_1d} ({total_1d/total_second*100:.1f}%)** | **{total_neither} ({total_neither/total_second*100:.1f}%)** | **{total_second}** |\n")
        f.write("\n")
        
        # Output Files Section
        f.write("## Output Files\n\n")
        f.write("The following files have been generated:\n\n")
        f.write("- **players_df_all.csv**: All participants (before filtering)\n")
        f.write("- **blocks_all.csv**: All blocks (before filtering)\n") 
        f.write("- **players_df_all_filtered.csv**: Participants excluding external aid users\n")
        f.write("- **blocks_all_filtered.csv**: Blocks excluding external aid users\n")
        f.write("- **processing_summary.md**: This summary file\n")
        f.write("\n")
        
        # Column Information
        f.write("## Data Structure\n\n")
        f.write(f"**Players dataframe columns** ({len(players_df_all.columns)} total):\n")
        f.write(f"`{', '.join(sorted(players_df_all.columns))}`\n\n")
        f.write(f"**Blocks dataframe columns** ({len(blocks_all.columns)} total):\n")
        f.write(f"`{', '.join(sorted(blocks_all.columns))}`\n")
    
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files saved to: {os.path.abspath(output_dir)}")
    print()


if __name__ == "__main__":
    main()