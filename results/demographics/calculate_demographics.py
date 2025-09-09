#!/usr/bin/env python3
"""
Calculate Demographics for Dyadic Learning Trap Experiment

This script calculates demographic statistics for each dataset:
- Number of participants before filtering (total and external aid users)
- Number of participants after filtering (total and by condition)

Output: Markdown file with demographic statistics
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import ALL_DATASETS

def load_raw_data_counts():
    """Load raw data files to get participant counts before any processing"""
    data_files = {
        'sim': '../../simulation/outputs/simulated_data.json',
        '1': '../../data/exp-1-data.json',
        '2': '../../data/exp-2-data.json', 
        '3': '../../data/exp-3-data.json',
        '4': '../../data/exp-4-data.json',
        '5': '../../data/exp-5-data.json',
        '6': '../../data/exp-6-data.json'
    }
    
    raw_counts = {}
    for dataset, filepath in data_files.items():
        print(f"  Loading {dataset}: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            raw_counts[dataset] = len(data)
        print(f"    -> {raw_counts[dataset]} participants")
    
    total_raw = sum(raw_counts.values())
    print(f"Total raw participants across all datasets: {total_raw}")
    print()
    
    return raw_counts

def load_processed_data():
    """Load the processed players dataframes"""
    players_all_path = "../../preprocessing/outputs/players_df_all.csv"
    players_filtered_path = "../../preprocessing/outputs/players_df_all_filtered.csv"
    
    players_all = pd.read_csv(players_all_path)
    players_filtered = pd.read_csv(players_filtered_path)
    
    return players_all, players_filtered

def calculate_demographics(players_all, players_filtered, raw_counts):
    """Calculate demographics for each dataset"""
    results = {}
    
    # Process datasets in specified order (sim first, then numbered datasets)
    for dataset in ALL_DATASETS:
        dataset_all = players_all[players_all['dataset'] == dataset]
        dataset_filtered = players_filtered[players_filtered['dataset'] == dataset]
        
        # Raw data count (convert dataset number to string to match raw_counts keys)
        raw_total = raw_counts.get(str(dataset), 0)
        
        # After initial processing (completion + partner filtering for d1)
        processed_total = len(dataset_all)
        external_aid_users = len(dataset_all[dataset_all['external_aid'] == True])
        
        # After external aid filtering
        final_total = len(dataset_filtered)
        solo_count = len(dataset_filtered[dataset_filtered['game_type'] == 'solo'])
        duo_count = len(dataset_filtered[dataset_filtered['game_type'] == 'duo'])
        
        results[dataset] = {
            'raw_data': {
                'total': raw_total
            },
            'after_initial_processing': {
                'total': processed_total,
                'external_aid': external_aid_users
            },
            'after_external_aid_filtering': {
                'total': final_total,
                'solo': solo_count,
                'duo': duo_count
            }
        }
    
    return results

def calculate_dataset1_dyads(players_df_filtered):
    """
    Calculate unique dyads for dataset 1.
    
    Parameters:
    -----------
    players_df_filtered : pandas.DataFrame
        Filtered players dataframe
        
    Returns:
    --------
    dict : Dictionary with dyad statistics
    """
    # Filter for dataset 1
    d1_data = players_df_filtered[players_df_filtered['dataset'] == '1'].copy()
    
    if len(d1_data) == 0:
        return {
            'total_participants': 0,
            'total_dyads': 0,
            'complete_dyads': 0,
            'incomplete_dyads': 0,
            'solo_participants': 0,
            'duo_participants': 0
        }
    
    # Get participant counts by game type
    solo_participants = len(d1_data[d1_data['game_type'] == 'solo'])
    duo_participants = len(d1_data[d1_data['game_type'] == 'duo'])
    
    # Create dyad identifiers for all participants
    dyad_ids = set()
    participant_set = set(d1_data['id'].values)
    
    for _, row in d1_data.iterrows():
        participant_id = row['id']
        partner_id = row.get('partner_id', None)
        
        if partner_id and pd.notna(partner_id):
            # Create sorted tuple as dyad identifier
            dyad_id = tuple(sorted([participant_id, partner_id]))
            dyad_ids.add(dyad_id)
    
    # Count complete vs incomplete dyads
    complete_dyads = 0
    incomplete_dyads = 0
    
    for dyad_id in dyad_ids:
        member1, member2 = dyad_id
        if member1 in participant_set and member2 in participant_set:
            complete_dyads += 1
        else:
            incomplete_dyads += 1
    
    return {
        'total_participants': len(d1_data),
        'total_dyads': len(dyad_ids),
        'complete_dyads': complete_dyads,
        'incomplete_dyads': incomplete_dyads,
        'solo_participants': solo_participants,
        'duo_participants': duo_participants
    }

def generate_markdown_report(demographics_data, dyad_stats=None):
    """Generate markdown report with demographics"""
    
    # Create markdown content
    md_content = []
    md_content.append("# Demographics Report")
    md_content.append("")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")
    md_content.append("This report shows participant demographics for each dataset in the dyadic learning trap experiment.")
    md_content.append("")
    md_content.append("## Filtering Process")
    md_content.append("")
    md_content.append("The data goes through multiple filtering stages:")
    md_content.append("")
    md_content.append("1. **Raw Data**: All participants in the original JSON files")
    md_content.append("2. **Initial Processing**: Filters for completed participants and valid partner pairs (for d1 duo condition only)")
    md_content.append("3. **External Aid Filtering**: Additionally excludes participants who reported using external assistance")
    md_content.append("")
    md_content.append("Note: For dataset 1 (d1), the initial processing stage filters duo participants to only include those with valid partner pairs, while solo participants are retained regardless of partner status.")
    md_content.append("")
    
    # Summary table
    md_content.append("## Summary Table")
    md_content.append("")
    md_content.append("| Dataset | Raw Data | After Initial Processing |  | After External Aid Filtering |  |  |")
    md_content.append("|---------|----------|--------------------------|--|------------------------------|--|--|")
    md_content.append("| | **Total** | **Total** | **External Aid** | **Total** | **Solo** | **Duo** |")
    
    # Process datasets in order
    for dataset in ALL_DATASETS:
        data = demographics_data[dataset]
        raw = data['raw_data']
        initial = data['after_initial_processing']
        final = data['after_external_aid_filtering']
        
        row = f"| {dataset} | {raw['total']} | {initial['total']} | {initial['external_aid']} | {final['total']} | {final['solo']} | {final['duo']} |"
        md_content.append(row)
    
    md_content.append("")
    
    # Detailed breakdown
    md_content.append("## Detailed Breakdown")
    md_content.append("")
    
    for dataset in ALL_DATASETS:
        data = demographics_data[dataset]
        md_content.append(f"### Dataset {dataset}")
        md_content.append("")
        
        raw = data['raw_data']
        initial = data['after_initial_processing']
        final = data['after_external_aid_filtering']
        
        md_content.append("**Raw Data:**")
        md_content.append(f"- Total participants in JSON file: {raw['total']}")
        md_content.append("")
        
        md_content.append("**After Initial Processing (completion + partner filtering for d1):**")
        md_content.append(f"- Total participants: {initial['total']}")
        md_content.append(f"- Participants using external aid: {initial['external_aid']}")
        md_content.append("")
        
        md_content.append("**After External Aid Filtering:**")
        md_content.append(f"- Total participants: {final['total']}")
        md_content.append(f"- Solo (asocial) condition: {final['solo']}")
        md_content.append(f"- Duo (social) condition: {final['duo']}")
        md_content.append("")
    
    # Overall totals
    total_raw = sum(data['raw_data']['total'] for data in demographics_data.values())
    total_initial = sum(data['after_initial_processing']['total'] for data in demographics_data.values())
    total_external_aid = sum(data['after_initial_processing']['external_aid'] for data in demographics_data.values())
    total_final = sum(data['after_external_aid_filtering']['total'] for data in demographics_data.values())
    total_solo = sum(data['after_external_aid_filtering']['solo'] for data in demographics_data.values())
    total_duo = sum(data['after_external_aid_filtering']['duo'] for data in demographics_data.values())
    
    md_content.append("## Overall Totals")
    md_content.append("")
    md_content.append("**Raw Data:**")
    md_content.append(f"- Total participants across all JSON files: {total_raw}")
    md_content.append("")
    md_content.append("**After Initial Processing:**")
    md_content.append(f"- Total participants across all datasets: {total_initial}")
    md_content.append(f"- Total participants using external aid: {total_external_aid}")
    md_content.append(f"- Participants excluded in initial processing: {total_raw - total_initial}")
    md_content.append("")
    md_content.append("**After External Aid Filtering:**")
    md_content.append(f"- Total participants across all datasets: {total_final}")
    md_content.append(f"- Total solo (asocial) participants: {total_solo}")
    md_content.append(f"- Total duo (social) participants: {total_duo}")
    md_content.append(f"- Participants excluded due to external aid: {total_external_aid}")
    md_content.append(f"- Total participants excluded across all stages: {total_raw - total_final}")
    
    # Add dyad analysis section if provided
    if dyad_stats is not None and dyad_stats['total_participants'] > 0:
        md_content.append("")
        md_content.append("## Experiment 1 Dyad Analysis")
        md_content.append("")
        md_content.append("In experiment 1, all participants (both solo and duo conditions) are organized into dyads.")
        md_content.append("This analysis counts unique dyads in the final filtered sample.")
        md_content.append("")
        md_content.append("**Dyad Statistics:**")
        md_content.append(f"- Total participants in experiment 1: {dyad_stats['total_participants']}")
        md_content.append(f"  - Solo condition: {dyad_stats['solo_participants']}")
        md_content.append(f"  - Duo condition: {dyad_stats['duo_participants']}")
        md_content.append(f"- Total unique dyads: {dyad_stats['total_dyads']}")
        md_content.append(f"- Complete dyads (both members present): {dyad_stats['complete_dyads']}")
        md_content.append(f"- Incomplete dyads (one member filtered out): {dyad_stats['incomplete_dyads']}")
        md_content.append("")
        md_content.append("**Methodology:**")
        md_content.append("Dyads are identified using sorted tuples of (participant_id, partner_id), ensuring the same")
        md_content.append("dyad identifier regardless of which member is accessed. Complete dyads have both members")
        md_content.append("present in the filtered dataset, while incomplete dyads have one member filtered out due to")
        md_content.append("completion requirements, partner pairing issues, or external aid usage.")
    
    return "\n".join(md_content)

def main():
    """Main function to calculate demographics and save results"""
    print("Calculating demographics for dyadic learning trap experiment...")
    
    # Load data
    print("Loading raw data counts...")
    raw_counts = load_raw_data_counts()
    
    print("Loading processed data...")
    players_all, players_filtered = load_processed_data()
    
    # Calculate demographics
    print("Calculating demographics...")
    demographics_data = calculate_demographics(players_all, players_filtered, raw_counts)
    
    # Calculate dyad statistics for dataset 1
    print("Calculating dyad statistics for experiment 1...")
    dyad_stats = calculate_dataset1_dyads(players_filtered)
    
    # Generate markdown report
    print("Generating markdown report...")
    markdown_report = generate_markdown_report(demographics_data, dyad_stats)
    
    # Create outputs directory and save
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/demographics_report.md"
    
    with open(output_path, 'w') as f:
        f.write(markdown_report)
    
    print(f"Demographics report saved to: {output_path}")
    
    # Also print summary to console
    print("\nSummary:")
    for dataset in ALL_DATASETS:
        data = demographics_data[dataset]
        raw = data['raw_data']
        initial = data['after_initial_processing']
        final = data['after_external_aid_filtering']
        print(f"Dataset {dataset}: {raw['total']} -> {initial['total']} -> {final['total']} participants (excluded {raw['total'] - initial['total']} in initial processing, {initial['external_aid']} for external aid)")
    
    # Print dyad summary
    if dyad_stats['total_participants'] > 0:
        print(f"\nExperiment 1 dyad analysis:")
        print(f"  Total unique dyads: {dyad_stats['total_dyads']}")
        print(f"  Complete dyads (both members present): {dyad_stats['complete_dyads']}")
        print(f"  Incomplete dyads (one member filtered): {dyad_stats['incomplete_dyads']}")

if __name__ == "__main__":
    main()