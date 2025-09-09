#!/usr/bin/env python3
"""
Analyze and visualize points per block across different experimental conditions.

This script analyzes participant performance (points per block) across different
phases of the dyadic learning trap experiment, focusing on trapped learners and
their performance under different social learning conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import ALL_DATASETS, DATASET_PHASES, PUBLICATION_DATASETS

# Define which datasets to process
DATASETS_TO_PROCESS = ALL_DATASETS


def load_data():
    """Load preprocessed player and block data."""
    print("Loading preprocessed data...")
    players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
    blocks_df = pd.read_csv('../../preprocessing/outputs/blocks_all_filtered.csv')
    print(f"Loaded {len(players_df)} players and {len(blocks_df)} blocks")
    return players_df, blocks_df


def get_condition_masks(players_df):
    """Create masks for different experimental conditions."""
    masks = {
        'trapped': players_df['first_test_drule_gen'] == '1d',
        'duo': players_df['game_type'] == 'duo',
        'solo': players_df['game_type'] == 'solo',
        'other_1d': players_df['partner_rule_rel'] == 'other-1d',
        'optimal': players_df['partner_rule_rel'] == '2d'
    }
    return masks


def get_condition_subsets(players_df, blocks_df):
    """Get player and block subsets for different conditions."""
    masks = get_condition_masks(players_df)
    
    # Get trapped learner subsets
    trapped_learners = players_df[masks['trapped']]
    trapped_learners_asocial = players_df[masks['trapped'] & masks['solo']]
    trapped_learners_2d = players_df[masks['trapped'] & masks['optimal']]
    trapped_learners_other_1d = players_df[masks['trapped'] & masks['other_1d']]
    
    # Get corresponding blocks
    trapped_learners_blocks = blocks_df[blocks_df['id'].isin(trapped_learners['id'])]
    trapped_learners_asocial_blocks = blocks_df[blocks_df['id'].isin(trapped_learners_asocial['id'])]
    trapped_learners_2d_blocks = blocks_df[blocks_df['id'].isin(trapped_learners_2d['id'])]
    trapped_learners_other_1d_blocks = blocks_df[blocks_df['id'].isin(trapped_learners_other_1d['id'])]
    
    return {
        'all_trapped': (trapped_learners, trapped_learners_blocks),
        'asocial': (trapped_learners_asocial, trapped_learners_asocial_blocks),
        '2d_partner': (trapped_learners_2d, trapped_learners_2d_blocks),
        'other_1d_partner': (trapped_learners_other_1d, trapped_learners_other_1d_blocks)
    }


def calculate_block_mean_points(filtered_blocks):
    """Calculate means and standard errors for each block.
    
    Args:
        filtered_blocks (pd.DataFrame): DataFrame containing block data
        
    Returns:
        tuple: (means, sems, x_pos, phases, phase_blocks, n_blocks_total) arrays containing mean points, 
        standard errors per block, x positions for plotting, phases, blocks per phase, and total blocks
    """
    # Get unique phases and blocks
    phases = sorted(filtered_blocks['phase'].unique())
    phase_blocks = [len(filtered_blocks[filtered_blocks['phase']==p]['block'].unique()) for p in phases]
    n_blocks_total = sum(phase_blocks)

    means = []
    sems = []  # Standard error of the mean
    x_pos = np.arange(n_blocks_total)
    
    for phase_idx, phase in enumerate(phases):
        offset = sum(phase_blocks[:phase_idx])
        for block in range(phase_blocks[phase_idx]):
            curr_blocks = filtered_blocks[
                (filtered_blocks['phase'] == phase) & 
                (filtered_blocks['block'] == block)
            ]
            points = curr_blocks['points']
            if len(points) > 0:
                means.append(points.mean())
                sems.append(points.std() / np.sqrt(len(points)))
            else:
                means.append(np.nan)
                sems.append(np.nan)
            
    return np.array(means), np.array(sems), x_pos, phases, phase_blocks, n_blocks_total


def plot_points_by_block(blocks_df, id_lists, labels, title, output_path, colors=None, phases=None):
    """Plot mean points per block with error bars for multiple conditions.
    
    Args:
        blocks_df (pd.DataFrame): Full blocks dataframe
        id_lists (list): List of lists containing participant IDs for each condition
        labels (list): List of labels for each condition
        title (str): Plot title
        output_path (str): Path to save the figure
        colors (list, optional): List of colors for plotting each condition
        phases (list, optional): List of phases to include in the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get all data to determine phases and blocks
    all_ids = [id for sublist in id_lists for id in sublist]
    all_blocks = blocks_df[blocks_df['id'].isin(all_ids)]
    
    # Plot means and error bars for each condition
    for i, (ids, label) in enumerate(zip(id_lists, labels)):
        condition_blocks = blocks_df[blocks_df['id'].isin(ids)]
        
        # Filter by selected phases
        if phases is not None:
            condition_blocks = condition_blocks[condition_blocks['phase'].isin(phases)]
        
        means, sems, x_pos, phases_found, phase_blocks, n_blocks_total = calculate_block_mean_points(condition_blocks)
        color = colors[i] if colors is not None else None
        plt.errorbar(x_pos, means, yerr=sems, fmt='o-', capsize=5, 
                    label=f"{label} (N={len(ids)})", color=color, linewidth=2, markersize=8)

    # Add vertical lines between phases
    curr_x = 0
    for blocks in phase_blocks[:-1]:
        curr_x += blocks
        plt.axvline(x=curr_x-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add horizontal lines for 1D and 2D performance
    plt.axhline(y=8, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)  # 1D rule performance
    plt.axhline(y=12, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)  # 2D rule performance

    # Customize plot
    plt.xlabel('Block', fontsize=20)
    plt.ylabel('Points Per Block (Mean)', fontsize=20)
    if title:
        plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=14)
    
    # Set y-ticks to go up to 12 and limit y-axis to 13
    ymin = int(np.floor(plt.gca().get_ylim()[0] / 4) * 4)  # Round down to multiple of 4
    plt.yticks(range(ymin, 13, 4), fontsize=18)  # From min to 12 by steps of 4
    plt.ylim(top=13)  # Set upper limit to 13

    # Set x-ticks
    block_labels = []
    for phase_idx, phase in enumerate(phases_found):
        for i in range(1, phase_blocks[phase_idx] + 1):
            if phase % 2 == 0:  # Learn phases (0, 2)
                block_labels.append(f'L{i}')
            else:  # Test phases (1, 3)
                block_labels.append(f'T{i}')
    plt.xticks(range(n_blocks_total), block_labels, fontsize=18)
    
    # Make tick marks thicker
    plt.tick_params(width=1.5)
    
    # Make axes lines thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()


def plot_cumulative_points_by_block(blocks_df, id_lists, labels, title, output_path, colors=None, phases=None):
    """Plot cumulative mean points across blocks with error bars for multiple conditions.
    
    Args:
        blocks_df (pd.DataFrame): Full blocks dataframe
        id_lists (list): List of lists containing participant IDs for each condition
        labels (list): List of labels for each condition
        title (str): Plot title
        output_path (str): Path to save the figure
        colors (list, optional): List of colors for plotting each condition
        phases (list, optional): List of phases to include in the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot cumulative means and error bars for each condition
    for i, (ids, label) in enumerate(zip(id_lists, labels)):
        condition_blocks = blocks_df[blocks_df['id'].isin(ids)]
        
        # Filter by selected phases
        if phases is not None:
            condition_blocks = condition_blocks[condition_blocks['phase'].isin(phases)]
            
        means, sems, x_pos, phases_found, phase_blocks, n_blocks_total = calculate_block_mean_points(condition_blocks)
        
        # Calculate cumulative means and sems
        cumulative_means = np.nancumsum(means)
        cumulative_sems = np.sqrt(np.nancumsum(np.square(sems)))  # Error propagation for independent variables
        
        color = colors[i] if colors is not None else None
        plt.errorbar(x_pos, cumulative_means, yerr=cumulative_sems, fmt='o-', capsize=5,
                    label=f"{label} (N={len(ids)})", color=color, linewidth=2, markersize=8)

    # Add vertical lines between phases
    curr_x = 0
    for blocks in phase_blocks[:-1]:
        curr_x += blocks
        plt.axvline(x=curr_x-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    # Customize plot
    plt.xlabel('Block', fontsize=20)
    plt.ylabel('Cumulative Points (Mean)', fontsize=20)
    if title:
        plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=14)
    
    # Set integer y-ticks
    ymax = int(np.ceil(plt.gca().get_ylim()[1]))
    plt.yticks(range(0, ymax + 1, 10), fontsize=18)  # From 0 to max by steps of 10
    
    # Set x-ticks
    block_labels = []
    for phase_idx, phase in enumerate(phases_found):
        for i in range(1, phase_blocks[phase_idx] + 1):
            if phase % 2 == 0:  # Learn phases (0, 2)
                block_labels.append(f'L{i}')
            else:  # Test phases (1, 3)
                block_labels.append(f'T{i}')
    plt.xticks(range(n_blocks_total), block_labels, fontsize=18)
    
    # Make tick marks thicker
    plt.tick_params(width=1.5)
    
    # Make axes lines thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()


def plot_points_by_dataset_grid(blocks_df, condition_data, output_path, phase_type='all', colors=None):
    """Create a grid of plots showing points per block for each dataset.
    
    Args:
        blocks_df (pd.DataFrame): Full blocks dataframe
        condition_data (dict): Dictionary of condition data
        output_path (str): Path to save the figure
        phase_type (str): Type of phases to plot ('all', 'learning', 'test')
        colors (dict): Dictionary of colors for each condition
    """
    # Set up the grid: 2 rows x 3 columns for 6 datasets
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, dataset in enumerate(PUBLICATION_DATASETS):
        ax = axes[i]
        
        # Get phases for this dataset
        dataset_phases_info = DATASET_PHASES[dataset]
        if phase_type == 'learning':
            phases_to_plot = dataset_phases_info['learning']
        elif phase_type == 'test':
            phases_to_plot = dataset_phases_info['test']
        else:  # 'all'
            # For dataset 5, exclude partner prediction phase when plotting all phases
            if dataset == '5':
                phases_to_plot = dataset_phases_info['learning'] + dataset_phases_info['test']
                phases_to_plot.sort()  # Ensure phases are in order: [0, 1, 3, 4]
            else:
                phases_to_plot = None
        
        # Filter data for this dataset
        dataset_blocks = blocks_df[blocks_df['dataset'] == dataset]
        
        # Plot each condition for this dataset
        for condition_name in ['asocial', '2d_partner', 'other_1d_partner']:
            condition_players = condition_data[condition_name][0]
            condition_dataset_players = condition_players[condition_players['dataset'] == dataset]
            
            if len(condition_dataset_players) == 0:
                continue
                
            condition_blocks = dataset_blocks[dataset_blocks['id'].isin(condition_dataset_players['id'])]
            
            # Filter by phases if specified
            if phases_to_plot is not None:
                condition_blocks = condition_blocks[condition_blocks['phase'].isin(phases_to_plot)]
            
            if len(condition_blocks) == 0:
                continue
                
            means, sems, x_pos, phases_found, phase_blocks, n_blocks_total = calculate_block_mean_points(condition_blocks)
            
            # Get condition label and color
            label_map = {
                'asocial': 'Asocial',
                '2d_partner': '2D Partner', 
                'other_1d_partner': 'Other-1D Partner'
            }
            label = label_map[condition_name]
            color = colors[condition_name] if colors else None
            
            ax.errorbar(x_pos, means, yerr=sems, fmt='o-', capsize=5,
                       label=f"{label} (N={len(condition_dataset_players)})", 
                       color=color, linewidth=2, markersize=6)
        
        # Add vertical lines between phases if plotting all phases
        if phases_to_plot is None and len(condition_blocks) > 0:
            curr_x = 0
            for blocks in phase_blocks[:-1]:
                curr_x += blocks
                ax.axvline(x=curr_x-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add horizontal reference lines
        ax.axhline(y=8, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # 1D rule performance
        ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # 2D rule performance
        
        # Customize subplot
        ax.set_title(f'Dataset {dataset}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 13)
        
        # Set x-ticks
        if len(condition_blocks) > 0:
            block_labels = []
            for phase_idx, phase in enumerate(phases_found):
                for j in range(1, phase_blocks[phase_idx] + 1):
                    # For dataset 5, handle the unique phase structure
                    if dataset == '5':
                        if phase in [0, 3]:  # Learning phases for dataset 5
                            block_labels.append(f'L{j}')
                        elif phase in [1, 4]:  # Test phases for dataset 5
                            block_labels.append(f'T{j}')
                    else:
                        # Standard structure for other datasets
                        if phase % 2 == 0:  # Learning phases (0, 2)
                            block_labels.append(f'L{j}')
                        else:  # Test phases (1, 3)
                            block_labels.append(f'T{j}')
            ax.set_xticks(range(n_blocks_total))
            ax.set_xticklabels(block_labels, fontsize=10)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(fontsize=10, loc='upper left')
    
    # Add overall title and labels
    phase_title_map = {
        'all': 'All Phases',
        'learning': 'Learning Phases Only', 
        'test': 'Test Phases Only'
    }
    fig.suptitle(f'Trapped Learners: Points Per Block ({phase_title_map[phase_type]})', 
                 fontsize=16, fontweight='bold')
    
    # Add common axis labels
    fig.text(0.5, 0.04, 'Block', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Points Per Block (Mean)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1, left=0.1)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()


def plot_cumulative_points_by_dataset_grid(blocks_df, condition_data, output_path, colors=None):
    """Create a grid of plots showing cumulative points for each dataset.
    
    Args:
        blocks_df (pd.DataFrame): Full blocks dataframe
        condition_data (dict): Dictionary of condition data
        output_path (str): Path to save the figure
        colors (dict): Dictionary of colors for each condition
    """
    # Set up the grid: 2 rows x 3 columns for 6 datasets
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, dataset in enumerate(PUBLICATION_DATASETS):
        ax = axes[i]
        
        # Get phases for this dataset
        dataset_phases_info = DATASET_PHASES[dataset]
        # For dataset 5, exclude partner prediction phase
        if dataset == '5':
            phases_to_plot = dataset_phases_info['learning'] + dataset_phases_info['test']
            phases_to_plot.sort()  # Ensure phases are in order: [0, 1, 3, 4]
        else:
            phases_to_plot = None
        
        # Filter data for this dataset
        dataset_blocks = blocks_df[blocks_df['dataset'] == dataset]
        
        # Plot each condition for this dataset
        for condition_name in ['asocial', '2d_partner', 'other_1d_partner']:
            condition_players = condition_data[condition_name][0]
            condition_dataset_players = condition_players[condition_players['dataset'] == dataset]
            
            if len(condition_dataset_players) == 0:
                continue
                
            condition_blocks = dataset_blocks[dataset_blocks['id'].isin(condition_dataset_players['id'])]
            
            # Filter by phases if specified
            if phases_to_plot is not None:
                condition_blocks = condition_blocks[condition_blocks['phase'].isin(phases_to_plot)]
            
            if len(condition_blocks) == 0:
                continue
                
            means, sems, x_pos, phases_found, phase_blocks, n_blocks_total = calculate_block_mean_points(condition_blocks)
            
            # Calculate cumulative means and sems
            cumulative_means = np.nancumsum(means)
            cumulative_sems = np.sqrt(np.nancumsum(np.square(sems)))  # Error propagation
            
            # Get condition label and color
            label_map = {
                'asocial': 'Asocial',
                '2d_partner': '2D Partner', 
                'other_1d_partner': 'Other-1D Partner'
            }
            label = label_map[condition_name]
            color = colors[condition_name] if colors else None
            
            ax.errorbar(x_pos, cumulative_means, yerr=cumulative_sems, fmt='o-', capsize=5,
                       label=f"{label} (N={len(condition_dataset_players)})", 
                       color=color, linewidth=2, markersize=6)
        
        # Add vertical lines between phases
        if len(condition_blocks) > 0:
            curr_x = 0
            for blocks in phase_blocks[:-1]:
                curr_x += blocks
                ax.axvline(x=curr_x-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Customize subplot
        ax.set_title(f'Dataset {dataset}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks
        if len(condition_blocks) > 0:
            block_labels = []
            for phase_idx, phase in enumerate(phases_found):
                for j in range(1, phase_blocks[phase_idx] + 1):
                    # For dataset 5, handle the unique phase structure
                    if dataset == '5':
                        if phase in [0, 3]:  # Learning phases for dataset 5
                            block_labels.append(f'L{j}')
                        elif phase in [1, 4]:  # Test phases for dataset 5
                            block_labels.append(f'T{j}')
                    else:
                        # Standard structure for other datasets
                        if phase % 2 == 0:  # Learning phases (0, 2)
                            block_labels.append(f'L{j}')
                        else:  # Test phases (1, 3)
                            block_labels.append(f'T{j}')
            ax.set_xticks(range(n_blocks_total))
            ax.set_xticklabels(block_labels, fontsize=10)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(fontsize=10, loc='upper left')
    
    # Add overall title and labels
    fig.suptitle('Trapped Learners: Cumulative Points by Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Add common axis labels
    fig.text(0.5, 0.04, 'Block', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Cumulative Points (Mean)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1, left=0.1)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()


def save_analysis_summary(condition_data, output_dir):
    """Save a summary of the analysis to a markdown file."""
    summary_path = os.path.join(output_dir, 'analysis_summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# Points Per Block Analysis Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This analysis examines participant performance (points per block) across different ")
        f.write("phases of the dyadic learning trap experiment, focusing on trapped learners ")
        f.write("and their performance under different social learning conditions.\n\n")
        
        f.write("## Conditions Analyzed\n\n")
        for condition_name, (players, blocks) in condition_data.items():
            f.write(f"### {condition_name}\n")
            f.write(f"- Participants: {len(players)}\n")
            f.write(f"- Blocks: {len(blocks)}\n")
            if len(players) > 0:
                f.write(f"- Datasets represented: {sorted(players['dataset'].unique())}\n")
            f.write("\n")
        
        f.write("## Output Files\n\n")
        f.write("- `points_per_block_by_dataset.png`: Grid plot showing points per block for each dataset (excluding partner prediction phase for dataset 5)\n")
        f.write("- `cumulative_points_by_dataset.png`: Grid plot showing cumulative points across blocks for each dataset\n")
        f.write("\n")
        f.write("Each grid contains 6 subplots (2x3), one for each dataset in PUBLICATION_DATASETS.\n")
        f.write("Dataset-specific phase structures are used (e.g., dataset 5 has different learning/test phases).\n\n")
        
        f.write("## Performance Benchmarks\n\n")
        f.write("- **1D Rule Performance**: 8 points per block (gray dashed line)\n")
        f.write("- **2D Rule Performance**: 12 points per block (gray dashed line)\n")
        f.write("\n")
        
        f.write("## Dataset-Specific Phase Structures\n\n")
        f.write("Different datasets have different phase structures:\n\n")
        for dataset in PUBLICATION_DATASETS:
            phases_info = DATASET_PHASES[dataset]
            f.write(f"### Dataset {dataset}\n")
            f.write(f"- Total phases: {phases_info['total']}\n")
            f.write(f"- Learning phases: {phases_info['learning']}\n")
            f.write(f"- Test phases: {phases_info['test']}\n")
            if 'partner_prediction' in phases_info:
                f.write(f"- Partner prediction phase: {phases_info['partner_prediction']}\n")
            f.write("\n")
        
        f.write("## Block Labels\n\n")
        f.write("- **L1-L4**: Learning blocks\n")
        f.write("- **T1-T2**: Test blocks\n")
        f.write("- Phase numbers determine which learning/test phase blocks belong to\n")


def main():
    """Main analysis function."""
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    players_df, blocks_df = load_data()
    
    # Get condition subsets
    print("\nPreparing condition subsets...")
    condition_data = get_condition_subsets(players_df, blocks_df)
    
    # Print condition summaries
    print("\nCondition summaries:")
    for condition_name, (players, blocks) in condition_data.items():
        print(f"  {condition_name}: {len(players)} participants, {len(blocks)} blocks")
    
    # Define colors for consistent plotting
    colors = {
        'asocial': 'C0',  # Blue
        '2d_partner': 'C1',  # Orange
        'other_1d_partner': 'C2',  # Green
        'all_trapped': 'C3'  # Red
    }
    
    # Create grid plots for each dataset
    print("\nCreating grid plots by dataset...")
    
    # Plot 1: Points per block
    print("  - Points per block grid...")
    plot_points_by_dataset_grid(
        blocks_df,
        condition_data,
        os.path.join(output_dir, 'points_per_block_by_dataset.svg'),
        phase_type='all',
        colors=colors
    )
    
    # Plot 2: Cumulative points
    print("  - Cumulative points grid...")
    plot_cumulative_points_by_dataset_grid(
        blocks_df,
        condition_data,
        os.path.join(output_dir, 'cumulative_points_by_dataset.svg'),
        colors=colors
    )
    
    # Save analysis summary
    print("Saving analysis summary...")
    save_analysis_summary(condition_data, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()