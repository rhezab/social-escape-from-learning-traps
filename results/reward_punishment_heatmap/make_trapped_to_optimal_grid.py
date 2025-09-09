#!/usr/bin/env python3
"""
Generate Trapped-to-Optimal Heatmap Grid

This script generates a 2x3 grid of heatmaps with marginals showing reward vs punishment patterns
for participants who were trapped in phase 1 but achieved optimal performance in phase 2.

Author: Generated with Claude Code
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append('../../')
from constants import PUBLICATION_DATASETS, DATASET_PHASES

# Import heatmap utility functions
from heatmap import get_reward_punishment_counts, plot_heatmap_with_marginals_grid


def load_data():
    """Load the preprocessed data files."""
    players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
    blocks_df = pd.read_csv('../../preprocessing/outputs/blocks_all_filtered.csv')
    return players_df, blocks_df


def main():
    """Generate the trapped-to-optimal heatmap grid."""
    print("Loading data...")
    players_df, blocks_df = load_data()
    
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
    plt.rcParams['ps.fonttype'] = 42   # Force TrueType fonts
    
    # Create 2x3 figure with GridSpec
    fig = plt.figure(figsize=(18, 12))
    outer_gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    print("Generating heatmaps for trapped → optimal participants...")
    
    # Process each dataset
    for idx, dataset in enumerate(PUBLICATION_DATASETS):
        row = idx // 3
        col = idx % 3
        panel_letter = chr(ord('A') + idx)
        
        # Create inner GridSpec for this heatmap with marginals
        inner_gs = outer_gs[row, col].subgridspec(2, 2, 
                                                  height_ratios=[1, 4], 
                                                  width_ratios=[4, 1],
                                                  hspace=0.05,
                                                  wspace=0.05)
        
        # Get the second learning phase from constants
        phase = DATASET_PHASES[dataset]['learning'][1]
        
        # Create mask for trapped → optimal participants
        mask = (
            (players_df['dataset'] == dataset) & 
            (players_df['first_test_drule_gen'] == '1d') &  # Trapped in phase 1
            (players_df['game_type'] == 'duo') &  # Paired participants
            ((players_df['partner_rule_rel'] == '2d') |  # With informative partner
             (players_df['partner_rule_rel'] == 'other-1d')) &
            (players_df['second_test_drule_gen'] == '2d')  # Achieved optimal in phase 2
        )
        
        # Get participant IDs
        ids = players_df[mask]['id'].values
        
        if len(ids) > 0:
            # Get reward and punishment counts
            punishment_counts, rewards_counts = get_reward_punishment_counts(ids, phase, blocks_df)
            
            # Create title with panel letter
            if dataset == 'sim':
                title = f'$\\mathbf{{{panel_letter}}}$: Simulation (N = {len(ids)})'
            else:
                title = f'$\\mathbf{{{panel_letter}}}$: Experiment {dataset} (N = {len(ids)})'
            
            # Define bin edges to have integers at the center (same as plot_rewards_vs_punishment_heatmap)
            x_bins = np.linspace(-0.5, 23.5, 25)  # Bins centered at integers 0-23
            y_bins = np.linspace(-1, 49, 26)  # Bins centered at even integers 0, 2, 4, ..., 48
            
            # Calculate bin centers for ticks
            x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
            y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
            
            # Limits
            x_lim = (0, 6.5)
            y_lim = (30, 49)
            
            # Plot heatmap with marginals using modified approach
            # Get axes directly from the plot function
            axes_dict = plot_heatmap_with_marginals_grid(
                np.array(punishment_counts), 
                np.array(rewards_counts),
                fig=fig,
                gs=inner_gs,
                x_bin_edges=x_bins, 
                y_bin_edges=y_bins,
                xlim=x_lim, 
                ylim=y_lim,
                title=None,  # We'll add title manually with better positioning
                xlabel="Dangerous bees approached (/16)",
                ylabel="Friendly bees approached (/48)",
                show_marginal_ticks=False,
                cmap='Greys',
                hline=32,
                xticks=x_bin_centers,
                yticks=y_bin_centers,
                cbar_tick_step=10
            )
            
            # Adjust font sizes for grid layout
            ax_heat = axes_dict['ax_heat']
            ax_histx = axes_dict['ax_histx']
            
            # Reduce axis label font sizes
            ax_heat.set_xlabel("Dangerous bees approached (/16)", fontsize=12)
            ax_heat.set_ylabel("Friendly bees approached (/48)", fontsize=12)
            
            # Reduce tick label font sizes
            ax_heat.tick_params(axis='both', which='major', labelsize=10)
            
            # Add title to the top marginal histogram with better positioning
            ax_histx.set_title(title, fontsize=15, pad=10)
        else:
            # No data for this condition - create empty subplot
            ax = fig.add_subplot(inner_gs[:, :])
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            if dataset == 'sim':
                title = f'$\\mathbf{{{panel_letter}}}$: Simulation'
            else:
                title = f'$\\mathbf{{{panel_letter}}}$: Experiment {dataset}'
            ax.set_title(title, fontsize=15)
    
    # Add overall title
    # fig.suptitle('Exploration Patterns of Trapped Learners Who Learned 2D Rule via Social Learning', 
    #             fontsize=20, y=0.94)
    
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Save figure in both formats
    svg_path = outputs_dir / "trapped_to_optimal_grid.svg"
    pdf_path = outputs_dir / "trapped_to_optimal_grid.pdf"
    
    print(f"Saving figure to {svg_path} and {pdf_path}")
    fig.savefig(svg_path, bbox_inches='tight', transparent=True)
    fig.savefig(pdf_path, bbox_inches='tight', transparent=True)
    
    plt.close(fig)
    
    print("✓ Trapped-to-optimal heatmap grid created successfully!")


if __name__ == "__main__":
    main()