import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import PUBLICATION_DATASETS

def plot_stacked_proportions(prop_2d, prop_neither, prop_1d, title, ax=None, show_legend=True):
    """
    Create a stacked bar plot showing proportions of decision rule usage across blocks.
    
    Parameters:
    -----------
    prop_2d, prop_neither, prop_1d : array-like
        Proportions for each decision rule type
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    show_legend : bool
        Whether to show legend
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    x = range(1, len(prop_2d) + 1)
    
    # Custom x-tick labels
    x_labels = ['L1', 'L2', 'L3', 'L4', 'T1', 'T2'][:len(prop_2d)]
    
    # Plot stacked bars
    ax.bar(x, prop_1d, color='C0', label='1D', edgecolor='black')
    ax.bar(x, prop_neither, bottom=prop_1d, color='white', label='Neither', edgecolor='black')
    ax.bar(x, prop_2d, bottom=[prop_1d[i] + prop_neither[i] for i in range(len(prop_2d))], 
           color='C1', label='2D', edgecolor='black')
    
    # Add a dashed vertical line between L4 and T1
    if len(x) > 4:  # Make sure we have at least 5 blocks (L1-L4 + T1)
        ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Block', fontsize=22)
    ax.set_ylabel('Proportion', fontsize=22)
    ax.set_ylim(0, 1)  # Set y-axis limits
    if title is not None: 
        ax.set_title(title, fontsize=24, pad=15)  # Added padding between title and plot
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    # Only apply tight_layout if we created a new figure
    if ax is None:
        plt.tight_layout()
    
    return ax


def plot_grouped_stacked_bars(results_df, ax=None):
    """
    Create grouped stacked bar plot comparing high vs low partner prediction performers.
    Shows T1 (first test) and T2 (second test) results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: condition, outcome, high_rate, low_rate, high_n, low_n, p_value
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Set matplotlib parameters for consistent styling
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Define conditions - we'll show aggregated T1 and separate T2 for each partner type
    # Order: T1 (aggregated), T2: 2D, T2: Other-1D
    x_positions = [0, 1, 2]
    x_labels = ['T1: All', 'T2: 2D', 'T2: Other-1D']
    
    x = np.array(x_positions)
    width = 0.35
    
    # Extract data for each condition
    high_data = {'2d': [], '1d': [], 'neither': [], 'n': []}
    low_data = {'2d': [], '1d': [], 'neither': [], 'n': []}
    p_values = {'2d': [], '1d': [], 'neither': []}
    
    # First bar: T1 (aggregated)
    # For T1, all trapped learners show 100% 1D rule
    high_data['2d'].append(0.0)
    high_data['1d'].append(1.0)
    high_data['neither'].append(0.0)
    low_data['2d'].append(0.0)
    low_data['1d'].append(1.0)
    low_data['neither'].append(0.0)
    p_values['2d'].append(1.0)  # No significance for T1
    
    # Get total sample sizes for T1 (combine both partner types)
    total_high_n = 0
    total_low_n = 0
    for partner_type in ['optimal_partner', 'other_1d_partner']:
        cond_df = results_df[results_df['condition'] == partner_type]
        if len(cond_df) > 0:
            total_high_n += cond_df.iloc[0]['high_n']
            total_low_n += cond_df.iloc[0]['low_n']
    high_data['n'].append(total_high_n)
    low_data['n'].append(total_low_n)
    
    # Next bars: T2 for each partner type
    for partner_type in ['optimal_partner', 'other_1d_partner']:
        cond_df = results_df[results_df['condition'] == partner_type]
        
        # Get rates for each outcome
        for outcome in ['2d', '1d', 'neither']:
            row = cond_df[cond_df['outcome'] == outcome].iloc[0]
            high_data[outcome].append(row['high_rate'])
            low_data[outcome].append(row['low_rate'])
            
        # Get p-value for 2D outcome only
        p_values['2d'].append(cond_df[cond_df['outcome'] == '2d'].iloc[0]['p_value'])
        
        # Get sample sizes (same for all outcomes)
        high_data['n'].append(cond_df.iloc[0]['high_n'])
        low_data['n'].append(cond_df.iloc[0]['low_n'])
    
    # Create stacked bars for low performers (F) on the left
    bars_low_1d = ax.bar(x - width/2, low_data['1d'], width,
                        label='1D', color='C0', edgecolor='black')
    bars_low_neither = ax.bar(x - width/2, low_data['neither'], width,
                             bottom=low_data['1d'],
                             label='Neither', color='white', edgecolor='black')
    bars_low_2d = ax.bar(x - width/2, low_data['2d'], width,
                        bottom=[low_data['1d'][i] + low_data['neither'][i] 
                               for i in range(len(x_positions))],
                        label='2D', color='C1', edgecolor='black')
    
    # Create stacked bars for high performers (S) on the right
    bars_high_1d = ax.bar(x + width/2, high_data['1d'], width, 
                          color='C0', edgecolor='black')
    bars_high_neither = ax.bar(x + width/2, high_data['neither'], width,
                               bottom=high_data['1d'], 
                               color='white', edgecolor='black')
    bars_high_2d = ax.bar(x + width/2, high_data['2d'], width,
                         bottom=[high_data['1d'][i] + high_data['neither'][i] 
                                for i in range(len(x_positions))],
                         color='C1', edgecolor='black')
    
    # Add significance stars (only for T2)
    for i in range(len(x_positions)):
        # Only show stars for T2 (indices 1 and 2)
        if i > 0:  # T2 positions
            # Check 2D outcome significance
            if p_values['2d'][i] < 0.001:
                stars = '***'
            elif p_values['2d'][i] < 0.01:
                stars = '**'
            elif p_values['2d'][i] < 0.05:
                stars = '*'
            else:
                stars = ''
            
            if stars:
                # Position stars above the bars
                y_pos = max(1.0, 
                           high_data['1d'][i] + high_data['neither'][i] + high_data['2d'][i],
                           low_data['1d'][i] + low_data['neither'][i] + low_data['2d'][i]) + 0.02
                ax.text(x[i], y_pos, stars, ha='center', va='bottom', fontsize=20)
    
    # Add combined S/F labels with sample sizes
    for i in range(len(x_positions)):
        ax.text(x[i] - width/2, -0.08, f'F\n(N={low_data["n"][i]})', 
                ha='center', va='top', fontsize=16, transform=ax.get_xaxis_transform())
        ax.text(x[i] + width/2, -0.08, f'S\n(N={high_data["n"][i]})', 
                ha='center', va='top', fontsize=16, transform=ax.get_xaxis_transform())
    
    # Customize plot
    ax.set_ylabel('Proportion', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create custom legend
    legend_elements = [
        mpatches.Patch(facecolor='C1', edgecolor='black', label='2D'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Neither'),
        mpatches.Patch(facecolor='C0', edgecolor='black', label='1D')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=18, 
              title='Decision Rule', title_fontsize=20)
    
    # Add vertical dashed line to separate T1 from T2
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Only apply tight_layout if we created a new figure
    if ax is None or ax.figure is None:
        plt.tight_layout()
    
    return fig, ax


def plot_partner_prediction_histogram(ax=None):
    """
    Create histogram of partner prediction performance for trapped learners.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Set matplotlib parameters for consistent styling
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Load the data
    players_df = pd.read_csv('../../preprocessing/outputs/players_df_all_filtered.csv')
    blocks_df = pd.read_csv('../../preprocessing/outputs/blocks_all_filtered.csv')
    
    # Get partner prediction data for trapped learners
    last_prediction_blocks_df = blocks_df[(blocks_df['dataset'] == '5') & 
                                         (blocks_df['phase'] == 2) & 
                                         (blocks_df['block'] == 3)]
    
    # Get trapped learner IDs
    trapped_ids = players_df[players_df['first_test_drule_gen'] == '1d']['id'].unique()
    
    # Get informative partner IDs (2d or other-1d)
    informative_partner_ids = players_df[
        players_df['partner_rule_rel'].isin(['2d', 'other-1d'])
    ]['id'].unique()
    
    # Get intersection: trapped learners with informative partners
    trapped_informative_ids = set(trapped_ids) & set(informative_partner_ids)
    
    # Filter for trapped learners with informative partners only
    trapped_prediction_data = last_prediction_blocks_df[
        last_prediction_blocks_df['id'].isin(trapped_informative_ids)
    ]
    
    # Extract partner prediction scores
    prediction_scores = trapped_prediction_data['partner_predictions_correct'].values
    
    # Create histogram with discrete bins
    bins = np.arange(-0.5, 17.5, 1)  # Bins centered on integers 0-16
    ax.hist(prediction_scores, bins=bins, alpha=0.7, color='skyblue', 
            edgecolor='black', linewidth=1)
    
    # Add threshold line
    threshold = 15
    ax.axvline(x=threshold - 0.5, color='red', linestyle='--', linewidth=2)
    
    # Customize plot
    ax.set_xlabel('Partner Predictions Correct', fontsize=22)
    ax.set_ylabel('Number of Participants', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlim(-0.5, 16.5)
    ax.set_xticks(range(0, 17))  # Integer ticks from 0 to 16
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample size to plot
    n_participants = len(prediction_scores)
    ax.text(0.02, 0.98, f'N = {n_participants}', transform=ax.transAxes, 
            fontsize=18, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Only apply tight_layout if we created a new figure
    if ax is None or ax.figure is None:
        plt.tight_layout()
    
    return fig, ax


def create_partner_prediction_comparison_plot():
    """
    Create and save the grouped stacked bar plot comparing partner prediction performance.
    """
    # Load the results data
    results_df = pd.read_csv('outputs/partner_prediction_results.csv')
    
    # Create the plot
    fig, ax = plot_grouped_stacked_bars(results_df)
    
    # Save the plot
    os.makedirs('outputs', exist_ok=True)
    
    # Save as PDF
    pdf_path = 'outputs/partner_prediction_grouped_comparison.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    
    # Save as SVG
    svg_path = 'outputs/partner_prediction_grouped_comparison.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved SVG: {svg_path}")
    
    plt.close()


def create_partner_prediction_histogram():
    """
    Create and save the histogram of partner prediction performance for trapped learners.
    """
    # Create the plot
    fig, ax = plot_partner_prediction_histogram()
    
    # Save the plot
    os.makedirs('outputs', exist_ok=True)
    
    # Save as PDF
    pdf_path = 'outputs/partner_prediction_histogram_informative_partners.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    
    # Save as SVG
    svg_path = 'outputs/partner_prediction_histogram_informative_partners.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved SVG: {svg_path}")
    
    plt.close()


def create_combined_partner_prediction_figure():
    """
    Create a combined figure with histogram (A) and grouped stacked bars (B).
    """
    # Set matplotlib parameters for consistent styling
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                   gridspec_kw={'wspace': 0.3})
    
    # Plot A: Histogram
    plot_partner_prediction_histogram(ax=ax1)
    # Add panel label A
    ax1.text(-0.12, 1.05, '$\\mathbf{A}$', transform=ax1.transAxes, 
             fontsize=24, va='top', ha='left')
    
    # Plot B: Grouped stacked bars
    results_df = pd.read_csv('outputs/partner_prediction_results.csv')
    plot_grouped_stacked_bars(results_df, ax=ax2)
    # Add panel label B
    ax2.text(-0.08, 1.05, '$\\mathbf{B}$', transform=ax2.transAxes, 
             fontsize=24, va='top', ha='left')
    
    # Save the combined figure
    os.makedirs('outputs', exist_ok=True)
    
    # Save as PDF
    pdf_path = 'outputs/partner_prediction_combined_figure.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    
    # Save as SVG
    svg_path = 'outputs/partner_prediction_combined_figure.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved SVG: {svg_path}")
    
    plt.close()


if __name__ == "__main__":
    create_combined_partner_prediction_figure()
