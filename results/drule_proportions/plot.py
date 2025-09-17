import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import global constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import PUBLICATION_DATASETS

def plot_stacked_proportions(prop_2d, prop_neither, prop_1d, title, ax=None, show_legend=True, 
                           ci_2d=None, ci_1d=None):
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
    ci_2d, ci_1d : list of tuples, optional
        Confidence intervals for 2D and 1D proportions. Each should be a list of 
        (lower, upper) tuples, one for each bar.
        
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
    
    # Add confidence interval error bars
    if ci_1d is not None:
        # Position error bars at center of 1D bars
        y_1d = [prop_1d[i] / 2 for i in range(len(prop_1d))]
        yerr_1d = [[y_1d[i] - ci_1d[i][0], ci_1d[i][1] - y_1d[i]] for i in range(len(prop_1d))]
        yerr_1d = np.array(yerr_1d).T  # Transpose for matplotlib format
        ax.errorbar(x, y_1d, yerr=yerr_1d, fmt='none', color='black', 
                   capsize=3, capthick=1, linewidth=1)
    
    if ci_2d is not None:
        # Position error bars at bottom of 2D bars (top of neither segment)
        bottom_2d = [prop_1d[i] + prop_neither[i] for i in range(len(prop_2d))]
        y_2d = bottom_2d  # Position at bottom edge of 2D bars
        
        # Error bars show CI for the 2D proportion, positioned at the bottom of the 2D segment
        yerr_lower = [prop_2d[i] - ci_2d[i][0] for i in range(len(prop_2d))]  # How far down from baseline
        yerr_upper = [ci_2d[i][1] - prop_2d[i] for i in range(len(prop_2d))]  # How far up from baseline  
        yerr_2d = [yerr_lower, yerr_upper]
        
        ax.errorbar(x, y_2d, yerr=yerr_2d, fmt='none', color='black',
                   capsize=3, capthick=1, linewidth=1)
    
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
    if ax.figure is not None:
        ax.figure.tight_layout()
    
    return ax


def create_drule_proportions_bar_plot(results_df):
    """
    Create the main decision rule proportions bar plot showing proportions 
    across datasets and conditions.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing proportion data with columns: dataset, condition, 
        block, prop_2d, prop_1d, prop_neither, N
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
    plt.rcParams['ps.fonttype'] = 42   # Force TrueType fonts

    # Get unique datasets from the data (handles both sim and empirical)
    unique_datasets = sorted(results_df['dataset'].unique())
    
    # Create figure and axes with reduced horizontal spacing
    fig, axs = plt.subplots(6, 4, figsize=(24, 40),  # Reduced height from 48 to 24
                           gridspec_kw={
                               'wspace': 0.3,  # reduced horizontal spacing between subplots
                               'hspace': 0.8,  # reduced vertical spacing between rows
                               'width_ratios': [1, 1, 1, 1],  # equal width for all subplots
                               'left': 0.15,   # left margin
                               'right': 0.95,  # right margin
                           })
    
    # For each dataset (row)
    for dataset_idx, dataset in enumerate(PUBLICATION_DATASETS):
        # Get the axes for this row
        ax1, ax2, ax3, ax4 = axs[dataset_idx]

        for condition, ax in zip(['first', '2d', 'other_1d', 'asocial'], [ax1, ax2, ax3, ax4]):
            # Check if this dataset/condition combination exists in the data
            this_df = results_df[(results_df['dataset'] == dataset) & (results_df['condition'] == condition)]
            
            if len(this_df) == 0:
                # No data for this combination - leave subplot blank
                ax.axis('off')
                continue
                
            prop_2d = this_df['prop_2d'].values
            prop_1d = this_df['prop_1d'].values
            prop_neither = this_df['prop_neither'].values
            N = this_df['N'].values[0]

            title_dict = {
                'first': 'First Learning Phase',
                '2d': '2D partner',
                'other_1d': 'Other-1D partner',
                'asocial': 'Asocial control'
            }

            title = title_dict[condition] + f' (N = {N})'

            ax = plot_stacked_proportions(
                prop_2d, prop_neither, prop_1d,
                ax=ax,
                title=title,
                show_legend=False
            )
        
        # Adjust the position of the plots in this row to create more space after the first plot
        extra_space = 0.05
        pos2 = ax2.get_position()
        ax2.set_position([pos2.x0 + extra_space, pos2.y0, pos2.width, pos2.height])
        pos3 = ax3.get_position()
        ax3.set_position([pos3.x0 + extra_space, pos3.y0, pos3.width, pos3.height])
        pos4 = ax4.get_position()
        ax4.set_position([pos4.x0 + extra_space, pos4.y0, pos4.width, pos4.height])

    return fig


def create_first_test_overview_plot(test_results_df, show_cis=False):
    """
    Create overview plot showing first test phase decision rule proportions 
    across all datasets for all participants.
    
    Parameters:
    -----------
    test_results_df : pandas.DataFrame
        DataFrame containing test proportion data
    show_cis : bool, optional
        Whether to show confidence intervals on the bars (default: False)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
    plt.rcParams['ps.fonttype'] = 42   # Force TrueType fonts

    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get first test phase data for all participants (condition='first', phase='first_gen')
    first_test_data = test_results_df[
        (test_results_df['condition'] == 'first') & 
        (test_results_df['test_phase_type'] == 'first_gen')
    ]
    
    # Prepare data for plotting
    datasets = []
    prop_2d_values = []
    prop_1d_values = []
    prop_neither_values = []
    n_values = []
    ci_2d_values = []
    ci_1d_values = []
    
    for dataset in PUBLICATION_DATASETS:
        dataset_data = first_test_data[first_test_data['dataset'] == dataset]
        
        if len(dataset_data) > 0:
            row = dataset_data.iloc[0]
            datasets.append(dataset)
            prop_2d_values.append(row['prop_2d'])
            prop_1d_values.append(row['prop_1d'])
            prop_neither_values.append(row['prop_neither'])
            n_values.append(row['N'])
            
            # Extract CIs if requested
            if show_cis:
                ci_2d_values.append((row['ci_2d_lower'], row['ci_2d_upper']))
                ci_1d_values.append((row['ci_1d_lower'], row['ci_1d_upper']))
    
    # Create x positions for bars with reduced spacing
    x_spacing = 0.8
    x = [i * x_spacing for i in range(len(datasets))]
    
    # Plot stacked bars with reduced width
    bar_width = 0.6
    ax.bar(x, prop_1d_values, width=bar_width, color='C0', label='1D', edgecolor='black')
    ax.bar(x, prop_neither_values, bottom=prop_1d_values, width=bar_width, color='white', label='Neither', edgecolor='black')
    ax.bar(x, prop_2d_values, bottom=[prop_1d_values[i] + prop_neither_values[i] for i in range(len(prop_2d_values))], 
           width=bar_width, color='C1', label='2D', edgecolor='black')
    
    # Add confidence interval error bars if requested
    if show_cis and ci_1d_values and ci_2d_values:
        # Check for overlap between 1D and 2D CIs for each bar
        x_1d = []
        x_2d = []
        for i in range(len(prop_1d_values)):
            # 1D CI upper bound (in absolute position)
            ci_1d_upper = ci_1d_values[i][1]
            # 2D CI lower bound (in absolute position from bottom of bar)
            bottom_2d_pos = prop_1d_values[i] + prop_neither_values[i]
            ci_2d_lower_abs = bottom_2d_pos - (ci_2d_values[i][1] - prop_2d_values[i])
            
            # Check if they overlap: 1D upper extends into 2D lower region
            if ci_1d_upper > ci_2d_lower_abs:
                # Apply offset to avoid overlap
                x_1d.append(x[i] - 0.05)
                x_2d.append(x[i] + 0.05)
            else:
                # No overlap, use original x position
                x_1d.append(x[i])
                x_2d.append(x[i])
        
        # Plot 1D CIs
        y_1d = [prop_1d_values[i] / 2 for i in range(len(prop_1d_values))]
        yerr_1d = [[y_1d[i] - ci_1d_values[i][0], ci_1d_values[i][1] - y_1d[i]] for i in range(len(prop_1d_values))]
        yerr_1d = np.array(yerr_1d).T  # Transpose for matplotlib format
        ax.errorbar(x_1d, y_1d, yerr=yerr_1d, fmt='none', color='black', 
                   capsize=3, capthick=1, linewidth=1)
        
        # Plot 2D CIs
        bottom_2d = [prop_1d_values[i] + prop_neither_values[i] for i in range(len(prop_2d_values))]
        y_2d = bottom_2d  # Position at bottom edge of 2D bars
        
        # Error bars show CI for the 2D proportion, positioned at the bottom of the 2D segment
        yerr_lower = [ci_2d_values[i][1] - prop_2d_values[i] for i in range(len(prop_2d_values))]  # Use CI upper for downward extension
        yerr_upper = [prop_2d_values[i] - ci_2d_values[i][0] for i in range(len(prop_2d_values))]  # Use CI lower for upward extension
        yerr_2d = [yerr_lower, yerr_upper]
        
        ax.errorbar(x_2d, y_2d, yerr=yerr_2d, fmt='none', color='black',
                   capsize=3, capthick=1, linewidth=1)
    elif show_cis:
        # Handle case where only one CI type is available
        if ci_1d_values:
            y_1d = [prop_1d_values[i] / 2 for i in range(len(prop_1d_values))]
            yerr_1d = [[y_1d[i] - ci_1d_values[i][0], ci_1d_values[i][1] - y_1d[i]] for i in range(len(prop_1d_values))]
            yerr_1d = np.array(yerr_1d).T
            ax.errorbar(x, y_1d, yerr=yerr_1d, fmt='none', color='black', 
                       capsize=3, capthick=1, linewidth=1)
        
        if ci_2d_values:
            bottom_2d = [prop_1d_values[i] + prop_neither_values[i] for i in range(len(prop_2d_values))]
            y_2d = bottom_2d
            yerr_lower = [ci_2d_values[i][1] - prop_2d_values[i] for i in range(len(prop_2d_values))]  # Use CI upper for downward extension
            yerr_upper = [prop_2d_values[i] - ci_2d_values[i][0] for i in range(len(prop_2d_values))]  # Use CI lower for upward extension
            yerr_2d = [yerr_lower, yerr_upper]
            ax.errorbar(x, y_2d, yerr=yerr_2d, fmt='none', color='black',
                       capsize=3, capthick=1, linewidth=1)
    
    # Create x-tick labels with N values
    xtick_labels = []
    for dataset, n in zip(datasets, n_values):
        if dataset == 'sim':
            dataset_title = 'Simulation'
        else:
            dataset_title = f'Experiment {dataset}' 
        xtick_labels.append(f"{dataset_title}\n(N = {n})")
    
    # Customize the plot with larger font sizes
    # ax.set_xlabel('Dataset', fontsize=20)
    ax.set_ylabel('Proportion', fontsize=20)
    ax.set_ylim(0, 1)
    ax.set_title('Decision Rule Proportions after First Learning Phase', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=18)
    
    plt.tight_layout()
    return fig


def get_significance_stars(p_value):
    """
    Convert p-value to significance asterisks.
    it 
    Parameters:
    -----------
    p_value : float or None
        P-value from statistical test
        
    Returns:
    --------
    str : Asterisks indicating significance level
    """
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def create_trapped_learner_comparison_plot(test_results_df, show_cis=False):
    """
    Create comparison plot showing trapped learner decision rule proportions
    for baseline (T1) vs different partner conditions (T2).
    
    Parameters:
    -----------
    test_results_df : pandas.DataFrame
        DataFrame containing test proportion data with p-value columns
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
    plt.rcParams['ps.fonttype'] = 42   # Force TrueType fonts

    # Create figure with 2x3 grid for 6 datasets
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()  # Flatten to make indexing easier
    
    # For each dataset
    for dataset_idx, dataset in enumerate(PUBLICATION_DATASETS):
        ax = axs[dataset_idx]
        
        # Get data for this dataset
        dataset_data = test_results_df[test_results_df['dataset'] == dataset]
        
        if len(dataset_data) == 0:
            ax.axis('off')
            continue
            
        # Prepare data for plotting - 4 bars: T1 baseline + 3 T2 conditions
        conditions = ['baseline', '2d', 'other_1d', 'asocial']
        prop_2d_values = []
        prop_1d_values = []
        prop_neither_values = []
        n_values = []
        ci_2d_values = []
        ci_1d_values = []
        
        for condition in conditions:
            if condition == 'baseline':
                # For baseline, calculate combined first_gen proportions for trapped learners
                # Get participants from 2d, other_1d, and asocial conditions
                baseline_conditions = ['2d', 'other_1d', 'asocial']
                baseline_data = dataset_data[
                    (dataset_data['condition'].isin(baseline_conditions)) & 
                    (dataset_data['test_phase_type'] == 'first_gen')
                ]
                
                if len(baseline_data) > 0:
                    # Calculate weighted average proportions
                    total_n = baseline_data['N'].sum()
                    if total_n > 0:
                        prop_2d = (baseline_data['prop_2d'] * baseline_data['N']).sum() / total_n
                        prop_1d = (baseline_data['prop_1d'] * baseline_data['N']).sum() / total_n
                        prop_neither = (baseline_data['prop_neither'] * baseline_data['N']).sum() / total_n
                    else:
                        prop_2d = prop_1d = prop_neither = 0
                    
                    prop_2d_values.append(prop_2d)
                    prop_1d_values.append(prop_1d)
                    prop_neither_values.append(prop_neither)
                    n_values.append(total_n)
                    
                    # No CIs for baseline (aggregated data)
                    if show_cis:
                        ci_2d_values.append(None)
                        ci_1d_values.append(None)
                else:
                    prop_2d_values.append(0)
                    prop_1d_values.append(0)
                    prop_neither_values.append(0)
                    n_values.append(0)
                    
                    # No CIs for missing baseline data
                    if show_cis:
                        ci_2d_values.append(None)
                        ci_1d_values.append(None)
            else:
                # For T2 conditions, use second_gen proportions
                condition_data = dataset_data[
                    (dataset_data['condition'] == condition) & 
                    (dataset_data['test_phase_type'] == 'second_gen')
                ]
                
                if len(condition_data) > 0:
                    row = condition_data.iloc[0]
                    prop_2d_values.append(row['prop_2d'])
                    prop_1d_values.append(row['prop_1d'])
                    prop_neither_values.append(row['prop_neither'])
                    n_values.append(row['N'])
                    
                    # Extract CIs for T2 conditions if requested
                    if show_cis:
                        ci_2d_values.append((row['ci_2d_lower'], row['ci_2d_upper']))
                        ci_1d_values.append((row['ci_1d_lower'], row['ci_1d_upper']))
                else:
                    prop_2d_values.append(0)
                    prop_1d_values.append(0)
                    prop_neither_values.append(0)
                    n_values.append(0)
                    
                    # No CIs for missing T2 data
                    if show_cis:
                        ci_2d_values.append(None)
                        ci_1d_values.append(None)
        
        # Create x positions for bars with reduced spacing
        x_spacing = 0.8
        x = [i * x_spacing for i in range(len(conditions))]
        
        # Plot stacked bars with reduced width
        bar_width = 0.6
        ax.bar(x, prop_1d_values, width=bar_width, color='C0', label='1D', edgecolor='black')
        ax.bar(x, prop_neither_values, bottom=prop_1d_values, width=bar_width, color='white', label='Neither', edgecolor='black')
        ax.bar(x, prop_2d_values, bottom=[prop_1d_values[i] + prop_neither_values[i] for i in range(len(prop_2d_values))], 
               width=bar_width, color='C1', label='2D', edgecolor='black')
        
        # Add confidence interval error bars if requested
        if show_cis and ci_1d_values and ci_2d_values:
            # Check for overlap between 1D and 2D CIs for each bar (skip baseline)
            x_1d = []
            x_2d = []
            valid_1d_cis = []
            valid_2d_cis = []
            valid_1d_props = []
            valid_2d_props = []
            valid_neither_props = []
            
            for i in range(len(prop_1d_values)):
                # Skip baseline (no CIs) and bars with no CI data
                if conditions[i] == 'baseline' or ci_1d_values[i] is None or ci_2d_values[i] is None:
                    continue
                
                # 1D CI upper bound (in absolute position)
                ci_1d_upper = ci_1d_values[i][1]
                # 2D CI lower bound (in absolute position from bottom of bar)
                bottom_2d_pos = prop_1d_values[i] + prop_neither_values[i]
                ci_2d_lower_abs = bottom_2d_pos - (ci_2d_values[i][1] - prop_2d_values[i])
                
                # Check if they overlap: 1D upper extends into 2D lower region
                if ci_1d_upper > ci_2d_lower_abs:
                    # Apply offset to avoid overlap
                    x_1d.append(x[i] - 0.05)
                    x_2d.append(x[i] + 0.05)
                else:
                    # No overlap, use original x position
                    x_1d.append(x[i])
                    x_2d.append(x[i])
                
                # Store valid data for plotting
                valid_1d_cis.append(ci_1d_values[i])
                valid_2d_cis.append(ci_2d_values[i])
                valid_1d_props.append(prop_1d_values[i])
                valid_2d_props.append(prop_2d_values[i])
                valid_neither_props.append(prop_neither_values[i])
            
            if valid_1d_cis:
                # Plot 1D CIs
                y_1d = [valid_1d_props[i] / 2 for i in range(len(valid_1d_props))]
                yerr_1d = [[y_1d[i] - valid_1d_cis[i][0], valid_1d_cis[i][1] - y_1d[i]] for i in range(len(valid_1d_props))]
                yerr_1d = np.array(yerr_1d).T  # Transpose for matplotlib format
                ax.errorbar(x_1d, y_1d, yerr=yerr_1d, fmt='none', color='black', 
                           capsize=3, capthick=1, linewidth=1)
            
            if valid_2d_cis:
                # Plot 2D CIs
                bottom_2d = [valid_1d_props[i] + valid_neither_props[i] for i in range(len(valid_2d_props))]
                y_2d = bottom_2d  # Position at bottom edge of 2D bars
                
                # Error bars show CI for the 2D proportion, positioned at the bottom of the 2D segment
                # Apply corrected logic: yerr_lower uses CI_upper, yerr_upper uses CI_lower
                yerr_lower = [valid_2d_cis[i][1] - valid_2d_props[i] for i in range(len(valid_2d_props))]  # Use CI upper for downward extension
                yerr_upper = [valid_2d_props[i] - valid_2d_cis[i][0] for i in range(len(valid_2d_props))]  # Use CI lower for upward extension
                yerr_2d = [yerr_lower, yerr_upper]
                
                ax.errorbar(x_2d, y_2d, yerr=yerr_2d, fmt='none', color='black',
                           capsize=3, capthick=1, linewidth=1)
        
        # Add significance asterisks for conditions vs asocial (excluding baseline)
        for i, condition in enumerate(conditions):
            if condition in ['asocial', 'baseline']:
                continue  # Skip asocial (can't compare to itself) and baseline
                
            # Get p-values for this condition vs asocial
            # For 2d and other_1d, use second_gen data
            condition_data = dataset_data[
                (dataset_data['condition'] == condition) & 
                (dataset_data['test_phase_type'] == 'second_gen')
            ]
            
            if len(condition_data) > 0:
                row = condition_data.iloc[0]
                
                # Get significance for 2D outcome (most important)
                p_2d = row.get('p_vs_asocial_2d', None)
                stars = get_significance_stars(p_2d)
                
                if stars:
                    # Position asterisks above the bar
                    bar_height = prop_1d_values[i] + prop_neither_values[i] + prop_2d_values[i]
                    y_pos = bar_height + 0.02
                    ax.text(x[i], y_pos, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Create x-tick labels with N values
        condition_labels = {
            'baseline': 'T1: Trapped',
            '2d': 'T2: 2D',
            'other_1d': 'T2: Other-1D',
            'asocial': 'T2: Asocial'
        }
        
        xtick_labels = []
        for condition, n in zip(conditions, n_values):
            xtick_labels.append(f"{condition_labels[condition]}\n(N = {n})")
        
        # Customize the plot with larger font sizes
        ax.set_xlabel('T1: Rule / T2: Partner', fontsize=14, labelpad=10)
        ax.set_ylabel('Proportion', fontsize=14)
        ax.set_ylim(0, 1.1)  # Increased upper limit to provide space for asterisks
        # Create title with bolded panel letter
        panel_letter = chr(ord('A') + dataset_idx)
        if dataset == 'sim':
            dataset_title = 'Simulation'
        else:
            dataset_title = f'Experiment {dataset}' 
        title_with_panel = f'$\\mathbf{{{panel_letter}}}$: {dataset_title}'
        ax.set_title(title_with_panel, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=16)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add individual legend in top left of each subplot
        ax.legend(loc='upper left', fontsize=12)
        
        # Add a dashed vertical line between baseline and T2 conditions (adjusted for new spacing)
        ax.axvline(x=0.5 * x_spacing, color='gray', linestyle='--', alpha=0.7)
        
        # Add panel header (A-F) - commented out, now in title
        # panel_letter = chr(ord('A') + dataset_idx)
        # ax.text(0.02, 1.05, panel_letter, transform=ax.transAxes, fontsize=16, 
        #         weight='bold', va='top', ha='left')
    
    # Individual legends are now added to each subplot, so no need for single legend
    
    # Add overall title with larger font
    # fig.suptitle('Effect of Social Learning on Trapped Learners', fontsize=22, y=0.98)
    
    plt.tight_layout()
    return fig


def create_test_drule_proportions_bar_plot(test_results_df):
    """
    Create test decision rule proportions bar plot showing proportions 
    across datasets and conditions for test phases.
    
    Parameters:
    -----------
    test_results_df : pandas.DataFrame
        DataFrame containing test proportion data with columns: dataset, condition, 
        test_phase_type, prop_2d, prop_1d, prop_neither, N
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
    plt.rcParams['ps.fonttype'] = 42   # Force TrueType fonts

    # Create figure with 2x3 grid for 6 datasets
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()  # Flatten to make indexing easier
    
    # For each dataset
    for dataset_idx, dataset in enumerate(PUBLICATION_DATASETS):
        ax = axs[dataset_idx]
        
        # Get data for this dataset
        dataset_data = test_results_df[test_results_df['dataset'] == dataset]
        
        if len(dataset_data) == 0:
            ax.axis('off')
            continue
            
        # Prepare data for plotting - 4 bars for 4 conditions
        conditions = ['first', '2d', 'other_1d', 'asocial']
        prop_2d_values = []
        prop_1d_values = []
        prop_neither_values = []
        n_values = []
        
        for condition in conditions:
            # For 'first' condition, use first_gen; for others, use second_gen
            if condition == 'first':
                phase_type = 'first_gen'
            else:
                phase_type = 'second_gen'
                
            # Get the specific row for this condition and phase type
            condition_data = dataset_data[
                (dataset_data['condition'] == condition) & 
                (dataset_data['test_phase_type'] == phase_type)
            ]
            
            if len(condition_data) > 0:
                row = condition_data.iloc[0]
                prop_2d_values.append(row['prop_2d'])
                prop_1d_values.append(row['prop_1d'])
                prop_neither_values.append(row['prop_neither'])
                n_values.append(row['N'])
            else:
                # No data for this condition/phase combination
                prop_2d_values.append(0)
                prop_1d_values.append(0)
                prop_neither_values.append(0)
                n_values.append(0)
        
        # Create x positions for bars
        x = range(len(conditions))
        
        # Plot stacked bars
        ax.bar(x, prop_1d_values, color='C0', label='1D', edgecolor='black')
        ax.bar(x, prop_neither_values, bottom=prop_1d_values, color='white', label='Neither', edgecolor='black')
        ax.bar(x, prop_2d_values, bottom=[prop_1d_values[i] + prop_neither_values[i] for i in range(len(prop_2d_values))], 
               color='C1', label='2D', edgecolor='black')
        
        # Create x-tick labels with N values
        condition_labels = {
            'first': 'Test 1',
            '2d': 'T2: 2D',
            'other_1d': 'T2: Other-1D',
            'asocial': 'T2: Asocial'
        }
        
        xtick_labels = []
        for condition, n in zip(conditions, n_values):
            xtick_labels.append(f"{condition_labels[condition]}\n(N = {n})")
        
        # Customize the plot
        ax.set_xlabel('Condition', fontsize=14)
        ax.set_ylabel('Proportion', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_title(f'Dataset {dataset}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add a dashed vertical line between 'first' and other conditions
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add legend to the last subplot
    axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add overall title
    fig.suptitle('Test Decision Rule Proportions by Dataset and Condition', fontsize=18, y=0.98)
    
    plt.tight_layout()
    return fig


def main():
    """
    Load decision rule proportion data and create decision rule proportions bar plot.
    """
    print("Loading decision rule proportion data...")
    
    # Load the results dataframe
    results_df = pd.read_csv('outputs/decision_rule_proportions.csv')
    
    print(f"Loaded {len(results_df)} rows of proportion data")
    print(f"Datasets: {sorted(results_df['dataset'].unique())}")
    print(f"Conditions: {sorted(results_df['condition'].unique())}")
    
    # Create the figure
    print("Creating decision rule proportions bar plot...")
    fig = create_drule_proportions_bar_plot(results_df)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save the figure in both SVG and PDF formats
    svg_output_path = 'outputs/drule-proportions-bar-plot.svg'
    pdf_output_path = 'outputs/drule-proportions-bar-plot.pdf'
    print(f"Saving figure to {svg_output_path} and {pdf_output_path}")
    fig.savefig(svg_output_path, bbox_inches='tight', transparent=True)
    fig.savefig(pdf_output_path, bbox_inches='tight', transparent=True)
    plt.close(fig)
    
    print("Figure created successfully!")
    
    # Also create test proportions figures if test data exists
    test_file = 'outputs/test_decision_rule_proportions.csv'
    if os.path.exists(test_file):
        print("\nLoading test decision rule proportion data...")
        test_results_df = pd.read_csv(test_file)
        
        print(f"Loaded {len(test_results_df)} rows of test proportion data")
        
        # Create the first test overview figure WITHOUT CIs
        print("Creating first test phase overview plot (without CIs)...")
        first_test_fig = create_first_test_overview_plot(test_results_df, show_cis=False)
        
        # Save the first test figure in both SVG and PDF formats
        first_test_svg_path = 'outputs/first_test_overview.svg'
        first_test_pdf_path = 'outputs/first_test_overview.pdf'
        print(f"Saving first test overview figure to {first_test_svg_path} and {first_test_pdf_path}")
        first_test_fig.savefig(first_test_svg_path, bbox_inches='tight', transparent=True)
        first_test_fig.savefig(first_test_pdf_path, bbox_inches='tight', transparent=True)
        plt.close(first_test_fig)
        
        # Create the first test overview figure WITH CIs
        print("Creating first test phase overview plot (with CIs)...")
        first_test_fig_ci = create_first_test_overview_plot(test_results_df, show_cis=True)
        
        # Save the first test figure with CIs in both SVG and PDF formats
        first_test_ci_svg_path = 'outputs/first_test_overview_with_cis.svg'
        first_test_ci_pdf_path = 'outputs/first_test_overview_with_cis.pdf'
        print(f"Saving first test overview figure with CIs to {first_test_ci_svg_path} and {first_test_ci_pdf_path}")
        first_test_fig_ci.savefig(first_test_ci_svg_path, bbox_inches='tight', transparent=True)
        first_test_fig_ci.savefig(first_test_ci_pdf_path, bbox_inches='tight', transparent=True)
        plt.close(first_test_fig_ci)
        
        # Create the trapped learner comparison figure (without CIs)
        print("Creating trapped learner comparison plot (without CIs)...")
        trapped_learner_fig = create_trapped_learner_comparison_plot(test_results_df, show_cis=False)
        
        # Save the trapped learner figure in both SVG and PDF formats
        trapped_learner_svg_path = 'outputs/trapped_learner_comparison.svg'
        trapped_learner_pdf_path = 'outputs/trapped_learner_comparison.pdf'
        print(f"Saving trapped learner comparison figure to {trapped_learner_svg_path} and {trapped_learner_pdf_path}")
        trapped_learner_fig.savefig(trapped_learner_svg_path, bbox_inches='tight', transparent=True)
        trapped_learner_fig.savefig(trapped_learner_pdf_path, bbox_inches='tight', transparent=True)
        plt.close(trapped_learner_fig)
        
        # Create the trapped learner comparison figure (with CIs)
        print("Creating trapped learner comparison plot (with CIs)...")
        trapped_learner_fig_ci = create_trapped_learner_comparison_plot(test_results_df, show_cis=True)
        
        # Save the trapped learner figure with CIs in both SVG and PDF formats
        trapped_learner_ci_svg_path = 'outputs/trapped_learner_comparison_with_cis.svg'
        trapped_learner_ci_pdf_path = 'outputs/trapped_learner_comparison_with_cis.pdf'
        print(f"Saving trapped learner comparison figure with CIs to {trapped_learner_ci_svg_path} and {trapped_learner_ci_pdf_path}")
        trapped_learner_fig_ci.savefig(trapped_learner_ci_svg_path, bbox_inches='tight', transparent=True)
        trapped_learner_fig_ci.savefig(trapped_learner_ci_pdf_path, bbox_inches='tight', transparent=True)
        plt.close(trapped_learner_fig_ci)
        
        # Create the original test figure (for reference)
        print("Creating original test decision rule proportions bar plot...")
        test_fig = create_test_drule_proportions_bar_plot(test_results_df)
        
        # Save the original test figure in both SVG and PDF formats
        test_svg_path = 'outputs/test_decision_rule_proportions.svg'
        test_pdf_path = 'outputs/test_decision_rule_proportions.pdf'
        print(f"Saving original test figure to {test_svg_path} and {test_pdf_path}")
        test_fig.savefig(test_svg_path, bbox_inches='tight', transparent=True)
        test_fig.savefig(test_pdf_path, bbox_inches='tight', transparent=True)
        plt.close(test_fig)
        
        print("All test figures created successfully!")
    else:
        print(f"\nTest data file {test_file} not found, skipping test proportions plots.")


if __name__ == "__main__":
    main()