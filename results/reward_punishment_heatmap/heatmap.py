import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

def plot_heatmap_with_marginals(x, y, x_bin_edges=None, y_bin_edges=None, cmap='viridis', xlim=None, ylim=None, density=False, 
                               title=None, xlabel='X', ylabel='Y', show_marginal_ticks=False,
                               show_unity_line=False, hline=None, vline=None,
                               xticks=None, yticks=None, xticklabels=None, yticklabels=None,
                               cbar_ticks=None, cbar_ticklabels=None, cbar_tick_step=None, filename=None):
    """
    Plots a heatmap with marginal histograms for two 1D arrays using Matplotlib.
    
    Parameters:
    - x: 1D array of x data.
    - y: 1D array of y data.
    - x_bin_edges: Optional array of bin edges for the x-axis. If None, defaults to 30 bins.
    - y_bin_edges: Optional array of bin edges for the y-axis. If None, defaults to 30 bins.
    - cmap: Colormap for the heatmap.
    - xlim: Optional tuple (xmin, xmax) specifying the x-axis limits.
    - ylim: Optional tuple (ymin, ymax) specifying the y-axis limits.
    - density: If True, plot probability density instead of counts.
    - title: Optional title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - show_marginal_ticks: If True, show tick labels on marginal histograms.
    - show_unity_line: If True, add a diagonal unity line (y=x).
    - hline: Optional value for a horizontal line to add to the plot.
    - vline: Optional value for a vertical line to add to the plot.
    - xticks: Optional array of positions for x-axis tick marks.
    - yticks: Optional array of positions for y-axis tick marks.
    - xticklabels: Optional array of labels for x-axis ticks.
    - yticklabels: Optional array of labels for y-axis ticks.
    - cbar_ticks: Optional array of positions for colorbar tick marks.
    - cbar_ticklabels: Optional array of labels for colorbar ticks.
    - cbar_tick_step: Optional step size for colorbar ticks (e.g., 3 will create ticks at 0, 3, 6, ..., max).
    - filename: Optional filename/path to save the plot as SVG. If None, displays the plot instead.
    """
    # Set default bin edges if not provided.
    if x_bin_edges is None:
        x_min = np.min(x) if xlim is None else xlim[0]
        x_max = np.max(x) if xlim is None else xlim[1]
        x_bin_edges = np.linspace(x_min, x_max, 31)
    if y_bin_edges is None:
        y_min = np.min(y) if ylim is None else ylim[0]
        y_max = np.max(y) if ylim is None else ylim[1]
        y_bin_edges = np.linspace(y_min, y_max, 31)
    
    # Compute the 2D histogram.
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bin_edges, y_bin_edges], density=density)
    
    # Generate colorbar ticks if cbar_tick_step is specified but cbar_ticks is not
    if cbar_tick_step is not None and cbar_ticks is None:
        max_value = np.max(H)
        # Generate ticks from 0 to max_value with the specified step
        cbar_ticks = list(np.arange(0, max_value, cbar_tick_step))
        # Add the max value if it's not already included and the list is not empty
        if len(cbar_ticks) > 0 and cbar_ticks[-1] != max_value:
            cbar_ticks.append(max_value)
        # If cbar_ticks is empty (max_value is 0 or very small), create a minimal tick list
        elif len(cbar_ticks) == 0:
            cbar_ticks = [0, max_value] if max_value > 0 else [0]
    
    # Compute marginal counts or densities.
    histx = np.sum(H, axis=1)  # Sum over y bins for each x bin.
    histy = np.sum(H, axis=0)  # Sum over x bins for each y bin.
    
    # If density is True, normalize the marginals to make them proper PDFs
    if density:
        # Normalize marginals to integrate to 1
        dx = np.diff(xedges)
        dy = np.diff(yedges)
        histx = histx / (np.sum(histx) * dx)
        histy = histy / (np.sum(histy) * dy)
    
    # Compute bin centers for plotting.
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Create figure with GridSpec:
    # - Top row: marginal histogram for x spanning first two columns.
    # - Bottom row: heatmap in first column, marginal histogram for y in second column, and colorbar in third column.
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  wspace=0.05, hspace=0.05)
    
    # Axes definition.
    ax_histx = fig.add_subplot(gs[0, 0])  # Top marginal histogram.
    ax_heat  = fig.add_subplot(gs[1, 0])  # Main heatmap.
    ax_histy = fig.add_subplot(gs[1, 1])  # Right marginal histogram.
    
    # Plot the heatmap. 'extent' ensures that the bin edges match the plotted image.
    im = ax_heat.imshow(H.T, origin='lower', cmap=cmap, aspect='auto',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # Plot marginal histograms.
    ax_histx.bar(x_centers, histx, width=np.diff(xedges), color='gray', align='center')
    ax_histy.barh(y_centers, histy, height=np.diff(yedges), color='gray', align='center')
    
    # Always remove tick labels from marginals
    ax_histx.set_xticklabels([])
    ax_histx.set_yticklabels([])
    ax_histy.set_xticklabels([])
    ax_histy.set_yticklabels([])
    
    # Configure tick visibility based on show_marginal_ticks
    if not show_marginal_ticks:
        ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax_histy.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
    else:
        # Only show y-ticks on the right histogram if requested
        ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax_histy.tick_params(axis="x", which="both", bottom=False, top=False)
        
        # If showing counts (not density), use integer ticks
        if not density:
            # Set integer ticks for marginal histograms
            max_x_count = int(np.ceil(np.max(histx)))
            max_y_count = int(np.ceil(np.max(histy)))
            
            # Set y-ticks for x-histogram to integers
            ax_histx.set_yticks(np.arange(0, max_x_count + 1, 1))
            
            # Set x-ticks for y-histogram to integers
            ax_histy.set_xticks(np.arange(0, max_y_count + 1, 1))
    
    # Set labels for the main heatmap with larger font size
    ax_heat.set_xlabel(xlabel, fontsize=20)
    ax_heat.set_ylabel(ylabel, fontsize=20)
    
    # Increase tick label font size
    ax_heat.tick_params(axis='both', which='major', labelsize=16)
    
    # Set custom ticks and tick labels if provided
    if xticks is not None:
        ax_heat.set_xticks(xticks)
    if yticks is not None:
        ax_heat.set_yticks(yticks)
    if xticklabels is not None:
        ax_heat.set_xticklabels(xticklabels, fontsize=16)
    if yticklabels is not None:
        ax_heat.set_yticklabels(yticklabels, fontsize=16)
    
    # Set title if provided with larger font size
    if title:
        fig.suptitle(title, fontsize=22, y=0.98)
    
    # Set axis limits if provided
    if xlim is not None:
        ax_heat.set_xlim(xlim)
        ax_histx.set_xlim(xlim)
    
    if ylim is not None:
        ax_heat.set_ylim(ylim)
        ax_histy.set_ylim(ylim)
    
    # Ensure the marginal histograms align with the heatmap
    ax_histx.set_xlim(ax_heat.get_xlim())
    ax_histy.set_ylim(ax_heat.get_ylim())
    
    # Add reference lines if requested
    if show_unity_line:
        # Get the common range for both axes
        min_val = max(ax_heat.get_xlim()[0], ax_heat.get_ylim()[0])
        max_val = min(ax_heat.get_xlim()[1], ax_heat.get_ylim()[1])
        ax_heat.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Unity Line (y=x)')
    
    if hline is not None:
        ax_heat.axhline(y=hline, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    if vline is not None:
        ax_heat.axvline(x=vline, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    # Remove ticks from right side of y-axis
    ax_heat.tick_params(axis='y', which='both', right=False)
    
    # Add colorbar - moved further to the right to avoid overlapping with histogram
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Density' if density else f'Count (/{len(x)})', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Set custom colorbar ticks and labels if provided
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    elif not density:
        # If not using density and no custom ticks provided, set integer ticks for the colorbar
        max_count = int(np.ceil(np.max(H)))
        cbar.set_ticks(np.arange(0, max_count + 1, 1))
        
    if cbar_ticklabels is not None:
        cbar.set_ticklabels(cbar_ticklabels)
    
    # Save or show the plot
    if filename is not None:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def plot_heatmap_with_marginals_grid(x, y, fig, gs, colorbar_ax=None, x_bin_edges=None, y_bin_edges=None, 
                                   cmap='viridis', xlim=None, ylim=None, density=False, 
                                   title=None, xlabel='X', ylabel='Y', show_marginal_ticks=False,
                                   show_unity_line=False, hline=None, vline=None,
                                   xticks=None, yticks=None, xticklabels=None, yticklabels=None,
                                   cbar_ticks=None, cbar_ticklabels=None, cbar_tick_step=None):
    """
    Plots a heatmap with marginal histograms for use in a grid layout.
    
    This is a grid-compatible version of plot_heatmap_with_marginals that works
    within an existing figure and gridspec layout.
    
    Parameters:
    - x: 1D array of x data.
    - y: 1D array of y data.
    - fig: Matplotlib figure object to use.
    - gs: GridSpec object defining the layout for this subplot.
    - colorbar_ax: Optional axes object for colorbar. If None, no colorbar is added.
    - x_bin_edges: Optional array of bin edges for the x-axis. If None, defaults to 30 bins.
    - y_bin_edges: Optional array of bin edges for the y-axis. If None, defaults to 30 bins.
    - cmap: Colormap for the heatmap.
    - xlim: Optional tuple (xmin, xmax) specifying the x-axis limits.
    - ylim: Optional tuple (ymin, ymax) specifying the y-axis limits.
    - density: If True, plot probability density instead of counts.
    - title: Optional title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - show_marginal_ticks: If True, show tick labels on marginal histograms.
    - show_unity_line: If True, add a diagonal unity line (y=x).
    - hline: Optional value for a horizontal line to add to the plot.
    - vline: Optional value for a vertical line to add to the plot.
    - xticks: Optional array of positions for x-axis tick marks.
    - yticks: Optional array of positions for y-axis tick marks.
    - xticklabels: Optional array of labels for x-axis ticks.
    - yticklabels: Optional array of labels for y-axis ticks.
    - cbar_ticks: Optional array of positions for colorbar tick marks.
    - cbar_ticklabels: Optional array of labels for colorbar ticks.
    - cbar_tick_step: Optional step size for colorbar ticks (e.g., 3 will create ticks at 0, 3, 6, ..., max).
    
    Returns:
    - dict: Dictionary containing axes objects {'ax_heat', 'ax_histx', 'ax_histy', 'im'}
    """
    # Set default bin edges if not provided.
    if x_bin_edges is None:
        x_min = np.min(x) if xlim is None else xlim[0]
        x_max = np.max(x) if xlim is None else xlim[1]
        x_bin_edges = np.linspace(x_min, x_max, 31)
    if y_bin_edges is None:
        y_min = np.min(y) if ylim is None else ylim[0]
        y_max = np.max(y) if ylim is None else ylim[1]
        y_bin_edges = np.linspace(y_min, y_max, 31)
    
    # Compute the 2D histogram.
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bin_edges, y_bin_edges], density=density)
    
    # Generate colorbar ticks if cbar_tick_step is specified but cbar_ticks is not
    if cbar_tick_step is not None and cbar_ticks is None:
        max_value = np.max(H)
        # Generate ticks from 0 to max_value with the specified step
        cbar_ticks = list(np.arange(0, max_value, cbar_tick_step))
        # Add the max value if it's not already included and the list is not empty
        if len(cbar_ticks) > 0 and cbar_ticks[-1] != max_value:
            cbar_ticks.append(max_value)
        # If cbar_ticks is empty (max_value is 0 or very small), create a minimal tick list
        elif len(cbar_ticks) == 0:
            cbar_ticks = [0, max_value] if max_value > 0 else [0]
    
    # Compute marginal counts or densities.
    histx = np.sum(H, axis=1)  # Sum over y bins for each x bin.
    histy = np.sum(H, axis=0)  # Sum over x bins for each y bin.
    
    # If density is True, normalize the marginals to make them proper PDFs
    if density:
        # Normalize marginals to integrate to 1
        dx = np.diff(xedges)
        dy = np.diff(yedges)
        histx = histx / (np.sum(histx) * dx)
        histy = histy / (np.sum(histy) * dy)
    
    # Compute bin centers for plotting.
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Axes definition using provided GridSpec
    ax_histx = fig.add_subplot(gs[0, 0])  # Top marginal histogram.
    ax_heat  = fig.add_subplot(gs[1, 0])  # Main heatmap.
    ax_histy = fig.add_subplot(gs[1, 1])  # Right marginal histogram.
    
    # Plot the heatmap. 'extent' ensures that the bin edges match the plotted image.
    im = ax_heat.imshow(H.T, origin='lower', cmap=cmap, aspect='auto',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # Plot marginal histograms.
    ax_histx.bar(x_centers, histx, width=np.diff(xedges), color='gray', align='center')
    ax_histy.barh(y_centers, histy, height=np.diff(yedges), color='gray', align='center')
    
    # Always remove tick labels from marginals
    ax_histx.set_xticklabels([])
    ax_histx.set_yticklabels([])
    ax_histy.set_xticklabels([])
    ax_histy.set_yticklabels([])
    
    # Configure tick visibility based on show_marginal_ticks
    if not show_marginal_ticks:
        ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax_histy.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
    else:
        # Only show y-ticks on the right histogram if requested
        ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax_histy.tick_params(axis="x", which="both", bottom=False, top=False)
        
        # If showing counts (not density), use integer ticks
        if not density:
            # Set integer ticks for marginal histograms
            max_x_count = int(np.ceil(np.max(histx)))
            max_y_count = int(np.ceil(np.max(histy)))
            
            # Set y-ticks for x-histogram to integers
            ax_histx.set_yticks(np.arange(0, max_x_count + 1, 1))
            
            # Set x-ticks for y-histogram to integers
            ax_histy.set_xticks(np.arange(0, max_y_count + 1, 1))
    
    # Set labels for the main heatmap with larger font size
    ax_heat.set_xlabel(xlabel, fontsize=20)
    ax_heat.set_ylabel(ylabel, fontsize=20)
    
    # Increase tick label font size
    ax_heat.tick_params(axis='both', which='major', labelsize=16)
    
    # Set custom ticks and tick labels if provided
    if xticks is not None:
        ax_heat.set_xticks(xticks)
    if yticks is not None:
        ax_heat.set_yticks(yticks)
    if xticklabels is not None:
        ax_heat.set_xticklabels(xticklabels, fontsize=16)
    if yticklabels is not None:
        ax_heat.set_yticklabels(yticklabels, fontsize=16)
    
    # Set title if provided with larger font size
    if title:
        ax_heat.set_title(title, fontsize=22, pad=20)
    
    # Set axis limits if provided
    if xlim is not None:
        ax_heat.set_xlim(xlim)
        ax_histx.set_xlim(xlim)
    
    if ylim is not None:
        ax_heat.set_ylim(ylim)
        ax_histy.set_ylim(ylim)
    
    # Ensure the marginal histograms align with the heatmap
    ax_histx.set_xlim(ax_heat.get_xlim())
    ax_histy.set_ylim(ax_heat.get_ylim())
    
    # Add reference lines if requested
    if show_unity_line:
        # Get the common range for both axes
        min_val = max(ax_heat.get_xlim()[0], ax_heat.get_ylim()[0])
        max_val = min(ax_heat.get_xlim()[1], ax_heat.get_ylim()[1])
        ax_heat.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Unity Line (y=x)')
    
    if hline is not None:
        ax_heat.axhline(y=hline, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    if vline is not None:
        ax_heat.axvline(x=vline, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    # Remove ticks from right side of y-axis
    ax_heat.tick_params(axis='y', which='both', right=False)
    
    # Add colorbar if colorbar_ax is provided
    if colorbar_ax is not None:
        cbar = fig.colorbar(im, cax=colorbar_ax)
        cbar.set_label('Density' if density else f'Count (/{len(x)})', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        # Set custom colorbar ticks and labels if provided
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        elif not density:
            # If not using density and no custom ticks provided, set integer ticks for the colorbar
            max_count = int(np.ceil(np.max(H)))
            cbar.set_ticks(np.arange(0, max_count + 1, 1))
            
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)
    
    # Return axes objects for further customization
    return {
        'ax_heat': ax_heat,
        'ax_histx': ax_histx, 
        'ax_histy': ax_histy,
        'im': im
    }

def plot_rewards_vs_punishment(punishment_avoided_means, rewards_approached_means, title, filename=None):
    plt.figure(figsize=(10,6))
    
    plt.scatter(punishment_avoided_means, rewards_approached_means, alpha=0.5)
    
    plt.xlabel('Proportion of Punishment Avoided')
    plt.ylabel('Proportion of Rewards Approached')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add reference lines at x=0.5 and y=0.5
    # plt.axhline(y=0.67, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0.67, color='gray', linestyle='--', alpha=0.3)
    plt.ylim(0.6, 1.02)
    plt.xlim(0.6, 1.02)
    
    # Save or show the plot
    if filename is not None:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_rewards_vs_punishment_heatmap(punishment_approached_counts, rewards_approached_counts, title, cmap='Greys', 
                                       x_lim = (0, 6.5), y_lim = (30, 49), cbar_ticks=None, cbar_tick_step=None, filename=None):
    # Create the figure using the helper function
    x = np.array(punishment_approached_counts)
    y = np.array(rewards_approached_counts)
    
    # Define bin edges to have integers at the center
    x_bins = np.linspace(-0.5, 23.5, 25)  # Bins centered at integers 0-23
    y_bins = np.linspace(-1, 49, 26)  # Bins centered at even integers 0, 2, 4, ..., 48
    
    # Calculate bin centers for ticks
    x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2

    # Limits
    x_lim = x_lim
    y_lim = y_lim
    
    # Call the plot_heatmap_with_marginals function
    plot_heatmap_with_marginals(x, y, 
                                    x_bin_edges=x_bins, y_bin_edges=y_bins,
                                    xlim=x_lim, ylim=y_lim,
                                    title=title,
                                    xlabel="Dangerous bees approached (/16)",
                                    ylabel="Friendly bees approached (/48)",
                                    show_marginal_ticks=False,
                                    cmap=cmap,
                                    hline=32,
                                    xticks=x_bin_centers,
                                    yticks=y_bin_centers,
                                    cbar_ticks=cbar_ticks,
                                    cbar_tick_step=cbar_tick_step,
                                    filename=filename)
    
    # result.show()

def get_reward_punishment_counts(ids, phases, blocks_all):
    """
    Count rewards approached and punishment avoided for given ids and phases.
    
    Args:
        ids: Array-like of participant IDs
        phases: Phase number or list of phase numbers to filter blocks
        blocks_all: DataFrame containing all blocks

    Returns:
        Tuple of (rewards_approached_counts, punishment_approached_counts) lists
    """
    rewards_approached_counts = []
    punishment_approached_counts = []
    
    # Convert single phase to list for consistent handling
    if not isinstance(phases, list):
        phases = [phases]
        
    for id in ids:
        filtered_blocks = blocks_all[(blocks_all['id']==id) & (blocks_all['phase'].isin(phases))]
        rewards_approached_counts.append(filtered_blocks['rewards_approached'].sum())
        punishment_approached_counts.append(filtered_blocks['punishment_approached'].sum())
    return punishment_approached_counts, rewards_approached_counts

def plot_heatmap_grid(data_list, nrows, ncols, titles=None, colorbar='shared', 
                     figsize=None, overall_title=None, filename=None,
                     row_labels=None, col_labels=None,
                     x_bin_edges=None, y_bin_edges=None, cmap='viridis', 
                     xlim=None, ylim=None, density=False, xlabel='X', ylabel='Y',
                     show_marginal_ticks=False, show_unity_line=False, 
                     hline=None, vline=None, xticks=None, yticks=None, 
                     xticklabels=None, yticklabels=None, cbar_ticks=None, 
                     cbar_ticklabels=None, cbar_tick_step=None):
    """
    Create a grid of heatmaps with marginal histograms.
    
    Parameters:
    - data_list: List of (x, y) tuples, one for each subplot
    - nrows: Number of rows in the grid
    - ncols: Number of columns in the grid
    - titles: Optional list of titles for each subplot
    - colorbar: Colorbar option - 'shared', 'individual', or 'none'
      - 'shared': Single colorbar for all subplots (default)
      - 'individual': Individual colorbar for each subplot
      - 'none': No colorbars displayed
    - figsize: Optional figure size tuple. If None, calculated automatically
    - overall_title: Optional title for the entire figure
    - filename: Optional filename to save the plot
    - row_labels: Optional list of labels for each row
    - col_labels: Optional list of labels for each column
    - All other parameters are passed to plot_heatmap_with_marginals_grid
    
    Returns:
    - fig: The matplotlib figure object
    - axes_list: List of axes dictionaries for each subplot
    """
    
    # Calculate figure size if not provided
    if figsize is None:
        # Each subplot needs space for heatmap + marginals, plus colorbar space
        subplot_width = 4
        subplot_height = 4
        if colorbar == 'shared':
            colorbar_width = 0.5
        elif colorbar == 'individual':
            colorbar_width = 0.3 * ncols  # Small colorbar for each subplot
        else:  # colorbar == 'none'
            colorbar_width = 0
        figsize = (ncols * subplot_width + colorbar_width, nrows * subplot_height)
    
    # Create the main figure
    fig = plt.figure(figsize=figsize)
    
    # Create overall GridSpec accounting for colorbar
    if colorbar == 'shared':
        main_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], 
                          wspace=0.02)
        grid_gs = main_gs[0, 0].subgridspec(nrows, ncols, wspace=0.1, hspace=0.1)
        cbar_gs = main_gs[0, 1]
    elif colorbar == 'individual':
        # For individual colorbars, we need space for each subplot + its colorbar
        grid_gs = GridSpec(nrows, ncols, figure=fig, wspace=0.1, hspace=0.1)
        cbar_gs = None
    else:  # colorbar == 'none'
        grid_gs = GridSpec(nrows, ncols, figure=fig, wspace=0.1, hspace=0.1)
        cbar_gs = None
    
    axes_list = []
    all_images = []  # Store all images for shared colorbar
    
    # Create each subplot
    for i in range(len(data_list)):
        if i >= nrows * ncols:
            break
            
        row = i // ncols
        col = i % ncols
        
        # Create subplot GridSpec for this position
        if colorbar == 'individual':
            # For individual colorbars, we need 3 columns: marginal, heatmap, colorbar
            subplot_gs = grid_gs[row, col].subgridspec(
                2, 3, width_ratios=[4, 1, 0.2], height_ratios=[1, 4],
                wspace=0.05, hspace=0.05
            )
            # Create colorbar axes
            colorbar_ax = fig.add_subplot(subplot_gs[1, 2])
        else:
            # For shared or no colorbar, use 2 columns: marginal, heatmap
            subplot_gs = grid_gs[row, col].subgridspec(
                2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                wspace=0.05, hspace=0.05
            )
            colorbar_ax = None
        
        # Get data for this subplot
        x, y = data_list[i]
        
        # Get title for this subplot
        title = titles[i] if titles and i < len(titles) else None
        
        # Plot the heatmap with marginals
        axes_dict = plot_heatmap_with_marginals_grid(
            x, y, fig, subplot_gs, colorbar_ax=colorbar_ax,
            x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges,
            cmap=cmap, xlim=xlim, ylim=ylim, density=density,
            title=title, xlabel=xlabel, ylabel=ylabel,
            show_marginal_ticks=show_marginal_ticks,
            show_unity_line=show_unity_line, hline=hline, vline=vline,
            xticks=xticks, yticks=yticks, xticklabels=xticklabels,
            yticklabels=yticklabels, cbar_ticks=cbar_ticks,
            cbar_ticklabels=cbar_ticklabels, cbar_tick_step=cbar_tick_step
        )
        
        axes_list.append(axes_dict)
        all_images.append(axes_dict['im'])
    
    # Add shared colorbar if requested
    if colorbar == 'shared' and all_images:
        # Use the first image to create the colorbar
        # You might want to normalize all images to the same scale
        cbar_ax = fig.add_subplot(cbar_gs)
        cbar = fig.colorbar(all_images[0], cax=cbar_ax)
        cbar.set_label('Density' if density else 'Count', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        # Set custom colorbar ticks if provided
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)
    
    # Add row and column labels if provided
    if row_labels is not None:
        for i, label in enumerate(row_labels):
            if i < nrows:
                # Add row label on the left side
                fig.text(0.02, 0.5 - (i / nrows) + (0.5 / nrows), label, 
                        rotation=90, va='center', ha='center', fontsize=18, weight='bold')
    
    if col_labels is not None:
        for i, label in enumerate(col_labels):
            if i < ncols:
                # Add column label at the top
                fig.text(0.1 + (i / ncols) + (0.4 / ncols), 0.95, label, 
                        va='center', ha='center', fontsize=18, weight='bold')
    
    # Add overall title if provided
    if overall_title:
        fig.suptitle(overall_title, fontsize=24, y=0.98)
    
    # Save or show the plot
    if filename is not None:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, axes_list



if __name__ == "__main__":
    # This module provides plotting functions but does not generate outputs when run directly.
    # Use make_heatmaps.py or generate_outputs.py to generate plots.
    print("heatmap.py provides plotting functions.")
    print("To generate heatmap outputs, run:")
    print("  python make_heatmaps.py")
    print("  python generate_outputs.py")