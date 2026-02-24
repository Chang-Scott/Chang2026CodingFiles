"""
MCMC Plotting Functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, uniform
import corner
# Import configuration
from helpers.mcmc_functions import (
    PARAM_KEYS, BLOB_KEYS, ALL_KEYS,
    PARAM_BOUNDS, DERIVED_BOUNDS, OBSERVABLE_BOUNDS, ALL_BOUNDS,
    PARAM_LABELS, DERIVED_LABELS, OBSERVABLE_LABELS, ALL_LABELS,
    BURN_IN
)


def plot_2d_corner(mcmc_data, var_names, var1_name, var2_name, burn_in=BURN_IN, 
                   true_values=None, use_bounds=True, output_filename="mcmc_figures/2d_corner.png"):
    """
    Create a simple 2D corner plot between any two variables.
    
    Parameters
    ----------
    mcmc_data : ndarray
        Combined MCMC data with shape (nsteps, nwalkers, nvars)
    var_names : list
        List of variable names corresponding to columns in mcmc_data
    var1_name : str
        First variable name
    var2_name : str
        Second variable name
    burn_in : int
        Number of burn-in steps to discard
    true_values : dict, optional
        True values for overlay
    use_bounds : bool, optional
        Whether to apply bounds to axes
    output_filename : str
        Output file path
    """
    # Flatten data after burn-in
    flat_data = mcmc_data[burn_in:].reshape(-1, mcmc_data.shape[-1])
    
    # Get variable indices
    idx1 = var_names.index(var1_name)
    idx2 = var_names.index(var2_name)
    
    var1_data = flat_data[:, idx1]
    var2_data = flat_data[:, idx2]
    
    # Remove NaNs
    valid_mask = ~(np.isnan(var1_data) | np.isnan(var2_data))
    var1_data = var1_data[valid_mask]
    var2_data = var2_data[valid_mask]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                          left=0.12, right=0.95, bottom=0.12, top=0.95)

    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    ax_main.set_facecolor("black")
    fig.patch.set_facecolor("white")

    hb = ax_main.hexbin(
        var1_data,
        var2_data,
        gridsize=200,
        cmap="magma",
        mincnt=1,
        norm=LogNorm(),
    )

    if true_values and var1_name in true_values and var2_name in true_values:
        true_x = true_values[var1_name]
        true_y = true_values[var2_name]
        ax_main.scatter(true_x, true_y, marker='*', s=400, c='red',
                        edgecolors='black', linewidths=1.5, zorder=10,
                        label='True value')
        ax_main.axvline(true_x, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_main.axhline(true_y, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_main.legend(loc='best', fontsize=10)

    ax_main.set_xlabel(ALL_LABELS.get(var1_name, var1_name), fontsize=12)
    ax_main.set_ylabel(ALL_LABELS.get(var2_name, var2_name), fontsize=12)
    ax_main.grid(alpha=0.3)

    if use_bounds:
        x_bounds = ALL_BOUNDS.get(var1_name)
        y_bounds = ALL_BOUNDS.get(var2_name)
        
        if x_bounds is not None:
            ax_main.set_xlim(x_bounds)
        if y_bounds is not None:
            ax_main.set_ylim(y_bounds)

    if use_bounds and var1_name in ALL_BOUNDS:
        bins = np.linspace(ALL_BOUNDS[var1_name][0], ALL_BOUNDS[var1_name][1], 50)
    else:
        bins = 50
    ax_top.hist(var1_data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    if true_values and var1_name in true_values:
        ax_top.axvline(true_values[var1_name], color='red', linestyle='--', linewidth=2)
    ax_top.set_ylabel('Count', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(alpha=0.3)

    if use_bounds and var2_name in ALL_BOUNDS:
        bins = np.linspace(ALL_BOUNDS[var2_name][0], ALL_BOUNDS[var2_name][1], 50)
    else:
        bins = 50
    ax_right.hist(var2_data, bins=bins, orientation='horizontal',
                  color='steelblue', alpha=0.7, edgecolor='black')
    if true_values and var2_name in true_values:
        ax_right.axhline(true_values[var2_name], color='red', linestyle='--', linewidth=2)
    ax_right.set_xlabel('Count', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_filename}")


def plot_blob_distributions(mcmc_data, var_names, burn_in=BURN_IN, 
                            output_filename="mcmc_figures/blob_distributions.png"):
    """
    Plot distributions of derived quantities (blobs).
    
    Parameters
    ----------
    mcmc_data : ndarray
        Combined MCMC data with shape (nsteps, nwalkers, nvars)
    var_names : list
        List of variable names corresponding to columns in mcmc_data
    burn_in : int
        Number of burn-in steps to discard
    output_filename : str
        Output file path
    """
    # Flatten data after burn-in
    flat_data = mcmc_data[burn_in:].reshape(-1, mcmc_data.shape[-1])
    
    # Plot derived quantities (first 5 blob keys)
    derived_keys = BLOB_KEYS[:5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, key in enumerate(derived_keys):
        idx = var_names.index(key)
        data = flat_data[:, idx]
        valid_data = data[~np.isnan(data)]
        
        axes[i].hist(valid_data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(DERIVED_LABELS.get(key, key), fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].grid(alpha=0.3)
    
    # Hide the last subplot (we only have 5 derived quantities)
    axes[-1].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_filename}")


def plot_custom_corner(mcmc_data, var_names, plot_vars=None, burn_in=BURN_IN, true_values=None, 
                       gridsize=200, cmap="magma", output_filename=None):
    """
    Plot custom corner plot with hexbin off-diagonal plots and no diagonal plots.
    
    Parameters
    ----------
    mcmc_data : ndarray
        Combined MCMC data with shape (nsteps, nwalkers, nvars)
    var_names : list
        List of variable names corresponding to columns in mcmc_data
    plot_vars : list of str, optional
        List of variable names to plot. If None, plots all parameters.
    burn_in : int
        Number of burn-in steps to discard
    true_values : dict, optional
        Dictionary of true parameter values {var_name: value}
    gridsize : int
        Hexbin grid size
    cmap : str
        Colormap name
    output_filename : str, optional
        Output file path. If None, uses default naming.
    """
    # Flatten data after burn-in
    flat_data = mcmc_data[burn_in:].reshape(-1, mcmc_data.shape[-1])
    
    # Default to all parameters if not specified
    if plot_vars is None:
        plot_vars = PARAM_KEYS
    
    # Build data array for selected variables
    data_list = []
    for var_name in plot_vars:
        if var_name not in var_names:
            raise ValueError(f"Variable '{var_name}' not found in var_names")
        idx = var_names.index(var_name)
        data_list.append(flat_data[:, idx])
    
    data = np.column_stack(data_list)
    
    # Remove rows with NaN
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    
    n_vars = len(plot_vars)
    
    # Create figure with white background
    fig = plt.figure(figsize=(3 * n_vars, 3 * n_vars), facecolor='white')
    
    # Create grid of subplots (lower triangular, no diagonal)
    # Grid is (n_vars-1) x (n_vars-1) since we skip diagonal
    axes = []
    for i in range(1, n_vars):  # Start from 1 to skip first row (diagonal)
        row_axes = []
        for j in range(i):  # Only create plots where j < i (lower triangular)
            ax = plt.subplot(n_vars - 1, n_vars - 1, (i - 1) * (n_vars - 1) + j + 1)
            ax.set_facecolor('black')
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Plot hexbins
    for i in range(1, n_vars):  # Start from 1 (skip first variable on y-axis)
        for j in range(i):  # j < i (lower triangular)
            ax = axes[i - 1][j]
            
            x_data = data[:, j]
            y_data = data[:, i]
            
            # Create hexbin
            hb = ax.hexbin(
                x_data, y_data,
                gridsize=gridsize,
                cmap=cmap,
                mincnt=1,
                norm=LogNorm()
            )
            
            # Overlay true values if provided
            if true_values:
                x_var = plot_vars[j]
                y_var = plot_vars[i]
                
                if x_var in true_values and y_var in true_values:
                    true_x = true_values[x_var]
                    true_y = true_values[y_var]
                    
                    ax.scatter(true_x, true_y, marker='*', s=200, c='red',
                              edgecolors='white', linewidths=1, zorder=10)
                    ax.axvline(true_x, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                    ax.axhline(true_y, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # Set bounds if available
            x_var = plot_vars[j]
            y_var = plot_vars[i]
            
            if x_var in ALL_BOUNDS:
                ax.set_xlim(ALL_BOUNDS[x_var])
            if y_var in ALL_BOUNDS:
                ax.set_ylim(ALL_BOUNDS[y_var])
            
            # Determine if this is bottom row or left column
            is_bottom_row = (i == n_vars - 1)
            is_left_column = (j == 0)
            
            # Labels and ticks
            if is_bottom_row:
                ax.set_xlabel(ALL_LABELS.get(plot_vars[j], plot_vars[j]), 
                             fontsize=10, color='black')
                ax.tick_params(axis='x', colors='black', labelsize=8, labelbottom=True)
            else:
                ax.tick_params(axis='x', colors='black', labelsize=8, labelbottom=False)
            
            if is_left_column:
                ax.set_ylabel(ALL_LABELS.get(plot_vars[i], plot_vars[i]), 
                             fontsize=10, color='black')
                ax.tick_params(axis='y', colors='black', labelsize=8, labelleft=True)
            else:
                ax.tick_params(axis='y', colors='black', labelsize=8, labelleft=False)
            
            # Spine colors (keep white to contrast with black plot background)
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(0.5)
    
    # Add a single shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    # Create a dummy mappable for the colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=LogNorm(vmin=1, vmax=data.shape[0]/100))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Counts', fontsize=12, color='black')
    cbar.ax.tick_params(colors='black', labelsize=8)
    cbar.outline.set_edgecolor('black')
    
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.98, 
                       hspace=0.05, wspace=0.05)
    
    # Save
    if output_filename is None:
        var_str = "_".join(plot_vars)
        output_filename = f"mcmc_figures/custom_corner_{var_str}.png"
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {output_filename}")


def plot_posterior_vs_prior(mcmc_data, var_names, var_name, burn_in=BURN_IN, true_values=None, 
                            output_filename=None, figsize=(10, 6)):
    """
    Plot posterior distribution vs prior distribution for a single variable.
    
    Parameters
    ----------
    mcmc_data : ndarray
        Combined MCMC data with shape (nsteps, nwalkers, nvars)
    var_names : list
        List of variable names corresponding to columns in mcmc_data
    var_name : str
        Name of the variable to plot
    burn_in : int
        Number of burn-in steps to discard
    true_values : dict, optional
        Dictionary of true parameter values {var_name: value}
    output_filename : str, optional
        Output file path. If None, uses default naming.
    figsize : tuple
        Figure size
    """
    # Flatten data after burn-in
    flat_data = mcmc_data[burn_in:].reshape(-1, mcmc_data.shape[-1])
    
    # Extract data for the variable
    if var_name not in var_names:
        raise ValueError(f"Variable '{var_name}' not found in var_names")
    
    idx = var_names.index(var_name)
    data = flat_data[:, idx]
    
    # Remove NaNs
    data = data[~np.isnan(data)]
    
    # Get bounds
    bounds = ALL_BOUNDS.get(var_name)
    if bounds is None:
        bounds = [data.min(), data.max()]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Create x-axis for plotting
    x = np.linspace(bounds[0], bounds[1], 1000)
    
    # Prior (uniform)
    prior_height = 1.0 / (bounds[1] - bounds[0])
    ax.fill_between(x, 0, prior_height, alpha=0.3, color='gray', label='Prior (Uniform)')
    ax.plot(x, np.full_like(x, prior_height), color='black', linewidth=1.5)
    # Posterior
    kde = gaussian_kde(data)
    posterior = kde(x)
    ax.plot(x, posterior, 'b-', linewidth=2, label='Posterior')
    ax.fill_between(x, 0, posterior, alpha=0.3, color='blue')
    
    # True value
    if true_values and var_name in true_values:
        true_val = true_values[var_name]
        ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='True Value')
    
    # Compute statistics
    median_val = np.median(data)
    

    ax.axvline(median_val, color='darkblue', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Annotations
    y_pos = ax.get_ylim()[1]
    ax.text(median_val, y_pos * 0.95, f'Median: {median_val:.3f}',
            ha='center', va='top', fontsize=9, color='darkblue',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if true_values and var_name in true_values:
        ax.text(true_val, y_pos * 0.75, f'True: {true_val:.3f}',
                ha='center', va='top', fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and styling
    ax.set_xlabel(ALL_LABELS.get(var_name, var_name), fontsize=12, color='black')
    ax.set_ylabel('Probability Density', fontsize=12, color='black')
    ax.set_title(f'Posterior vs Prior: {ALL_LABELS.get(var_name, var_name)}', 
                fontsize=14, color='black', weight='bold')
    ax.set_xlim(bounds)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, color='gray')
    ax.tick_params(colors='black')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    # Save
    if output_filename is None:
        output_filename = f"mcmc_figures/posterior_vs_prior_{var_name}.png"
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {output_filename}")


def plot_mcmc_results(samples, log_prob, burn_in=BURN_IN):
    """
    Generate diagnostic plots for MCMC results.
    
    Parameters
    ----------
    samples : ndarray
        MCMC samples with shape (nsteps, nwalkers, ndim)
    log_prob : ndarray
        Log probability values with shape (nsteps, nwalkers)
    burn_in : int
        Number of burn-in steps
    """
    nsteps, nwalkers, n_dim = samples.shape
    
    os.makedirs('mcmc_figures', exist_ok=True)
    
    # Trace plots for parameters
    print("Creating trace plots...")
    fig, axes = plt.subplots(n_dim + 1, 1, figsize=(12, 2 * (n_dim + 1)), sharex=True)
    
    for i, param_key in enumerate(PARAM_KEYS):
        for walker in range(nwalkers):
            axes[i].plot(samples[:, walker, i], alpha=0.3, linewidth=0.5)
        axes[i].axvline(burn_in, color='red', linestyle='--', linewidth=2, label='Burn-in')
        axes[i].set_ylabel(PARAM_LABELS.get(param_key, param_key), fontsize=10)
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper right')
    
    # Log probability
    for walker in range(nwalkers):
        axes[-1].plot(log_prob[:, walker], alpha=0.3, linewidth=0.5)
    axes[-1].axvline(burn_in, color='red', linestyle='--', linewidth=2)
    axes[-1].set_ylabel('Log Probability', fontsize=10)
    axes[-1].set_xlabel('Step', fontsize=12)
    axes[-1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mcmc_figures/trace_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved trace_plots.png")
    
    # Corner plot
    print("Creating corner plot...")
    flat_samples = samples[burn_in:].reshape(-1, n_dim)
    
    fig = corner.corner(
        flat_samples,
        labels=[PARAM_LABELS.get(key, key) for key in PARAM_KEYS],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={"fontsize": 12}
    )
    plt.savefig('mcmc_figures/corner_plot.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("Saved corner_plot.pdf")


def plot_variable_histograms(mcmc_data, var_names, plot_vars=None, burn_in=BURN_IN, 
                             true_values=None, bins=50, output_filename="mcmc_figures/variable_histograms.png"):
    """
    Plot histograms of selected variables for diagnostic purposes.
    
    This function creates a grid of histograms showing the posterior distributions
    of selected variables, optionally overlaying true values and prior bounds.
    Useful for verifying that MCMC is exploring the parameter space as expected.
    
    Parameters
    ----------
    mcmc_data : ndarray
        Combined MCMC data with shape (nsteps, nwalkers, nvars)
    var_names : list
        List of variable names corresponding to columns in mcmc_data
    plot_vars : list of str, optional
        List of variable names to plot. If None, plots all parameters.
    burn_in : int
        Number of burn-in steps to discard
    true_values : dict, optional
        Dictionary of true parameter values {var_name: value}
    bins : int or str
        Number of bins for histograms (or 'auto')
    output_filename : str
        Output file path
    """
    # Flatten data after burn-in
    flat_data = mcmc_data[burn_in:].reshape(-1, mcmc_data.shape[-1])
    
    # Default to all parameters if not specified
    if plot_vars is None:
        plot_vars = PARAM_KEYS
    
    n_vars = len(plot_vars)
    
    # Determine grid layout
    n_cols = min(3, n_vars)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor('white')
    
    # Flatten axes array for easy iteration
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_vars > n_cols else axes
    
    for idx, var_name in enumerate(plot_vars):
        ax = axes[idx]
        
        # Get variable data
        if var_name not in var_names:
            ax.text(0.5, 0.5, f"Variable '{var_name}'\nnot found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        
        var_idx = var_names.index(var_name)
        data = flat_data[:, var_idx]
        
        # Remove NaNs
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f"No valid data\nfor '{var_name}'", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Get bounds for this variable
        bounds = ALL_BOUNDS.get(var_name)
        
        # Create histogram
        if bounds is not None:
            bin_edges = np.linspace(bounds[0], bounds[1], bins + 1)
            counts, edges, patches = ax.hist(data, bins=bin_edges, color='steelblue', 
                                            alpha=0.7, edgecolor='black', linewidth=0.5)
        else:
            counts, edges, patches = ax.hist(data, bins=bins, color='steelblue', 
                                            alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Overlay true value if provided
        if true_values and var_name in true_values:
            true_val = true_values[var_name]
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2, 
                      label='True Value', zorder=10)
        
        # Add vertical lines for statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color='darkblue', linestyle=':', linewidth=1.5, 
                  alpha=0.7, label='Mean')
        ax.axvline(median_val, color='darkblue', linestyle='-', linewidth=1.5, 
                  alpha=0.7, label='Median')
        
        # Set bounds if available
        if bounds is not None:
            ax.set_xlim(bounds)
        else:
            ax.set_xlim(data.min(), data.max())
        # Labels and title
        ax.set_xlabel(ALL_LABELS.get(var_name, var_name), fontsize=11, weight='bold')
        ax.set_ylabel('Count', fontsize=10)
    
        
        # Grid
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend (only for first plot to avoid clutter)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {output_filename}")
