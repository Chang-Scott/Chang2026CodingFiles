import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, uniform

try:
    import corner
except Exception:
    corner = None


def plot_2d_corner(
    samples,
    blobs,
    var1_name,
    var2_name,
    true_values=None,
    burn_in=0,
    param_keys=None,
    blob_keys=None,
    label_map=None,
    param_bounds=None,
    blob_bounds=None,
    use_bounds=True,
):
    """
    Create a simple 2D corner plot between any two variables.
    """
    if param_keys is None or blob_keys is None or label_map is None:
        raise ValueError("param_keys, blob_keys, and label_map are required")

    param_map = {name: i for i, name in enumerate(param_keys)}
    blob_map = {name: i for i, name in enumerate(blob_keys)}

    if var1_name in param_map:
        var1_data = samples[burn_in:, :, param_map[var1_name]].flatten()
    elif var1_name in blob_map:
        var1_data = blobs[burn_in:, :, blob_map[var1_name]].flatten()
    else:
        raise ValueError(f"Variable {var1_name} not found in parameters or blobs")

    if var2_name in param_map:
        var2_data = samples[burn_in:, :, param_map[var2_name]].flatten()
    elif var2_name in blob_map:
        var2_data = blobs[burn_in:, :, blob_map[var2_name]].flatten()
    else:
        raise ValueError(f"Variable {var2_name} not found in parameters or blobs")

    valid_mask = ~(np.isnan(var1_data) | np.isnan(var2_data))
    var1_data = var1_data[valid_mask]
    var2_data = var2_data[valid_mask]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                          left=0.12, right=0.95, bottom=0.12, top=0.95)

    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    
    # Dark background like your figure
    ax_main.set_facecolor("black")
    fig.patch.set_facecolor("white")

    # Hexbin with better brightness control
    hb = ax_main.hexbin(
        var1_data,
        var2_data,
        gridsize=200,
        cmap="magma",          # brighter than Blues
        mincnt=1,
        norm=LogNorm(),        # key for brightness
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

    ax_main.set_xlabel(label_map.get(var1_name, var1_name), fontsize=12)
    ax_main.set_ylabel(label_map.get(var2_name, var2_name), fontsize=12)
    ax_main.grid(alpha=0.3)

    if use_bounds:
        x_bounds = None
        y_bounds = None
        if param_bounds and var1_name in param_bounds:
            x_bounds = param_bounds[var1_name]
        elif blob_bounds and var1_name in blob_bounds:
            x_bounds = blob_bounds[var1_name]

        if param_bounds and var2_name in param_bounds:
            y_bounds = param_bounds[var2_name]
        elif blob_bounds and var2_name in blob_bounds:
            y_bounds = blob_bounds[var2_name]

        if x_bounds is not None:
            ax_main.set_xlim(x_bounds)
        if y_bounds is not None:
            ax_main.set_ylim(y_bounds)

    if use_bounds and param_bounds and var1_name in param_bounds:
        bins = np.linspace(param_bounds[var1_name][0], param_bounds[var1_name][1], 50)
    elif use_bounds and blob_bounds and var1_name in blob_bounds:
        bins = np.linspace(blob_bounds[var1_name][0], blob_bounds[var1_name][1], 50)
    else:
        bins = 50
    ax_top.hist(var1_data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    if true_values and var1_name in true_values:
        ax_top.axvline(true_values[var1_name], color='red', linestyle='--', linewidth=2)
    ax_top.set_ylabel('Count', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(alpha=0.3)

    if use_bounds and param_bounds and var2_name in param_bounds:
        bins = np.linspace(param_bounds[var2_name][0], param_bounds[var2_name][1], 50)
    elif use_bounds and blob_bounds and var2_name in blob_bounds:
        bins = np.linspace(blob_bounds[var2_name][0], blob_bounds[var2_name][1], 50)
    else:
        bins = 50
    ax_right.hist(var2_data, bins=bins, color='steelblue', alpha=0.7,
                  edgecolor='black', orientation='horizontal')
    if true_values and var2_name in true_values:
        ax_right.axhline(true_values[var2_name], color='red', linestyle='--', linewidth=2)
    ax_right.set_xlabel('Count', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(alpha=0.3)

    os.makedirs('mcmc_figures', exist_ok=True)
    filename = f'mcmc_figures/corner_{var1_name}_vs_{var2_name}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved {filename}")

    print(f"\n{label_map.get(var1_name, var1_name)}:")
    print(f"  Median: {np.median(var1_data):.4f}")
    print(f"  68% CI: [{np.percentile(var1_data, 16):.4f}, {np.percentile(var1_data, 84):.4f}]")
    if true_values and var1_name in true_values:
        print(f"  True: {true_values[var1_name]:.4f}")

    print(f"\n{label_map.get(var2_name, var2_name)}:")
    print(f"  Median: {np.median(var2_data):.4f}")
    print(f"  68% CI: [{np.percentile(var2_data, 16):.4f}, {np.percentile(var2_data, 84):.4f}]")
    if true_values and var2_name in true_values:
        print(f"  True: {true_values[var2_name]:.4f}")



def plot_custom_corner(
    samples,
    blobs,
    var_names,
    burn_in=0,
    param_keys=None,
    blob_keys=None,
    label_map=None,
    true_values=None,
    param_bounds=None,
    blob_bounds=None,
    use_bounds=True,
    gridsize=200,
    cmap="magma",
    output_filename=None,
):
    """
    Create a custom corner plot with hexbin visualizations for all variable pairs.
    No diagonal plots are shown, only off-diagonal hexbin plots.
    
    Parameters
    ----------
    samples : ndarray
        MCMC samples array with shape (nsteps, nwalkers, ndim)
    blobs : ndarray
        Blob data array with shape (nsteps, nwalkers, nblobs)
    var_names : list of str
        List of variable names to include in corner plot (can be parameters or blobs)
    burn_in : int, optional
        Number of burn-in steps to discard
    param_keys : list of str, optional
        List of parameter names corresponding to samples dimensions
    blob_keys : list of str, optional
        List of blob names corresponding to blobs dimensions
    label_map : dict, optional
        Mapping from variable names to display labels
    true_values : dict, optional
        Dictionary of true/reference values for each variable
    param_bounds : dict, optional
        Dictionary of bounds for parameters
    blob_bounds : dict, optional
        Dictionary of bounds for blobs
    use_bounds : bool, optional
        Whether to apply bounds to axes
    gridsize : int, optional
        Hexbin grid size (default: 200)
    cmap : str, optional
        Colormap for hexbin plots (default: "magma")
    output_filename : str, optional
        Custom output filename (default: auto-generated from var_names)
    """
    if param_keys is None or blob_keys is None or label_map is None:
        raise ValueError("param_keys, blob_keys, and label_map are required")
    
    # Create mappings for parameter and blob indices
    param_map = {name: i for i, name in enumerate(param_keys)}
    blob_map = {name: i for i, name in enumerate(blob_keys)}
    
    # Extract data for all variables
    var_data = {}
    for var_name in var_names:
        if var_name in param_map:
            data = samples[burn_in:, :, param_map[var_name]].flatten()
        elif var_name in blob_map:
            data = blobs[burn_in:, :, blob_map[var_name]].flatten()
        else:
            raise ValueError(f"Variable {var_name} not found in parameters or blobs")
        
        # Remove NaNs
        var_data[var_name] = data[~np.isnan(data)]
    
    n_vars = len(var_names)
    
    # Create figure with gridspec
    fig_size = 3 * n_vars
    fig = plt.figure(figsize=(fig_size, fig_size))
    gs = fig.add_gridspec(n_vars, n_vars, hspace=0.05, wspace=0.05,
                          left=0.1, right=0.92, bottom=0.1, top=0.95)
    
    # Create all subplots
    axes = np.empty((n_vars, n_vars), dtype=object)
    hexbin_plots = []
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i > j:  # Lower triangular only
                # Share axes appropriately
                sharex = axes[n_vars-1, j] if i < n_vars-1 else None
                sharey = axes[i, 0] if j > 0 else None
                axes[i, j] = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
            else:
                # Create subplot but will hide it
                axes[i, j] = fig.add_subplot(gs[i, j])
                axes[i, j].axis('off')
    
    # Plot hexbins for lower triangular
    for i in range(n_vars):
        for j in range(n_vars):
            if i > j:  # Lower triangular only
                ax = axes[i, j]
                
                # Get variable names for this subplot
                x_var = var_names[j]
                y_var = var_names[i]
                
                # Get data
                x_data = var_data[x_var]
                y_data = var_data[y_var]
                
                # Filter NaNs for this pair
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data_clean = x_data[valid_mask]
                y_data_clean = y_data[valid_mask]
                
                # Set black background
                ax.set_facecolor("black")
                
                # Create hexbin plot
                hb = ax.hexbin(
                    x_data_clean,
                    y_data_clean,
                    gridsize=gridsize,
                    cmap=cmap,
                    mincnt=1,
                    norm=LogNorm(),
                )
                hexbin_plots.append(hb)
                
                # Add true values if available
                if true_values and x_var in true_values and y_var in true_values:
                    true_x = true_values[x_var]
                    true_y = true_values[y_var]
                    ax.scatter(true_x, true_y, marker='*', s=400, c='red',
                              edgecolors='black', linewidths=1.5, zorder=10)
                    ax.axvline(true_x, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.axhline(true_y, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
                # Apply bounds if requested
                if use_bounds:
                    x_bounds = None
                    y_bounds = None
                    
                    if param_bounds and x_var in param_bounds:
                        x_bounds = param_bounds[x_var]
                    elif blob_bounds and x_var in blob_bounds:
                        x_bounds = blob_bounds[x_var]
                    
                    if param_bounds and y_var in param_bounds:
                        y_bounds = param_bounds[y_var]
                    elif blob_bounds and y_var in blob_bounds:
                        y_bounds = blob_bounds[y_var]
                    
                    if x_bounds is not None:
                        ax.set_xlim(x_bounds)
                    if y_bounds is not None:
                        ax.set_ylim(y_bounds)
                
                # Set labels only on edges
                if i == n_vars - 1:  # Bottom row
                    ax.set_xlabel(label_map.get(x_var, x_var), fontsize=11)
                else:
                    ax.tick_params(labelbottom=False)
                
                if j == 0:  # Leftmost column
                    ax.set_ylabel(label_map.get(y_var, y_var), fontsize=11)
                else:
                    ax.tick_params(labelleft=False)
                
                ax.grid(alpha=0.3)
    
    # Add shared colorbar on the right
    if hexbin_plots:
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.85])
        fig.colorbar(hexbin_plots[-1], cax=cbar_ax, label='Count')
    
    # Save figure
    os.makedirs('mcmc_figures', exist_ok=True)
    if output_filename is None:
        var_str = '_'.join(var_names[:3])  # Use first 3 vars to avoid too long filename
        if len(var_names) > 3:
            var_str += f'_plus{len(var_names)-3}more'
        filename = f'mcmc_figures/custom_corner_{var_str}.pdf'
    else:
        filename = output_filename
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved {filename}")
    
    # Print statistics for each variable
    print("\n" + "=" * 60)
    print("CORNER PLOT VARIABLE STATISTICS")
    print("=" * 60)
    for var_name in var_names:
        data = var_data[var_name]
        median = np.median(data)
        q16, q84 = np.percentile(data, [16, 84])
        
        print(f"\n{label_map.get(var_name, var_name)}:")
        print(f"  Median: {median:.4f}")
        print(f"  68% CI: [{q16:.4f}, {q84:.4f}]")
        if true_values and var_name in true_values:
            print(f"  True: {true_values[var_name]:.4f}")
    print("=" * 60)


def plot_posterior_vs_prior(
    samples,
    blobs,
    var_name,
    burn_in=0,
    param_keys=None,
    blob_keys=None,
    label_map=None,
    true_values=None,
    param_bounds=None,
    blob_bounds=None,
    output_filename=None,
    figsize=(10, 6),
):
    """
    Create a publication-quality plot showing the evolution from prior to posterior
    distribution for a single variable, with optional true value overlay.
    
    Parameters
    ----------
    samples : ndarray
        MCMC samples array with shape (nsteps, nwalkers, ndim)
    blobs : ndarray
        Blob data array with shape (nsteps, nwalkers, nblobs)
    var_name : str
        Name of variable to plot (can be parameter or blob)
    burn_in : int, optional
        Number of burn-in steps to discard
    param_keys : list of str, optional
        List of parameter names corresponding to samples dimensions
    blob_keys : list of str, optional
        List of blob names corresponding to blobs dimensions
    label_map : dict, optional
        Mapping from variable names to display labels
    true_values : dict, optional
        Dictionary of true/reference values for each variable
    param_bounds : dict, optional
        Dictionary of bounds for parameters (used as prior bounds)
    blob_bounds : dict, optional
        Dictionary of bounds for blobs (used as prior bounds)
    output_filename : str, optional
        Custom output filename (default: auto-generated from var_name)
    figsize : tuple, optional
        Figure size (default: (10, 6))
    """
    if param_keys is None or blob_keys is None or label_map is None:
        raise ValueError("param_keys, blob_keys, and label_map are required")
    
    # Create mappings for parameter and blob indices
    param_map = {name: i for i, name in enumerate(param_keys)}
    blob_map = {name: i for i, name in enumerate(blob_keys)}
    
    # Extract data for the variable
    bounds = None
    if var_name in param_map:
        var_data = samples[burn_in:, :, param_map[var_name]].flatten()
        bounds = param_bounds.get(var_name) if param_bounds else None
    elif var_name in blob_map:
        var_data = blobs[burn_in:, :, blob_map[var_name]].flatten()
        bounds = blob_bounds.get(var_name) if blob_bounds else None
    else:
        raise ValueError(f"Variable {var_name} not found in parameters or blobs")
    
    if bounds is None:
        raise ValueError(f"Bounds not found for variable {var_name}")
    
    # Clean data
    var_data_clean = var_data[~np.isnan(var_data)]
    
    # Calculate statistics
    posterior_median = np.median(var_data_clean)
    posterior_std = np.std(var_data_clean)
    q16, q84 = np.percentile(var_data_clean, [16, 84])
    q10, q90 = np.percentile(var_data_clean, [10, 90])  # 80% credible interval
    
    # Create x range for plotting
    prior_low, prior_high = bounds
    prior_center = (prior_low + prior_high) / 2
    x_range = np.linspace(prior_low - 0.15*(prior_high-prior_low), 
                          prior_high + 0.15*(prior_high-prior_low), 1000)
    
    # Prior distribution (uniform)
    prior_pdf = uniform.pdf(x_range, loc=prior_low, scale=prior_high-prior_low)
    # Normalize for better visual comparison
    prior_pdf = prior_pdf / prior_pdf.max() * 0.8
    
    # Posterior distribution (KDE)
    kde = gaussian_kde(var_data_clean, bw_method='scott')
    posterior_pdf = kde(x_range)
    # Normalize
    posterior_pdf = posterior_pdf / posterior_pdf.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Plot prior (orange)
    ax.fill_between(x_range, prior_pdf, alpha=0.5, color='#E67E22', 
                    label='Prior', zorder=1)
    ax.plot(x_range, prior_pdf, color='#E67E22', linewidth=2.5, zorder=1)
    
    # Plot posterior (green)
    ax.fill_between(x_range, posterior_pdf, alpha=0.5, color='#2ECC71',
                    label='Posterior', zorder=2)
    ax.plot(x_range, posterior_pdf, color='#2ECC71', linewidth=2.5, zorder=2)
    
    # Get max height for annotations
    max_height = max(posterior_pdf.max(), prior_pdf.max())
    
    # Plot true value if available (blue dashed line)
    true_val = None
    if true_values and var_name in true_values:
        true_val = true_values[var_name]
    
    # Add vertical dashed lines at key positions
    ax.axvline(prior_center, color='#E67E22', linestyle='--', 
               linewidth=1.5, alpha=0.5, zorder=0)
    ax.axvline(posterior_median, color='#2ECC71', linestyle='--',
               linewidth=1.5, alpha=0.5, zorder=0)
    if true_val is not None:
        ax.axvline(true_val, color='#3498DB', linestyle='--',
                   linewidth=1.5, alpha=0.5, zorder=0)
    
    # Add text annotations below x-axis
    y_text = -0.08 * max_height
    ax.text(prior_center, y_text, 'Expectation', ha='center', va='top',
            fontsize=11, fontweight='normal')
    ax.text(posterior_median, y_text, 'Estimate', ha='center', va='top',
            fontsize=11, fontweight='normal')
    if true_val is not None:
        ax.text(true_val, y_text, 'Reality', ha='center', va='top',
                fontsize=11, fontweight='normal')
    
    # Add arrow annotations
    arrow_y = max_height * 1.08
    
    # 80% posterior distribution arrow (middle 80%)
    ax.annotate('', xy=(q90, arrow_y * 0.85), 
                xytext=(q10, arrow_y * 0.85),
                arrowprops=dict(arrowstyle='<->', color='#2ECC71', lw=2))
    ax.text((q10 + q90) / 2, arrow_y * 0.9,
            '80% Posterior', ha='center', va='bottom', fontsize=10, color='#2ECC71')
    
    # Prediction error arrow (if true value available)
    if true_val is not None:
        ax.annotate('', xy=(true_val, arrow_y), 
                    xytext=(posterior_median, arrow_y),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text((posterior_median + true_val) / 2, arrow_y * 1.05,
                'Prediction error', ha='center', va='bottom', fontsize=10)
    
    # Styling
    ax.set_xlabel(label_map.get(var_name, var_name), fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray')
    ax.set_ylim(bottom=-0.12*max_height, top=max_height*1.15)
    ax.set_xlim(x_range[0], x_range[-1])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Thicker tick marks
    ax.tick_params(width=1.5, labelsize=11)
    
    # Save figure
    os.makedirs('mcmc_figures', exist_ok=True)
    if output_filename is None:
        filename = f'mcmc_figures/posterior_vs_prior_{var_name}.pdf'
    else:
        filename = output_filename
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {filename}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"POSTERIOR VS PRIOR: {label_map.get(var_name, var_name)}")
    print(f"{'='*60}")
    print(f"Prior bounds: [{prior_low:.4f}, {prior_high:.4f}]")
    print(f"Prior center: {prior_center:.4f}")
    print(f"Posterior median: {posterior_median:.4f}")
    print(f"Posterior std: {posterior_std:.4f}")
    print(f"Posterior 68% CI: [{q16:.4f}, {q84:.4f}]")
    print(f"Posterior 80% CI: [{q10:.4f}, {q90:.4f}]")
    if true_val is not None:
        print(f"True value: {true_val:.4f}")
        print(f"Prediction error: {abs(posterior_median - true_val):.4f}")
        print(f"Error as % of prior range: {100 * abs(posterior_median - true_val) / (prior_high - prior_low):.2f}%")
    print(f"{'='*60}\n")


def plot_blob_distributions(samples, blobs, burn_in=0, blob_keys=None, blob_labels=None, blob_bounds=None, use_bounds=True):
    """
    Plot distributions of additional properties saved as blobs.
    """
    if blob_keys is None or blob_labels is None:
        raise ValueError("blob_keys and blob_labels are required")

    _, _, n_blobs = blobs.shape

    flat_blobs = {}
    for i, key in enumerate(blob_keys):
        if i >= n_blobs:
            break
        flat_data = blobs[burn_in:, :, i].flatten()
        flat_blobs[key] = flat_data[~np.isnan(flat_data)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, key in enumerate(blob_keys):
        ax = axes[idx]
        data = flat_blobs.get(key, np.array([]))

        if len(data) > 0:
            if use_bounds and blob_bounds and key in blob_bounds:
                bins = np.linspace(blob_bounds[key][0], blob_bounds[key][1], 30)
            else:
                bins = 30

            ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')

            median = np.median(data)
            q16, q84 = np.percentile(data, [16, 84])
            ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
            ax.axvline(q16, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvline(q84, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

            ax.set_xlabel(blob_labels.get(key, key), fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(
                f'{blob_labels.get(key, key)}\n{median:.2f} +{q84 - median:.2f} -{median - q16:.2f}',
                fontsize=10
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

            if use_bounds and blob_bounds and key in blob_bounds:
                ax.set_xlim(blob_bounds[key])
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(blob_labels.get(key, key), fontsize=11)

    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig('mcmc_figures/blob_distributions.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("Saved blob_distributions.pdf")

    print("\n" + "=" * 60)
    print("DERIVED QUANTITIES (BLOBS) STATISTICS")
    print("=" * 60)
    for key in blob_keys:
        data = flat_blobs.get(key, np.array([]))
        if len(data) > 0:
            q16, median, q84 = np.percentile(data, [16, 50, 84])
            print(f"{blob_labels.get(key, key)}:")
            print(f"  Median: {median:.4f}")
            print(f"  68% CI: [{q16:.4f}, {q84:.4f}]")
            print(f"  Range: [{np.min(data):.4f}, {np.max(data):.4f}]")
        else:
            print(f"{blob_labels.get(key, key)}: No valid data")
    print("=" * 60)


def plot_mcmc_results(
    samples,
    log_prob,
    param_labels,
    yobs,
    burn_in=0,
    param_keys=None,
    param_bounds=None,
    use_bounds=True,
):
    """
    Create comprehensive plots of MCMC results.
    """
    nsteps, _, n_dim = samples.shape

    os.makedirs('mcmc_figures', exist_ok=True)

    print("Creating trace plots...")
    fig, axes = plt.subplots(n_dim + 1, 1, figsize=(10, 2.5 * (n_dim + 1)), sharex=True)

    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3, color='k', linewidth=0.5)
        ax.set_ylabel(param_labels[i])
        if burn_in > 0:
            ax.axvline(burn_in, color='r', linestyle='--', label='Burn-in')
        ax.grid(alpha=0.3)

        if use_bounds and param_keys and param_bounds and i < len(param_keys):
            param_key = param_keys[i]
            if param_key in param_bounds:
                ax.set_ylim(param_bounds[param_key])

    axes[n_dim].plot(log_prob, alpha=0.3, color='k', linewidth=0.5)
    axes[n_dim].set_ylabel('Log Probability')
    axes[n_dim].set_xlabel('Step')
    if burn_in > 0:
        axes[n_dim].axvline(burn_in, color='r', linestyle='--', label='Burn-in')
    axes[n_dim].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcmc_figures/trace_plots.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("Saved trace_plots.pdf")

    flat_samples = samples[burn_in:, :, :].reshape(-1, n_dim)
    print(f"Chain shape after burn-in: {flat_samples.shape}")

    if corner is not None:
        print("Creating corner plot...")
        fig = corner.corner(
            flat_samples,
            labels=param_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12}
        )
        plt.savefig('mcmc_figures/corner_plot.pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        print("Saved corner_plot.pdf")

    print("\n" + "=" * 60)
    print("PARAMETER STATISTICS")
    print("=" * 60)
    for i, name in enumerate(param_labels):
        mcmc_samples = flat_samples[:, i]
        q = np.percentile(mcmc_samples, [16, 50, 84])
        q_m, q_med, q_p = q
        print(f"{name}:")
        print(f"  Median: {q_med:.4f}")
        print(f"  -1σ: {q_med - q_m:.4f}")
        print(f"  +1σ: {q_p - q_med:.4f}")
        print(f"  68% CI: [{q_m:.4f}, {q_p:.4f}]")
    print("=" * 60)

    print("\nCreating posterior predictive plot...")
    # plot_posterior_predictive(flat_samples, yobs)


def plot_posterior_predictive(samples, yobs, run_planetprofile, n_samples=100):
    """
    Plot posterior predictive distribution vs observations.
    """
    indices = np.random.randint(len(samples), size=n_samples)

    k2_pred = []
    h2_pred = []

    print(f"Generating {n_samples} posterior predictions...")
    for idx in indices:
        theta = samples[idx]
        try:
            ysim, _ = run_planetprofile(theta)
            if not np.isnan(ysim).any():
                k2_pred.append(ysim[0])
                h2_pred.append(ysim[1])
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue

    k2_pred = np.array(k2_pred)
    h2_pred = np.array(h2_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(k2_pred, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(yobs[0], color='red', linestyle='--', linewidth=2, label='Observed')
    axes[0].axvline(np.median(k2_pred), color='green', linestyle='-', linewidth=2, label='Predicted median')
    axes[0].set_xlabel('k2 (Love number)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Posterior Predictive: k2', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(h2_pred, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(yobs[1], color='red', linestyle='--', linewidth=2, label='Observed')
    axes[1].axvline(np.median(h2_pred), color='green', linestyle='-', linewidth=2, label='Predicted median')
    axes[1].set_xlabel('h2 (Love number)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Posterior Predictive: h2', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcmc_figures/posterior_predictive.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("Saved posterior_predictive.pdf")

    print("\nPosterior Predictive Statistics:")
    print(f"k2: observed={yobs[0]:.4f}, predicted median={np.median(k2_pred):.4f} ± {np.std(k2_pred):.4f}")
    print(f"h2: observed={yobs[1]:.4f}, predicted median={np.median(h2_pred):.4f} ± {np.std(h2_pred):.4f}")

def plot_methanogenesis_affinities(samples, blobs, true_affinity):
    """
    Plot methanogenesis affinities.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(blobs[:, 0], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(true_affinity[0], color='red', linestyle='--', linewidth=2, label='True')
    axes[0].set_xlabel('Methanogenesis Affinity', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Methanogenesis Affinity', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)