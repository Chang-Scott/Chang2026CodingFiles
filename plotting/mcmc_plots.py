import os

import numpy as np
import matplotlib.pyplot as plt

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

    ax_main.hexbin(var1_data, var2_data, gridsize=50, cmap='Blues', mincnt=1, alpha=0.8)
    ax_main.scatter(var1_data[::10], var2_data[::10], alpha=0.1, s=1, c='steelblue')

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