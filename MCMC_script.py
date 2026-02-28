"""
MCMC Analysis for Europa Interior Structure
Main orchestration script.
"""
import numpy as np
import emcee
import os
import importlib
from multiprocessing import Pool
import logging
import copy

# Import PlanetProfile
from PlanetProfile.GetConfig import Params as globalParams, FigLbl
from PlanetProfile.Utilities.defineStructs import Constants
from PlanetProfile.Main import LoadPPfiles, PlanetProfile

# Import local modules
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2, SetSettings
SetSettings(save_to_txt_file=False, output_figures=False, mat_output_dir='./', txt_output_dir='./', figure_output_dir='./')
from helpers.mcmc_functions import *
from helpers.mcmc_functions import INVERSION_TYPE, OBSERVABLE_INDICES

from helpers.pp_common import loadUserSettings, CopyCarefully
from plotting.mcmc_plots import (
    plot_mcmc_results, plot_2d_corner, plot_custom_corner, plot_posterior_vs_prior,
    plot_variable_histograms
)

# Setup
logger = logging.getLogger('PlanetProfile')
logger.setLevel(logging.PROFILE)
np.random.seed(123)

# Setup directories
this_dir = os.path.dirname(os.path.abspath(__file__))
Europa_dir = os.path.join(this_dir, 'Europa')
baseModelFileName = 'PPEuropa_ExploreBaseModel_wFeCore_Fixed.py'

# Global variables for the worker processes
loadUserSettings('Inversion')
CopyCarefully(os.path.join('ModelFiles', baseModelFileName), os.path.join('Europa', baseModelFileName))
globalParams, loadNames = LoadPPfiles(globalParams, fNames=[baseModelFileName], bodyname='Europa')
Planet = importlib.import_module(loadNames[0]).Planet


def forward_model_wrapper(theta):
    """
    Wrapper for run_planetprofile that captures global context.
    This function is picklable for multiprocessing.
    
    Parameters
    ----------
    theta : array
        Parameter values
        
    Returns
    -------
    observables : array
        Forward model observables (filtered based on INVERSION_TYPE)
    blobs : array
        Derived quantities and observables
    """
    return run_planetprofile(theta, Planet, globalParams)


def initialize_walkers(n_walkers):
    """Initialize walker positions from uniform distributions within bounds."""
    param_bounds_array = np.array([PARAM_BOUNDS[key] for key in PARAM_KEYS])
    return np.random.uniform(
        low=param_bounds_array[:, 0], 
        high=param_bounds_array[:, 1], 
        size=(n_walkers, N_DIM)
    )


def run_mcmc(yobs, n_walkers, n_steps, burn_in):
    """
    Run MCMC sampling.
    
    Parameters
    ----------
    yobs : array
        Observed values (full array, will be filtered based on INVERSION_TYPE)
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of production steps
    burn_in : int
        Number of burn-in steps
        
    Returns
    -------
    samples : ndarray
        MCMC samples with shape (nsteps, nwalkers, ndim)
    blobs : ndarray
        Blob data with shape (nsteps, nwalkers, nblobs)
    log_prob : array
        Log probability values
    """
    # Filter observables and covariance based on inversion type
    indices = OBSERVABLE_INDICES[INVERSION_TYPE]
    yobs_filtered = yobs[indices]
    cov_filtered = COV[np.ix_(indices, indices)]
    
    print("="*60)
    print("MCMC SAMPLING WITH PLANETPROFILE")
    print("="*60)
    print(f"Inversion type: {INVERSION_TYPE}")
    print(f"Number of walkers: {n_walkers}")
    print(f"Number of dimensions: {N_DIM}")
    print(f"\nObservables being used:")
    obs_names = [OBSERVABLE_KEYS[i] for i in indices]
    for name, val in zip(obs_names, yobs_filtered):
        print(f"  {name}: {val:.4f}")
    print("="*60)
    
    # Initialize walkers
    p0 = initialize_walkers(n_walkers)
    p0[2:5, 2] = np.random.uniform(-3, -4, 3)   
    p0[0:2, 2] = np.random.uniform(-12, -10, 2)
    p0[5:n_walkers, 2] = np.random.uniform(-8, -6, n_walkers - 5)
    print(f"\nUsing {N_PROCESSES} parallel processes")
    
    with Pool(processes=N_PROCESSES) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, 
            N_DIM,
            log_probability,
            args=[yobs_filtered, cov_filtered, forward_model_wrapper],
            pool=pool,
            moves=MOVES
        )
        
        # Run burn-in
        print(f"\nRunning burn-in for {burn_in} steps...")
        state = sampler.run_mcmc(p0, burn_in, progress=True)
        
        # Run production
        print(f"\nRunning production for {n_steps} steps...")
        sampler.run_mcmc(state, n_steps, progress=True)
    
    print("\nMCMC sampling complete!")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Get results
    samples = sampler.get_chain()
    blobs = sampler.get_blobs()
    log_prob = sampler.get_log_prob()
    
    # Save results
    print("\nSaving results...")
    np.save(f'mcmc_chain_{INVERSION_TYPE}.npy', samples)
    np.save(f'mcmc_blobs_{INVERSION_TYPE}.npy', blobs)
    np.save(f'mcmc_log_prob_{INVERSION_TYPE}.npy', log_prob)
    np.save(f'mcmc_acceptance_fraction_{INVERSION_TYPE}.npy', sampler.acceptance_fraction)
    np.save(f'mcmc_burn_in_{INVERSION_TYPE}.npy', burn_in)
    print(f"Saved: mcmc_chain_{INVERSION_TYPE}.npy, mcmc_blobs_{INVERSION_TYPE}.npy, mcmc_log_prob_{INVERSION_TYPE}.npy, mcmc_acceptance_fraction_{INVERSION_TYPE}.npy, mcmc_burn_in_{INVERSION_TYPE}.npy")
    
    return samples, blobs, log_prob


def mcmc(inversion_type):
    global INVERSION_TYPE
    INVERSION_TYPE = inversion_type
    # Calculate true observations
    print("Calculating true observations...")
    true_params = {
        'rho_core': 5150.0,
        'rho_sil': 3500.0,
        'log_fH2': -10.2,
        'Tb_K': 266.0
    }
    
    TruePlanet = copy.deepcopy(Planet)
    TruePlanet.Do.ICEIh_THICKNESS = False
    TruePlanet.Core.rhoFe_kgm3 = true_params['rho_core']
    TruePlanet.Sil.rhoSilWithCore_kgm3 = true_params['rho_sil']
    TruePlanet.Ocean.comp = Replicate_Zolotov_H2([true_params['log_fH2']])[0]
    TruePlanet.Bulk.Tb_K = true_params['Tb_K']
    
    TruePlanet, _ = PlanetProfile(TruePlanet, globalParams)
    
    # Extract true observables
    yobs = np.array([
        TruePlanet.Gravity.kAmp,
        TruePlanet.Gravity.hAmp,
        np.real(TruePlanet.Magnetic.Bi1Tot_nT[0]),
        np.imag(TruePlanet.Magnetic.Bi1Tot_nT[0]),
        np.real(TruePlanet.Magnetic.Bi1Tot_nT[1]),
        np.imag(TruePlanet.Magnetic.Bi1Tot_nT[1])
    ])
    
    print(f"True observations: k2={yobs[0]:.4f}, h2={yobs[1]:.4f}")
    print(f"Magnetic (orbital): {yobs[2]:.2f} + {yobs[3]:.2f}i nT")
    print(f"Magnetic (synodic): {yobs[4]:.2f} + {yobs[5]:.2f}i nT")
    
    # Add derived quantities and observables to true_params for plotting
    true_params.update({
        'ice_thickness_km': TruePlanet.zb_km,
        'ocean_thickness_km': TruePlanet.D_km,
        'core_radius_km': TruePlanet.Core.Rmean_m / 1e3,
        'ocean_mean_density_kgm3': TruePlanet.Ocean.rhoMean_kgm3,
        'mean_conductivity_Sm': TruePlanet.Ocean.sigmaMean_Sm,
        'k2': yobs[0],
        'h2': yobs[1],
        'mag_r_orb': yobs[2],
        'mag_i_orb': yobs[3],
        'mag_r_syn': yobs[4],
        'mag_i_syn': yobs[5],
    })
    
    # Run or load MCMC
    calc_new = True
    
    if calc_new:
        samples, blobs, log_prob = run_mcmc(yobs, N_WALKERS, N_STEPS, BURN_IN)
    else:
        print(f"\nLoading MCMC data for {INVERSION_TYPE} inversion...")
        samples = np.load(f'mcmc_chain_{INVERSION_TYPE}.npy')
        blobs = np.load(f'mcmc_blobs_{INVERSION_TYPE}.npy')
        log_prob = np.load(f'mcmc_log_prob_{INVERSION_TYPE}.npy')
        print(f"  Total steps: {samples.shape[0]}")
        print(f"  Burn-in steps: {BURN_IN}")
        print(f"  Production steps: {samples.shape[0] - BURN_IN}")
    
    # Combine samples and blobs into single array with variable names
    print("\nCombining samples and blobs into unified array...")
    mcmc_data, var_names = combine_samples_blobs(samples, blobs)
    print(f"  Combined data shape: {mcmc_data.shape}")
    print(f"  Variable names: {var_names}")
    

    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    
    # Trace plots and corner plot
    plot_mcmc_results(samples, log_prob)
    
    # Histogram diagnostic plots
    print("\nGenerating histogram diagnostic plots...")
    
    """# Filter data based on redox state (log_fH2) between -4 and -3
    log_fH2_idx = var_names.index('log_fH2')
    mask = (mcmc_data[BURN_IN:, :, log_fH2_idx] >= -4) & (mcmc_data[BURN_IN:, :, log_fH2_idx] <= -3)
    
    # Apply filter and reshape to maintain (steps, walkers, variables) format
    filtered_data = mcmc_data[BURN_IN:][mask]
    n_filtered = filtered_data.shape[0]
    n_vars = mcmc_data.shape[-1]
    mcmc_data = filtered_data.reshape(n_filtered, 1, n_vars)
    # Print statistics for mag_i_syn
    mag_i_syn_idx = var_names.index('mag_i_orb')
    mag_i_syn_data = filtered_data[:, mag_i_syn_idx]
    print(f"\nMag_i_orb statistics (filtered data):")
    print(f"  Min: {np.min(mag_i_syn_data):.3f}")
    print(f"  Max: {np.max(mag_i_syn_data):.3f}")
    print(f"  Mean: {np.mean(mag_i_syn_data):.3f}")
    print(f"  Std: {np.std(mag_i_syn_data):.3f}")"""
    plot_variable_histograms(
        mcmc_data,
        var_names=var_names,
        plot_vars=['k2', 'h2', 'mag_r_orb', 'mag_i_orb', 'mag_r_syn', 'mag_i_syn'],  # Plot all parameters
        true_values=true_params,
    )
    
    # Custom corner plot
    print("\nGenerating custom corner plot...")
    plot_custom_corner(
        mcmc_data,
        var_names=var_names,
        plot_vars=['log_fH2', 'Tb_K'],
        true_values=true_params,
    )
    
    print("\nGenerating custom corner plot...")
    plot_custom_corner(
        mcmc_data,
        var_names=var_names,
        plot_vars=['log_fH2', ''],
        true_values=true_params,
    )
    # Posterior vs prior plots
    print("\nGenerating posterior vs prior plots...")
    plot_posterior_vs_prior(
        mcmc_data,
        var_names=var_names,
        var_name='log_fH2',
        true_values=true_params,
    )
    
    print("\n" + "="*60)
    print("MCMC RUN COMPLETE!")
    print("Check the 'mcmc_figures' directory for plots.")
    print("="*60)

if __name__ == "__main__":
    mcmc(inversion_type='Joint')
    mcmc(inversion_type='Gravity')
    mcmc(inversion_type='MagneticInduction')