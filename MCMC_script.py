import numpy as np
import emcee
import os
import importlib
from multiprocessing import Pool
import logging
import matplotlib.pyplot as plt
from PlanetProfile.GetConfig import Params as globalParams, FigLbl
from PlanetProfile.Utilities.defineStructs import Constants
from PlanetProfile.Main import LoadPPfiles, PlanetProfile
from PlanetProfile.Thermodynamics.OceanProps import LiquidOceanPropsCalcs
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2, SetSettings
from helpers.pp_common import loadUserSettings, CopyCarefully
from plotting.mcmc_plots import plot_mcmc_results, plot_blob_distributions, plot_2d_corner, plot_custom_corner, plot_posterior_vs_prior
import time
import copy
from scipy.interpolate import make_interp_spline
SetSettings(save_to_txt_file=False, output_figures=False, mat_output_dir='./', txt_output_dir='./', figure_output_dir='./')
logger = logging.getLogger('PlanetProfile')
logger.setLevel(logging.PROFILE)
np.random.seed(123)


# Choose the 'measurement' uncertainties
k2_err = 0.018
h2_err = 0.1
magnetic_err = 1.5
cov = np.array([[k2_err**2, 0, 0, 0, 0, 0], 
                [0, h2_err**2, 0, 0, 0, 0], 
                [0, 0, magnetic_err**2, 0, 0, 0], 
                [0, 0, 0, magnetic_err**2, 0, 0], 
                [0, 0, 0, 0, magnetic_err**2, 0], 
                [0, 0, 0, 0, 0, magnetic_err**2]]) 

# Setup directories
this_dir = os.path.dirname(os.path.abspath(__file__))
Europa_dir = os.path.join(this_dir, 'Europa')
baseModelFileName = 'PPEuropa_ExploreBaseModel_wFeCore_Fixed.py'

# Global constants for blob indexing
BLOB_KEYS = ['ice_thickness_km', 'ocean_thickness_km', 'core_radius_km',
             'ocean_mean_density_kgm3', 'mean_conductivity_Sm']
BLOB_LABELS = {
    'ice_thickness_km': 'Ice Shell Thickness (km)',
    'ocean_thickness_km': 'Ocean Thickness (km)',
    'core_radius_km': 'Core Radius (km)',
    'ocean_mean_density_kgm3': 'Ocean Mean Density',
    'mean_conductivity_Sm': 'Mean Conductivity'
}


# Parameter bounds
PARAM_KEYS = ['rho_core', 'rho_sil', 'log_fH2', 'Tb_K']
PARAM_BOUNDS = {
    'rho_core': [5150, 8000],
    'rho_sil': [3000, 4000],
    'log_fH2': [-12.0, -3.0],
    'Tb_K': [250, 273]
}

# Blob bounds (expected physical ranges for plotting)
BLOB_BOUNDS = {
    'ice_thickness_km': [0, 90],
    'ocean_thickness_km': [0, 200],
    'core_radius_km': [200, 600],
    'ocean_mean_density_kgm3': [1000, 1300],
    'mean_conductivity_Sm': [0, 5]
}

# ============================================================================
# MCMC Setup
# ============================================================================

N_DIM = 4  # number of parameters [coreDensity, silicateDensity, logfH2, iceThickness]
N_WALKERS = globalParams.maxCores * 2 # number of MCMC walkers (Set to 2x number of CPU cores)
BURN_IN = 1000 # number of burn-in steps
N_STEPS = 10000 # number of production steps
# Get parameter bounds as array for initialization
param_bounds_array = np.array([PARAM_BOUNDS[key] for key in PARAM_KEYS])

# Initialize walker positions from uniform distributions within bounds
p0 = np.random.uniform(
    low=param_bounds_array[:, 0], 
    high=param_bounds_array[:, 1], 
    size=(N_WALKERS, N_DIM)
)

# Global variables for the worker processes
loadUserSettings('Inversion')
CopyCarefully(os.path.join('ModelFiles', baseModelFileName), os.path.join('Europa', baseModelFileName))
globalParams, loadNames = LoadPPfiles(globalParams, fNames=[baseModelFileName], bodyname='Europa')
Planet = importlib.import_module(loadNames[0]).Planet


def run_MCMC(yobs_param, planet_base, configParams):
    """
    Calculate the MCMC results.
    
    Parameters
    ----------
    burn_in : int
        Number of burn-in steps
    yobs_param : np.array
        Observed values to use for MCMC
    """
    yobs = yobs_param
    
    print("="*60)
    print("MCMC SAMPLING WITH PLANETPROFILE")
    print("="*60)
    print("N_WAL")
    print(f"Number of walkers: {N_WALKERS}")
    print(f"Number of dimensions: {N_DIM}")
    print(f"\nObserved values: k2={yobs[0]:.4f}, h2={yobs[1]:.4f}")
    print("="*60)
    
    # Set number of parallel processes
    # Use None to automatically detect CPU count, or set manually
    n_processes = configParams.maxCores  # Adjust based on your CPU cores
    nburn = BURN_IN
    nsteps = N_STEPS
    configParams.DO_PARALLEL = True
    print(f"\nUsing {n_processes} parallel processes")
    if configParams.DO_PARALLEL:
        pool = Pool(processes=n_processes)
        
        sampler = emcee.EnsembleSampler(
            N_WALKERS, 
            N_DIM, 
            log_probability, 
            args=[yobs, cov],
            pool=pool
        )
        
        print(f"\nRunning burn-in for {nburn} steps...")
        state = sampler.run_mcmc(p0, nburn, progress=True)
        
        # Run production (DO NOT reset - keep burn-in data)
        print(f"\nRunning production for {nsteps} steps...")
        sampler.run_mcmc(state, nsteps, progress=True)
        
        pool.close()
        pool.join()
    else:
        sampler = emcee.EnsembleSampler(
            N_WALKERS, 
            N_DIM, 
            log_probability, 
            args=[yobs, cov]
        )
        # Run burn-in
        print(f"\nRunning burn-in for {nburn} steps...")
        state = sampler.run_mcmc(p0, nburn, progress=True)
        
        # Run production (DO NOT reset - keep burn-in data)
        print(f"\nRunning production for {nsteps} steps...")
        sampler.run_mcmc(state, nsteps, progress=True)
    
    print("\nMCMC sampling complete!")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Save results
    print("\nSaving results...")
    
    # Get all data from sampler - includes BOTH burn-in and production
    # Chain shape: (nburn + nsteps, N_WALKERS, N_DIM)
    full_chain = sampler.get_chain()
    full_log_prob = sampler.get_log_prob()
    raw_blobs = sampler.get_blobs()
    
    total_steps, n_walkers_actual, ndims = full_chain.shape
    print(f"Total steps (burn-in + production): {total_steps}")
    print(f"Burn-in steps: {nburn}")
    print(f"Production steps: {nsteps}")
    
    print("Extracting scalar blobs...")
    scalar_blobs = np.zeros((total_steps, n_walkers_actual, len(BLOB_KEYS)))
    
    for i in range(total_steps):
        for j in range(n_walkers_actual):
            scalar_blobs[i, j, :] = raw_blobs[i, j][0]
    
    # Save all results (including burn-in)
    np.save('mcmc_chain.npy', full_chain)
    np.save('mcmc_log_prob.npy', full_log_prob)
    np.save('mcmc_acceptance_fraction.npy', sampler.acceptance_fraction)
    np.save('mcmc_blobs.npy', scalar_blobs)
    np.save('mcmc_burn_in.npy', nburn)  # Save burn-in count for reference
    print("Saved: mcmc_chain.npy, mcmc_log_prob.npy, mcmc_acceptance_fraction.npy, mcmc_blobs.npy, mcmc_affinity_data.npy, mcmc_burn_in.npy")
    
    return full_chain, full_log_prob, sampler.acceptance_fraction, scalar_blobs



def run_planetprofile(theta):
    """
    Wrapper function to run PlanetProfile with given theta parameters.
    
    Parameters
    ----------
    theta : array-like
        [coreDensity, silicateDensity, logfH2redoxState, iceShellThickness]
        
    Returns
    -------
    ysim : np.array
        Simulated observables [k2, h2, rOrbital, iOrbital, rSynodic, iSynodic] - Love numbers and magnetic field components
    scalar_blobs : np.array
        Scalar properties (shape: 5) - ice thickness, ocean thickness, core radius, ocean density, conductivity
    methanogenesis_affinities : np.array
        Affinity values for methanogenesis (shape: 6) - one per H2/H2O ratio
    """
    planetRun = copy.deepcopy(Planet)
    coreDensity, silicateDensity, logfH2redoxState, Tb_K = theta
    
    # Update Planet parameters
    planetRun.Core.rhoFe_kgm3 = coreDensity
    planetRun.Do.ICEIh_THICKNESS = False
    planetRun.Sil.rhoSilWithCore_kgm3 = silicateDensity
    planetRun.Bulk.Tb_K = Tb_K
    planetRun.Ocean.comp = Replicate_Zolotov_H2([logfH2redoxState])[0]
    
    
    time_start = time.time()
    planetRun, _ = PlanetProfile(planetRun, globalParams)
    time_end = time.time()
    total_time = time_end - time_start

    if total_time > 1.3:
        print(f"HERE")
       
    # Extract Love numbers from PlanetProfile results
    # These should be extracted from the Exploration or Planet object
    # Adjust based on actual PlanetProfile output structure
    if planetRun.Do.VALID is False:
        ysim = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        scalar_blobs = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    else:
        k2_sim = planetRun.Gravity.kAmp
        
        h2_sim = planetRun.Gravity.hAmp
        rOrbital_sim = np.real(planetRun.Magnetic.Bi1Tot_nT[0])
        iOrbital_sim = np.imag(planetRun.Magnetic.Bi1Tot_nT[0])
        rSynodic_sim = np.real(planetRun.Magnetic.Bi1Tot_nT[1])
        iSynodic_sim = np.imag(planetRun.Magnetic.Bi1Tot_nT[1])
        ysim = np.array([k2_sim, h2_sim, rOrbital_sim, iOrbital_sim, rSynodic_sim, iSynodic_sim])
        
        # Extract additional properties to save as "blobs"
        # These are NOT used in the likelihood but are saved for analysis
        # Store as array in consistent order (see BLOB_KEYS global)
        scalar_blobs = np.array([
            planetRun.zb_km,
            planetRun.D_km,
            planetRun.Core.Rmean_m / 1e3,
            planetRun.Ocean.rhoMean_kgm3,
            planetRun.Ocean.sigmaMean_Sm
        ])
    
    return ysim, scalar_blobs


def log_likelihood(ysim, yobs, cov):
    """Compute log likelihood for Gaussian errors."""
    residual = ysim - yobs
    return -0.5 * residual.T @ np.linalg.inv(cov) @ residual


def log_prior(theta):
    """Define prior probability bounds on parameters."""
    # Check bounds for each parameter
    for i, (param_key, value) in enumerate(zip(PARAM_KEYS, theta)):
        bounds = PARAM_BOUNDS[param_key]
        if not (bounds[0] <= value <= bounds[1]):
            return -np.inf
    return 0.0


def log_probability(theta, yobs, cov):
    """
    Compute log posterior probability for emcee.
    
    This function combines prior and likelihood, and runs PlanetProfile
    as the forward model.
    
    Returns
    -------
    log_prob : float
        Log posterior probability
    blobs : tuple
        (scalar_blobs, affinity_array) - both arrays to save
    """
    # Check prior first (fast rejection)
    lp = log_prior(theta)
    if not np.isfinite(lp):
        # Return -inf and NaN blobs for rejected samples
        nan_scalar_blobs = np.full(len(BLOB_KEYS), np.nan)
        return -np.inf, nan_scalar_blobs

    ysim, scalar_blobs = run_planetprofile(theta)
    
    if np.isnan(ysim).any():
        return -np.inf, scalar_blobs
    
    # Compute likelihood
    ll = log_likelihood(ysim, yobs, cov)
    
    return lp + ll, scalar_blobs

if __name__ == "__main__":
    CopyCarefully(os.path.join('ModelFiles', baseModelFileName), os.path.join('Europa', baseModelFileName))
    # Calculate true observations ONLY in main process
    print("Calculating true observations...")
    rhoSil_true = 3500.0
    rhoCore_true = 5150.0
    logfH2_true = -10.2
    Tb_true = 266
    TruePlanet = copy.deepcopy(Planet)
    
    TruePlanet.Do.ICEIh_THICKNESS = False
    TruePlanet.Core.rhoFe_kgm3 = rhoCore_true
    TruePlanet.Sil.rhoSilWithCore_kgm3 = rhoSil_true
    TruePlanet.Ocean.comp = Replicate_Zolotov_H2([logfH2_true])[0]
    TruePlanet.Bulk.Tb_K = Tb_true
    time_start = time.time()
    TruePlanet, _ = PlanetProfile(TruePlanet, globalParams)
    time_end = time.time()
    print(f"PlanetProfile run time: {time_end - time_start:.2f} seconds")
    k2_true = TruePlanet.Gravity.kAmp
    h2_true = TruePlanet.Gravity.hAmp
    rOrbital = np.real(TruePlanet.Magnetic.Bi1Tot_nT[0])
    iOrbital = np.imag(TruePlanet.Magnetic.Bi1Tot_nT[0])
    rSynodic = np.real(TruePlanet.Magnetic.Bi1Tot_nT[1])
    iSynodic = np.imag(TruePlanet.Magnetic.Bi1Tot_nT[1])
    yobs_main = np.array([k2_true, h2_true, rOrbital, iOrbital, rSynodic, iSynodic])
    
    print(f"True observations: k2={k2_true:.4f}, h2={h2_true:.4f}")
    print(f"Magnetic (orbital): {rOrbital:.2f} + {iOrbital:.2f}i nT")
    print(f"Magnetic (synodic): {rSynodic:.2f} + {iSynodic:.2f}i nT")
    
    # Parameter names for plotting
    param_names = [
        r'$\rho_{\mathrm{core}}$ (kg/m³)',
        r'$\rho_{\mathrm{sil}}$ (kg/m³)',
        r'log $f_{\mathrm{H}_2}$',
        r'$T_b$ (K)'
    ]
    calcNew = True
    if calcNew == True:
        time_start = time.time()
        samples, log_prob, acceptance_fraction, scalar_blobs = run_MCMC(yobs_main, Planet, globalParams)
        time_end = time.time()
        print(f"Time taken to calculate mcmc results: {time_end - time_start:.2f} seconds")
    else:
        samples = np.load('mcmc_chain.npy')
        log_prob = np.load('mcmc_log_prob.npy')
        acceptance_fraction = np.load('mcmc_acceptance_fraction.npy')
        scalar_blobs = np.load('mcmc_blobs.npy')
        burn_in = int(np.load('mcmc_burn_in.npy'))  # Load the burn-in count
        yobs = yobs_main  # Set for plotting
        print(f"\nLoaded MCMC data:")
        print(f"  Burn-in steps: {burn_in}")
        print(f"  Total steps: {samples.shape[0]}")
        print(f"  Production steps: {samples.shape[0] - burn_in}")
    
    # Prepare true values dictionary for plotting
    true_vals = {
        'rho_core': rhoCore_true,
        'rho_sil': rhoSil_true,
        'log_fH2': logfH2_true,
        'Tb_K': Tb_true,
        'ice_thickness_km': TruePlanet.zb_km,
        'ocean_thickness_km': TruePlanet.D_km,
        'core_radius_km': TruePlanet.Core.Rmean_m / 1e3,
        'ocean_mean_density_kgm3': TruePlanet.Ocean.rhoMean_kgm3,
        'mean_conductivity_Sm': TruePlanet.Ocean.sigmaMean_Sm
    }
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_mcmc_results(
        samples,
        log_prob,
        param_names,
        yobs_main,
        burn_in=BURN_IN,
        param_keys=PARAM_KEYS,
        param_bounds=PARAM_BOUNDS,
    )
    
    # Plot scalar blob distributions
    plot_blob_distributions(
        samples,
        scalar_blobs,
        burn_in=BURN_IN,
        blob_keys=BLOB_KEYS,
        blob_labels=BLOB_LABELS,
        blob_bounds=BLOB_BOUNDS,
    )

     # Plot affinity scatter plot
    #plot_affinity_scatter(samples, affinity_data, methanogenesis_affinities_reference, burn_in=burn_in)
    
    # Example 2D corner plots
    print("\nGenerating 2D corner plots...")
    # Plot parameter vs ice thickness
    label_map = {
        'rho_core': r'$\rho_{\mathrm{core}}$ (kg/m³)',
        'rho_sil': r'$\rho_{\mathrm{sil}}$ (kg/m³)',
        'log_fH2': r'log $f_{\mathrm{H}_2}$',
        'Tb_K': r'$T_b$ (K)',
        **BLOB_LABELS,
    }
    
    # Example: Custom corner plot with selected variables
    print("\nGenerating custom corner plot...")
    plot_custom_corner(
        samples,
        scalar_blobs,
        var_names=['log_fH2', 'ice_thickness_km', 'ocean_thickness_km'],
        burn_in=BURN_IN,
        param_keys=PARAM_KEYS,
        blob_keys=BLOB_KEYS,
        label_map=label_map,
        true_values=true_vals,
        param_bounds=PARAM_BOUNDS,
        blob_bounds=BLOB_BOUNDS,
    )
    
    # Example: Posterior vs Prior plots for key parameters
    plot_posterior_vs_prior(
        samples,
        scalar_blobs,
        var_name='log_fH2',
        burn_in=BURN_IN,
        param_keys=PARAM_KEYS,
        blob_keys=BLOB_KEYS,
        label_map=label_map,
        true_values=true_vals,
        param_bounds=PARAM_BOUNDS,
        blob_bounds=BLOB_BOUNDS,
    )
    
    print("\n" + "="*60)
    print("MCMC RUN COMPLETE!")
    print("Check the 'mcmc_figures' directory for plots.")
    print("="*60)
