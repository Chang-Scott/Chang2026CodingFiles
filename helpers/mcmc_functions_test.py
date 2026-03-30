"""
MCMC Functions Test
Combined module containing configuration, forward model, and likelihood functions.
"""
import numpy as np
import copy
import time
import os
from multiprocessing import Pool
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from PlanetProfile.Main import PlanetProfile
from PlanetProfile.Utilities.defineStructs import EOSlist
from helpers.pp_common import loadUserSettings
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2
import emcee
from PlanetProfile.GetConfig import Params as globalParams
loadUserSettings('Inversion')
# ============================================================================
# CONFIGURATION - DATA STRUCTURE
# ============================================================================

# Define all variable names in order
DO_PARALLEL = True
TEST_MODE = 'CoreDensityAndRadius'
if TEST_MODE == 'CoreDensityAndRadius':
    PARAM_KEYS = ['rho_core', 'r_core_m', 'ice_thickness_km', 'ocean_thickness_km', 'rhoOcean_kgm3']
elif TEST_MODE == 'Densities':
    PARAM_KEYS = ['rho_core', 'rho_sil', 'ice_thickness_km', 'ocean_thickness_km', 'rhoOcean_kgm3']
DERIVED_KEYS = ['ice_thickness_km', 'ocean_thickness_km', 'core_radius_km',
                'ocean_mean_density_kgm3', 'mean_conductivity_Sm', 'hydrosphere_thickness_km', 'rho_sil']
OBSERVABLE_KEYS = ['MoI', 'k2', 'h2', 'mag_r_orb', 'mag_i_orb', 'mag_r_syn', 'mag_i_syn']

# Combined blob keys (derived + observables)
BLOB_KEYS = DERIVED_KEYS + OBSERVABLE_KEYS

# Observable indices for different inversion types
OBSERVABLE_INDICES = {
    'Gravity': [0], # MoI only
    'GravityandTides': [0, 1, 2],  # MoI and tides
    'MagneticInduction': [3, 4, 5, 6],  # Magnetic induction
    'Joint': [0, 1, 2, 3, 4, 5, 6]  # MoI, tides, mag_r_orb, mag_i_orb, mag_r_syn, mag_i_syn
}

# ============================================================================
# CONFIGURATION - PARAMETER BOUNDS
# ============================================================================
PARAM_BOUNDS = {
    'rho_core': [5150, 8000],
    'rho_sil': [2500, 4500],
    'ice_thickness_km': [1, 200],
    'ocean_thickness_km': [1, 200],
    'rhoOcean_kgm3': [1000, 1300],
    'r_core_m': [1, 1000*1e3]
}

DERIVED_PLOTTING_BOUNDS = {
    'hydrosphere_thickness_km': [1, 300],
    'core_radius_km': [200, 600],
    'ocean_mean_density_kgm3': [1000, 1300],
    'mean_conductivity_Sm': [0, 5],
    'hydrosphere_thickness_km': [0, 300],
}

OBSERVABLE_PLOTTING_BOUNDS = {
    'MoI':  [0.350, 0.357],
    'k2': [0.25, 0.35],
    'h2': [1.1, 1.3],
    'mag_r_orb': [7.5, 12.5],
    'mag_i_orb': [0, 5],
    'mag_r_syn': [200, 215],
    'mag_i_syn': [6.5, 19],
}

ALL_BOUNDS = {**PARAM_BOUNDS, **DERIVED_PLOTTING_BOUNDS, **OBSERVABLE_PLOTTING_BOUNDS}


# ============================================================================
# CONFIGURATION - LABELS FOR PLOTTING
# ============================================================================

PARAM_LABELS = {
    'rho_core': r'$\rho_{\mathrm{core}}$ (kg/m³)',
    'rho_sil': r'$\rho_{\mathrm{sil}}$ (kg/m³)',
    'log_fH2': r'log $f_{\mathrm{H}_2}$',
    'Tb_K': r'$T_b$ (K)',
}

DERIVED_LABELS = {
    'ice_thickness_km': 'Ice Shell Thickness (km)',
    'ocean_thickness_km': 'Ocean Thickness (km)',
    'core_radius_km': 'Core Radius (km)',
    'ocean_mean_density_kgm3': 'Ocean Mean Density (kg/m³)',
    'mean_conductivity_Sm': 'Mean Conductivity (S/m)'
}

OBSERVABLE_LABELS = {
    'MoI': r'Moment of Inertia',
    'k2': r'$k_2$ Love number',
    'h2': r'$h_2$ Love number',
    'mag_r_orb': 'Mag Real Orbital (nT)',
    'mag_i_orb': 'Mag Imag Orbital (nT)',
    'mag_r_syn': 'Mag Real Synodic (nT)',
    'mag_i_syn': 'Mag Imag Synodic (nT)',
}

ALL_LABELS = {**PARAM_LABELS, **DERIVED_LABELS, **OBSERVABLE_LABELS}


# ============================================================================
# CONFIGURATION - MCMC SETTINGS
# ============================================================================

N_DIM = len(PARAM_KEYS)
BURN_IN = 1000
N_STEPS = 10000
N_MC_SAMPLES = 10000
# Set number of parallel processes
N_PROCESSES = globalParams.maxCores
N_WALKERS = N_PROCESSES * 2 - 2
N_PRIOR_SAMPLES = 5000

# Observation uncertainties
MOI_ERR = 0.001
K2_ERR = 0.018
H2_ERR = 0.1
MAG_ERR = 1.5

# Covariance matrix (6x6 for all observables)
COV = np.diag([MOI_ERR**2, K2_ERR**2, H2_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2])

MOVES = [(emcee.moves.StretchMove(a = 4), 0.7), (emcee.moves.DEMove(), 0.3)]

# Prior-only Monte Carlo (same uniform priors as initialize_walkers / log_prior).
# Independent draws; not tied to MCMC length. Scale up/down as needed.
PRIOR_N_SAMPLES = min(10000, max(500, N_WALKERS * 100))
PRIOR_BURN_IN = 0

# ============================================================================
# FORWARD MODEL
# ============================================================================
    
def run_planetprofile(theta, planet_template, global_params, inversion_type):
    """
    Run PlanetProfile forward model.
    
    Parameters
    ----------
    theta : array-like
        Parameter values [rho_core, rho_sil, log_fH2, Tb_K]
    planet_template : Planet object
        Template planet to copy
    global_params : Params object
        Global configuration
    inversion_type : str
        Type of inversion ('Gravity', 'MagneticInduction', or 'Joint')
        
    Returns
    -------
    observables : array
        Filtered observables based on inversion_type
    blobs : array
        Combined array of derived quantities and observables
        (ice_thickness_km, ocean_thickness_km, core_radius_km,
         ocean_mean_density_kgm3, mean_conductivity_Sm,
         k2, h2, mag_r_orb, mag_i_orb, mag_r_syn, mag_i_syn)
    """
    planetRun = copy.deepcopy(planet_template)
    # Unpack parameters
    if TEST_MODE == 'CoreDensityAndRadius':
        rho_core, r_core_m, ice_thickness_km, ocean_thickness_km, rhoOcean_kgm3 = theta
        # Set core density and radius
        planetRun.Core.rhoFe_kgm3 = rho_core
        planetRun.Core.Rset_m = r_core_m
        planetRun.Do.ConstantProps['Inner'] = True
        planetRun.Do.ConstantProps['Ice'] = True
        planetRun.Do.ConstantProps['Ocean'] = True
        planetRun.Ocean.IceConstantProps['Ih'].rho_kgm3 = 920
        planetRun.Ocean.ConstantProps.rho_kgm3 = rhoOcean_kgm3
        planetRun.Do.SPECIFY_CORE_DENSITY_AND_RADIUS = True
    elif TEST_MODE == 'Densities':
        
        rho_core, rho_sil, ice_thickness_km, ocean_thickness_km, rhoOcean_kgm3 = theta
        # Set densities
        planetRun.Core.rhoFe_kgm3 = rho_core
        planetRun.Sil.rhoSilWithCore_kgm3 = rho_sil
        planetRun.Do.ConstantProps['Inner'] = True
        planetRun.Do.ConstantProps['Ice'] = True
        planetRun.Do.ConstantProps['Ocean'] = True
        planetRun.Ocean.IceConstantProps['Ih'].rho_kgm3 = 920
        planetRun.Ocean.ConstantProps.rho_kgm3 = rhoOcean_kgm3
        planetRun.Do.SPECIFY_CORE_DENSITY_AND_RADIUS = False
        
    # Set ice thickness
    g_ms2 = 1.315
    planetRun.Bulk.PbSet_MPa = 900 * g_ms2 * ice_thickness_km * 1e3 / 1e6
    
    # set ocean thickness
    planetRun.Do.SPECIFY_HYDROSPHERE_SEAFLOOR_PRESSURE = True
    planetRun.Ocean.PHydroSeafloorSet_MPa = rhoOcean_kgm3 * g_ms2 * ocean_thickness_km * 1e3 / 1e6 + planetRun.Bulk.PbSet_MPa
    
    # Set other settings
    planetRun.Do.ICEIh_THICKNESS = False
    planetRun.Ocean.comp = 'PureH2O'
    global_params.SKIP_INDUCTION = True
    global_params.SKIP_GRAVITY = True
    global_params.CALC_CONDUCT = False
    global_params.CALC_VISCOSITY = False
    global_params.CALC_SEISMIC = False
    global_params.CALC_OCEAN_PROPS = False
    global_params.CALC_ASYM = False
    global_params.CALC_NEW_ASYM = False
    global_params.CALC_NEW_INDUCT = False
    global_params.CALC_NEW_GRAVITY = False
    global_params.CALC_NEW_REF = False
    planetRun.Do.NO_ICE_CONVECTION = True
    
    # Run forward model
    time_start = time.time()
    planetRun, _ = PlanetProfile(planetRun, global_params)
    time_end = time.time()
    print(f"Time taken for PlanetProfile: {time_end - time_start} seconds")
    if not planetRun.Do.VALID:
        # Return NaN arrays (size based on inversion type)
        n_obs = len(OBSERVABLE_INDICES[inversion_type])
        observables = np.full(n_obs, np.nan)
        blobs = np.full(len(BLOB_KEYS), np.nan)
        return observables, blobs
    
    # Extract observables based on inversion type
    gravity_obs = np.full(1, np.nan)
    tides_obs = np.full(2, np.nan)
    magnetic_obs = np.full(4, np.nan)
    
    if inversion_type in ['Gravity', 'GravityandTides', 'Joint']:
        gravity_obs = np.array([
            planetRun.CMR2mean
        ])
    if inversion_type in ['GravityandTides', 'Joint']:
        tides_obs = np.array([
            planetRun.Gravity.kAmp,
            planetRun.Gravity.hAmp
        ])
    
    if inversion_type in ['MagneticInduction', 'Joint']:
        magnetic_obs = np.array([
            np.real(planetRun.Magnetic.Bi1Tot_nT[0]),
            np.imag(planetRun.Magnetic.Bi1Tot_nT[0]),
            np.real(planetRun.Magnetic.Bi1Tot_nT[1]),
            np.imag(planetRun.Magnetic.Bi1Tot_nT[1])
        ])
    
    # Combine all observables and filter based on inversion type
    all_observables = np.concatenate([gravity_obs, tides_obs, magnetic_obs])
    indices = OBSERVABLE_INDICES[inversion_type]
    filtered_observables = all_observables[indices]
    
    # Combine derived quantities and observables for blobs (always include all)
    blobs = np.array([
        # Derived quantities
        planetRun.zb_km,
        planetRun.D_km,
        planetRun.Core.Rmean_m / 1e3,
        planetRun.Ocean.rhoMean_kgm3,
        planetRun.Ocean.sigmaMean_Sm,
        planetRun.zb_km + planetRun.D_km,
        planetRun.Sil.rhoSilWithCore_kgm3,
        # Observables (saved again for easy access in plotting)
        planetRun.CMR2mean,
        planetRun.Gravity.kAmp,
        planetRun.Gravity.hAmp,
        np.real(planetRun.Magnetic.Bi1Tot_nT[0]) if not global_params.SKIP_INDUCTION else np.nan,
        np.imag(planetRun.Magnetic.Bi1Tot_nT[0]) if not global_params.SKIP_INDUCTION else np.nan,
        np.real(planetRun.Magnetic.Bi1Tot_nT[1]) if not global_params.SKIP_INDUCTION else np.nan,
        np.imag(planetRun.Magnetic.Bi1Tot_nT[1]) if not global_params.SKIP_INDUCTION else np.nan,
    ])
    
    return filtered_observables, blobs


# ============================================================================
# DATA ACCESS UTILITIES
# ============================================================================

# Combined list of all variable names (params + blobs)
ALL_KEYS = PARAM_KEYS + BLOB_KEYS


def combine_samples_blobs(samples, blobs):
    """
    Combine samples and blobs into a single array for easier handling.
    
    Parameters
    ----------
    samples : ndarray
        MCMC samples with shape (nsteps, nwalkers, ndim)
    blobs : ndarray
        Blob data with shape (nsteps, nwalkers, nblobs)
        
    Returns
    -------
    combined : ndarray
        Combined array with shape (nsteps, nwalkers, ndim + nblobs)
    var_names : list
        List of variable names corresponding to columns in combined array
    """
    combined = np.concatenate([samples, blobs], axis=2)
    var_names = ALL_KEYS
    return combined, var_names


def stack_prior_as_mcmc_chains(thetas, blobs_list):
    """
    Stack independent prior draws into arrays with the same layout as emcee's
    get_chain / get_blobs for use with combine_samples_blobs and plotting.

    Parameters
    ----------
    thetas : ndarray, shape (n, ndim)
        Rows are independent uniform prior draws (same as initialize_walkers).
    blobs_list : sequence of ndarray, length n
        Each element is shape (nblobs,) from run_planetprofile.

    Returns
    -------
    samples : ndarray, shape (n, 1, ndim)
    blobs : ndarray, shape (n, 1, nblobs)
    log_prob : ndarray, shape (n, 1)
        Filled with NaN (no posterior).
    """
    thetas = np.asarray(thetas)
    blobs_arr = np.stack(blobs_list, axis=0)
    samples = thetas[:, np.newaxis, :]
    blobs = blobs_arr[:, np.newaxis, :]
    log_prob = np.full((samples.shape[0], 1), np.nan, dtype=float)
    return samples, blobs, log_prob


# ============================================================================
# LIKELIHOOD FUNCTIONS
# ============================================================================

def initialize_walkers(n_walkers):
    """Initialize walker positions from uniform distributions within bounds."""
    param_bounds_array = np.array([PARAM_BOUNDS[key] for key in PARAM_KEYS])
    return np.random.uniform(
        low=param_bounds_array[:, 0], 
        high=param_bounds_array[:, 1], 
        size=(n_walkers, N_DIM)
    )
    
def log_prior(theta):
    """
    Check if parameters are within bounds.
    
    Parameters
    ----------
    theta : array-like
        Parameter values
        
    Returns
    -------
    float
        0.0 if within bounds, -inf otherwise
    """
    for i, (param_key, value) in enumerate(zip(PARAM_KEYS, theta)):
        bounds = PARAM_BOUNDS[param_key]
        if not (bounds[0] <= value <= bounds[1]):
            return -np.inf
    return 0.0


def log_likelihood(observables, yobs, cov):
    """
    Compute log likelihood for Gaussian errors.
    
    Parameters
    ----------
    observables : array
        Forward model observables (k2, h2, mag_r_orb, mag_i_orb, mag_r_syn, mag_i_syn)
    yobs : array
        Observed values
    cov : array
        Covariance matrix
        
    Returns
    -------
    float
        Log likelihood value
    """
    if np.isnan(observables).any():
        return -np.inf
    
    residual = observables - yobs
    return -0.5 * residual.T @ np.linalg.inv(cov) @ residual

def log_probability(theta, yobs, cov, forward_model_fn, inversion_type):
    
    lp = log_prior(theta)
    if not np.isfinite(lp):
        # Return NaN blobs for rejected samples
        nan_blobs = np.full(len(BLOB_KEYS), np.nan)
        return -np.inf, nan_blobs
    
    observables, blobs = forward_model_fn(theta, inversion_type)
    ll = log_likelihood(observables, yobs, cov)
    
    return lp + ll, blobs


# ============================================================================
# PRIOR SAMPLING
# ============================================================================

def run_prior_sampling(forward_model_fn, inversion_type, n_samples=N_PRIOR_SAMPLES,
                       n_processes=None, save_dir='.', do_parallel=DO_PARALLEL):
    """
    Draw samples uniformly from prior bounds and evaluate the forward model in parallel.

    Outputs are saved and returned in the same (nsteps, nwalkers, nvars) format as the
    MCMC chain so that all existing plotting functions can be reused with burn_in=0.

    Parameters
    ----------
    forward_model_fn : callable
        Function with signature (theta, inversion_type) -> (observables, blobs).
        Use the same forward_model_wrapper defined in the main script.
    inversion_type : str
        One of 'Gravity', 'GravityandTides', 'MagneticInduction', 'Joint'.
    n_samples : int
        Total number of prior draws to evaluate.
    n_processes : int or None
        Number of parallel worker processes. Defaults to N_PROCESSES.
    save_dir : str
        Directory in which to save .npy output files.
    do_parallel : bool
        Whether to use multiprocessing.

    Returns
    -------
    samples_arr : ndarray, shape (n_samples, 1, ndim)
    blobs_arr   : ndarray, shape (n_samples, 1, nblobs)
    log_prob_arr: ndarray, shape (n_samples, 1)  -- all zeros (flat prior)
    """
    if n_processes is None:
        n_processes = N_PROCESSES

    print("=" * 60)
    print("PRIOR SAMPLING WITH PLANETPROFILE")
    print("=" * 60)
    print(f"Inversion type  : {inversion_type}")
    print(f"Number of samples: {n_samples}")
    print(f"Parallel         : {do_parallel} ({n_processes} processes)")
    print("=" * 60)

    # Draw uniformly from prior bounds
    param_bounds_array = np.array([PARAM_BOUNDS[key] for key in PARAM_KEYS])
    theta_samples = np.random.uniform(
        low=param_bounds_array[:, 0],
        high=param_bounds_array[:, 1],
        size=(n_samples, N_DIM)
    )

    args_list = [(theta, inversion_type) for theta in theta_samples]

    if do_parallel:
        with Pool(processes=n_processes) as pool:
            results = pool.starmap(forward_model_fn, args_list)
    else:
        results = [forward_model_fn(theta, inversion_type) for theta, _ in args_list]

    blobs_flat = np.array([r[1] for r in results], dtype=float)  # (n_samples, nblobs)

    # Reshape to (n_samples, 1, *) so existing plot helpers work with burn_in=0
    samples_arr = theta_samples[:, np.newaxis, :]       # (n_samples, 1, ndim)
    blobs_arr = blobs_flat[:, np.newaxis, :]             # (n_samples, 1, nblobs)
    log_prob_arr = np.zeros((n_samples, 1))              # flat prior -> log prob = 0

    # Save outputs
    os.makedirs(save_dir, exist_ok=True)
    tag = f'prior_{inversion_type}'
    np.save(os.path.join(save_dir, f'mcmc_chain_{tag}.npy'), samples_arr)
    np.save(os.path.join(save_dir, f'mcmc_blobs_{tag}.npy'), blobs_arr)
    np.save(os.path.join(save_dir, f'mcmc_log_prob_{tag}.npy'), log_prob_arr)
    print(f"\nSaved: mcmc_chain_{tag}.npy, mcmc_blobs_{tag}.npy, mcmc_log_prob_{tag}.npy")

    return samples_arr, blobs_arr, log_prob_arr
