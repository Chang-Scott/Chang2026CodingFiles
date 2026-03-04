"""
MCMC Functions
Combined module containing configuration, forward model, and likelihood functions.
"""
import numpy as np
import copy
import time
import os
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
PARAM_KEYS = ['rho_core', 'rho_sil', 'log_fH2', 'Tb_K']
DERIVED_KEYS = ['ice_thickness_km', 'ocean_thickness_km', 'core_radius_km',
                'ocean_mean_density_kgm3', 'mean_conductivity_Sm']
OBSERVABLE_KEYS = ['k2', 'h2', 'mag_r_orb', 'mag_i_orb', 'mag_r_syn', 'mag_i_syn']

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
    'rho_sil': [3000, 4000],
    'log_fH2': [-12.0, -3.0],
    'Tb_K': [250, 273]
}

DERIVED_PLOTTING_BOUNDS = {
    'ice_thickness_km': [0, 140],
    'ocean_thickness_km': [0, 150],
    'core_radius_km': [200, 600],
    'ocean_mean_density_kgm3': [1000, 1300],
    'mean_conductivity_Sm': [0, 5]
}

OBSERVABLE_PLOTTING_BOUNDS = {
    'MoI':  [0.350, 0.360],
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
BURN_IN = 5000
N_STEPS = 10000
# Set number of parallel processes
N_PROCESSES = globalParams.maxCores
N_WALKERS = N_PROCESSES * 2

# Observation uncertainties
MOI_ERR = 0.001
K2_ERR = 0.018
H2_ERR = 0.1
MAG_ERR = 1.5

# Covariance matrix (6x6 for all observables)
COV = np.diag([MOI_ERR**2, K2_ERR**2, H2_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2])

MOVES = [(emcee.moves.StretchMove(a = 5.0), 0.7), (emcee.moves.DEMove(), 0.3)]
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
    rho_core, rho_sil, log_fH2, Tb_K = theta
    
    # Set parameters
    planetRun.Core.rhoFe_kgm3 = rho_core
    planetRun.Do.ICEIh_THICKNESS = False
    planetRun.Sil.rhoSilWithCore_kgm3 = rho_sil
    planetRun.Bulk.Tb_K = Tb_K
    
    # Round redox state to nearest 0.05 to reduce computat
    log_fH2 = round(log_fH2 / 0.05) * 0.05
    planetRun.Ocean.comp = Replicate_Zolotov_H2([log_fH2])[0]
    
    # Run forward model
    time_start = time.time()
    planetRun, _ = PlanetProfile(planetRun, global_params)
    time_end = time.time()
    print(f"Time taken for PlanetProfile: {time_end - time_start} seconds")
    if log_fH2 < -8 and log_fH2 > -5:
        print(planetRun.Do.VALID)
    if not planetRun.Do.VALID:
        # Return NaN arrays (size based on inversion type)
        n_obs = len(OBSERVABLE_INDICES[inversion_type])
        observables = np.full(n_obs, np.nan)
        blobs = np.full(11, np.nan)
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
        # Observables (saved again for easy access in plotting)
        planetRun.CMR2mean,
        planetRun.Gravity.kAmp,
        planetRun.Gravity.hAmp,
        np.real(planetRun.Magnetic.Bi1Tot_nT[0]),
        np.imag(planetRun.Magnetic.Bi1Tot_nT[0]),
        np.real(planetRun.Magnetic.Bi1Tot_nT[1]),
        np.imag(planetRun.Magnetic.Bi1Tot_nT[1])
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
