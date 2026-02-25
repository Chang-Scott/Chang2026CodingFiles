"""
MCMC Functions
Combined module containing configuration, forward model, and likelihood functions.
"""
import numpy as np
import copy
from PlanetProfile.Main import PlanetProfile
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2
import emcee

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


# ============================================================================
# CONFIGURATION - PARAMETER BOUNDS
# ============================================================================

PARAM_BOUNDS = {
    'rho_core': [5150, 8000],
    'rho_sil': [3000, 4000],
    'log_fH2': [-12.0, -3.0],
    'Tb_K': [250, 273]
}

DERIVED_BOUNDS = {
    'ice_thickness_km': [0, 90],
    'ocean_thickness_km': [0, 200],
    'core_radius_km': [200, 600],
    'ocean_mean_density_kgm3': [1000, 1300],
    'mean_conductivity_Sm': [0, 5]
}

OBSERVABLE_BOUNDS = {
    'k2': [0.25, 0.35],
    'h2': [1.1, 1.3],
    'mag_r_orb': [7.5, 12.5],
    'mag_i_orb': [0, 5],
    'mag_r_syn': [200, 215],
    'mag_i_syn': [6.5, 19],
}

ALL_BOUNDS = {**PARAM_BOUNDS, **DERIVED_BOUNDS, **OBSERVABLE_BOUNDS}


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
    'k2': r'$k_2$ (Love number)',
    'h2': r'$h_2$ (Love number)',
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
BURN_IN = 100
N_STEPS = 5000

# Observation uncertainties
K2_ERR = 0.018
H2_ERR = 0.1
MAG_ERR = 1.5

# Covariance matrix (6x6 for all observables)
COV = np.diag([K2_ERR**2, H2_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2])


MOVES = [
    (emcee.moves.StretchMove(), 0.9),
    (emcee.moves.DEMove(), 0.1)
]

# ============================================================================
# FORWARD MODEL
# ============================================================================

def run_planetprofile(theta, planet_template, global_params):
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
    replicate_zolotov_fn : callable
        Function to generate ocean composition
        
    Returns
    -------
    observables : array
        (k2, h2, mag_r_orb, mag_i_orb, mag_r_syn, mag_i_syn)
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
    planetRun, _ = PlanetProfile(planetRun, global_params)
    if log_fH2 < -8 and log_fH2 > -5:
        print(planetRun.Do.VALID)
    if not planetRun.Do.VALID:
        # Return NaN arrays
        observables = np.full(6, np.nan)
        blobs = np.full(11, np.nan)
        return observables, blobs
    
    # Extract observables
    observables = np.array([
        planetRun.Gravity.kAmp,
        planetRun.Gravity.hAmp,
        np.real(planetRun.Magnetic.Bi1Tot_nT[0]),
        np.imag(planetRun.Magnetic.Bi1Tot_nT[0]),
        np.real(planetRun.Magnetic.Bi1Tot_nT[1]),
        np.imag(planetRun.Magnetic.Bi1Tot_nT[1])
    ])
    
    # Combine derived quantities and observables for blobs
    blobs = np.array([
        # Derived quantities
        planetRun.zb_km,
        planetRun.D_km,
        planetRun.Core.Rmean_m / 1e3,
        planetRun.Ocean.rhoMean_kgm3,
        planetRun.Ocean.sigmaMean_Sm,
        # Observables (saved again for easy access in plotting)
        planetRun.Gravity.kAmp,
        planetRun.Gravity.hAmp,
        np.real(planetRun.Magnetic.Bi1Tot_nT[0]),
        np.imag(planetRun.Magnetic.Bi1Tot_nT[0]),
        np.real(planetRun.Magnetic.Bi1Tot_nT[1]),
        np.imag(planetRun.Magnetic.Bi1Tot_nT[1])
    ])
    
    return observables, blobs


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
# Add at module level in mcmc_functions.py
_last_log_fH2 = {}  # Dictionary keyed by thread/process ID

def log_probability(theta, yobs, cov, forward_model_fn):
    
    lp = log_prior(theta)
    if not np.isfinite(lp):
        # Return NaN blobs for rejected samples
        nan_blobs = np.full(len(BLOB_KEYS), np.nan)
        return -np.inf, nan_blobs
    
    observables, blobs = forward_model_fn(theta)
    ll = log_likelihood(observables, yobs, cov)
    
    return lp + ll, blobs
