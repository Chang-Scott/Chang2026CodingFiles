"""
MCMC Functions
Combined module containing configuration, forward model, and likelihood functions.
"""
from PlanetProfile.Thermodynamics.Reaktoro.reaktoroProps import SetupReaktoroDatabases
import numpy as np
import copy
import time
import os
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from PlanetProfile.Main import PlanetProfile
from PlanetProfile.Utilities.defineStructs import EOSlist
from helpers.pp_common import loadUserSettings
from PlanetProfile.Thermodynamics.Reaktoro.reaktoroProps import SetupReaktoroDatabases
from PlanetProfile.Thermodynamics.HydroEOS import GetOceanEOS, GetTfreeze, GetPfreeze
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2
import emcee
from PlanetProfile.GetConfig import Params as globalParams
loadUserSettings('Inversion')
# ============================================================================
# MELTING CURVE LOOKUP TABLE
# ============================================================================

# Module-level constants for lookup table
MELTING_TABLE_FILE = 'melting_lookup_table.npz'
MELTING_TABLE = None  # Will be lazily loaded
_EOS_CACHE = {}  # Cache for ocean EOS objects



def generate_melting_lookup_table(
):
    """
    Generate lookup table of melting temperatures for different redox states and pressures.
    
    Parameters
    ----------
    redox_range : tuple
        (min, max) log_fH2 values
    n_redox : int
        Number of redox state points
    pressure_range : tuple
        (min, max) pressure values in MPa
    n_pressure : int
        Number of pressure points
        
    Returns
    -------
    dict
        Dictionary with 'redox_states', 'pressures', and 'Tfreeze' arrays
    """
    # Create grid
    redox_states = np.linspace(-12.0, -3.0, 36)
    Tb_K = np.arange(240, 274, 0.05)
    P_MPa = np.arange(0.1, 200, 0.1)
    # Initialize output array
    Tfreeze = np.zeros((len(redox_states), len(P_MPa)))
    
    # Loop through redox states
    EOSlist.loaded['ReaktoroDatabases'] = SetupReaktoroDatabases()
    for i, log_fH2 in enumerate(redox_states):

        # Get ocean EOS for this redox state
        # Round to reduce cache misses
        oceanComp = Replicate_Zolotov_H2([log_fH2])[0]
        
        oceanEOS = GetOceanEOS(P_MPa = P_MPa, T_K = Tb_K, compstr = oceanComp, wOcean_ppt = None, elecType = None, MELT = True)
        
        phases = oceanEOS.fn_phase(P_MPa, Tb_K, grid=True).astype(int)

        # Find where phase decreases along temperature
        phase_jump = (np.diff(phases, axis=1) == -1)   # shape (nP, nT-1)

        # Since exactly one per row, argmax gives correct column index
        j = phase_jump.argmax(axis=1)                  # shape (nP,)

        Tfreeze[i, :] = Tb_K[j + 1]
        
    
    # Save to file
    print(f"Saving lookup table to {MELTING_TABLE_FILE}...")
    np.savez(MELTING_TABLE_FILE,
             redox_states=redox_states,
             pressures=P_MPa,
             Tfreeze=Tfreeze)
    
    print("Lookup table generation complete!")
    
    return {
        'redox_states': redox_states,
        'pressures': P_MPa,
        'Tfreeze': Tfreeze
    }


def _ensure_melting_table_loaded():
    """
    Ensure melting lookup table is loaded or generated.
    
    Returns
    -------
    dict
        Melting table dictionary
    """
    global MELTING_TABLE
    
    if MELTING_TABLE is None:
        if os.path.exists(MELTING_TABLE_FILE):
            print(f"Loading melting lookup table from {MELTING_TABLE_FILE}")
            data = np.load(MELTING_TABLE_FILE)
            MELTING_TABLE = {
                'redox_states': data['redox_states'],
                'pressures': data['pressures'],
                'Tfreeze': data['Tfreeze']
            }
            print(f"  Loaded table with {len(MELTING_TABLE['redox_states'])} redox states "
                  f"and {len(MELTING_TABLE['pressures'])} pressure points")
        else:
            print(f"Melting lookup table not found at {MELTING_TABLE_FILE}")
            MELTING_TABLE = generate_melting_lookup_table()
    
    return MELTING_TABLE


def interpolate_melting_temperature(log_fH2, P_anchor_MPa):
    """
    Interpolate melting temperature from lookup table.
    
    Parameters
    ----------
    log_fH2 : float
        Log fugacity of H2
    P_anchor_MPa : float
        Anchor pressure in MPa
        
    Returns
    -------
    float
        Interpolated melting temperature in K
    """
    table = _ensure_melting_table_loaded()
    
    # Create interpolator (cached internally by scipy for repeated calls)
    interp = RectBivariateSpline(
        table['redox_states'],
        table['pressures'],
        table['Tfreeze'],
        kx=1, ky=1  # Linear interpolation
    )
    
    # Evaluate at requested point
    T_melt = float(interp(log_fH2, P_anchor_MPa))
    
    return T_melt

def interpolate_melting_pressure(log_fH2, T_anchor_K):
    """
    Interpolate pressure from lookup table.
    
    Parameters
    ----------
    log_fH2 : float
    """
    table = _ensure_melting_table_loaded()
    
    # Create interpolator (cached internally by scipy for repeated calls)
    interp = RectBivariateSpline(
        table['redox_states'],
        table['pressures'],
        table['Tfreeze'],
        kx=1, ky=1  # Linear interpolation
    )
    
    # Evaluate at requested point
    P_melt = float(interp(log_fH2, T_anchor_K))
    
    return P_melt

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
# Set number of parallel processes
N_PROCESSES = 4
N_WALKERS = N_PROCESSES * 2

# Observation uncertainties
K2_ERR = 0.018
H2_ERR = 0.1
MAG_ERR = 1.5

# Covariance matrix (6x6 for all observables)
COV = np.diag([K2_ERR**2, H2_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2, MAG_ERR**2])

# Custom move proposal widths
SIGMA_RHO_CORE = 100.0
SIGMA_RHO_SIL = 50.0
SIGMA_LOG_FH2 = 3.0
SIGMA_TB = 0.5


# ============================================================================
# CUSTOM MCMC MOVE
# ============================================================================

class JointMoveWithMeltingCurve(emcee.moves.Move):
    """
    Custom MCMC move that proposes all parameters jointly while respecting
    the thermodynamic coupling between redox state (log_fH2) and ice-ocean
    boundary temperature (Tb) via melting curves.
    
    The move proposes:
    - rho_core, rho_sil, log_fH2 from independent Gaussian proposals
    - Tb from a Gaussian centered on the melting temperature corresponding
      to the new redox state at the anchor pressure from the old state
    
    This is an asymmetric proposal requiring a Hastings correction.
    """
    
    def __init__(self, sigma_rho_core, sigma_rho_sil, sigma_log_fH2, sigma_Tb):
        """
        Initialize the custom move.
        
        Parameters
        ----------
        sigma_rho_core : float
            Proposal width for core density (kg/m³)
        sigma_rho_sil : float
            Proposal width for silicate density (kg/m³)
        sigma_log_fH2 : float
            Proposal width for log H2 fugacity
        sigma_Tb : float
            Proposal width for ice-ocean boundary temperature (K)
        """
        self.sigma_rho_core = sigma_rho_core
        self.sigma_rho_sil = sigma_rho_sil
        self.sigma_log_fH2 = sigma_log_fH2
        self.sigma_Tb = sigma_Tb
        
        # Ensure lookup table is loaded
        _ensure_melting_table_loaded()
    
    def propose(self, model, state):
        """
        Propose new positions for walkers.
        
        Parameters
        ----------
        model : emcee.Model
            The model being sampled
        state : emcee.State
            Current state of the ensemble
            
        Returns
        -------
        state : emcee.State
            Updated state with accepted proposals
        accepted : ndarray
            Boolean array indicating which proposals were accepted
        """
        time_start = time.time()
        # Get current positions
        coords = state.coords  # Shape: (n_walkers, n_dim)
        n_walkers, n_dim = coords.shape
        
        # Initialize arrays for new positions and log proposal ratios
        new_coords = np.copy(coords)
        log_q_ratio = np.zeros(n_walkers)
        
        # Process each walker to generate proposals
        for i in range(n_walkers):
            # Current parameters
            rho_core_old, rho_sil_old, log_fH2_old, Tb_old = coords[i]
            
            # Propose rho_core, rho_sil, log_fH2 from independent Gaussians
            rho_core_new = rho_core_old + model.random.randn() * self.sigma_rho_core
            rho_sil_new = rho_sil_old + model.random.randn() * self.sigma_rho_sil
            log_fH2_new = log_fH2_old + model.random.randn() * self.sigma_log_fH2
            
            # Compute anchor pressure from old state
            # This is the pressure on the old melting curve at Tb_old
            P_anchor_MPa = interpolate_melting_pressure(log_fH2_old, Tb_old)
            
            # Get melting temperature at anchor pressure for new redox state
            mu_Tb_fwd = interpolate_melting_temperature(log_fH2_new, P_anchor_MPa)
            
            # Propose Tb_new from Gaussian centered at mu_Tb_fwd
            Tb_new = mu_Tb_fwd + model.random.randn() * self.sigma_Tb
            
            # Compute backward proposal mean for Hastings correction
            # Need to find what mu would be for backward proposal
            P_anchor_bwd_MPa = interpolate_melting_pressure(log_fH2_new, Tb_new)
            mu_Tb_bwd = interpolate_melting_temperature(log_fH2_old, P_anchor_bwd_MPa)
            
            # Compute log proposal ratio (backward / forward)
            # Forward: q(theta_new | theta_old) includes N(Tb_new; mu_Tb_fwd, sigma_Tb^2)
            # Backward: q(theta_old | theta_new) includes N(Tb_old; mu_Tb_bwd, sigma_Tb^2)
            # For other parameters, proposals are symmetric so they cancel
            
            log_q_fwd = norm.logpdf(Tb_new, loc=mu_Tb_fwd, scale=self.sigma_Tb)
            log_q_bwd = norm.logpdf(Tb_old, loc=mu_Tb_bwd, scale=self.sigma_Tb)
            log_q_ratio[i] = log_q_bwd - log_q_fwd
            
            # Store new coordinates
            new_coords[i] = [rho_core_new, rho_sil_new, log_fH2_new, Tb_new]
        
        # Compute log probabilities for proposed positions
        new_log_probs, new_blobs = model.compute_log_prob_fn(new_coords)
        
        # Perform Metropolis-Hastings acceptance test
        accepted = np.zeros(n_walkers, dtype=bool)
        for i in range(n_walkers):
            # Acceptance probability: min(1, exp(log_q_ratio + new_log_prob - old_log_prob))
            # In log space: accept if log_q_ratio + new_log_prob - old_log_prob > log(uniform)
            lnpdiff = log_q_ratio[i] + new_log_probs[i] - state.log_prob[i]
            if lnpdiff > np.log(model.random.rand()):
                accepted[i] = True
        
        # Update state with accepted proposals
        new_state = state.copy()
        new_state.coords[accepted] = new_coords[accepted]
        new_state.log_prob[accepted] = new_log_probs[accepted]
        if new_blobs is not None and state.blobs is not None:
            new_state.blobs[accepted] = new_blobs[accepted]
        time_end = time.time()
        print(f"Time taken for JointMoveWithMeltingCurve: {time_end - time_start} seconds")
        return new_state, accepted


# MCMC move mixture
MOVES = [
    (emcee.moves.StretchMove(a=5.0), 0.5),
    (emcee.moves.DEMove(), 0.2),
    """(JointMoveWithMeltingCurve(
        sigma_rho_core=SIGMA_RHO_CORE,
        sigma_rho_sil=SIGMA_RHO_SIL,
        sigma_log_fH2=SIGMA_LOG_FH2,
        sigma_Tb=SIGMA_TB
    ), 0.3)"""
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
    time_start = time.time()
    planetRun, _ = PlanetProfile(planetRun, global_params)
    time_end = time.time()
    print(f"Time taken for PlanetProfile: {time_end - time_start} seconds")
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
