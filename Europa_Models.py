"""
Europa Models Run
Script to run Explore-O-Grams for Europa with different ocean compositions.
Based on PPEuropaExplore.py configuration.

This script explores how different ocean compositions affect Europa's interior structure
by running PlanetProfile's ExploreOgram functionality with varying ocean compositions.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import importlib
import copy
from scipy.interpolate import make_interp_spline
from scipy.stats import norm, gaussian_kde
# PlanetProfile imports
from PlanetProfile.GetConfig import Params as globalParams, FigMisc, Color, FigLbl, Style
from PlanetProfile.Main import run, ExploreOgram, ReloadExploreOgram, PlanetProfile, LoadPPfiles
from PlanetProfile.Plotting.ExplorationPlots import GenerateExplorationPlots
from PlanetProfile.Inversion.Inversion import InvertBestPlanet
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2, SetSettings
from helpers.pp_common import CopyCarefully, loadUserSettings
from PlanetProfile.Thermodynamics.OceanProps import LiquidOceanPropsCalcs
from PlanetProfile.Thermodynamics.Reaktoro.reaktoroPropsHelperFunctions import SupcrtGenerator
from PlanetProfile.Utilities.defineStructs import Constants, EOSlist

# Local directory setup
this_dir = os.path.dirname(os.path.abspath(__file__))
Europa_dir = os.path.join(this_dir, 'Europa')

# Configure output directories and settings
_MATOUTPUTDIR = Europa_dir
_TXTOUTPUTDIR = Europa_dir
_FIGUREOUTPUTDIR = this_dir
saveToTxtFile = False
OUTPUTFIGURES = False
SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)

# Methanogenesis parameters
logfH2RedoxStateInterval = 0.5 # interval for logfH2RedoxState x axis steps for methanogenesis calculations

def run_spot_models():
    loadUserSettings('SpotModels')
    LogH2Fugacities = [-12, -9.5, 
                    -9, -6.01, -3
                    ]
    FigMisc.SHOW_GEOTHERM = False
    FigMisc.SCALE_HYDRO_LW = False

    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    globalParams.Explore.oceanCompRangeList = CustomSolutionComps
    setupPlotColorSettings(LogH2Fugacities, CustomSolutionComps, changeColorSpacing=True)
    globalParams.Explore.nx = len(LogH2Fugacities)
    globalParams.Explore.ny = 1
    globalParams.Explore.xRange = ['CustomSolutionfH212', 'CustomSolutionfH23'] # This is simply for naming files, where xRange is used to generate file names
    globalParams.Explore.yRange = [30, 30]

    run(bodyname = 'Europa', fNames = [spotModelFileName])
    
def run_interior_densities(doPlots = True):
    global globalParams
    loadUserSettings('LargeScaleExploration')
    if not doPlots:
        globalParams.SKIP_PLOTS = True
    
    # Define interior configurations to test
    coreDensities = np.linspace(5862.5, 8000, 10)
    silicateDensities = np.linspace(3750, 4000, 10)
    
    # Add "no core" case to the beginning of core densities list
    all_core_densities = [None] + list(coreDensities)  # None represents "no core"
    
    ExplorationGrid = np.empty((len(all_core_densities), len(silicateDensities)), dtype=object)
    currentModelFile = 'PPEuropa_ExploreBaseModel_wFeCore_Fixed.py'
    LogH2Fugacities = np.linspace(-12, -3, 30)
    
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    globalParams.Explore.oceanCompRangeList = CustomSolutionComps
    
    globalParams.Explore.ny = 36
    globalParams.Explore.yRange = [10, 90]
    globalParams.Explore.xRange = [LogH2Fugacities[0], LogH2Fugacities[-1]]

    noCoreExploration = None
    DONE_NOCORE = False
    
    # Loop through all core and silicate density combinations
    for i, coreDensity in enumerate(all_core_densities):
        for j, silicateDensity in enumerate(silicateDensities):
            
            # Create file name and figure base name
            if coreDensity is None and DONE_NOCORE:
                # We have already run the no core case, so skip it
                continue
            elif coreDensity is None:  # No core case has not been run yet
                fName = f'PPEuropa_ExploreBaseModel_noCore.py'
                globalParams.OverrideFigureBase = 'NoCoreExplore'
                FigLbl.titleAddendum = f"assuming no core"
            else:  # With core case
                fName = f'PPEuropa_ExploreBaseModel_wFeCore_Fixed_Fe{int(coreDensity)}kgm3_Sil{int(silicateDensity)}kgm3.py'
                globalParams.OverrideFigureBase = f'Fe{coreDensity}kgm3_Sil{silicateDensity}kgm3Explore'
                FigLbl.titleAddendum = ("assuming " + r"$\rho_{\mathrm{core}} = " + f"{coreDensity}" + r"\,\mathrm{{kg\,m^{-3}}}, \quad "
                                      r"\rho_{\mathrm{sil}} = " + f"{silicateDensity}" + r"\,\mathrm{{kg\,m^{-3}}}$")
            
            # Copy and modify the model file
            CopyCarefully(os.path.join('ModelFiles', currentModelFile), os.path.join('Europa', fName))
            
            with open(os.path.join('Europa', fName), 'r') as f:
                content = f.read()
            
            if coreDensity is None:  # No core case
                content = re.sub(r'Planet\.Do\.Fe_CORE\s*=\s*[^\n]*', f'Planet.Do.Fe_CORE = False', content)
            else:  # With core case
                content = re.sub(r'Planet\.Core\.rhoFe_kgm3\s*=\s*[^\n]*', f'Planet.Core.rhoFe_kgm3 = {coreDensity}', content)
                content = re.sub(r'Planet\.Sil\.rhoSilWithCore_kgm3\s*=\s*[^\n]*', f'Planet.Sil.rhoSilWithCore_kgm3 = {silicateDensity}', content)
            
            with open(os.path.join('Europa', fName), 'w') as f:
                f.write(content)
            
            setupPlotColorSettings(LogH2Fugacities, CustomSolutionComps)
            globalParams.Explore.nx = len(LogH2Fugacities)
            FigLbl.axisCustomScalesExplore = {'oceanComp': LogH2Fugacities}
        
            try:
                Exploration, globalParams = ReloadExploreOgram(bodyname = 'Europa', Params = globalParams, fNameOverride = fName)
            except FileNotFoundError:
                Exploration, globalParams = ExploreOgram(bodyname = 'Europa', Params = globalParams, fNameOverride = fName)
            except Exception as e:
                raise ValueError(f"Error running {fName}: {e}")
            
            ExplorationGrid[i, j] = Exploration
            
            # Generate all plot types for manuscript
            if doPlots:
                plot_types = [
                    ['rhoOceanMean_kgm3', 'D_km', 'Dconv_m', 'sigmaMean_Sm'],
                    ['kLoveAmp', 'hLoveAmp'],
                    ['kLovePhase', 'hLovePhase'],
                    ['InductionBi1Tot_nT']
                ]
            
                for plot_type in plot_types:
                    globalParams.Explore.zName = plot_type
                    _, globalParams = ReloadExploreOgram(bodyname = 'Europa', Params = globalParams, fNameOverride = fName)
                    GenerateExplorationPlots([Exploration], [globalParams.FigureFiles], globalParams)
            
            # Save no core case separately
            if coreDensity is None and not DONE_NOCORE:
                noCoreExploration = Exploration
                DONE_NOCORE = True
    ExplorationGrid = ExplorationGrid[1:, :] # Remove no core case from grid since we save separately as noCoreExploration
    return ExplorationGrid, noCoreExploration

                
  
def run_best_fit_model(ExplorationGrid, noCoreExploration):
    # Generate best fit planet models
    loadUserSettings('SpotModels')
    LogH2Fugacities = np.linspace(-10.2, -4.8, 2)
    globalParams.Explore.nx = len(LogH2Fugacities)
    globalParams.Explore.ny = 2
    globalParams.Explore.yRange = [20, 50]
    globalParams.Explore.xRange = ['CustomSolutionfH210BestFit', 'CustomSolutionfH25BestFit'] # This is simply for naming files, where xRange is used to generate file names
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    globalParams.Explore.oceanCompRangeList = CustomSolutionComps
    setupPlotColorSettings(LogH2Fugacities, CustomSolutionComps)
    globalParams.CALC_NEW = False
    PlanetGrid, _, _ = ExploreOgram(bodyname = 'Europa', Params = globalParams, fNameOverride = spotModelFileName, RETURN_GRID=True)
    BestPlanetList = PlanetGrid.flatten()
    
    
    loadUserSettings('Inversion')
    globalParams.Inversion.ExplorationGridResultionScale = 10
    
    for i, TruePlanet in enumerate(BestPlanetList):
        if i == 0:
            globalParams.Inversion.GRID_RESOLUTION_SCALE = 10
            interpolateGrid = True
            saveGrid = True
        else:
            globalParams.Inversion.GRID_RESOLUTION_SCALE = 1
            interpolateGrid = False
            saveGrid = False
        newH2Fugacities = np.linspace(-12, -3, 10 * globalParams.Inversion.GRID_RESOLUTION_SCALE)
        globalParams.Explore.xRange = [-12, -3]
        FigLbl.axisCustomScalesExplore = {'oceanComp': newH2Fugacities}
        ExplorationGrid = InvertBestPlanet(TruePlanet, globalParams, ExplorationGrid, otherExplorationList=[noCoreExploration], interpolateGrid=interpolateGrid, saveGrid=saveGrid)
        
def setupPlotColorSettings(fugacity_list, CustomSolutionList, changeColorSpacing=False, showColormap=True):
    """
    Setup color settings for the plots based on H2 fugacity, and optionally show a colormap figure.

    Parameters:
    - fugacity_list: List of fugacities corresponding to CustomSolutionList order.
    - changeColorSpacing: Improve color separation if fugacity values are clustered.
    - showColormap: If True, displays a color bar showing how color maps to fugacity.
    """

    fugacityMax = max(fugacity_list)
    fugacityMin = min(fugacity_list)
    fugacityRange = fugacityMax - fugacityMin
    cmap_name = 'coolwarm'
    epsilon = 0.01  # Small offset to avoid zero-width bounds

    cmap = get_cmap(cmap_name)

    for i, comp in enumerate(CustomSolutionList):
        fugacity = fugacity_list[i]

        # Optionally perturb spacing
        if changeColorSpacing:
            if i < len(fugacity_list) - 1 and (fugacity + 1) > fugacity_list[i + 1]:
                ColorIndex = (fugacity - 0.5 - fugacityMin) / fugacityRange
            elif i > 0 and (fugacity - 1) < fugacity_list[i - 1]:
                ColorIndex = (fugacity + 0.5 - fugacityMin) / fugacityRange
            else:
                ColorIndex = (fugacity - fugacityMin) / fugacityRange
        else:
            if fugacity - fugacityMin == 0:
                ColorIndex = 0.5
            else:
                ColorIndex = (fugacity - fugacityMin) / fugacityRange

        # Clamp color index
        ColorIndex = float(np.clip(ColorIndex, 0.0, 1.0))

        # Assign color info
        Color.cmapName[comp] = cmap_name
        Color.cmapBounds[comp] = [max(0.0, ColorIndex - epsilon), min(1.0, ColorIndex + epsilon)]
        Color.saturation[comp] = 1.0

    Color.SetCmaps()

    # Optional colormap figure
    if showColormap:
        _plot_fugacity_colormap(fugacity_list, cmap_name=cmap_name)

    return


def plot_affinity_density_distribution(logfH2RedoxStateRanges, affinities_seafloor_kJ, 
                                        CH4_CO2_mixing_ratios, hMixingDistance_km,
                                        modelType, output_dir,
                                        redox_state_mean=-7.5, redox_state_std=2.0):
    """
    Plot affinity as a density function, weighted by the probability distribution of Europa's redox state.
    
    Parameters:
    - logfH2RedoxStateRanges: Array of redox states sampled
    - affinities_seafloor_kJ: Array of affinities [redox_state, h_mixing, CH4_CO2_ratio]
    - CH4_CO2_mixing_ratios: List of CH4/CO2 mixing ratios
    - hMixingDistance_km: List of mixing distances
    - modelType: 'serpentization' or 'plume'
    - output_dir: Directory to save figure
    - redox_state_mean: Mean of the Gaussian distribution for inferred redox state
    - redox_state_std: Standard deviation of the Gaussian distribution for inferred redox state
    """
    
    # Calculate Gaussian probability weights for each redox state
    redox_probabilities = norm.pdf(logfH2RedoxStateRanges, loc=redox_state_mean, scale=redox_state_std)
    # Normalize to sum to 1
    redox_probabilities = redox_probabilities / np.sum(redox_probabilities)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for different CH4_CO2 mixing ratios
    colors = plt.cm.viridis(np.linspace(0, 1, len(CH4_CO2_mixing_ratios)))
    
    # For each CH4/CO2 ratio, create a weighted density distribution
    for j, CH4_CO2_ratio in enumerate(CH4_CO2_mixing_ratios):
        color = colors[j]
        
        # Create label
        if CH4_CO2_ratio == 0.4:
            label = f'CH$_4$/CO$_2$ = 0.4 (Enceladus)'
        else:
            exponent = int(np.log10(CH4_CO2_ratio))
            label = f'CH$_4$/CO$_2$ = $10^{{{exponent}}}$'
        
        # Collect all affinity values across all h_mixing distances, weighted by redox probability
        # We'll average over h_mixing distances for simplicity
        affinity_values = []
        weights = []
        
        for i, logfH2 in enumerate(logfH2RedoxStateRanges):
            # Average affinity across all h_mixing distances for this redox state and CH4/CO2 ratio
            avg_affinity = np.mean(affinities_seafloor_kJ[i, :, j])
            affinity_values.append(avg_affinity)
            weights.append(redox_probabilities[i])
        
        affinity_values = np.array(affinity_values)
        weights = np.array(weights)
        
        # Create a fine grid for the density plot
        affinity_range = np.linspace(affinity_values.min() - 10, affinity_values.max() + 10, 500)
        
        # Calculate weighted density using kernel density estimation
        # Repeat each affinity value proportional to its weight for KDE
        n_samples = 10000
        weighted_samples = np.random.choice(affinity_values, size=n_samples, p=weights)
        
        # Apply KDE
        kde = gaussian_kde(weighted_samples, bw_method='scott')
        density = kde(affinity_range)
        
        # Plot the density
        ax.plot(affinity_range, density, color=color, linewidth=2.5, label=label)
        ax.fill_between(affinity_range, 0, density, color=color, alpha=0.2)
    
    # Add vertical line at x=0 to indicate equilibrium
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Equilibrium', zorder=10)
    
    # Configure axes
    ax.set_xlabel(r'Methanogenesis Affinity (kJ mol$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'Probability Density', fontsize=12)
    ax.set_title(f'Probability Distribution of Methanogenesis Affinity at Seafloor ({modelType})\n' + 
                 f'Redox State: log fH$_2$ ~ N({redox_state_mean}, {redox_state_std}$^2$)', fontsize=14)
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f'methanogenesis_affinity_density_{modelType}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved affinity density plot to: {fig_path}")


def calculate_methanogenesis_affinities():
    global globalParams
    loadUserSettings('AffinityCalculations')

    # Range of excess H2 concentrations to explore (molal), from low-serpentization to high-plume end
    deltaT_ranges_K = np.logspace(0, 3, 4)
    H2Flux_mols_yr = 10**8  # Vance et. al, 2016
    serpentizationHeat_J_yr = 0.01 * 4 * np.pi * (1451 * 1000)**2 * 60 * 60 * 24 * 365
    H2_concentrations_range_molal = H2Flux_mols_yr * 4184 * deltaT_ranges_K / serpentizationHeat_J_yr

    logfH2RedoxStateRanges = np.linspace(-12, -3, 10)
    globalParams, loadNames = LoadPPfiles(globalParams, fNames=[spotModelFileName], bodyname='Europa')
    Planet = importlib.import_module(loadNames[0]).Planet
    Planet.Do.ICEIh_THICKNESS = True
    Planet.Bulk.zb_approximate_km = 30

    # Arrays indexed [redox_state, H2_concentration]
    equilibrium_constants_seafloor_array = np.full(len(logfH2RedoxStateRanges), np.nan)
    affinities_seafloor_kJ = np.full((len(logfH2RedoxStateRanges), len(deltaT_ranges_K)), np.nan)

    for i, logfH2RedoxState in enumerate(logfH2RedoxStateRanges):
        oceanComp = Replicate_Zolotov_H2([logfH2RedoxState])[0]
        planetRun = copy.deepcopy(Planet)
        planetRun.Ocean.comp = oceanComp
        planetRun.Ocean.reactionEquation = "CO2(aq) + 4 H2(aq) = Methane(aq) + 2 H2O(aq)"
        planetRun, _ = PlanetProfile(planetRun, globalParams)

        equilibrium_constants_seafloor = planetRun.Ocean.equilibriumReactionConstant[-1]
        equilibrium_constants_seafloor_array[i] = equilibrium_constants_seafloor

        Pseafloor_MPa = planetRun.P_MPa[planetRun.Steps.nHydro - 1]
        Tseafloor_K = planetRun.T_K[planetRun.Steps.nHydro - 1]

        H2index = np.where(np.array(planetRun.Ocean.aqueousSpecies) == 'H2(aq)')[0][0]
        H2_equilibrium_molal_seafloor = planetRun.Ocean.aqueousSpeciesAmount_mol[H2index, -1]
        CO2index = np.where(np.array(planetRun.Ocean.aqueousSpecies) == 'CO2(aq)')[0][0]
        CO2_equilibrium_molal_seafloor = planetRun.Ocean.aqueousSpeciesAmount_mol[CO2index, -1]
        CH4index = np.where(np.array(planetRun.Ocean.aqueousSpecies) == 'Methane(aq)')[0][0]
        CH4_equilibrium_molal_seafloor = planetRun.Ocean.aqueousSpeciesAmount_mol[CH4index, -1]

        # Equilibrium CH4/CO2 ratio from PlanetProfile
        CH4_CO2_eq_ratio = CH4_equilibrium_molal_seafloor / CO2_equilibrium_molal_seafloor
        print(f'CH4/CO2 (eq) = {CH4_CO2_eq_ratio:.4e} for redox state {logfH2RedoxState}')

        disequilibriumConcentrations_molal = {'H2O(aq)': 55.51, 'H2(aq)': 0, 'H+': 0, 'OH-': 0, 'CO2(aq)': 0, 'Methane(aq)': 0}
        db, system, state, conditions, solver, props = SupcrtGenerator(
            'H2O(aq) H2(aq) H+ OH-', disequilibriumConcentrations_molal, "mol",
            "supcrt16-organics", None, Constants.PhreeqcToSupcrtNames, 200, rktDatabase=None)
        state.pressure(Pseafloor_MPa, "MPa")
        state.temperature(Tseafloor_K, "K")

        for k, H2_conc in enumerate(H2_concentrations_range_molal):
            H2_disequilibrium_molal = H2_conc + H2_equilibrium_molal_seafloor
            state.setSpeciesAmount('H2(aq)', H2_disequilibrium_molal, "mol")
            props.update(state)
            Q = (CH4_CO2_eq_ratio
                 * float(props.speciesActivity('H2O(aq)'))**2
                 / float(props.speciesActivity('H2(aq)'))**4)
            affinities_seafloor_kJ[i, k] = (
                2.3026 * 8.314 * Tseafloor_K
                * (np.log10(equilibrium_constants_seafloor) - np.log10(Q))
                / 1000)

    # --- Plot: one line per H2 concentration vs. redox state ---
    from matplotlib.ticker import FuncFormatter
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    deltaT_min = deltaT_ranges_K[0]
    deltaT_max = deltaT_ranges_K[-1]
    cmap = cm.get_cmap('viridis')
    norm_deltaT = mcolors.Normalize(vmin=deltaT_min, vmax=deltaT_max)

    # Wider figure; leave explicit room on the right for the twin axis + colorbar
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=0.08, right=0.68, top=0.90, bottom=0.10)

    logfH2_smooth = np.linspace(logfH2RedoxStateRanges[0], logfH2RedoxStateRanges[-1], 300)
    for k, deltaT_K in enumerate(deltaT_ranges_K):
        color = cmap(norm_deltaT(deltaT_K))
        spl = make_interp_spline(logfH2RedoxStateRanges, affinities_seafloor_kJ[:, k], k=3)
        affinity_smooth = spl(logfH2_smooth)
        ax.plot(logfH2_smooth, affinity_smooth, linewidth=2, color=color)

    # Equilibrium reference line
    ax.axhline(y=0, color='black', linestyle='solid', linewidth=2, label='Equilibrium')

    ax.set_xlim([-12, -3])
    ax.set_ylim([0, 250])
    ax.set_xticks(np.arange(-12, -3 + 1, 1))
    ax.set_yticks(np.arange(0, 251, 25))
    ax.set_xlabel(FigLbl.axisLabelsExplore['oceanComp'], fontsize=12)
    ax.set_ylabel(r'Methanogenesis Affinity (kJ (mol of reaction)$^{-1}$)', fontsize=12)
    ax.set_title('Methanogenesis Affinity at the Seafloor')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Second y-axis: biomass supported (twin of ax, shares the same plot area)
    ax2_biomass = ax.twinx()
    biomass_conversion = (H2Flux_mols_yr / 4) * (1 / 4.184) * 0.1 * (1 / 10) * (1 / 0.02)  # Jaksoyev and Shock, 2010
    y1_min, y1_max = ax.get_ylim()
    ax2_biomass.set_ylim([y1_min * biomass_conversion, y1_max * biomass_conversion])
    ax2_biomass.set_ylabel(r'Biomass Supported (g of cells yr$^{-1}$)', fontsize=12)

    def abs_sci_formatter(x, pos):
        if not np.isfinite(x) or x == 0:
            return "0"
        x = abs(x)
        exponent = int(np.floor(np.log10(x)))
        mantissa = x / 10**exponent
        return rf"${mantissa:.1f}\times10^{{{exponent}}}$"

    ax2_biomass.yaxis.set_major_formatter(FuncFormatter(abs_sci_formatter))

    # Colorbar in its own explicitly-positioned axes, to the right of both ax and ax2_biomass
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_deltaT)
    sm.set_array([])
    cax = fig.add_axes([0.87, 0.10, 0.025, 0.80])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$\Delta T$ (K)', fontsize=12)

    output_dir = globalParams.FigureFiles.figPath if hasattr(globalParams.FigureFiles, 'figPath') else '.'
    fig_path = os.path.join(output_dir, 'methanogenesis_affinity.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved affinity plot to: {fig_path}")


# ---------------------------
# Helper: Draw color gradient
# ---------------------------
def _plot_fugacity_colormap(fugacity_list, cmap_name='coolwarm', figsize=(2, 6)):
    fugacity_list = np.array(sorted(fugacity_list))
    fugacityMin = fugacity_list.min()
    fugacityMax = fugacity_list.max()

    norm = plt.Normalize(vmin=fugacityMin, vmax=fugacityMax)
    cmap = get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    # Transpose to make vertical
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)

    # Set extent so that the fugacity scale maps vertically
    ax.imshow(gradient, aspect='auto', cmap=cmap,
                extent=[0, 1, fugacityMin, fugacityMax])

    ax.set_xticks([])
    ax.set_ylabel(r'$f_{\mathrm{H}_2}$')

    plt.tight_layout()
    plt.savefig('FugacityColorMap.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    spotModelFileName = 'PPEuropa_SpotModel_wFeCore_3500rhokgm3_5150fekgm3.py'
    CopyCarefully(os.path.join('ModelFiles', spotModelFileName), os.path.join('Europa', spotModelFileName))
    #run_spot_models()
    calculate_methanogenesis_affinities()
    #ExplorationGrid, noCoreExploration = run_interior_densities(doPlots=True)
    #run_best_fit_model(ExplorationGrid, noCoreExploration)



    """  
def ocean_solid_species():
    Params = deepcopy(Params)
    Params.PLOT_INDIVIDUAL_PLANET_PLOTS = True
    Params.CALC_NEW = True
    Params.CustomSolution.SOLID_PHASES = True
    Params.CustomSolution.SOLID_PHASES_TO_CONSIDER = 'All'
    Params.CALC_OCEAN_PROPS = True
    LogH2Fugacities = [-12, -9.5, -9, -6.01, -3]
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    setupPlotColorSettings(LogH2Fugacities)


    Params.Explore.nx = len(LogH2Fugacities)
    Params.Explore.ny = 1
    Params.Explore.xRange = ['CustomSolutionfH212SolidSpecies', 'CustomSolutionfH23SolidSpecies'] # This is simply for naming files, where xRange is used to generate file names
    Params.Explore.yRange = [30, 30]
    
    Params.CustomSolution.EOS_deltaP = 5.0
    Params.CustomSolution.EOS_deltaT = 5.0
    FigMisc.figFormat = 'png'
    FigMisc.xtn = '.png'
    run(bodyname = 'Europa', fNames = [currentModelFile])
    Params.PLOT_INDIVIDUAL_PLANET_PLOTS = False
    Params.CustomSolution.SOLID_PHASES = False
    
    
def run_tidalLove_MiniExploration():
    Params = deepcopy(Params)
    Params.CALC_NEW = True
    Params.DO_PARALLEL = True
    Params.NO_SAVEFILE = True # Do not want to save all the file runs - will blow up disk
    Params.PLOT_INDIVIDUAL_PLANET_PLOTS = False # Do not want to plot individual planet plots - will blow up disk
    Params.LEGEND = True # Do not want to plot legend in such a large exploreogram
    Params.COMPARE = False
    Params.SKIP_PLOTS = False
    Params.CALC_NEW_INDUCT = True
    Params.CALC_OCEAN_PROPS = False
    LogH2Fugacities = [-12, -9.5,
                    -9, -6.01, -3
                    ]
    FigMisc.HIGHLIGHT_ICE_THICKNESSES = False
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    setupPlotColorSettings(LogH2Fugacities, changeColorSpacing=True)

    Params.Explore.nx = len(LogH2Fugacities)
    Params.Explore.ny = 9
    Params.Explore.yRange = [10, 90]
    Params.Explore.xRange = ['CustomSolutionfH212', 'CustomSolutionfH23'] # This is simply for naming files, where xRange is used to generate file names
    Params.Explore.zName = 'D_km'
    run(bodyname = 'Europa', fNames = [currentModelFile])
def run_tidalLoveInductionPlots():
    Params = deepcopy(Params)
    Params.CALC_NEW = False
    Params.DO_PARALLEL = True
    Params.NO_SAVEFILE = True # Do not want to save all the file runs - will blow up disk
    Params.PLOT_INDIVIDUAL_PLANET_PLOTS = False # Do not want to plot individual planet plots - will blow up disk
    Params.LEGEND = True # Do not want to plot legend in such a large exploreogram
    Params.COMPARE = False
    Params.SKIP_PLOTS = False
    Params.CALC_NEW_INDUCT = True
    FigMisc.EDGE_COLOR_K_IN_COMPLEX_PLOTS = True
    FigMisc.HIGHLIGHT_ICE_THICKNESSES = False
    Params.Induct.excSelectionPlot = {  # Which magnetic excitations to include in plotting
        'orbital': True,  # Key excitation
        'synodic': True,  # Key excitation
    }
    
    LogH2Fugacities = [-12, -9.5,
                    -9, -6.01, -3
            ]
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    setupPlotColorSettings(LogH2Fugacities, changeColorSpacing=True)
    Params.Explore.nx = len(LogH2Fugacities)
    Params.Explore.ny = 2
    Params.Explore.yRange = [30, 90]
    run(bodyname = 'Europa', fNames = [currentModelFile])
    
def run_full_responses(do_ice_convection = True):
    Params.CALC_NEW = True
    Params.LEGEND = True
    Params.NO_SAVEFILE = True # Do not want to save all the file runs - will blow up disk
    Params.PLOT_INDIVIDUAL_PLANET_PLOTS = False # Do not want to plot individual planet plots - will blow up disk
    Params.LEGEND = False # Do not want to plot legend in such a large exploreogram
    Params.CALC_OCEAN_PROPS = False
    Params.SKIP_GRAVITY = False
    Params.SKIP_INDUCTION = False
    Params.NO_SAVEFILE = True
    Params.DO_PARALLEL = True
    if do_ice_convection:
        Params.Explore.xRange = ['fH212', 'fH23'] # This is simply for naming files, where xRange is used to generate file names
        Params.Explore.xRange = ['CustomSolutionfH212', 'CustomSolutionfH23']
    else:
        from PlanetProfile.Thermodynamics.Reaktoro.WIP.Europa_Exploreogram.Europa.PPEuropa_ExploreBaseModel_wFeCore import Planet as PPEuropa_ExploreBaseModel
        PPEuropa_ExploreBaseModel.Do.NO_ICE_CONVECTION = True
        Params.Explore.xRange = ['NoConvectionfH212', 'NoConvectionfH23'] # This is simply for naming files, where xRange is used to generate file names
    LogH2Fugacities = np.linspace(-12, -3, 36)
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    setupPlotColorSettings(LogH2Fugacities)
    Params.Explore.nx = len(LogH2Fugacities)
    FigLbl.axisCustomScalesExplore = {'oceanComp': LogH2Fugacities}
    # We just run one of these models as we already have all .mat files to plot together in the comparison
    Params.Explore.ny = 80
    Params.Explore.yRange = [10, 90]
    #Params.Explore.zName = ['kLoveAmp', 'kLovePhase', 'hLoveAmp', 'hLovePhase']
    Params.Explore.zName = ['rhoOceanMean_kgm3', 'D_km', 'Dconv_m','sigmaMean_Sm']
    run(bodyname = 'Europa', fNames = [currentModelFile])
    Params.CALC_NEW = False
    Params.Explore.zName = ['kLoveAmp', 'hLoveAmp']
    run(bodyname = 'Europa', fNames = [currentModelFile])
    Params.Explore.zName = ['kLovePhase', 'hLovePhase']
    run(bodyname = 'Europa', fNames = [currentModelFile])
    Params.Explore.zName = ['InductionBi1x_nT']
    run(bodyname = 'Europa', fNames = [currentModelFile])
    


def run_induction():
    Params = deepcopy(Params)
    Params.DO_INDUCTOGRAM = True
    Params.DO_EXPLOREOGRAM = False
    Params.CALC_NEW = False
    Params.CALC_NEW_INDUCT = False
    LogH2Fugacities = np.linspace(-12, -3, 25)
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    setupPlotColorSettings(LogH2Fugacities)

    Params.Explore.nx = len(LogH2Fugacities)
    
    Params.COMPARE = True
    Params.Induct.zbMin['Europa'] = 10
    Params.Induct.zbMax['Europa'] = 20
    Params.Induct.nZbPts = 2
    run(bodyname = 'Europa', fNames = [currentModelFile])
    Params.Induct.zbMin['Europa'] = 30
    Params.Induct.zbMax['Europa'] = 40
    Params.Induct.nZbPts = 2
    Params.COMPARE = True
    run(bodyname = 'Europa', fNames = [currentModelFile])
"""
