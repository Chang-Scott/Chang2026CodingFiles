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
# PlanetProfile imports
from PlanetProfile.GetConfig import Params as globalParams, FigMisc, Color, FigLbl, Style
from PlanetProfile.Main import run, ExploreOgram, ReloadExploreOgram
from PlanetProfile.Plotting.ExplorationPlots import GenerateExplorationPlots
from PlanetProfile.Inversion.Inversion import InvertBestPlanet
from Replicate_Zolotov_2008_Elemental import Replicate_Zolotov_H2, SetSettings
from helpers.pp_common import CopyCarefully, loadUserSettings

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
    LogH2Fugacities = np.linspace(-12, -3, 2)
    
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    globalParams.Explore.oceanCompRangeList = CustomSolutionComps
    
    globalParams.Explore.ny = 2
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


def run_methanogenesis_plot():
    Params.CALC_OCEAN_PROPS = True
    Params.SKIP_GRAVITY = True
    Params.SKIP_INDUCTION = True
    Params.CALC_NEW = True
    Params.DO_PARALLEL = True
    Params.DO_EXPLOREOGRAM = True
    Params.PLOT_Zb_D = False
    Params.PLOT_Zb_Y = False
    Params.CustomSolution.SOLID_PHASES = True
    Params.CustomSolution.SOLID_PHASES_TO_CONSIDER = 'All'
    Params.CustomSolution.EOS_deltaP = 2.0
    Params.CustomSolution.EOS_deltaT = 2.0
    Constants.SupcrtTmax_K = 320
    Constants.SupcrtTmin_K = 240
    Constants.SupcrtPmax_MPa = 250
    Constants.FrezchemPmax_MPa = 200
    Color.cmapName['default'] = 'bwr'
    LogH2Fugacities = np.linspace(-12, -3, 36)
    Params.Explore.contourName = 'pHSeafloor'
    _matFileName = 'yRangeData.mat'
    SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR, matFileName=_matFileName)
    CustomSolutionComps = Replicate_Zolotov_H2(LogH2Fugacities)
    FigLbl.axisCustomScalesExplore = {'oceanComp': LogH2Fugacities}
    setupPlotColorSettings(LogH2Fugacities)
    Params.Explore.xName = 'mixingRatioToH2O'
    Params.Explore.yName = 'oceanComp'
    Params.Explore.ny = len(LogH2Fugacities)
    Params.Explore.nx = 36
    Params.Explore.xRange = [-5, -1]
    Params.Explore.zName = ['affinitySeafloor_kJ', 'affinityTop_kJ']
    Params.Explore.yRange = ['CustomSolutionfH212', 'CustomSolutionfH23'] # This is simply for naming files, where xRange is used to generate file names
    run(bodyname = 'Europa', fNames = [fName])
    
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
    run_spot_models()
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
