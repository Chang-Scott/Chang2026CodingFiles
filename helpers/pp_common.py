import os
import shutil

from PlanetProfile.GetConfig import Params as globalParams, FigLbl, FigMisc, Style


    
def CopyCarefully(source, destination):
    try:
        if os.path.dirname(destination) != '':
            os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
    except OSError as err:
        raise OSError(
            f'Unable to copy from {source} to {destination}. '
            f'Check that you have write permission for {os.getcwd()}. '
            f'The error reported was:\n{err}'
        )
    else:
        print(f'{destination} was copied from default at {source}.')

    return


def loadUserSettings(configModule: str = ''):
    # Set uniform settings for all models
    globalParams.DO_PARALLEL = True
    globalParams.CALC_ASYM = False
    globalParams.CALC_OCEAN_PROPS = False
    globalParams.CustomSolution.REMOVE_SPECIES_NA_IN_FREZCHEM = True
    globalParams.minPres_MPa = 0.5
    globalParams.minTres_K = 0.5
    globalParams.CustomSolution.SUPCRT_DATABASE = 'supcrt16-organics'
    globalParams.CustomSolution.SOLID_PHASES_TO_SUPPRESS = [
        'Aegerine', 'Akermanite', 'Albite', 'Albite,high', 'Albite,low', 'Andradite',
        'Antigorite', 'Chloritoid', 'Clinozoisite', 'Coesite', 'Cristobalite', 'Cristobalite,alpha',
        'Cristobalite,beta', 'Diaspore', 'Dolomite', 'Dolomite,disordered', 'Dolomite,ordered', 'Edenite',
        'Glaucophane', 'Grossular', 'Grunerite', 'Kalsilite', 'Larnite', 'Laumontite', 'Lawsonite', 'Magnesite',
        'Margarite', 'Merwinite', 'Monticellite', 'Muscovite', 'Paragonite', 'Pargasite', 'Prehnite', 'Pyrophyllite',
        'Riebeckite', 'Tremolite', 'Wairakite', 'Wollastonite', 'Zoisite', 'Quartz', 'Anhydrite', 'Goethite'
    ]
    globalParams.REDUCED_LAYERS_SIZE = {'0': 5, 'Ih': 5, 'II': 5, 'III': 5, 'V': 5, 'VI': 5, 'Clath': 5, 'Sil': 5,
                                        'Fe': 5}

    # Only calculate orbital and synodic excitations for induction calculations
    globalParams.Induct.excSelectionCalc = {
        key: (key == 'orbital' or key == 'synodic')
        for key in globalParams.Induct.excSelectionCalc
    }
    if configModule == 'SpotModels':
        globalParams.DO_EXPLOREOGRAM = True
        globalParams.DO_INDUCTOGRAM = False
        globalParams.DO_MONTECARLO = False
        globalParams.NO_SAVEFILE = False
        globalParams.PLOT_INDIVIDUAL_PLANET_PLOTS = True
        globalParams.PLOT_GRAVITY = True
        globalParams.PLOT_HYDROSPHERE = True
        globalParams.PLOT_HYDROSPHERE_THERMODYNAMICS = True
        globalParams.PLOT_MELTING_CURVES = True
        globalParams.PLOT_SPECIES_HYDROSPHERE = True
        globalParams.PLOT_REF = False
        globalParams.PLOT_SIGS = True
        globalParams.PLOT_SOUNDS = True
        globalParams.PLOT_TRADEOFF = True
        globalParams.PLOT_POROSITY = True
        globalParams.PLOT_SEISMIC = True
        globalParams.PLOT_PRESSURE_DEPTH = False
        globalParams.PLOT_VISCOSITY = True
        globalParams.PLOT_WEDGE = True
        globalParams.Explore.xName = 'oceanComp'
        globalParams.Explore.yName = 'zb_approximate_km'
        globalParams.PLOT_PRESSURE_DEPTH = False
    elif configModule == 'LargeScaleExploration':
        globalParams.DO_EXPLOREOGRAM = True
        globalParams.NO_SAVEFILE = True
        globalParams.PLOT_INDIVIDUAL_PLANET_PLOTS = False
        globalParams.DO_INDUCTOGRAM = False
        globalParams.CALC_NEW_ASYM = False
        globalParams.PRELOAD_EOS = True
        globalParams.Explore.xName = 'oceanComp'
        globalParams.Explore.yName = 'zb_approximate_km'
        globalParams.zName = [
            'rhoOceanMean_kgm3', 'sigmaMean_Sm', 'D_km', 'Tb_K', 'kLoveAmp', 'kLovePhase', 'affinitySeafloor_kJ'
        ]
    elif configModule == 'Inversion':
        globalParams.DO_PARALLEL = False
        globalParams.ALLOW_BROKEN_MODELS = True
        globalParams.SKIP_PLOTS = True
        globalParams.NO_SAVEFILE = True
        globalParams.DO_PARALLEL = False
        globalParams.CustomSolution.EOS_deltaP = 10.0
        globalParams.CustomSolution.EOS_deltaT = 20.0
        globalParams.QUIET = True
        globalParams.SPEC_FILE = True
    elif configModule == 'AffinityCalculations':
        globalParams.DO_PARALLEL = False
        globalParams.CALC_OCEAN_PROPS = True
        globalParams.SKIP_GRAVITY = True
        globalParams.SKIP_INDUCTION = True
    # Set plotting settings
    FigLbl.axisLabelsExplore['oceanComp'] = r'Log $\text{H}_2$ Fugacity'
    FigLbl.oceanCompLabel = r'Log $\text{H}_2$ Fugacity'
    FigLbl.exploreDescrip['oceanComp'] = r'$\text{H}_2$ Fugacity'
    FigLbl.exploreDescrip['zb_approximate_km'] = FigLbl.exploreDescrip['zb_km']
    FigLbl.axisMultsExplore['Dconv_m'] = 1e-3
    FigMisc.figFormat = 'pdf'
    FigLbl.exploreDescrip['Dconv_m'] = r'Convecting layer thickness ($\si{km}$)'
    FigLbl.axisLabelsExplore['Dconv_m'] = r'Ice convective layer thickness $dz_\mathrm{conv}$ ($\si{km}$)'
    Style.TS_axis = 20
    Style.TS_super = 26
    Style.TS_ticks = 16
    FigLbl.TS_hydroLegendSize = 11

    return
