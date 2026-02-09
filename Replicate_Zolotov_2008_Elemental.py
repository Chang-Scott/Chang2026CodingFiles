from reaktoro import *
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterExponent
from scipy.io import savemat
from adjustText import adjust_text
import logging
# Assign logger
log = logging.getLogger('Replicate_Zolotov_2008_Elemental')

""" Define ROOT (for loading files) and OUTPUT_DIR (for saving files)"""
_ROOT = os.path.dirname(os.path.abspath(__file__))
_MATOUTPUTDIR = _ROOT # Same as _ROOT for now but can be changed to a different directory
_TXTOUTPUTDIR = _ROOT # Same as _ROOT for now but can be changed to a different directory
_FIGUREOUTPUTDIR = _ROOT # Same as _ROOT for now but can be changed to a different directory
saveToTxtFile = True # Whether to save the results to a txt file
SHOWPLOT = True # Whether to show the plots while running the script


""" Plotting labeling settings"""
# Enable LaTeX rendering and add mhchem to the preamble for plotting
stix = 'stix'
mhchem = 'mhchem'
siunitx = 'siunitx'
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': f'\\usepackage{{{stix}}}\\usepackage{{{mhchem}}}\\usepackage{{{siunitx}}}',
    'font.family': 'sans-serif',  # Changed from 'serif' to 'sans-serif'
    'font.sans-serif': 'Times New Roman',    # Set to Times New Roman
    'axes.labelsize': 16,      # Axis label font size
    'xtick.labelsize': 14,     # X tick label font size
    'ytick.labelsize': 14,     # Y tick label font size
    'axes.titlesize': 18,
        'font.weight': 'bold',           # Makes all text bold
    'axes.labelweight': 'bold',      # Makes axis labels bold
    'axes.titleweight': 'bold', 
})
species_font_size = 24
element_font_size = 24
mineral_font_size = 18
alpha_box = 0.7 # Alpha value for white box around text labels
OUTPUTFIGURES = True
_matFileName = 'xRangeData.mat'


""" Modeling setup"""
# Load the Core11 database into Reaktoro
path_file = os.path.join(_ROOT, 'Zolotov_2008_Recreation', 'core11_Diab_Original.dat')
db = PhreeqcDatabase.fromFile(path_file)
P_MPa = [137.5]  # Pressure in MPa
T_K = [273.2]  # Temperature in K
mass_g = 1000 # Mass of CI Chondrite to add in grams
# W/R ratio to use
water_rock_ratio = 1
# Elements to have in chemical system
elements = "O H C S Cl Na K Mg Fe Ca Si Al"
# CI Chondrite Elemental Composition
ci_chondrite_elemental = [
        ("O", 17.8409), ("H", 2.0863),
        ("K", 0.0184), ("Mg", 5.6838),
        ("Ca", 0.35), ("Al", 0.4813),
        ("C", 0.3648), ("Si", 5.4510),
        ("S", 1.4908 ), ("Na", 0.3540),
        ("Fe", 4.8704)
    ]
# Ionic species to consider
ionic_species_to_consider = [
    "H+", "Na+", "K+",
    "Mg+2", "Ca+2", 
    "Cl-", "SO4-2", "HCO3-", "CO3-2",  "OH-",
    "Al+3", "Fe+2", "Fe+3"
]
# Species to exclude that are not in supcrt16 (which we use to generate properties in PlanetProfile)
aqueous_species_to_exclude = ['FeCO3', 'NaCO3-', 'NaHCO3', "Al+3", "O2", "CH3COOH", "C2H4", "CO2", "C2H6", "C3H6O2", "C3H8",
    "C6H8O7", "CH2O", "CH3CH2OH", "CH3COCOOH", "CH3OH", "CH4", "CO",
    "Ca(CH3COO)2", "CaCl2", "CaSO4", "Fe+2", "Fe(CH3COO)2", "Fe+3",
    "FeCl2", "H2S", "HAlO2", "HCOOH", "HCl", "HSO4-", "KCH3COO",
    "KCl", "KHSO4", "KSO4-", "Mg(CH3COO)2", "MgSO4", "NaCH3COO", "NaCl",
    "NaOH", "SO2", "MgCO3"]




# Reaktoro specific setup options
maxIterations = 2000 # Maximum number of iterations for the equilibrium solver before determining non-convergence
Warnings.disable(548) # Disable convergence warnings since we handle those in the loop by restarting the calculation
Warnings.disable(906)

def generate_initial_speciation(db, elemental_concentrations, elements, P, T, do_H2, do_CO2, initial_fugacities):
    """
    Performs a preliminary equilibrium calculation to determine the initial aqueous speciation.
    This provides a good starting point for the main simulation.
    """
    # Generate a chemical system without minerals for initial speciation
    system, initial_state, conditions, solver, restrictions = \
        generate_chemical_system(db, elemental_concentrations, Do_minerals=False, DO_speciate=True,
                                 elements=elements, use_activity_model=False, DO_PH=False, DO_CO2=do_CO2,
                                 DO_H2=do_H2, add_elements=True)

    # Clone the initial state for the calculation
    aqueous_state = initial_state.clone()

    # Set solver options and equilibrium conditions for the preliminary run
    options = EquilibriumOptions()
    options.optima.maxiters = maxIterations
    solver.setOptions(options)
    for gas, fugacity in initial_fugacities.items():
        conditions.fugacity(gas, fugacity, "bar")
    conditions.pressure(P, "MPa")
    conditions.temperature(T, "K")

    # Solve for the initial equilibrium state
    result = solver.solve(aqueous_state, conditions)

    if result.succeeded():
        log.info("Initial speciation calculation succeeded.")
    else:
        log.warning("Warning: Initial speciation calculation failed.")

    return aqueous_state, system


def query_equilibrium_across_fugacity(system, initial_state, conditions, solver, restrictions, sweep_gas,
                        log_sweep_gas_fugacity, P_MPa, T_K, fixed_gases=None):
    """
    Runs the main simulation loop over a range of gas fugacities, calculating the equilibrium state at each step. Specify the gas to sweep over, the range of fugacities to sweep over, and the pressure and temperature ranges to sweep over.

    Returns:
        dict: A dictionary containing the simulation results.
    """
    # Prepare data structures to store results
    element_names = np.array([element.symbol() for element in system.elements()])
    aqueous_species_names = np.array(
        [s.name() for s in system.species() if s.aggregateState() == AggregateState.Aqueous])
    solid_species_names = np.array(
        [s.name() for s in system.species() if s.aggregateState() == AggregateState.Solid])
    solid_phases_names = np.array([p.name() for p in system.phases() if p.aggregateState() == AggregateState.Solid])

    # Initialize lists to store simulation output
    results = {
        'pH': [], 'elements': [[] for _ in element_names], 'aqueous_species': [[] for _ in aqueous_species_names],
        'solid_species_mol': [[] for _ in solid_species_names],
        'solid_phases_volume': [[] for _ in solid_phases_names]
    }

    state = initial_state.clone()
    props = ChemicalProps(state)

    # Loop over pressure, temperature, and H2 fugacity ranges
    for P in P_MPa:
        conditions.pressure(P, "MPa")
        for T in T_K:
            conditions.temperature(T, "K")

            # Set any fixed gas fugacities
            if fixed_gases:
                for gas, log_fugacity in fixed_gases.items():
                    conditions.fugacity(gas, 10 ** log_fugacity, "bar")

            for i, log_fugacity in enumerate(log_sweep_gas_fugacity):
                log.info(f'Performing calculation {i + 1} of {len(log_sweep_gas_fugacity)}')
                state.setPressure(P, "MPa")
                state.setTemperature(T, "K")
                conditions.fugacity(sweep_gas, 10 ** log_fugacity, "bar")

                # Here we increase the number of iterations Reaktoro will do while looking for thermodynamic equilibrium
                options = EquilibriumOptions()
                options.optima.maxiters = maxIterations
                solver.setOptions(options)
                result = solver.solve(state, conditions, restrictions)
                notSuccess = not result.succeeded()
                # If we did not succeed, then let's try again from the initial state
                if notSuccess:
                    log.info("RETRYING CALCULATIONS")
                    state = initial_state.clone()
                    conditions.fugacity(sweep_gas, 10 ** np.round(log_fugacity, 1), "bar") # Round the fugacity to 1 decimal place since this can help numerical stability
                    result = solver.solve(state, conditions, restrictions)
                    notSuccess = not result.succeeded()
                    while notSuccess:
                        state = initial_state.clone()
                        # try bumping log fugacity by 0.1
                        log_fugacity += 0.1
                        conditions.fugacity(sweep_gas, 10 ** np.round(log_fugacity, 1), "bar") # Round the fugacity to 1 decimal place since this can help numerical stability
                        result = solver.solve(state, conditions, restrictions)
                        notSuccess = not result.succeeded()
                # If we did succeed, then save all the equilibrium data to the previously created lists
                if result.succeeded():
                    # Update the properties
                    props.update(state)
                    aprops = AqueousProps(props)
                    results['pH'].append(float(aprops.pH()))
                    for k, molality in enumerate(aprops.elementMolalities()):
                        results['elements'][k].append(float(f'{molality:.2e}'))
                    for k, molality in enumerate(aprops.speciesMolalities()):
                        results['aqueous_species'][k].append(float(f'{molality:.2e}'))
                    for k, s_name in enumerate(solid_species_names):
                        results['solid_species_mol'][k].append(float(props.speciesAmount(s_name)))
                    for k, p_name in enumerate(solid_phases_names):
                        results['solid_phases_volume'][k].append(float(props.phaseProps(p_name).volume()) * 100 ** 3)
                else:
                    # Append NaN for failed calculations
                    log.error(f"ERROR at P = {P} MPa, T = {T} K, {sweep_gas} = {log_fugacity}")
                    results['pH'].append(np.nan)
                    for k in range(len(results['elements'])): results['elements'][k].append(np.nan)
                    for k in range(len(results['aqueous_species'])): results['aqueous_species'][k].append(np.nan)
                    for k in range(len(results['solid_species_mol'])): results['solid_species_mol'][k].append(np.nan)
                    for k in range(len(results['solid_phases_volume'])): results['solid_phases_volume'][k].append(
                        np.nan)

    # Convert lists to numpy arrays for easier manipulation
    return {
        'pH_array': np.array(results['pH']),
        'aqueous_species_array_molal': np.array(results['aqueous_species']),
        'element_species_array_molal': np.array(results['elements']),
        'solid_species_array_mol': np.array(results['solid_species_mol']),
        'solid_phases_volume_array': np.array(results['solid_phases_volume']),
        'element_names': element_names,
        'aqueous_species_names': aqueous_species_names,
        'solid_phases_names': solid_phases_names
    }

def Replicate_Zolotov_H2(log_H2_fugacity_range = None):
    """
    Replicates the Zolotov (2008) H2 fugacity model using the Core11 database.
    This function sets up the chemical system, runs a simulation across a range of
    H2 fugacities, and plots the resulting equilibrium compositions.
    """
    # --------------------------------------------------------------------------
    # 1. Define Simulation Parameters
    # --------------------------------------------------------------------------
    if log_H2_fugacity_range is None:
        log_H2_fugacity = np.linspace(-12, -3, 100)
    else:
        log_H2_fugacity = log_H2_fugacity_range

    # --------------------------------------------------------------------------
    # 2. Initial Speciation Calculation
    # --------------------------------------------------------------------------
    # Perform a preliminary equilibrium calculation to get a good initial guess
    # for the aqueous speciation before introducing minerals.
    speciated_state, speciated_system = generate_initial_speciation(
        db, ci_chondrite_elemental, elements, P_MPa[0], T_K[0],
        do_H2=True, do_CO2=False, initial_fugacities={'H2(g)': 10 ** -12}
    )

    # --------------------------------------------------------------------------
    # 3. Setup Main Chemical System with Minerals
    # --------------------------------------------------------------------------
    # Generate the full chemical system including mineral phases.
    system, initial_state, conditions, solver, restrictions = \
        generate_chemical_system(db, ci_chondrite_elemental, Do_minerals=True,
                                 DO_speciate=True, elements=elements, DO_PH=False,
                                 DO_CO2=False, DO_H2=True, add_elements=False,
                                 use_activity_model=True)

    # Populate the initial state with species amounts from the preliminary calculation.
    species_amounts = speciated_state.speciesAmounts()
    species_names = speciated_system.species()
    for i, amount in enumerate(species_amounts):
        if amount < 1e-16:
            amount = 1e-16
        initial_state.add(species_names[i].name(), amount, "mol")

    # --------------------------------------------------------------------------
    # 4. Run Fugacity Sweep Simulation
    # --------------------------------------------------------------------------
    # Run the main simulation loop over the specified H2 fugacity range.
    results = query_equilibrium_across_fugacity(system, initial_state, conditions, solver,
                                  restrictions, 'H2(g)', log_H2_fugacity, P_MPa, T_K)

    # --------------------------------------------------------------------------
    # 5. Process and Visualize Results
    # --------------------------------------------------------------------------
    # Define axis limits for the plots.
    species_lim = (1e-12, 1e1)
    pH_lim = (6, 14)

    # Save aqueous species data for external use (e.g., PlanetProfile).
    wppt_list, m_strings = save_aqueous_species_mat(results['aqueous_species_names'],
                                         results['aqueous_species_array_molal'],
                                         log_H2_fugacity, 'H2', results['pH_array'])

    # Generate and display plots based on the simulation results.
    figname = os.path.join(_FIGUREOUTPUTDIR, 'Zolotov_H2_Replication')
    if OUTPUTFIGURES:
        generate_Zolotov_plots(
        data=(results['element_names'], results['aqueous_species_names'],
              results['solid_phases_names'], results['pH_array'],
              results['aqueous_species_array_molal'], results['element_species_array_molal'],
              results['solid_species_array_mol'], results['solid_phases_volume_array'],
              wppt_list),
        x=log_H2_fugacity,
        species_lim=species_lim,
        pH_lim=pH_lim,
        figname=figname,
        xlabel=r'Log $\text{H}_2$ Fugacity'
    )

    return m_strings

def Replicate_Zolotov_CO2(log_CO2_fugacity_range = None):
    """
    Function to run to replicate the Zolotov CO2 fugacity model using the Core11 database and plot the results.
    """
    # --------------------------------------------------------------------------
    # 1. Define Simulation Parameters
    # --------------------------------------------------------------------------
    if log_CO2_fugacity_range is None:
        log_CO2_fugacity = np.linspace(-6, 2, 100)
    else:
        log_CO2_fugacity = log_CO2_fugacity_range
    log_H2_fugacity_fixed = -10

    # --------------------------------------------------------------------------
    # 2. Initial Speciation Calculation
    # --------------------------------------------------------------------------
    # Perform a preliminary equilibrium calculation to get a good initial guess.
    initial_fugacities = {
        'H2(g)': 10 ** log_H2_fugacity_fixed,
        'CO2(g)': 10 ** log_CO2_fugacity[0]
    }
    speciated_state, speciated_system = generate_initial_speciation(
        db, ci_chondrite_elemental, elements, P_MPa[0], T_K[0],
        do_H2=True, do_CO2=True, initial_fugacities=initial_fugacities
    )

    # --------------------------------------------------------------------------
    # 3. Setup Main Chemical System with Minerals
    # --------------------------------------------------------------------------
    # Generate the full chemical system including mineral phases.
    system, initial_state, conditions, solver, restrictions = \
        generate_chemical_system(db, ci_chondrite_elemental, Do_minerals=True,
                                 DO_speciate=True, elements=elements, DO_PH=False,
                                 DO_CO2=True, DO_H2=True, add_elements=False,
                                 use_activity_model=True)

    # Populate the initial state with species amounts from the preliminary calculation.
    species_amounts = speciated_state.speciesAmounts()
    species_names = speciated_system.species()
    for i, amount in enumerate(species_amounts):
        if amount < 1e-16: amount = 1e-16
        initial_state.add(species_names[i].name(), amount, "mol")

    # --------------------------------------------------------------------------
    # 4. Run Fugacity Sweep Simulation
    # --------------------------------------------------------------------------
    # Run the main simulation loop over the specified CO2 fugacity range.
    results = query_equilibrium_across_fugacity(system, initial_state, conditions, solver,
                                  restrictions, 'CO2(g)', log_CO2_fugacity, P_MPa, T_K,
                                  fixed_gases={'H2(g)': log_H2_fugacity_fixed})

    # --------------------------------------------------------------------------
    # 5. Process and Visualize Results
    # --------------------------------------------------------------------------
    # Define axis limits for the plots.
    species_lim = (1e-6, 1e1)
    pH_lim = (4, 8)

    # Save aqueous species data for external use (e.g., PlanetProfile).
    wppt_list, m_strings = save_aqueous_species_mat(results['aqueous_species_names'],
                                         results['aqueous_species_array_molal'],
                                         log_CO2_fugacity, 'CO2', results['pH_array'])

    # Generate and display plots based on the simulation results.
    figname = os.path.join(_FIGUREOUTPUTDIR, 'Zolotov_CO2_Replication')
    if OUTPUTFIGURES:
        generate_Zolotov_plots(
        data=(results['element_names'], results['aqueous_species_names'],
              results['solid_phases_names'], results['pH_array'],
              results['aqueous_species_array_molal'], results['element_species_array_molal'],
              results['solid_species_array_mol'], results['solid_phases_volume_array'],
              wppt_list),
        x=log_CO2_fugacity,
        species_lim=species_lim,
        pH_lim=pH_lim,
        figname=figname,
        xlabel=r'Log $\text{CO}_2$ Fugacity',
        exclude_species={'HS-+H2S'}
        )

    return m_strings


def generate_Zolotov_plots(data, x, species_lim, pH_lim, figname, xlabel, exclude_species=None):
    """
    Function to generate plots, modeled after Zolotov's
    :param data: data to plot
    :param x: x range to plot
    :param species_lim: species axis limits
    :param pH_lim: pH axis limits
    :param figname: output filename
    :param exclude_species: set of species names to exclude from plotting (optional)
    """
    if exclude_species is None:
        exclude_species = set()
    # Define the elements and aqeuous species we want to plot as lines
    lines = {
     ChemicalSpecies('tab:cyan', 'Fe', 'Fe'),
     ChemicalSpecies('tab:orange', 'Ca', 'Ca'),
    ChemicalSpecies('tab:green', 'K', 'K'),
     ChemicalSpecies('tab:red', 'Na', 'Na'),
     ChemicalSpecies('tab:red', 'Cl', 'Cl'),
    ChemicalSpecies('tab:purple', 'SO4^{2-}', 'SO4'),
        ChemicalSpecies('tab:olive', 'Mg', 'Mg'),
     ChemicalSpecies('tab:blue', 'Si', 'Si'),
    ChemicalSpecies('tab:olive', 'C', 'C'),
        ChemicalSpecies('tab:olive', 'HS^{-}+H_{2}S', 'HS-+H2S')
    }
    # Filter out excluded species
    desired_species_to_plot = [s for s in lines if s.species not in exclude_species]
    # Get data
    element_names, aqueous_species_names, solid_phases_names, pH_array, aqueous_species_array_molal, element_species_array_molal, solid_species_array_mol, solid_phases_volume_array, wppt_array = data
    fig2 = plt.figure(figsize=(8, 10))
    ax4 = fig2.add_subplot(1, 1, 1)  # Create a new subplot for fig2
    texts_ax4 = []
    for i, species_name in enumerate(aqueous_species_names):
        if species_name != 'H2O':
            aqueous_species_data = aqueous_species_array_molal[i]
            # Find the index of the 80th percentile value
            percentile_70 = np.percentile(aqueous_species_data, 70)
            max_index = np.abs(aqueous_species_data - percentile_70).argmin()
            line, = ax4.plot(x, aqueous_species_data, label=species_name)
            # Only add text label if the value is within species_lim
            y_value = aqueous_species_data[max_index]
            if species_lim[0] <= y_value <= species_lim[1]:
                # Format species name to properly display charges as superscripts
                formatted_species = species_name.replace('+', '^{+}').replace('-', '^{-}')
                # Handle multiple charges (e.g., +2, -2)
                formatted_species = formatted_species.replace('^{+}2', '^{2+}').replace('^{-}2', '^{2-}')
                formatted_species = formatted_species.replace('^{+}3', '^{3+}').replace('^{-}3', '^{3-}')
                text_obj = ax4.text(x[max_index], y_value, rf'$\ce{{{formatted_species}}}$', ha='left', va='top',
                         color=line.get_color(), fontsize=species_font_size,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=alpha_box, edgecolor='none'))
                texts_ax4.append(text_obj)
    # Set scales and add titles
    xlabelTitle = xlabel.replace('Log ', '')
    fig2.suptitle(f'Europa\'s Aqueous System  {xlabelTitle} Model')
    ax4.set_yscale('log')
    ax4.set_ylim(species_lim)
    adjust_text(texts_ax4, ax=ax4, expand=(2, 3), # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
             arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax4.set_ylabel(r'Log Mole (kg $\ce{H2O^{-1}}$)')
    ax4.set_xlabel(xlabel)


    plt.tight_layout()
    if SHOWPLOT:
        plt.show()  # Use plt.show() instead of fig2.show()

    # Set up the plot
    # Create subplots (one on top and one below)
    fig = plt.figure(figsize=(8,10))
    fig.suptitle(f'Europa\'s Aqueous System Across {xlabelTitle} Spectrum')
    grid = GridSpec(7, 1)
    ax1 = fig.add_subplot(grid[0:3, 0])
    ax2 = fig.add_subplot(grid[3, 0])
    ax3 = fig.add_subplot(grid[4:7, 0])
    texts_ax1 = []
    texts_ax3 = []
    for i,species in enumerate(desired_species_to_plot):
        if species.species == 'SO4':
            index_species = np.where(np.char.find(aqueous_species_names, species.species) != -1)
        elif species.species == 'HS-+H2S':
            index_species = np.where((np.char.find(aqueous_species_names, 'HS-') != -1) | (np.char.find(aqueous_species_names, 'H2S') != -1))

        else:
            index_species = np.where(aqueous_species_names == species.species)
        index_element = np.where(element_names == species.species)
        if index_species[0].size > 0:
            Plot = True
            species_molality = np.zeros(np.shape(aqueous_species_array_molal[index_species[0][0]]))
            for index in index_species[0]:
                species_molality += aqueous_species_array_molal[index]
        elif index_element[0].size > 0:
            Plot = True
            species_molality = element_species_array_molal[index_element[0][0]]
        else:
            Plot = False

        if Plot:
            line,  = ax1.plot(x, species_molality, label=species.species, color = species.color)
            xText = np.linspace(x[0], x[-1], len(desired_species_to_plot) + 2)
            xPosition = xText[i+1]
            # Find y position of closest y value associated with closest x value to xPosition
            closest_x_index = np.argmin(np.abs(x - xPosition))
            yPosition = species_molality[closest_x_index]
            text_obj = ax1.text(xPosition, yPosition, rf'$\ce{{{species.mhchem_name}}}$', ha='left',
                       va='top', color=line.get_color(), fontsize=element_font_size,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=alpha_box, edgecolor='none'))
            texts_ax1.append(text_obj)
    # Add labels and title
    ax1.set_yscale('log')
    # Apply LogFormatterExponent to show only exponents
    formatter = LogFormatterExponent(base=10)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.set_xticklabels([]) # hide labels but keep ticks
    y_limit_min, y_limit_max = species_lim  # Adjust this factor as needed; 10 is an example value for the log scale
    # Set y-limits for aqueous species plots
    ax1.set_ylim(bottom=y_limit_min, top = y_limit_max)
    ax1.set_ylabel(r'Log Mole (kg $\ce{H2O^{-1}}$)')
    xmax = np.max(x)
    xmin = np.min(x)
    ax2.set_xlim(xmin, xmax)
    ax1.set_xlim(xmin, xmax)

    
    # Plot pH on the left y-axis
    ax2.plot(x, pH_array, color='black', linestyle='-', label='pH')
    ax2.set_ylabel('pH', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Set ticks every 2 units
    min_pH = np.min(pH_array)
    max_pH = np.max(pH_array)
    pH_ticks = np.arange(np.floor(min_pH/2)*2, np.ceil(max_pH/2)*2 + 2, 2)
    ax2.set_yticks(pH_ticks)
    
    # Plot ppt on the right y-axis
    # Create a twin axis for ppt_array
    if True:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, wppt_array, color='red', linestyle='--', label='ppt')
        ax2_twin.set_ylabel('ppt', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Set ticks every 50ppm
        min_ppt = np.min(wppt_array)
        max_ppt = np.max(wppt_array)
        ppt_ticks = np.arange(np.floor(min_ppt/50)*50, np.ceil(max_ppt/50)*50 + 50, 50)
        ax2_twin.set_yticks(ppt_ticks)
    ax3.set_xlabel(xlabel)
    desired_solid_species_to_plot = solid_phases_names
    for i, species_name in enumerate(desired_solid_species_to_plot):
        species_molality = solid_phases_volume_array[i]
        Plot = False
        if np.any(species_molality > 1e-14):
            Plot = True
        # Plot the phase plot
        if Plot:
            # Find the index of the 80th percentile value
            percentile_100 = np.percentile(species_molality, 100)
            max_index = np.abs(species_molality - percentile_100).argmin()
            line, = ax3.plot(x, species_molality, label=species_name)
            text_obj = ax3.text(x[max_index], species_molality[max_index], species_name, ha='left', va='top',
                     color=line.get_color(), fontsize=mineral_font_size,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=alpha_box, edgecolor='none'))
            texts_ax3.append(text_obj)
    y_limit_min, y_limit_max = pH_lim
    ax2.set_ylim(bottom=y_limit_min, top = y_limit_max)
    ax2.set_xticklabels([]) # hide labels but keep ticks
    ax3.set_ylabel(r"Vol. of minerals, $\mathrm{cm^{3}}/\mathrm{kg}$ rock")
    # Set x-axis limits and ticks for ax3
    x_start = float(np.min(x))
    x_end = float(np.max(x))
    ax3.set_xlim(x_start, x_end)
    # Generate ticks every 2 units from min(x) up to max(x) (not exceeding max)
    tick_start = int(np.ceil(x_start))
    tick_end = int(np.floor(x_end))
    ticks = list(range(tick_start, tick_end + 1, 2))
    if ticks and ticks[-1] > x_end:
        ticks = ticks[:-1]
    ax3.set_xticks(ticks)
    # After plotting everything, synchronize x-ticks across all axes
    ax1.set_xticks(ticks)
    ax2.set_xticks(ticks)
    # Adjust text to prevent overlap and keep within bounds
    adjust_text(texts_ax1, ax=ax1, expand=(2, 3), # expand text bounding boxes by 1.2 fold in x direction and 3 fold in y direction
             arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    adjust_text(texts_ax3, ax=ax3, expand=(1, 1.2), # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
             arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    plt.tight_layout()
    if SHOWPLOT:
        plt.show()
    # Save figure
    fig2.savefig(f"{figname}_species.pdf", format='pdf', transparent=True)

    # Save figure
    fig.savefig(f"{figname}.pdf", format='pdf', transparent=True)






def generate_chemical_system(db, elemental_molal, Do_minerals = False, use_activity_model=False, DO_speciate=False,
                             elements=None, DO_PH=True, DO_CO2 = True, DO_H2 = True, add_elements = True):
    aqueous_species = ''
    for species in db.species().withElements(elements):
        if species.aggregateState() == AggregateState.Aqueous:
            if abs(species.charge()) > 0:
                aqueous_species += species.name() + ' ' if species.name() in ionic_species_to_consider else ''
            else:
                aqueous_species += species.name() + ' ' if species.name() not in aqueous_species_to_exclude else ''

    # Define the aqueous phase by speciating the elements
    solution = AqueousPhase(aqueous_species)


    # Define the gas phases by speciating the elements
    gases = GaseousPhase(speciate(elements))

    # Use pitzer and peng robinson activity models
    if use_activity_model:
        solution.set(chain(ActivityModelPitzer() , ActivityModelPhreeqcIonicStrengthPressureCorrection()))
        gases.set(ActivityModelPengRobinson())
    if Do_minerals:
        minerals = MineralPhases()
        # Define the solid solution phases in the system
        MgNaCaKSaponite = MineralPhase("Saponite-Mg-Mg Saponite-Mg-Na Saponite-Mg-K Saponite-Mg-Ca")
        MgNaCaKSaponite.setName("Mg-Na-Ca-K-saponite")
        Montmor = MineralPhase("Montmor-Ca Montmor-K Montmor-Mg Montmor-Na")
        Montmor.setName("Na-Mg-K-Ca-montmorillonite")

        # Define the mineral phases
        sepiolite = MineralPhase("Sepiolite")
        Chrysotile = MineralPhase("Chrysotile")
        goethite = MineralPhase("Goethite")
        amorphous_silica = MineralPhase("SiO2(am)")
        dolomite = MineralPhase("Dolomite")
        calcite = MineralPhase("Calcite")
        andradite = MineralPhase("Andradite")
        Sylvite = MineralPhase("Sylvite")
        gypsum = MineralPhase("Gypsum")
        magnetite = MineralPhase("Magnetite")
        pyrrhotite = MineralPhase("Pyrrhotite")
        magnesite = MineralPhase("Magnesite")
        pyrite = MineralPhase("Pyrite")
        daphnite = MineralPhase('Daphnite-14A')
        siderite = MineralPhase("Siderite")

        # Create the system
        #system = ChemicalSystem(db, solution, gases, minerals)
        system = ChemicalSystem(db, solution, gases, MgNaCaKSaponite, goethite, amorphous_silica, gypsum, daphnite, magnesite, siderite, Montmor, pyrite, dolomite, calcite, Chrysotile, Sylvite, andradite, pyrrhotite, magnetite, sepiolite)

    else:
        system = ChemicalSystem(db, solution, gases)

    # Define the state
    state = ChemicalState(system)

    # Set equilibrium specifications needed for Reaktoro
    specs = EquilibriumSpecs(system)
    specs.temperature()
    specs.pressure()
    if DO_H2:
        specs.fugacity("H2(g)")
    if DO_CO2:
        specs.fugacity("CO2(g)")
    if DO_PH:
        specs.pH()
        
    if add_elements:
        # Get the element amounts from the initial state
        element_amounts = state.componentAmounts()
        # Add the mineral species to the state
        for compound, mass in elemental_molal:
            system, element_amounts = add_molal_elements(system, element_amounts, compound, mass)
        # Add mass_g * water_rock_ratio (1kg) to the state
        add_compound_mass_to_elements(system, element_amounts, "H2O", mass_g * water_rock_ratio)

        # The Zolotov model separately adds Cl- as it assumes 'The majority of Cl was extracted from accreted rocks' rather than the CI chondrite, so we also add a determined Cl- amount
        add_compound_mass_to_elements(system, element_amounts, "Cl-", 4.25436)
        conditions = EquilibriumConditions(specs)
        conditions.setInitialComponentAmounts(element_amounts)
    else:
        conditions = EquilibriumConditions(specs)
    
    
    restrictions = EquilibriumRestrictions(system)
    # if Do_minerals:
    #     restrictions.cannotReact('Riebeckite')
    # Initialize solver
    solver = EquilibriumSolver(specs)

    # Return the system and objects needed to interact with the system
    return system, state, conditions, solver, restrictions

def scale_compound_masses(compound_masses, target_total_mass):
    """
    Scales the masses in compound_masses so their sum equals target_total_mass.
    
    Args:
        compound_masses (list): List of (compound, mass) tuples
        target_total_mass (float): Target total mass in grams
        
    Returns:
        list: New list of (compound, scaled_mass) tuples
        
    Raises:
        ValueError: If current total mass is zero
    """
    current_total = sum(mass for _, mass in compound_masses)
    if current_total == 0:
        raise ValueError("Current total mass is zero, cannot scale.")
    scale_factor = target_total_mass / current_total
    return [(compound, mass * scale_factor) for compound, mass in compound_masses]


def add_compound_mass_to_elements(system, element_amounts, compound, mass_g):
    """
    Adds the elemental moles from a given mass (g) of a compound to the element_amounts array.
    
    Args:
        system: Reaktoro ChemicalSystem object
        element_amounts: Array of element amounts (moles) to modify
        compound (str): Chemical formula of the compound
        mass_g (float): Mass of compound in grams
        
    Returns:
        tuple: (system, updated_element_amounts)
    """
    formula = ChemicalFormula(compound)
    elements = formula.elements()  # returns Pairs<String, double>
    compound_molar_mass = formula.molarMass()  # kg/mol
    moles_compound = mass_g / (compound_molar_mass * 1000)  # convert g to kg, then to mol
    
    for elem, coeff in elements:
        idx = system.elements().index(elem)
        element_amounts[idx] += moles_compound * coeff
    
    return system, element_amounts

def add_molal_elements(system, element_amounts, element, molal):
    """
    Adds the elemental moles from a given molal (mol/kg) of a compound to the element_amounts array.
    """
    idx = system.elements().index(element)
    element_amounts[idx] += molal
    return system, element_amounts


class ChemicalSpecies:
    def __init__(self, color, mhchem_name, species):
        """
        Initialize the ChemicalSpecies object with color, mhchem_name, and species.

        Parameters:
        color (str): The color associated with the species.
        mhchem_name (str): The name of the species in mhchem format.
        species (str): The species name.
        """
        self.color = color
        self.mhchem_name = mhchem_name
        self.species = species

    def __repr__(self):
        return f"ChemicalSpecies(color='{self.color}', mhchem_name='{self.mhchem_name}', species='{self.species}')"


def save_aqueous_species_mat(aqueous_species_names, aqueous_species_array_molal, fugacity_array, fugacity_label, pH_array):
    """
    Function to save the aqueous species to .mat file of strings formatted for compatibility with PlanetProfile
    :param aqueous_species_names: name of aqueous species
    :param aqueous_species_array_molal: data of molal (mol/kg) of each aqueous species
    :param fugacity_array: array of fugacity values
    :param fugacity_label: string label for the fugacity variable (e.g., 'H2' or 'CO2')
    :param pH_array: array of pH values
    :return: wppt_list
    """
    wppt_list = []
    m_strings = []  # For .mat file (LaTeX format)
    txt_strings = []  # For .txt file (normal format)
    
    # First, calculate wppt for each condition (column)
    for j in range(aqueous_species_array_molal.shape[1]):
        # Create species string for this condition
        species_string_with_ratios = ", ".join(
            f"{species}: {aqueous_species_array_molal[i, j]}" for i, species in
            enumerate(aqueous_species_names)
        )
        # Calculate wppt for this condition
        wppt = wpptCalculator(species_string_with_ratios)
        wppt_list.append(wppt)
        
        # Create the custom solution labels with fugacity, ppt, and pH
        fugacity_val = fugacity_array[j]
        pH_val = pH_array[j]
        
        # LaTeX format for .mat file
        latex_label = rf"CustomSolution${fugacity_val:.1f}\log f\mathrm{{H}}_2\!:\!{wppt:.1f}\mathrm{{ppt}},{pH_val:.1f}\,\mathrm{{pH}}$"
        
        # Normal format for .txt file
        normal_label = f"CustomSolution{fugacity_label}Fugacity{fugacity_val:.2f}ppt{wppt:.1f}pH{pH_val:.2f}"
        
        # Create the full strings
        latex_full_string = f"{latex_label} = {species_string_with_ratios}"
        normal_full_string = f"{normal_label} = {species_string_with_ratios}"
        
        m_strings.append(latex_full_string)
        txt_strings.append(normal_full_string)

    # Save labels to a text file with normal format
    if saveToTxtFile:
        save_custom_solution_labels_to_txt(txt_strings, fugacity_label)

    return wppt_list, m_strings

def save_custom_solution_labels_to_txt(custom_solution_labels, fugacity_label):
    """
    Saves a list of CustomSolutionLabels to a .txt file, with each label on a new line.

    :param custom_solution_labels: A list of strings, where each string is a CustomSolutionLabel.
    :param fugacity_label: The label for the fugacity variable (e.g., 'H2' or 'CO2'), used for the filename.
    """
    filename = os.path.join(_TXTOUTPUTDIR, f'{fugacity_label}CustomSolutionLabels.txt')
    with open(filename, 'w') as f:
        for label in custom_solution_labels:
            f.write(f"{label}\n")
    
def wpptCalculator(species_string_with_ratios_mol_kg):
    """
    Calculate the total solute mass in grams of the species string and return. Assumes that the species string is in mol/kg
     Parameters
     ----------
     species_string_with_ratios_mol_kg: String of all the species that should be considered in aqueous phase and their corresponding molal ratios.
        For example, "Cl-: 19.076, Na+: 5.002, Ca2+: 0.0"
     Returns
     -------
     total_solute_mass_g: total solute mass of solution in grams
    """
    # Convert the species string into a string of species and its corresponding dictionary of ratios
    aqueous_species_string, speciation_mol_kg = SpeciesFormatter(species_string_with_ratios_mol_kg)
    # Sum up total amount of mass in grams
    total_solute_mass_g = 0
    # Go through each species in dictionary and calculate total mass in grams
    for species, ratio_mol_kg in speciation_mol_kg.items():
        # Don't add H2O mass
        if species != "H2O":
            try:
                species_molar_mass_g_mol = Species(species).molarMass() * 1000
            except:
                try:
                    species_molar_mass_g_mol = Species(str(db.species(species).formula())).molarMass() * 1000
                except:
                    raise ValueError(f'Species {species} not found in Reaktoro database. Check that the species is spelled correctly and is in the database.')
            species_gram_kg = species_molar_mass_g_mol * ratio_mol_kg
            total_solute_mass_g += species_gram_kg
    return total_solute_mass_g

def SpeciesFormatter(species_string_with_ratios):
    '''
    Converts provided String of species and ratios into formats necessary for Reaktoro. Namely, creates
    a String of all the species in the list and a dictionary with 'active' species that are added to solution (the observer species are
    automatically generated to be 1e-16 moles in the solution by Reaktoro).

     Parameters
     ----------
     species_string_with_ratios: String of all the species that should be considered in aqueous phase and their corresponding ratios.
        For example, "Cl-: 19.076, Na+: 5.002, Ca2+: 0.0"
     Returns
     -------
     aqueous_species_string: String that has all species names that should be considered in aqueous phase
     speciation_ratio_mol_per_kg: Dictionary of active species and the values of their molar ratio (mol/kg of water)
    '''
    # Dictionary of speciation
    speciation_ratio_mol_per_kg = {}
    # Go through each species and corresponding ratio_mol_per_kg and add to corresponding lists
    for species_with_ratio in species_string_with_ratios.split(", "):
        species, ratio_mol_per_kg = species_with_ratio.split(": ")
        speciation_ratio_mol_per_kg[species] = float(ratio_mol_per_kg)
    # String of speciation
    aqueous_species_str = " ".join(speciation_ratio_mol_per_kg.keys())
    # Return string and species
    return aqueous_species_str, speciation_ratio_mol_per_kg

if __name__ == "__main__":
    Replicate_Zolotov_H2()
    #Replicate_Zolotov_CO2()

def SetSettings(save_to_txt_file=saveToTxtFile, output_figures=OUTPUTFIGURES, mat_output_dir=_MATOUTPUTDIR, txt_output_dir=_TXTOUTPUTDIR, figure_output_dir=_FIGUREOUTPUTDIR, matFileName=_matFileName):
    global saveToTxtFile, OUTPUTFIGURES, _MATOUTPUTDIR, _TXTOUTPUTDIR, _FIGUREOUTPUTDIR, _matFileName
    saveToTxtFile = save_to_txt_file
    OUTPUTFIGURES = output_figures
    _MATOUTPUTDIR = mat_output_dir
    _TXTOUTPUTDIR = txt_output_dir
    _FIGUREOUTPUTDIR = figure_output_dir
    _matFileName = matFileName
