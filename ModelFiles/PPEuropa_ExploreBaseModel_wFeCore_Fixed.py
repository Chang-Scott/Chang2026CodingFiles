"""
PPEuropa
Contains all body-specific parameters and information for PlanetProfile models of this body.
Import as a module and access information assigned to the attributes of the Planet struct.
Note that this file expects to be imported from the directory above.
"""
import numpy as np
from PlanetProfile.Utilities.defineStructs import PlanetStruct, Constants

Planet = PlanetStruct('Europa')

# Reduce search range for melting pressure to values realistic for Europa
Planet.PfreezeUpper_MPa = 150

""" Bulk planetary settings """
Planet.Bulk.R_m = 1560.8e3  # Value from mean radius in Archinal et al. (2018): https://doi.org/10.1007/s10569-017-9805-5
Planet.Bulk.M_kg = 4.800e22  # Value from Hussmann et al. (2006): http://dx.doi.org/10.1016/j.icarus.2006.06.005
Planet.Bulk.Tsurf_K = 110
Planet.Bulk.Psurf_MPa = 0.0
Planet.Bulk.Cmeasured = 0.3547  # Value from Anderson et al. (1998): https://doi.org/10.1126/science.281.5385.2019
Planet.Bulk.Cuncertainty = 0.0024
Planet.Bulk.Tb_K = 269.685  # 30 km ice with 1.0x Seawater
Planet.Do.ICEIh_THICKNESS = True
Planet.Bulk.zb_approximate_km = 30 # The approximate ice shell thickness desired
""" Layer step settings """
Planet.Steps.nIceI = 50
Planet.Steps.nSilMax = 300
Planet.Steps.nCore = 10
Planet.Steps.iSilStart = 50

""" Hydrosphere assumptions/settings """
Planet.Ocean.comp = 'CustomSolutionExplore = TBD'
Planet.Ocean.wOcean_ppt = None
# If using a custom solution, specify the species in the solution and the mol/kg of water.
# All desired species must be specified except H2O, which is assumed to be in solution at 1 mol equivalent for 1kg
Planet.Ocean.deltaP = 2.0
Planet.Ocean.deltaT = 1.0
Planet.Ocean.propsStepReductionFactor = 10
Planet.Ocean.PHydroMax_MPa = 1250.0
Planet.TfreezeLower_K = 240.0
Planet.TfreezeUpper_K = 274.0
Planet.TfreezeRes_K = 0.005
Planet.PfreezeUpper_MPa = 200.0
Planet.PfreezeRes_MPa = 0.5
Planet.PfreezeLower_MPa = 0.01

""" Silicate Mantle """
Planet.Do.CONSTANT_INNER_DENSITY = True
Planet.Sil.rhoSilWithCore_kgm3 = 3500
Planet.Do.Fe_CORE = True
Planet.Core.rhoFe_kgm3 = 8000
Planet.Core.rhoFeS_kgm3 = 5650
Planet.Core.xFeS = 0
Planet.Sil.mantleEOS = 'CI_undifferentiated_hhph_DEW17_nofluid_nomelt_685.tab'  # (2900 for Q= 100 GW, 3240 for Q= 220 GW)
Planet.Core.coreEOS = 'Fe-S_3D_EOS.mat'
Planet.Sil.GSmean_GPa = 50.0
Planet.Core.GSmean_GPa = 50.0
Planet.Core.etaFeSolid_Pas = Constants.etaFeLiquid_Pas
Planet.Core.kTherm_WmK = Constants.kThermFe_WmK
Planet.Sil.kTherm_WmK = Constants.kThermSil_WmK
Planet.Sil.etaRock_Pas = 1e20


""" Seismic properties of solids """
Planet.Seismic.lowQDiv = 1.0

""" Magnetic induction """
Planet.Bulk.J2 = 435.5e-6  # J2 and C22 values from Anderson et al. (1998): https://doi.org/10.1126/science.281.5385.2019
Planet.Bulk.C22 = 131.0e-6
Planet.Magnetic.ionosBounds_m = 100e3
Planet.Magnetic.sigmaIonosPedersen_Sm = 30/100e3
Planet.Magnetic.SCera = 'Juno'
Planet.Magnetic.extModel = 'JRM33C2020'

# The block below should be made into one single function that returns the FTdata struct if the file is found, and warns the user/downloads if not.
# try:
#     load(['FTdata' Planet.name],'FTdata')
# catch:
#     dlNow = FTdataMissingWarning()
#     if dlNow; load(['FTdata' Planet.name],'FTdata'); end

""" Interior constraints imposed in Vance et al. 2014 """
# Planet.Sil.mSi = 28.0855
# Planet.Sil.mS = 32.065
# Planet.Sil.mFe = 55.845
# Planet.Sil.mMg = 24.305
# Planet.Sil.xOl = 0.44  # Percentage of olivine - Javoy (1995) - Fe/Si = 0.909 Mg/Si = 0.531, Mg# = 0.369
# Planet.Sil.xSi = (Planet.Sil.xOl+2*(1-Planet.Sil.xOl))*Planet.Sil.mSi/(Planet.Sil.xOl*184.295+(1-Planet.Sil.xOl)*244.3805) # mass fraction of sulfur in silicates
# Planet.Sil.MEarth_kg = 5.97e24
# Planet.Sil.xSiEarth = 0.1923  # Javoy in kg/kg in Exoplanets paper20052006-xSiSolar only in mole
# Planet.Sil.xK = 1.0  # Enrichment in K
