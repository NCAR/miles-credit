"""
A collection of constants

Reference:

Harrop, B.E., Pritchard, M.S., Parishani, H., Gettelman, A., Hagos, S., Lauritzen, P.H., Leung, L.R., Lu, J., Pressel, K.G. and Sakaguchi, K., 2022. Conservation of dry air, water, and energy in CAM and its potential impact on tropical rainfall. Journal of Climate, 35(9), pp.2895-2917.
"""
import numpy as np
# Earth's radius
RAD_EARTH = np.float64(6371000 ) # m

# ideal gas constant of water vapor
RVGAS = np.float64(461.5)  # J/kg/K

# ideal gas constant of dry air
RDGAS = np.float64(287.05)  # J/kg/K

# gravity
GRAVITY = np.float64(9.80665)  # m/s^2

# density of water
RHO_WATER = np.float64(1000.0)  # kg/m^3

# ========================================================= #
# latent heat caused by the phase change of water (0 deg C)
LH_WATER = np.float64(2.501e6)  # J/kg
LH_ICE = np.float64(333700)  # J/kg

# ========================================================= #
# heat capacity on constant pressure
# dry air
CP_DRY = np.float64(1004.64)  # J/kg K
# water vapor
CP_VAPOR = np.float64(1810.0)  # J/kg K
# liquid
CP_LIQUID = np.float64(4188.0)  # J/kg K
# ice
CP_ICE = np.float64(2117.27)  # J/kg K
