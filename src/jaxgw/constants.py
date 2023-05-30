from astropy.constants import c,au,G,pc
from astropy.units import year as yr
from astropy.cosmology import WMAP9 as cosmo

Msun = 4.9255e-6
year = (1*yr).cgs.value
Mpc = 1e6*pc.value/c.value
euler_gamma = 0.577215664901532860606512090082
MR_sun = 1.476625061404649406193430731479084713e3
C_SI = 299792458.0

EARTH_SEMI_MAJOR_AXIS = 6378137  # for ellipsoid model of Earth, in m
EARTH_SEMI_MINOR_AXIS = 6356752.314  # in m

DAYSID_SI = 86164.09053133354
DAYJUL_SI = 86400.0