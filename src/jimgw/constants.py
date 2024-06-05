from astropy.constants import c, pc  # type: ignore TODO: fix astropy stubs
import astropy.units as u  # type: ignore

Msun = 4.9255e-6
year = (1 * u.yr).cgs.value  # type: ignore
Mpc = 1e6 * pc.value / c.value
euler_gamma = 0.577215664901532860606512090082
MR_sun = 1.476625061404649406193430731479084713e3
C_SI = 299792458.0

EARTH_SEMI_MAJOR_AXIS = 6378137.0  # for ellipsoid model of Earth, in m
EARTH_SEMI_MINOR_AXIS = 6356752.314  # in m

DAYSID_SI = 86164.09053133354
DAYJUL_SI = 86400.0
