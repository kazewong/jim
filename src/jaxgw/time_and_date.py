import jax.numpy as np

  {2444239.5,    -43200, 19},  /* 1980-Jan-01 */
  {2444786.5,  46828800, 20},  /* 1981-Jul-01 */
  {2445151.5,  78364801, 21},  /* 1982-Jul-01 */
  {2445516.5, 109900802, 22},  /* 1983-Jul-01 */
  {2446247.5, 173059203, 23},  /* 1985-Jul-01 */
#if 0
  /* NOTE: IF THIS WERE A NEGATIVE LEAP SECOND, INSERT AS FOLLOWS */
  {2447161.5, 252028803, 22},  /* 1988-Jan-01 EXAMPLE ONLY! */
#endif
  {2447161.5, 252028804, 24},  /* 1988-Jan-01 */
  {2447892.5, 315187205, 25},  /* 1990-Jan-01 */
  {2448257.5, 346723206, 26},  /* 1991-Jan-01 */
  {2448804.5, 393984007, 27},  /* 1992-Jul-01 */
  {2449169.5, 425520008, 28},  /* 1993-Jul-01 */
  {2449534.5, 457056009, 29},  /* 1994-Jul-01 */
  {2450083.5, 504489610, 30},  /* 1996-Jan-01 */
  {2450630.5, 551750411, 31},  /* 1997-Jul-01 */
  {2451179.5, 599184012, 32},  /* 1999-Jan-01 */
  {2453736.5, 820108813, 33},  /* 2006-Jan-01 */
  {2454832.5, 914803214, 34},  /* 2009-Jan-01 */
  {2456109.5, 1025136015, 35}, /* 2012-Jul-01 */
  {2457204.5, 1119744016, 36}, /* 2015-Jul-01 */
  {2457754.5, 1167264017, 37}, /* 2017-Jan-01 */


def gps_to_utc(gps_time):

def greenwich_mean_sidereal_time(gps_time):

def time_delay_geocentric(detector1, detector2, ra, dec, time):
    gmst = fmod(greenwich_mean_sidereal_time(time), 2 * np.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    omega = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    delta_d = detector2 - detector1
    return np.dot(omega, delta_d) / speed_of_light

