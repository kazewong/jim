'''
This module computes a JAX-compatible conversion from GPS to UTC time and Julian day.
Following the implementation in LALSuite and Bilby.

The conversion from GPS time to UTC date is typically done using the following packages:
    * datetime, astropy, numpy
but none of them are JAX-compatible.
There is the new `jax_datetime` package, but it does not compute the year and month.
See: https://github.com/google/jax-datetime/
'''
from jax import config
import jax.numpy as np
config.update("jax_enable_x64", True)
from jaxtyping import Float, Int

GPS_EPOCH: int = 315964800
EPOCH_J2000_0_JD: float = 2451545.0
'''
 * Leap seconds list
 *
 * JD and GPS time of leap seconds and the value of TAI-UTC.
 *
 * reference: http://maia.usno.navy.mil/
 *            http://maia.usno.navy.mil/ser7/tai-utc.dat
 *
 * notes: the list below must be updated whenever a leap second is added
 * See also: https://lscsoft.docs.ligo.org/lalsuite/lal/_x_l_a_l_leap_seconds_8h_source.html
 * https://data.iana.org/time-zones/data/leap-seconds.list
'''
LEAP_SECONDS = np.array([
    46828800,
    78364801,
    109900802,
    173059203,
    252028804,
    315187205,
    346723206,
    393984007,
    425520008,
    457056009,
    504489610,
    551750411,
    599184012,
    820108813,
    914803214,
    1025136015,
    1119744016,
    1167264017,
])

def int_div(a: Int, b: Int) -> Int:
    '''This is to emulate the C-style integer division in Python.
    See: https://stackoverflow.com/a/61386872
    '''
    q, r = a // b, a % b
    return np.where(
            ((a >= 0) != (b >= 0)) & r,
            q + 1, q)


def n_leap_seconds(date: Int) -> Int:
    """
    Find the number of leap seconds required for the specified date.

    Search in reverse order as in practice, almost all requested times will
    be after the most recent leap.
    
    The Bilby Cython implementation:
    NUM_LEAPS: int = 18
    n_leaps: int = NUM_LEAPS

    if date > LEAP_SECONDS[NUM_LEAPS - 1]:
        return NUM_LEAPS
    while (n_leaps > 0) and (date < LEAP_SECONDS[n_leaps - 1]):
        n_leaps -= 1
    return n_leaps

    Args:
        date (int): The date in seconds since the GPS epoch.
    Returns:
        int: The number of leap seconds.
    """
    return np.sum(date > LEAP_SECONDS).astype(np.float64)

def is_leap_year(year):
    '''Function to check if a year is a leap year.
    '''
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))

# Constants
SECONDS_IN_DAY = 24 * 60 * 60
SECONDS_IN_YEAR = 365 * SECONDS_IN_DAY
LEAP_YEAR_SECONDS = 366 * SECONDS_IN_DAY
MONTH_ARRAY = np.arange(1, 13)
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
UNIX_EPOCH_YEAR = 1970

# **Someone in the future please update this**
# This is surely not the most ideal way, but for the code to be jittable, 
# this seems to be a not too bad compromise.
YEAR_ARRAY = np.arange(UNIX_EPOCH_YEAR, 2500)
IS_LEAP_YEARS = is_leap_year(YEAR_ARRAY)
LEAP_SEC_ARRAY = np.where(IS_LEAP_YEARS, LEAP_YEAR_SECONDS, SECONDS_IN_YEAR)

def utc_date_from_timestamp(timestamp: Int) -> tuple[Int, Int, Int, Int]:
    '''
    This function converts a UTC timestamp to a UTC date (year, month, day, seconds).

    This sole intention of this function is to be JAX-compatible and be used within Jim .
    While it has been agressively tested against the C-implementation in LAL, 
    and the datetime module, it is advised to use those other modules for 
    other purposes.

    Note that the current implementation has an upper limit of year up to 2500, 
    which is sufficient for most practical purposes.
    '''
    # The lower bound assumes every year is a leap year
    year = timestamp // LEAP_YEAR_SECONDS + UNIX_EPOCH_YEAR
    seconds_before = np.where(YEAR_ARRAY < year, LEAP_SEC_ARRAY, 0).sum()

    # if (year < 2500):
    #     raise ValueError(
    #         f"This function is designed to work between years {UNIX_EPOCH_YEAR} and 2500. The year {year} is out of bounds."
    #     )

    remaining_seconds = timestamp - seconds_before
    sec_in_year = np.where(is_leap_year(year), LEAP_YEAR_SECONDS, SECONDS_IN_YEAR)
    is_more_than = (remaining_seconds > sec_in_year).astype(np.int32)
    year += is_more_than
    remaining_seconds -= is_more_than * sec_in_year
    is_negative = (remaining_seconds < 0).astype(np.int32)
    year -= is_negative
    remaining_seconds += is_negative * sec_in_year

    # Adjust for leap year
    sec_in_months = np.where(
        is_leap_year(year),
        DAYS_IN_MONTH.at[1].set(29),
        DAYS_IN_MONTH,
    ) * SECONDS_IN_DAY
    
    month = remaining_seconds // (28 * SECONDS_IN_DAY)
    seconds_before_mon = np.where(MONTH_ARRAY < month, sec_in_months, 0).sum()
    month += (month == 0).astype(np.int32)
    remaining_seconds -= seconds_before_mon

    sec_in_month = sec_in_months[month - 1]
    is_more_than = (remaining_seconds > sec_in_month).astype(np.int32)
    month += is_more_than
    remaining_seconds -= is_more_than * sec_in_month
    is_negative = (remaining_seconds < 0).astype(np.int32)
    month -= is_negative
    remaining_seconds += is_negative * sec_in_month

    # Calculate the day and seconds
    day, seconds = divmod(remaining_seconds, SECONDS_IN_DAY)
    return year, month, day + 1, seconds.astype(np.int32)


def gps_to_utc_date(gps_time: Float) -> tuple[Int, Int, Int, Int]:
    '''
    Args:
        gps_time (float): The GPS time to convert.
    Returns:
        tuple (int): A tuple containing the year, month, day, and seconds.
    '''
    _sec = gps_time - n_leap_seconds(gps_time)
    return utc_date_from_timestamp(GPS_EPOCH + _sec.astype(int))


def gps_to_julian_day(gps_time: Float) -> Float:
    """
    Convert from UTC to Julian day, this is a necessary intermediate step in
    converting from GPS to GMST.

    The type-cast on the second is necessary for consistency with the C
    implementation. Without which the error is of order 0.1.

    The `int_div` function is introduced to emulate the C-style integer division.

    Args:
        gps_time (float): The GPS time to convert.

    Returns:
        float: The Julian day corresponding to the given UTC time.
    """
    year, month, day, second = gps_to_utc_date(gps_time)
    return (
        367 * year
        - int_div(7 * (year + int_div(month + 9, 12)), 4)
        + int_div(275 * month, 9)
        + day
        + 1721014
        + second.astype(float) / (60 * 60 * 24)
        - 0.5
    )


def greenwich_mean_sidereal_time(gps_time: Float) -> Float:
    """
    Compute the Greenwich mean sidereal time from the GPS time.

    Ags:
        gps_time (float): The GPS time to convert
    Returns:
        float: The Greenwich mean sidereal time in radians.
    """
    return greenwich_sidereal_time(gps_time, 0.0)


def greenwich_sidereal_time(
        gps_time: Float, 
        equation_of_equinoxes: Float) -> Float:
    """
    Compute the Greenwich mean sidereal time from the GPS time and equation of
    equinoxes.

    Based on XLALGreenwichSiderealTime in lalsuite/lal/lib/XLALSiderealTime.c.

    Args:
        gps_time (float): The GPS time to convert
        equation_of_equinoxes (float): The equation of equinoxes
    Returns:
        float: The Greenwich sidereal time in radians.
    """
    julian_day = gps_to_julian_day(gps_time)
    t_hi = (julian_day - EPOCH_J2000_0_JD) / 36525.0
    t_lo = (gps_time % 1) / (36525.0 * 86400.0)

    t = t_hi + t_lo

    sidereal_time = equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t**2 + 67310.54841
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * np.pi / 43200.0