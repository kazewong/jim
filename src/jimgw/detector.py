import jax.numpy as jnp
from .constants import *
from .wave import Polarization
from scipy.signal.windows import tukey
from abc import abstractmethod
import equinox as eqx
from jaxtyping import Array


DEG_TO_RAD = jnp.pi/180

def np2(x):
    """Returns the next power of two as big as or larger than x."""
    p = 1
    while p < x:
        p = p << 1
    return p

class Detector(eqx.Module):
    """ Base class for all detectors.

    """

    @abstractmethod
    def load_data(self, data):
        raise NotImplementedError

    @abstractmethod
    def fd_response(self, frequency: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def td_response(self, time: Array) -> Array:
        raise NotImplementedError

class 

class Detector(object):
    """Defines a ground-based gravitational-wave detector.

    Argument
    --------
    name : str
        interferometer name, e.g., 'H1' for LIGO Hanford.
    coordinates : dict
        optionally, provide custom detector arm and vertex coordinates.
    """
    def __init__(self, name, coordinates=None):
        self.name = name.upper()
        self._coordinates = coordinates or {}

    @property
    def coordinates(self):
        """Coordinates defining a triangular detector (angles in radians).
        """
        if not self._coordinates:
            if self.name == 'H1':
                # LIGO Hanford
                self._coordinates = dict(
                    lat = (46 + 27. / 60 + 18.528 / 3600) * DEG_TO_RAD,
                    lon = -(119 + 24. / 60 + 27.5657 / 3600) * DEG_TO_RAD,
                    xarm_azimuth = 125.9994 * DEG_TO_RAD,
                    yarm_azimuth = 215.9994 * DEG_TO_RAD,
                    xarm_tilt = -6.195e-4,
                    yarm_tilt = 1.25e-5,
                    elevation = 142.554,
                )
            elif self.name == 'L1':
                # LIGO Livingston
                self._coordinates = dict(
                    lat = (30 + 33. / 60 + 46.4196 / 3600) * DEG_TO_RAD,
                    lon= -(90 + 46. / 60 + 27.2654 / 3600) * DEG_TO_RAD,
                    xarm_azimuth = 197.7165 * DEG_TO_RAD,
                    yarm_azimuth = 287.7165 * DEG_TO_RAD,
                    xarm_tilt = 0 ,
                    yarm_tilt = 0,
                    elevation = -6.574,
                )
            elif self.name == 'V1':
                # Virgo
                self._coordinates = dict(
                    lat = (43 + 37. / 60 + 53.0921 / 3600) * DEG_TO_RAD,
                    lon = (10 + 30. / 60 + 16.1878 / 3600) * DEG_TO_RAD,
                    xarm_azimuth = 70.5674 * DEG_TO_RAD,
                    yarm_azimuth = 160.5674 * DEG_TO_RAD,
                    xarm_tilt = 0,
                    yarm_tilt = 0,
                    elevation = 51.884,
                )
            elif not self._coordinates:
                raise ValueError(f"unknown detector {self.name}")
        return self._coordinates

    @staticmethod
    def _get_arm(lat, lon, tilt, azimuth):
        """Construct detector-arm vectors in Earth-centric Cartesian coordinates.

        Arguments
        ---------
        lat : float
            vertex latitude in rad.
        lon : float
            vertex longitude in rad.
        tilt : float
            arm tilt in rad.
        azimuth : float
            arm azimuth in rad.
        """
        e_lon = jnp.array([-jnp.sin(lon), jnp.cos(lon), 0])
        e_lat = jnp.array([-jnp.sin(lat) * jnp.cos(lon),
                          -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)])
        e_h = jnp.array([jnp.cos(lat) * jnp.cos(lon),
                        jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)])

        return (jnp.cos(tilt) * jnp.cos(azimuth) * e_lon +
                jnp.cos(tilt) * jnp.sin(azimuth) * e_lat +
                jnp.sin(tilt) * e_h)

    @property
    def arms(self):
        """Detector arm vectors (x, y).
        """
        c = self.coordinates
        x = self._get_arm(c['lat'], c['lon'], c['xarm_tilt'], c['xarm_azimuth'])
        y = self._get_arm(c['lat'], c['lon'], c['yarm_tilt'], c['yarm_azimuth'])
        return x, y
	
    @property
    def tensor(self):
        """Detector tensor defining the strain measurement.
        """
        #TODO: this could easily be generalized for other detector geometries
        arm1, arm2 = self.arms
        return  0.5 * (jnp.einsum('i,j->ij', arm1, arm1) - 
                       jnp.einsum('i,j->ij', arm2, arm2))

    @property
    def vertex(self):
        """Detector vertex coordinates in the reference celestial frame. Based
        on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.
        """
        # get detector and Earth parameters
        lat = self.coordinates['lat']
        lon = self.coordinates['lon']
        h = self.coordinates['elevation']
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2*(major**2*jnp.cos(lat)**2 + minor**2*jnp.sin(lat)**2)**(-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major)**2 * r + h)*jnp.sin(lat)
        return jnp.array([x, y, z])

    @property
    def delay_from_geocenter_constructor(self):
        """Gives function to compute the delay from geocenter for any sky
        location and GMST.
        """
        delta_d = -self.vertex
        def delay(ra, dec, gmst):
            """ Calculate time delay between two detectors in geocentric
            coordinates based on XLALArrivaTimeDiff in TimeDelay.c

            https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html
    
            Arguments
            ---------
            ra : float
                right ascension of the source in rad.
            dec : float
                declination of the source in rad.
            gmst : float
                Greenwich mean sidereal time in rad.
    
            Returns
            -------
            float: time delay from Earth center.
            """
            gmst = jnp.mod(gmst, 2 * jnp.pi)
            phi = ra - gmst
            theta = jnp.pi / 2 - dec
            omega = jnp.array([jnp.sin(theta)*jnp.cos(phi),
                               jnp.sin(theta)*jnp.sin(phi),
                               jnp.cos(theta)])
            return jnp.dot(omega, delta_d) / C_SI
        return delay

    def antenna_pattern_constructor(self, modes='pc'):
        """Gives function to compute antenna patterns for any sky location,
        polarization angle and GMST. The antenna pattern is defined 
        instantaneously under the long-wavelength approximation.
        
        Arguments
        ---------
        modes : list,str
            list of polarizations to include, defaults to tensor modes: 'pc'.
        aps : func
            function to compute antenna patterns for any sky location, 
            polarization angle and GMST.
        """
        detector_tensor = self.tensor
        wave_tensor_functions = [Polarization(m).tensor_from_sky_constructor
                                 for m in modes]
        def aps(ra, dec, psi, gmst):
            """Computes {name} antenna patterns for {modes} polarizations
            at the specified sky location, orientation and GMST.

            In the long-wavelength approximation, the antenna pattern for a
            given polarization is the dyadic product between the detector
            tensor and the corresponding polarization tensor.

            Arguments
            ---------
            ra : float
                source right ascension in radians.
            dec : float
                source declination in radians.
            psi : float
                source polarization angle in radians.
            gmst : float
                Greenwich mean sidereal time (GMST) in radians.
            
            Returns
            -------
            Fps : list
                antenna pattern values for {modes}.
            """
            antenna_patterns = []
            for pol_func in wave_tensor_functions:
                wave_tensor = pol_func(ra, dec, psi, gmst)
                ap = jnp.einsum('ij,ij->', detector_tensor, wave_tensor)
                antenna_patterns.append(ap)
            return antenna_patterns
        aps.__doc__ = aps.__doc__.format(name=self.name, modes=str(modes))
        return aps

    def construct_fd_response(self, modes='pc', epoch=0., earth_rotation=False,
                              earth_rotation_times=None, data_frequencies=None):
        """Generates a function to return the Fourier-domain projection of an
        arbitrary gravitational wave onto this detector, starting from FD 
        polarizations defined at geocenter.

        Arguments
        ---------
        modes : str
            polarizations to include in response, defaults to tensor modes 'pc'
        epoch : float
            time corresponding to beginning of segment, def. 0.
        earth_rotation : bool
            whether to account for Earth rotation in antenna patterns,
            def. False.

        Returns
        -------
        get_det_h : func
            function to produce the detector response for arbitrary input
            polarizations in the Fourier domain.
        """
        get_delay = self.delay_from_geocenter_constructor 
        get_aps = self.antenna_pattern_constructor(modes)
        if earth_rotation:
            if earth_rotation_times is None:
                if data_frequencies is None:
                    raise ValueError("Must provide data frequencies and epch,"
                                     "or explicit time grid to evaluate antenna "
                                     "patterns under Earth rotation.")
                else:
                    # TODO: move this to likelihood! construct_fd_response 
                    # should only accept a time grid.

                    # construct time grid on which to evaluate antenna patterns
                    # the time grid should as long as the data segment implied
                    # by the provided frequency array, i.e, T = 1/df.
                    seglen = 1/(data_frequencies[1] - data_frequencies[0])
                    # the grid spacing should be as coarse as possible while
                    # still resolving the evolution of the antenna patterns, 
                    # which has characterisitc frequency of up to 2/(sid_day).
                    dt_sid = np2(DAYSID_SI)/4
                    N = len(data_frequencies)
                    earth_rotation_times = jnp.arange(N)*dt_sid + epoch
                    w = tukey(N, 0.1) # TODO: do not hard code alpha!
            
        def get_det_h(f, polwaveforms, ra, dec, psi, gmst, tc):
            """Project Fourier-domain '{p}' polarizations onto {i} detector,
            taking into account antenna patterns and time of flight from
            geocenter.

            The response is defined by

            .. math:: h(f) = \\sum_p h_p(f) F_p(\\alpha, \\delta, \\psi) \\exp(2\\pi i \delta t)

            for polarization functions :math:`h_p(f)` delayed apropriately
            relative to geocenter by a time :math:`\\delta t(\\alpha,\\delta)`,
            and antenna patterns :math:`F_p(\\alpha, \\delta, \\psi)` for each
            included polarization :math:`p`.

            Arguments
            ---------
            f : array
                frequency array over which polarizations are evaluated.
            polwaveforms : list
                lenght-{n} list of arrays containing '{p}' polarizations, each
                assumed to be defined at geocenter and evaluated over the
                frequency grid `f`.
            ra : float
                source right ascension in radians.
            dec : float
                source declination in radians.
            psi : float
                source polarization angle in radians.
            gmst : float
                Greenwich mean sidereal time (GMST) in radians.
            tc : float
                time of arrival (coalescence) at geocenter in second measured
                from epoch {t}.

            Returns
            -------
            h : array
                Fourier domain detector response.
            """
            dt_geo = get_delay(ra, dec, gmst) 
            if earth_rotation:
                # antenna patterns are a function of time to be evaluated at
                # sparsely over a grid of times spanning the data segment
                # the result will be FFTed and convolved with the waveform
                aps = jnp.vectorize(get_aps)(ra, dec, psi, earth_rotation_times)
                delta_t = 0.5 / f[-1]
                aps_fd = jnp.fft.rfft(aps*w[:,jnp.newaxis], axis=0) * delta_t
                # TODO: ^do we want to zero pad?
                # now, we convolve the antenna patterns with the polarizations
                h = jnp.zeros_like(polwaveforms[0])
                for p in range(len(modes)):
                    # TODO: do we want jnp.fftconvolve?
                    h += jnp.convolve(polwaveforms[p], aps_fd[:,p], mode='same')
            else:
                aps = get_aps(ra, dec, psi, gmst)
                h = jnp.sum([aps[i]*polwaveforms[i] for i in len(aps)], axis=0)
            # note, under our sign convention for the Fourier transform the 
            # phase shift below corresponds to a time shift
            # ``t -> t - dt_geo - tc + epoch``
            # this makes sense: a waveform tha that peaks at t=0 at geocenter
            # will peak at t=dt_geo at the detector, so dt is indeed a delay.
            h *= jnp.exp(-2j*jnp.pi*f*(dt_geo + tc - epoch))
            return h
        get_det_h.__doc__ = get_det_h.__doc__.format(p=str(modes), i=self.name,
                                                     n=len(modes), t=epoch)
        return get_det_h
        
