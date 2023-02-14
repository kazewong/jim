import jax.numpy as jnp
from .constants import *
from .wave import Polarization


DEG_TO_RAD = jnp.pi/180

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
    def coordinates(self)
        """Coordinates defining a triangular detector (angles in radians).
        """
        if not self._coordinates
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

    @static
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
        x = (radius + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (radius + h) * jnp.cos(lat) * jnp.sin(lon)
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

    def antenna_pattern_constructor(modes='pc'):
        """Gives function to compute antenna patterns for any sky location,
        polarization angle and GMST.
        
        Arguments
        ---------
        modes : list,str
            list of polarizations to include, defaults to tensor modes: 'pc'.
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
        return antenna_patterns
        


def make_detector_response(detector_tensor, detector_vertex):
    antenna_response_plus = make_antenna_response(detector_tensor,'plus')
    antenna_response_cross = make_antenna_response(detector_tensor, 'cross')
    def detector_response(f, hp, hc, ra, dec, gmst, psi):
        output = antenna_response_plus(ra, dec, gmst, psi)*hp + antenna_response_cross(ra, dec, gmst, psi)*hc
        timeshift = time_delay_geocentric(detector_vertex, jnp.array([0.,0.,0.]), ra, dec, gmst)
        output = output * jnp.exp(-1j * 2 * jnp.pi * f * timeshift)
        return output
    return detector_response
        

