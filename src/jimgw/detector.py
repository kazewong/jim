import jax.numpy as jnp
from jimgw.constants import *
from jimgw.wave import Polarization
from scipy.signal.windows import tukey
from abc import ABC, abstractmethod
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
import jax
from gwpy.timeseries import TimeSeries
from typing import Callable
import requests
import numpy as np
from scipy.interpolate import interp1d

DEG_TO_RAD = jnp.pi/180

# TODO: Need to expand this list. Currently it is only O3.
psd_file_dict= {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}

def np2(x):
    """
    Returns the next power of two as big as or larger than x."""
    p = 1
    while p < x:
        p = p << 1
    return p

class Detector(ABC):
    """ 
    Base class for all detectors.

    """

    @abstractmethod
    def load_data(self, data):
        raise NotImplementedError

    @abstractmethod
    def fd_response(self, frequency: Array, h: Array, params: dict) -> Array:
        """
        Modulate the waveform in the sky frame by the detector response in the frequency domain."""
        pass

    @abstractmethod
    def td_response(self, time: Array, h: Array, params: dict) -> Array:
        """
        Modulate the waveform in the sky frame by the detector response in the time domain."""
        pass
    
class GroundBased2G(Detector):

    name: str
    polarization_mode: list[Polarization]
    frequencies: Array = None
    data : Array = None
    psd: Array = None

    latitude: float = 0
    longitude: float = 0
    xarm_azimuth: float = 0
    yarm_azimuth: float = 0
    xarm_tilt: float = 0
    yarm_tilt: float = 0
    elevation: float = 0

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name

        self.latitude = kwargs.get('latitude', 0)
        self.longitude = kwargs.get('longitude', 0)
        self.elevation = kwargs.get('elevation', 0)
        self.xarm_azimuth = kwargs.get('xarm_azimuth', 0)
        self.yarm_azimuth = kwargs.get('yarm_azimuth', 0)
        self.xarm_tilt = kwargs.get('xarm_tilt', 0)
        self.yarm_tilt = kwargs.get('yarm_tilt', 0)
        modes = kwargs.get('mode', 'pc')

        self.polarization_mode = [Polarization(m) for m in modes]

    @staticmethod
    def _get_arm(lat, lon, tilt, azimuth):
        """
        Construct detector-arm vectors in Earth-centric Cartesian coordinates.

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
        """
        Detector arm vectors (x, y).
        """
        x = self._get_arm(self.latitude, self.longitude, self.xarm_tilt, self.xarm_azimuth)
        y = self._get_arm(self.latitude, self.longitude, self.yarm_tilt, self.yarm_azimuth)
        return x, y
	
    @property
    def tensor(self):
        """
        Detector tensor defining the strain measurement.
        """
        #TODO: this could easily be generalized for other detector geometries
        arm1, arm2 = self.arms
        return  0.5 * (jnp.einsum('i,j->ij', arm1, arm1) - 
                       jnp.einsum('i,j->ij', arm2, arm2))

    @property
    def vertex(self):
        """
        Detector vertex coordinates in the reference celestial frame. Based
        on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.
        """
        # get detector and Earth parameters
        lat = self.latitude
        lon = self.longitude
        h = self.elevation
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2*(major**2*jnp.cos(lat)**2 + minor**2*jnp.sin(lat)**2)**(-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major)**2 * r + h)*jnp.sin(lat)
        return jnp.array([x, y, z])

    def load_data(self, trigger_time:float,
                gps_start_pad: int,
                gps_end_pad: int,
                f_min: float,
                f_max: float,
                psd_pad: int = 16,
                tukey_alpha: float = 0.2) -> None:
        """
        Load data from the detector.

        Parameters
        ----------
        trigger_time : float
            The GPS time of the trigger.
        gps_start_pad : int
            The amount of time before the trigger to fetch data.
        gps_end_pad : int
            The amount of time after the trigger to fetch data.
        f_min : float
            The minimum frequency to fetch data.
        f_max : float
            The maximum frequency to fetch data.
        psd_pad : int
            The amount of time to pad the PSD data.
        tukey_alpha : float
            The alpha parameter for the Tukey window.

        """

        print("Fetching data from {}...".format(self.name))
        data_td = TimeSeries.fetch_open_data(self.name, trigger_time - gps_start_pad, trigger_time + gps_end_pad, cache=True)
        segment_length = data_td.duration.value
        n = len(data_td)
        delta_t = data_td.dt.value
        data = jnp.fft.rfft(jnp.array(data_td.value)*tukey(n, tukey_alpha))*delta_t
        freq = jnp.fft.rfftfreq(n, delta_t)
        # TODO: Check if this is the right way to fetch PSD
        start_psd = int(trigger_time) - gps_start_pad - psd_pad # What does Int do here?
        end_psd = int(trigger_time) + gps_end_pad + psd_pad

        print("Fetching PSD data...")
        psd_data_td = TimeSeries.fetch_open_data(self.name, start_psd, end_psd, cache=True)
        psd = psd_data_td.psd(fftlength=segment_length).value # TODO: Check whether this is sright.

        print("Finished generating data.")

        self.frequencies = freq[(freq>f_min)&(freq<f_max)]
        self.data = data[(freq>f_min)&(freq<f_max)]
        self.psd = psd[(freq>f_min)&(freq<f_max)]

    def fd_response(self, frequency: Array, h_sky: dict, params: dict) -> Array:
        """
        Modulate the waveform in the sky frame by the detector response in the frequency domain."""
        ra, dec, psi, gmst = params['ra'], params['dec'], params['psi'], params['gmst']
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        timeshift = self.delay_from_geocenter(ra, dec, gmst)
        h_detector = jax.tree_util.tree_map(lambda h, antenna: h * antenna * jnp.exp(-2j * jnp.pi * frequency * timeshift), h_sky, antenna_pattern)
        return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(h_detector)),axis=0)

    def td_response(self, time: Array, h: Array, params: Array) -> Array:
        """
        Modulate the waveform in the sky frame by the detector response in the time domain."""
        pass



    def delay_from_geocenter(self, ra: float, dec: float, gmst: float) -> float:
        """ 
        Calculate time delay between two detectors in geocentric
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
        delta_d = -self.vertex
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array([jnp.sin(theta)*jnp.cos(phi),
                            jnp.sin(theta)*jnp.sin(phi),
                            jnp.cos(theta)])
        return jnp.dot(omega, delta_d) / C_SI

    def antenna_pattern(self, ra:float, dec:float, psi:float, gmst:float) -> dict:
        """
        Computes {name} antenna patterns for {modes} polarizations
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
        modes : str
            string of polarizations to include, defaults to tensor modes: 'pc'.
        
        Returns
        -------
        result : list
            antenna pattern values for {modes}.
        """  
        detector_tensor = self.tensor

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
            antenna_patterns[polarization.name] = jnp.einsum('ij,ij->', detector_tensor, wave_tensor)

        return antenna_patterns

    def inject_signal(self,
                      key: PRNGKeyArray,
                      freqs: Array,
                      h_sky: dict,
                      params: dict,
                      psd_file: str = None) -> None:
        """
        """
        self.frequencies = freqs
        self.psd = self.load_psd(freqs, psd_file)
        key, subkey = jax.random.split(key, 2)
        vars = self.psd / (freqs[1] - freqs[0])
        noise_real = jax.random.normal(subkey, shape=freqs.shape)*jnp.sqrt(vars)
        noise_imag = jax.random.normal(subkey, shape=freqs.shape)*jnp.sqrt(vars)
        signal = self.fd_response(freqs, h_sky, params)
        self.data = signal + noise_real + 1j*noise_imag

    def load_psd(self, freqs: Array, psd_file: str = None) -> None:
        if psd_file is None:
            print("Grabbing GWTC-2 PSD for "+self.name)
            url = psd_file_dict[self.name]
            data = requests.get(url)
            open(self.name+".txt", "wb").write(data.content)
            f, asd_vals = np.loadtxt(self.name+".txt", unpack=True)
        else:
            f, asd_vals = np.loadtxt(psd_file, unpack=True)
        psd_vals = asd_vals**2
        psd = interp1d(f, psd_vals, fill_value=(psd_vals[0], psd_vals[-1]))(freqs)
        return psd

H1 = GroundBased2G('H1',
latitude = (46 + 27. / 60 + 18.528 / 3600) * DEG_TO_RAD,
longitude = -(119 + 24. / 60 + 27.5657 / 3600) * DEG_TO_RAD,
xarm_azimuth = 125.9994 * DEG_TO_RAD,
yarm_azimuth = 215.9994 * DEG_TO_RAD,
xarm_tilt = -6.195e-4,
yarm_tilt = 1.25e-5,
elevation = 142.554,
mode='pc')

L1 = GroundBased2G('L1',
latitude = (30 + 33. / 60 + 46.4196 / 3600) * DEG_TO_RAD,
longitude = -(90 + 46. / 60 + 27.2654 / 3600) * DEG_TO_RAD,
xarm_azimuth = 197.7165 * DEG_TO_RAD,
yarm_azimuth = 287.7165 * DEG_TO_RAD,
xarm_tilt = 0 ,
yarm_tilt = 0,
elevation = -6.574,
mode='pc')

V1 = GroundBased2G('V1',
latitude = (43 + 37. / 60 + 53.0921 / 3600) * DEG_TO_RAD,
longitude = (10 + 30. / 60 + 16.1887 / 3600) * DEG_TO_RAD,
xarm_azimuth = 243. * DEG_TO_RAD,
yarm_azimuth = 333. * DEG_TO_RAD,
xarm_tilt = 0 ,
yarm_tilt = 0,
elevation = 51.884,
mode='pc')