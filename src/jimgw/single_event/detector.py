from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex, PRNGKeyArray, jaxtyped, Bool
from numpy import loadtxt
import requests
from beartype import beartype as typechecker
from typing import Optional

from jimgw.constants import (
    C_SI,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
    DEG_TO_RAD,
)
from jimgw.single_event.wave import Polarization
from jimgw.single_event.data import Data, PowerSpectrum
from jimgw.single_event.utils import inner_product, complex_inner_product

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}


class Detector(ABC):
    """Base class for all detectors.

    Attributes:
        name (str): Name of the detector.
        data (Data): Detector data object.
        psd (PowerSpectrum): Power spectral density object.
        frequency_bounds (tuple[float, float]): Lower and upper frequency bounds.
    """

    name: str

    # NOTE: for some detectors (e.g. LISA, ET) data could be a list of Data
    # objects so this might be worth revisiting
    data: Data
    psd: PowerSpectrum

    frequency_bounds: tuple[float, float] = (0.0, float("inf"))

    _sliced_frequencies: Float[Array, " n_sample"] = jnp.array([])
    _sliced_fd_data: Float[Array, " n_sample"] = jnp.array([])
    _sliced_psd: Float[Array, " n_sample"] = jnp.array([])

    @property
    def epoch(self) -> Float:
        """The epoch of the data."""
        return self.data.epoch

    @property
    def times(self) -> Float[Array, " n_sample"]:
        return self.data.times

    @property
    def frequencies(self) -> Float[Array, " n_sample"]:
        return self.data.frequencies

    @property
    def duration(self) -> Float:
        return self.data.duration

    @property
    def frequency_mask(self) -> Bool[Array, " n_sample"]:
        f_min, f_max = self.frequency_bounds
        return (f_min <= self.frequencies) & (self.frequencies <= f_max)

    @abstractmethod
    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        trigger_time: Float = 0.0,
        **kwargs,
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, " n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, " n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. The keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict): Dictionary of source parameters including:
                - ra (float): Right ascension in radians
                - dec (float): Declination in radians
                - psi (float): Polarization angle in radians
                - gmst (float): Greenwich mean sidereal time in radians
                - t_c (Float): Difference between geocent time and trigger time in sec
            trigger_time (Float): Trigger time of the data in seconds.
            **kwargs: Additional keyword arguments.

        Returns:
            Complex[Array, " n_sample"]: Complex strain measured by the detector in frequency domain.
        """
        pass

    @abstractmethod
    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        **kwargs,
    ) -> Float[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            Array of detector response in time domain.
        """
        pass

    def set_frequency_bounds(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ) -> None:
        """Set the frequency bounds for the detector.
        This also set the sliced frequencies, data and psd.

        Args:
            f_min: Minimum frequency.
            f_max: Maximum frequency.
        """
        bounds = list(self.frequency_bounds)
        if f_min is not None:
            bounds[0] = f_min
        if f_max is not None:
            bounds[1] = f_max
        self.frequency_bounds = tuple(bounds)  # type: ignore

        # Compute sliced frequencies, data and psd.
        data, freqs_1 = self.data.frequency_slice(*self.frequency_bounds)
        psd, freqs_2 = self.psd.frequency_slice(*self.frequency_bounds)

        assert all(
            freqs_1 == freqs_2
        ), f"The {self.name} data and PSD must have same frequencies"

        self._sliced_frequencies = freqs_1
        self._sliced_fd_data = data
        self._sliced_psd = psd

    def clear_data_and_psd(self) -> None:
        """Clear the data and PSD of the detector."""
        self.data = Data()
        self.psd = PowerSpectrum()
        self.frequency_bounds = (0.0, float("inf"))
        for attrname in [
            "sliced_frequencies",
            "sliced_fd_data",
            "sliced_psd",
        ]:
            if hasattr(self, attrname):
                delattr(self, attrname)

    @property
    def sliced_frequencies(self) -> Float[Array, " n_freq"]:
        """Get frequency-domain data slice based on frequency bounds.

        Returns:
            Float[Array, " n_sample"]: Sliced frequency-domain data.
        """
        return self._sliced_frequencies

    @property
    def sliced_fd_data(self) -> Complex[Array, " n_freq"]:
        """Get frequency-domain data slice based on frequency bounds.

        Returns:
            Complex[Array, " n_sample"]: Sliced frequency-domain data.
        """
        return self._sliced_fd_data

    @property
    def sliced_psd(self) -> Float[Array, " n_freq"]:
        """Get PSD slice based on frequency bounds.

        Returns:
            Float[Array, " n_sample"]: Sliced power spectral density.
        """
        return self._sliced_psd


class GroundBased2G(Detector):
    """Object representing a ground-based detector.

    Contains information about the location and orientation of the detector on Earth,
    as well as actual strain data and the PSD of the associated noise.

    Attributes:
        name (str): Name of the detector.
        latitude (Float): Latitude of the detector in radians.
        longitude (Float): Longitude of the detector in radians.
        xarm_azimuth (Float): Azimuth of the x-arm in radians.
        yarm_azimuth (Float): Azimuth of the y-arm in radians.
        xarm_tilt (Float): Tilt of the x-arm in radians.
        yarm_tilt (Float): Tilt of the y-arm in radians.
        elevation (Float): Elevation of the detector in meters.
        polarization_mode (list[Polarization]): List of polarization modes (`pc` for plus and cross) to be used in
            computing antenna patterns; in the future, this could be expanded to
            include non-GR modes.
        data (Data): Array of Fourier-domain strain data.
        psd (PowerSpectrum): Power spectral density object.
    """

    polarization_mode: list[Polarization]
    data: Data
    psd: PowerSpectrum

    latitude: Float = 0
    longitude: Float = 0
    xarm_azimuth: Float = 0
    yarm_azimuth: Float = 0
    xarm_tilt: Float = 0
    yarm_tilt: Float = 0
    elevation: Float = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(
        self,
        name: str,
        latitude: float = 0,
        longitude: float = 0,
        elevation: float = 0,
        xarm_azimuth: float = 0,
        yarm_azimuth: float = 0,
        xarm_tilt: float = 0,
        yarm_tilt: float = 0,
        modes: str = "pc",
    ):
        """Initialize a ground-based detector.

        Args:
            name (str): Name of the detector.
            latitude (float, optional): Latitude of the detector in radians. Defaults to 0.
            longitude (float, optional): Longitude of the detector in radians. Defaults to 0.
            elevation (float, optional): Elevation of the detector in meters. Defaults to 0.
            xarm_azimuth (float, optional): Azimuth of the x-arm in radians. Defaults to 0.
            yarm_azimuth (float, optional): Azimuth of the y-arm in radians. Defaults to 0.
            xarm_tilt (float, optional): Tilt of the x-arm in radians. Defaults to 0.
            yarm_tilt (float, optional): Tilt of the y-arm in radians. Defaults to 0.
            modes (str, optional): Polarization modes. Defaults to "pc".
        """
        self.name = name

        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt

        self.polarization_mode = [Polarization(m) for m in modes]
        self.data = Data()
        self.psd = PowerSpectrum()

    @staticmethod
    def _get_arm(
        lat: Float, lon: Float, tilt: Float, azimuth: Float
    ) -> Float[Array, " 3"]:
        """Construct detector-arm vectors in geocentric Cartesian coordinates.

        Args:
            lat (Float): Vertex latitude in radians.
            lon (Float): Vertex longitude in radians.
            tilt (Float): Arm tilt in radians.
            azimuth (Float): Arm azimuth in radians.

        Returns:
            Float[Array, " 3"]: Detector arm vector in geocentric Cartesian coordinates.
        """
        e_lon = jnp.array([-jnp.sin(lon), jnp.cos(lon), 0])
        e_lat = jnp.array(
            [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)]
        )
        e_h = jnp.array(
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)]
        )

        return (
            jnp.cos(tilt) * jnp.cos(azimuth) * e_lon
            + jnp.cos(tilt) * jnp.sin(azimuth) * e_lat
            + jnp.sin(tilt) * e_h
        )

    @property
    def arms(self) -> tuple[Float[Array, " 3"], Float[Array, " 3"]]:
        """Get the detector arm vectors.

        Returns:
            tuple[Float[Array, " 3"], Float[Array, " 3"]]: A tuple containing:
                - x: X-arm vector in geocentric Cartesian coordinates
                - y: Y-arm vector in geocentric Cartesian coordinates
        """
        x = self._get_arm(
            self.latitude, self.longitude, self.xarm_tilt, self.xarm_azimuth
        )
        y = self._get_arm(
            self.latitude, self.longitude, self.yarm_tilt, self.yarm_azimuth
        )
        return x, y

    @property
    def tensor(self) -> Float[Array, " 3 3"]:
        """Get the detector tensor defining the strain measurement.

        For a 2-arm differential-length detector, this is given by:

        .. math::

            D_{ij} = \\left(x_i x_j - y_i y_j\\right)/2

        for unit vectors :math:`x` and :math:`y` along the x and y arms.

        Returns:
            Float[Array, " 3 3"]: The 3x3 detector tensor in geocentric coordinates.
        """
        # TODO: this could easily be generalized for other detector geometries
        arm1, arm2 = self.arms
        return 0.5 * (
            jnp.einsum("i,j->ij", arm1, arm1) - jnp.einsum("i,j->ij", arm2, arm2)
        )

    @property
    def vertex(self) -> Float[Array, " 3"]:
        """Detector vertex coordinates in the reference celestial frame.

        Based on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.

        Returns:
            Float[Array, " 3"]: Detector vertex coordinates in geocentric Cartesian coordinates.
        """
        # get detector and Earth parameters
        lat = self.latitude
        lon = self.longitude
        h = self.elevation
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * jnp.cos(lat) ** 2 + minor**2 * jnp.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major) ** 2 * r + h) * jnp.sin(lat)
        return jnp.array([x, y, z])

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
        trigger_time: Float = 0.0,
        **kwargs,
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, " n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, " n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. Keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict[str, Float]): Dictionary of source parameters containing:
                - ra (Float): Right ascension in radians
                - dec (Float): Declination in radians
                - psi (Float): Polarization angle in radians
                - gmst (Float): Greenwich mean sidereal time in radians
                - t_c (Float): Difference between geocent time and trigger time in sec
            trigger_time (Float): Trigger time of the data in seconds.
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Complex strain measured by the detector in frequency domain, obtained by
                  combining the antenna patterns for each polarization mode.
        """
        ra, dec, psi, gmst = params["ra"], params["dec"], params["psi"], params["gmst"]
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        timeshift = self.delay_from_geocenter(ra, dec, gmst)
        h_detector = jax.tree_util.tree_map(
            lambda h, antenna: h
            * antenna
            * jnp.exp(-2j * jnp.pi * frequency * timeshift),
            h_sky,
            antenna_pattern,
        )
        projected_strain = jnp.sum(
            jnp.stack(jax.tree_util.tree_leaves(h_detector)), axis=0
        )
        trigger_time_shift = trigger_time - self.epoch + params["t_c"]
        phase_shift = jnp.exp(-2j * jnp.pi * frequency * trigger_time_shift)
        return projected_strain * phase_shift

    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        **kwargs,
    ) -> Array:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            Array of detector response in time domain.
        """
        raise NotImplementedError

    def delay_from_geocenter(self, ra: Float, dec: Float, gmst: Float) -> Float:
        """Calculate time delay between two detectors in geocentric coordinates.

        Based on XLALArrivaTimeDiff in TimeDelay.c
        https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html

        Args:
            ra (Float): Right ascension of the source in radians.
            dec (Float): Declination of the source in radians.
            gmst (Float): Greenwich mean sidereal time in radians.

        Returns:
            Float: Time delay from Earth center in seconds.
        """
        delta_d = -self.vertex
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return jnp.einsum("i...,i->...", omega, delta_d) / C_SI

    def antenna_pattern(
        self, ra: Float, dec: Float, psi: Float, gmst: Float
    ) -> dict[str, Complex]:
        """Compute antenna patterns for polarizations at specified sky location.

        In the long-wavelength approximation, the antenna pattern for a
        given polarization is the dyadic product between the detector
        tensor and the corresponding polarization tensor.

        Args:
            ra (Float): Source right ascension in radians.
            dec (Float): Source declination in radians.
            psi (Float): Source polarization angle in radians.
            gmst (Float): Greenwich mean sidereal time (GMST) in radians.

        Returns:
            dict[str, Complex]: Dictionary mapping polarization names to their antenna pattern values.
        """
        detector_tensor = self.tensor

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
            antenna_patterns[polarization.name] = jnp.einsum(
                "ij,ij...->...", detector_tensor, wave_tensor
            )

        return antenna_patterns

    @jaxtyped(typechecker=typechecker)
    def load_and_set_psd(self, psd_file: str = "", asd_file: str = "") -> PowerSpectrum:
        """Load power spectral density (PSD) from file or default GWTC-2 catalog,
            and set it to the detector.

        Args:
            psd_file (str, optional): Path to file containing PSD data. If empty, uses GWTC-2 PSD.

        Returns:
            Float[Array, " n_sample"]: Array of PSD values of the detector.
        """
        if psd_file != "":
            f, psd_vals = loadtxt(psd_file, unpack=True)
        elif asd_file != "":
            f, asd_vals = loadtxt(asd_file, unpack=True)
            psd_vals = asd_vals**2
        else:
            print("Grabbing GWTC-2 PSD for " + self.name)
            url = asd_file_dict[self.name]
            data = requests.get(url)
            tmp_file_name = f"fetched_default_asd_{self.name}.txt"
            open(tmp_file_name, "wb").write(data.content)
            f, asd_vals = loadtxt(tmp_file_name, unpack=True)
            psd_vals = asd_vals**2

        _loaded_psd = PowerSpectrum(psd_vals, f, name=f"{self.name}_psd")
        self.set_psd(_loaded_psd)
        return self.psd

    def _equal_data_psd_frequencies(self) -> Bool:
        """Check if the frequencies of the data and PSD match.
        A helper function for `set_data` and `set_psd`.

        Return:
            Bool: True if the frequencies match, False otherwise.
        """
        if self.psd.empty or self.data.empty:
            # In this case, we simply skip the check
            return True
        if self.psd.n_freq != self.data.n_freq:
            # Cannot proceed comparison, needs interpolation
            return False
        if (self.psd.frequencies == self.data.frequencies).all():
            # Frequencies match
            return True
        # This case means the frequencies are different
        return False

    def set_data(self, data: Data | Array, **kws) -> None:
        """Add data to the detector.

        Args:
            data (Union[Data, Array]): Data to be added to the detector, either as a `Data` object
                or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `Data` constructor.

        Returns:
            None
        """
        if isinstance(data, Data):
            self.data = data
        else:
            self.data = Data(data, **kws)
        # Assert PSD frequencies agree with data
        if not ((self.psd is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def set_psd(self, psd: PowerSpectrum | Array, **kws) -> None:
        """Add PSD to the detector.

        Args:
            psd (Union[PowerSpectrum, Array]): PSD to be added to the detector, either as a `PowerSpectrum`
                object or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `PowerSpectrum` constructor.

        Returns:
            None
        """
        if isinstance(psd, PowerSpectrum):
            self.psd = psd
        else:
            # not clear if we want to support this
            self.psd = PowerSpectrum(psd, **kws)
        # Assert PSD frequencies agree with data frequencies
        if not ((self.data is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def inject_signal(
        self,
        duration: float,
        sampling_frequency: float,
        epoch: float,
        waveform_model,
        parameters: dict[str, float],
        is_zero_noise: bool = False,
        rng_key: PRNGKeyArray = jax.random.PRNGKey(0),
    ) -> None:
        """Inject a signal into the detector data.

        Note: The power spectral density must be set beforehand.

        Args:
            waveform_model: The waveform model to be injected.
            parameters (dict): Dictionary of parameters for the waveform model.

        Returns:
            None
        """
        # 1. Set empty data to initialise the detector
        n_times = int(duration * sampling_frequency)
        self.set_data(
            Data(
                name=f"{self.name}_empty",
                td=jnp.zeros(n_times),
                delta_t=1 / sampling_frequency,
                epoch=epoch,
            )
        )

        # 2. Compute the projected strain from parameters
        polarisations = waveform_model(self.frequencies, parameters)
        projected_strain = self.fd_response(self.frequencies, polarisations, parameters)

        # 3. Set the new data
        strain_data = jnp.where(self.frequency_mask, projected_strain, 0.0 + 0.0j)
        if not is_zero_noise:
            strain_data += jnp.where(
                self.frequency_mask, self.psd.simulate_data(rng_key), 0.0 + 0.0j
            )

        self.set_data(
            Data.from_fd(
                name=f"{self.name}_injected",
                fd=strain_data,
                frequencies=self.frequencies,
                epoch=self.data.epoch,
            )
        )

        # 4. Update the sliced data and psd with the (potentially) new frequency bounds
        self.set_frequency_bounds()
        masked_signal = projected_strain[self.frequency_mask]

        _optimal_snr_sq = inner_product(
            masked_signal, masked_signal, self.sliced_psd, self.sliced_frequencies
        )
        optimal_snr = _optimal_snr_sq**0.5
        match_filtered_snr = complex_inner_product(
            masked_signal,
            self.sliced_fd_data,
            self.sliced_psd,
            self.sliced_frequencies,
        )
        match_filtered_snr /= optimal_snr

        # NOTE: Change this to logging later.
        print(f"For detector {self.name}, the injected signal has:")
        print(f"  - Optimal SNR: {optimal_snr:.4f}")
        print(f"  - Match filtered SNR: {match_filtered_snr:.4f}")

    def set_data_from_file(self, filename: str) -> None:
        """Set data from a file.

        Args:
            filename (str): Path to the file containing the data.

        Returns:
            None
        """
        pass

    def get_whitened_frequency_domain_strain(
        self, frequency_series: Complex[Array, " n_freq"]
    ) -> Complex[Array, " n_freq"]:
        """Get the whitened frequency-domain strain.
        Args:
            frequency_series (Complex[Array, " n_freq"]): Array of frequency domain data/signal.
        Returns:
            Complex[Array, " n_freq"]: Whitened frequency-domain strain.
        """
        scaled_asd = jnp.sqrt(self.psd.values * self.duration / 4)
        return (frequency_series / scaled_asd) * self.frequency_mask

    def whitened_frequency_to_time_domain_strain(
        self, whitened_frequency_series: Complex[Array, " n_time // 2 + 1"]
    ) -> Float[Array, " n_time"]:
        """Get the whitened frequency-domain strain.
        Args:
            whitened_frequency_series (Complex[Array, " n_time // 2 + 1"]):
                Array of whitened frequency domain data/signal.
        Returns:
            Float[Array, " n_time"]: Whitened time-domain strain/signal.
        """
        freq_mask_ratio = len(self.frequency_mask) / jnp.sqrt(
            jnp.sum(self.frequency_mask)
        )
        return jnp.fft.irfft(whitened_frequency_series) * freq_mask_ratio

    @property
    def whitened_frequency_domain_data(self) -> Complex[Array, " n_sample"]:
        """Get the whitened frequency-domain data.

        Args:
            frequency (Float[Array, " n_sample"]): Array of frequency samples.

        Returns:
            Float[Array, " n_sample"]: Whitened frequency-domain data.
        """

        return self.get_whitened_frequency_domain_strain(self.data.fd)

    @property
    def whitened_time_domain_data(self) -> Float[Array, " n_sample"]:
        """Get the whitened time-domain data.

        Args:
            time (Float[Array, " n_sample"]): Array of time samples.

        Returns:
            Float[Array, " n_sample"]: Whitened time-domain data.
        """
        return self.whitened_frequency_to_time_domain_strain(
            self.whitened_frequency_domain_data
        )


H1 = GroundBased2G(
    "H1",
    latitude=(46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD,
    longitude=-(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD,
    xarm_azimuth=125.9994 * DEG_TO_RAD,
    yarm_azimuth=215.9994 * DEG_TO_RAD,
    xarm_tilt=-6.195e-4,
    yarm_tilt=1.25e-5,
    elevation=142.554,
    modes="pc",
)

L1 = GroundBased2G(
    "L1",
    latitude=(30 + 33.0 / 60 + 46.4196 / 3600) * DEG_TO_RAD,
    longitude=-(90 + 46.0 / 60 + 27.2654 / 3600) * DEG_TO_RAD,
    xarm_azimuth=197.7165 * DEG_TO_RAD,
    yarm_azimuth=287.7165 * DEG_TO_RAD,
    xarm_tilt=-3.121e-4,
    yarm_tilt=-6.107e-4,
    elevation=-6.574,
    modes="pc",
)

V1 = GroundBased2G(
    "V1",
    latitude=(43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD,
    longitude=(10 + 30.0 / 60 + 16.1878 / 3600) * DEG_TO_RAD,
    xarm_azimuth=70.5674 * DEG_TO_RAD,
    yarm_azimuth=160.5674 * DEG_TO_RAD,
    xarm_tilt=0,
    yarm_tilt=0,
    elevation=51.884,
    modes="pc",
)

detector_preset = {
    "H1": H1,
    "L1": L1,
    "V1": V1,
}
