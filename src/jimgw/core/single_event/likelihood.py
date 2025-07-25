import jax
import jax.numpy as jnp
from flowMC.strategy.optimization import AdamOptimization
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from typing import Optional
from scipy.interpolate import interp1d
from jimgw.core.utils import log_i0
from jimgw.core.prior import Prior
from jimgw.core.base import LikelihoodBase
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.waveform import Waveform
from jimgw.core.single_event.utils import inner_product, complex_inner_product
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)
import logging
from typing import Sequence
from abc import abstractmethod


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform
    fixed_parameters: dict[str, Float] = {}

    @property
    def duration(self) -> Float:
        return self.detectors[0].data.duration

    @property
    def detector_names(self):
        """The interferometers for the likelihood."""
        return [detector.name for detector in self.detectors]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
    ) -> None:
        self.detectors = detectors
        self.waveform = waveform
        self.fixed_parameters = fixed_parameters if fixed_parameters is not None else {}

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the likelihood for a given set of parameters.

        This is a template method that calls the core likelihood evaluation method
        """
        params.update(self.fixed_parameters)
        return self._likelihood(params, data)

    @abstractmethod
    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Core likelihood evaluation method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class ZeroLikelihood(LikelihoodBase):
    def __init__(self):
        pass

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the likelihood, which is always zero."""
        return 0.0


class BaseTransientLikelihoodFD(SingleEventLikelihood):
    """Base class for frequency-domain transient gravitational wave likelihood.

    This class provides the basic likelihood evaluation for gravitational wave transient events
    in the frequency domain, using matched filtering across multiple detectors.

    Attributes:
        frequencies (Float[Array]): The frequency array used for likelihood evaluation.
        trigger_time (Float): The GPS time of the event trigger.
        gmst (Float): Greenwich Mean Sidereal Time computed from the trigger time.

    Args:
        detectors (Sequence[Detector]): List of detector objects containing data and metadata.
        waveform (Waveform): Waveform model to evaluate.
        f_min (Float, optional): Minimum frequency for likelihood evaluation. Defaults to 0.
        f_max (Float, optional): Maximum frequency for likelihood evaluation. Defaults to infinity.
        trigger_time (Float, optional): GPS time of the event trigger. Defaults to 0.

    Example:
        >>> likelihood = BaseTransientLikelihoodFD(detectors, waveform, f_min=20, f_max=1024, trigger_time=1234567890)
        >>> logL = likelihood.evaluate(params, data)
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: Float = 0,
    ) -> None:
        """Initializes the BaseTransientLikelihoodFD class.

        Sets up the frequency bounds for the detectors and computes the Greenwich Mean Sidereal Time.

        Args:
            detectors (Sequence[Detector]): List of detector objects.
            waveform (Waveform): Waveform model.
            f_min (Float, optional): Minimum frequency. Defaults to 0.
            f_max (Float, optional): Maximum frequency. Defaults to infinity.
            trigger_time (Float, optional): Event trigger time. Defaults to 0.
        """
        super().__init__(detectors, waveform, fixed_parameters)
        # Set the frequency bounds for the detectors
        _frequencies = []
        for detector in detectors:
            detector.set_frequency_bounds(f_min, f_max)
            _frequencies.append(detector.sliced_frequencies)
        _frequencies = jnp.array(_frequencies)
        assert jnp.all(
            jnp.array(_frequencies)[:-1] == jnp.array(_frequencies)[1:]
        ), "The frequency arrays are not all the same."
        self.frequencies = _frequencies[0]
        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the log-likelihood for a given set of parameters.

        Computes the log-likelihood by matched filtering the model waveform against the data
        for each detector, using the frequency-domain inner product.

        Args:
            params (dict[str, Float]): Dictionary of model parameters.
            data (dict): Dictionary containing data (not used in this implementation).

        Returns:
            Float: The log-likelihood value.
        """
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Core likelihood evaluation method for frequency-domain transient events."""
        waveform_sky = self.waveform(self.frequencies, params)
        log_likelihood = 0.0
        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            match_filter_SNR = inner_product(h_dec, ifo_data, psd, df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += match_filter_SNR - optimal_SNR / 2
        return log_likelihood


class TimeMarginalizedLikelihoodFD(BaseTransientLikelihoodFD):
    """Frequency-domain likelihood class with analytic marginalization over coalescence time.

    This class implements a likelihood function for gravitational wave transient events,
    marginalized over the coalescence time parameter (`t_c`). The marginalization is performed
    using a fast Fourier transform (FFT) over the frequency domain inner product between the
    model and the data. The likelihood is computed for a set of detectors and a waveform model.

    Attributes:
        tc_range (tuple[Float, Float]): The range of coalescence times to marginalize over.
        tc_array (Float[Array, "duration*f_sample/2"]): Array of time shifts corresponding to FFT bins.
        pad_low (Float[Array, "n_pad_low"]): Zero-padding array for frequencies below the minimum frequency.
        pad_high (Float[Array, "n_pad_high"]): Zero-padding array for frequencies above the maximum frequency.

    Args:
        detectors (Sequence[Detector]): List of detector objects containing data and metadata.
        waveform (Waveform): Waveform model to evaluate.
        f_min (Float, optional): Minimum frequency for likelihood evaluation. Defaults to 0.
        f_max (Float, optional): Maximum frequency for likelihood evaluation. Defaults to infinity.
        trigger_time (Float, optional): GPS time of the event trigger. Defaults to 0.
        tc_range (tuple[Float, Float], optional): Range of coalescence times to marginalize over. Defaults to (-0.12, 0.12).

    Example:
        >>> likelihood = TimeMarginalizedLikelihoodFD(detectors, waveform, f_min=20, f_max=1024, trigger_time=1234567890)
        >>> logL = likelihood.evaluate(params, data)
    """

    tc_range: tuple[Float, Float]
    tc_array: Float[Array, " duration*f_sample/2"]
    pad_low: Float[Array, " n_pad_low"]
    pad_high: Float[Array, " n_pad_high"]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: Float = 0,
        tc_range: tuple[Float, Float] = (-0.12, 0.12),
    ) -> None:
        """Initializes the TimeMarginalizedLikelihoodFD class.

        Sets up the frequency bounds, coalescence time range, FFT time array, and zero-padding
        arrays for the likelihood calculation.

        Args:
            detectors (Sequence[Detector]): List of detector objects.
            waveform (Waveform): Waveform model.
            f_min (Float, optional): Minimum frequency. Defaults to 0.
            f_max (Float, optional): Maximum frequency. Defaults to infinity.
            trigger_time (Float, optional): Event trigger time. Defaults to 0.
            tc_range (tuple[Float, Float], optional): Marginalization range for coalescence time. Defaults to (-0.12, 0.12).
        """
        super().__init__(
            detectors, waveform, fixed_parameters, f_min, f_max, trigger_time
        )
        assert (
            "t_c" not in self.fixed_parameters
        ), "Cannot have t_c fixed while marginalizing over t_c"
        self.tc_range = tc_range
        fs = self.detectors[0].data.sampling_frequency
        duration = self.detectors[0].data.duration
        self.tc_array = jnp.fft.fftfreq(int(duration * fs / 2), 1.0 / duration)
        self.pad_low = jnp.zeros(int(self.frequencies[0] * duration))
        if jnp.isclose(self.frequencies[-1], fs / 2.0 - 1.0 / duration):
            self.pad_high = jnp.array([])
        else:
            self.pad_high = jnp.zeros(
                int((fs / 2.0 - 1.0 / duration - self.frequencies[-1]) * duration)
            )

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params["t_c"] = 0.0  # Fixing t_c to 0 for time marginalization
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the time-marginalized likelihood for a given set of parameters.
        Computes the log-likelihood marginalized over coalescence time by:
        - Calculating the frequency-domain inner product between the model and data for each detector.
        - Padding the inner product array to cover the full frequency range.
        - Applying FFT to obtain the likelihood as a function of coalescence time.
        - Restricting the FFT output to the specified `tc_range`.
        - Marginalizing using logsumexp over the allowed coalescence times.
        Args:
            params (dict[str, Float]): Dictionary of model parameters.
            data (dict): Dictionary containing data (not used in this implementation).
        Returns:
            Float: The marginalized log-likelihood value.
        """

        log_likelihood = 0.0
        complex_h_inner_d = jnp.zeros_like(self.detectors[0].sliced_frequencies)
        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        waveform_sky = self.waveform(self.frequencies, params)
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            # using <h|d> instead of <d|h>
            complex_h_inner_d += 4 * h_dec * jnp.conj(ifo_data) / psd * df
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += -optimal_SNR / 2

        # Padding the complex_h_inner_d to cover the full frequency range
        complex_h_inner_d_positive_f = jnp.concatenate(
            (self.pad_low, complex_h_inner_d, self.pad_high)
        )

        # FFT to obtain <h|d> exp(-i2πf t_c) as a function of t_c
        fft_h_inner_d = jnp.fft.fft(complex_h_inner_d_positive_f, norm="backward")

        # Restrict FFT output to the allowed tc_range, set others to -inf
        fft_h_inner_d = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            fft_h_inner_d.real,
            jnp.zeros_like(fft_h_inner_d.real) - jnp.inf,
        )

        # Marginalize over t_c using logsumexp
        log_likelihood += logsumexp(fft_h_inner_d) - jnp.log(len(self.tc_array))
        return log_likelihood


class PhaseMarginalizedLikelihoodFD(BaseTransientLikelihoodFD):
    """This has not been tested by a human yet."""

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["phase_c"] = 0.0  # Fixing phase_c to 0 for phase marginalization
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood = 0.0
        complex_d_inner_h = 0.0 + 0.0j

        waveform_sky = self.waveform(self.frequencies, params)
        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            complex_d_inner_h += complex_inner_product(h_dec, ifo_data, psd, df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += -optimal_SNR / 2

        log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))
        return log_likelihood


class PhaseTimeMarginalizedLikelihoodFD(TimeMarginalizedLikelihoodFD):
    """This has not been tested by a human yet."""

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params["t_c"] = 0.0  # Fix t_c for marginalization
        params["phase_c"] = 0.0
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        # Refactored: use self.detectors, self.frequencies, self.tc_array, self.pad_low, self.pad_high, self.tc_range
        log_likelihood = 0.0
        complex_h_inner_d = 0.0 + 0.0j

        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        waveform_sky = self.waveform(self.frequencies, params)
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            complex_h_inner_d += complex_inner_product(h_dec, ifo_data, psd, df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += -optimal_SNR / 2

        # Pad the complex_h_inner_d to cover the full frequency range
        complex_h_inner_d_positive_f = jnp.concatenate(
            (self.pad_low, complex_h_inner_d, self.pad_high)
        )

        # FFT to obtain <h|d> exp(-i2πf t_c) as a function of t_c
        fft_h_inner_d = jnp.fft.fft(complex_h_inner_d_positive_f, norm="backward")

        # Restrict FFT output to the allowed tc_range, set others to -inf
        log_i0_abs_fft = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            log_i0(jnp.absolute(fft_h_inner_d)),
            jnp.zeros_like(fft_h_inner_d.real) - jnp.inf,
        )

        # Marginalize over t_c using logsumexp
        log_likelihood += logsumexp(log_i0_abs_fft) - jnp.log(len(self.tc_array))
        return log_likelihood


class HeterodynedTransientLikelihoodFD(BaseTransientLikelihoodFD):
    n_bins: int  # Number of bins to use for the likelihood
    ref_params: dict  # Reference parameters for the likelihood
    freq_grid_low: Array  # Heterodyned frequency grid
    freq_grid_center: Array  # Heterodyned frequency grid at the center of the bin
    waveform_low_ref: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the low edge of the frequency bin, keyed by detector name
    waveform_center_ref: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the center of the frequency bin, keyed by detector name
    A0_array: dict[
        str, Float[Array, " n_bin"]
    ]  # A0 array for the likelihood, keyed by detector name
    A1_array: dict[
        str, Float[Array, " n_bin"]
    ]  # A1 array for the likelihood, keyed by detector name
    B0_array: dict[
        str, Float[Array, " n_bin"]
    ]  # B0 array for the likelihood, keyed by detector name
    B1_array: dict[
        str, Float[Array, " n_bin"]
    ]  # B1 array for the likelihood, keyed by detector name

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: float = 0,
        n_bins: int = 100,
        popsize: int = 100,
        n_steps: int = 2000,
        ref_params: dict = {},
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        sample_transforms: list[BijectiveTransform] = [],
        likelihood_transforms: list[NtoMTransform] = [],
    ):

        super().__init__(
            detectors, waveform, fixed_parameters, f_min, f_max, trigger_time
        )

        logging.info("Initializing heterodyned likelihood..")

        # Can use another waveform to use as reference waveform, but if not provided, use the same waveform
        if reference_waveform is None:
            reference_waveform = waveform

        if ref_params:
            self.ref_params = ref_params.copy()
            logging.info(f"Reference parameters provided, which are {self.ref_params}")
        elif prior:
            logging.info("No reference parameters are provided, finding it...")
            ref_params = self.maximize_likelihood(
                prior=prior,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
                popsize=popsize,
                n_steps=n_steps,
            )
            self.ref_params = {key: float(value) for key, value in ref_params.items()}
            logging.info(f"The reference parameters are {self.ref_params}")
        else:
            raise ValueError(
                "Either reference parameters or parameter names must be provided"
            )
        # safe guard for the reference parameters
        # since ripple cannot handle eta=0.25
        if jnp.isclose(self.ref_params["eta"], 0.25):
            self.ref_params["eta"] = 0.249995
            logging.info("The eta of the reference parameter is close to 0.25")
            logging.info(f"The eta is adjusted to {self.ref_params['eta']}")

        logging.info("Constructing reference waveforms..")

        self.ref_params["trigger_time"] = self.trigger_time
        self.ref_params["gmst"] = self.gmst

        self.waveform_low_ref = {}
        self.waveform_center_ref = {}
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}

        # Get the original frequency grid
        frequency_original = self.frequencies
        # Get the grid of the relative binning scheme (contains the final endpoint)
        # and the center points
        freq_grid, self.freq_grid_center = self.make_binning_scheme(
            jnp.array(frequency_original), n_bins
        )
        self.freq_grid_low = freq_grid[:-1]

        h_sky = reference_waveform(frequency_original, self.ref_params)

        # Get frequency masks to be applied, for both original
        # and heterodyne frequency grid
        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[pol]) for pol in h_sky.keys()]), axis=0
        )
        f_valid = frequency_original[jnp.where(h_amp > 0)[0]]
        f_max = jnp.max(f_valid)
        f_min = jnp.min(f_valid)

        mask_heterodyne_grid = jnp.where((freq_grid <= f_max) & (freq_grid >= f_min))[0]
        mask_heterodyne_low = jnp.where(
            (self.freq_grid_low <= f_max) & (self.freq_grid_low >= f_min)
        )[0]
        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_max) & (self.freq_grid_center >= f_min)
        )[0]
        freq_grid = freq_grid[mask_heterodyne_grid]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_low]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]

        # Ensure frequency grids have same length
        if len(self.freq_grid_low) > len(self.freq_grid_center):
            self.freq_grid_low = self.freq_grid_low[: len(self.freq_grid_center)]

        h_sky_low = reference_waveform(self.freq_grid_low, self.ref_params)
        h_sky_center = reference_waveform(self.freq_grid_center, self.ref_params)

        for detector in self.detectors:
            # Get the reference waveforms
            waveform_ref = detector.fd_response(
                frequency_original, h_sky, self.ref_params
            )
            self.waveform_low_ref[detector.name] = detector.fd_response(
                self.freq_grid_low, h_sky_low, self.ref_params
            )
            self.waveform_center_ref[detector.name] = detector.fd_response(
                self.freq_grid_center, h_sky_center, self.ref_params
            )
            A0, A1, B0, B1 = self.compute_coefficients(
                detector.sliced_fd_data,
                waveform_ref,
                detector.sliced_psd,
                frequency_original,
                freq_grid,
                self.freq_grid_center,
            )
            self.A0_array[detector.name] = A0[mask_heterodyne_center]
            self.A1_array[detector.name] = A1[mask_heterodyne_center]
            self.B0_array[detector.name] = B0[mask_heterodyne_center]
            self.B1_array[detector.name] = B1[mask_heterodyne_center]

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params.update(self.fixed_parameters)
        # evaluate the waveforms as usual
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        log_likelihood = 0.0
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)
        for detector in self.detectors:
            waveform_low = detector.fd_response(
                frequencies_low, waveform_sky_low, params
            )
            waveform_center = detector.fd_response(
                frequencies_center, waveform_sky_center, params
            )

            r0 = waveform_center / self.waveform_center_ref[detector.name]
            r1 = (waveform_low / self.waveform_low_ref[detector.name] - r0) / (
                frequencies_low - frequencies_center
            )
            match_filter_SNR = jnp.sum(
                self.A0_array[detector.name] * r0.conj()
                + self.A1_array[detector.name] * r1.conj()
            )
            optimal_SNR = jnp.sum(
                self.B0_array[detector.name] * jnp.abs(r0) ** 2
                + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
            )
            log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        return log_likelihood

    @staticmethod
    def max_phase_diff(
        freqs: Float[Array, " n_freq"],
        f_low: float,
        f_high: float,
        chi: float = 1.0,
    ):
        """
        Compute the maximum phase difference between the frequencies in the array.

        See Eq.(7) in arXiv:2302.05333.

        Parameters
        ----------
        freqs: Float[Array, "n_freq"]
            Array of frequencies to be binned.
        f_low: float
            Lower frequency bound.
        f_high: float
            Upper frequency bound.
        chi: float
            Power law index.

        Returns
        -------
        Float[Array, "n_freq"]
            Maximum phase difference between the frequencies in the array.
        """
        gamma = jnp.arange(-5, 6) / 3.0
        # Promotes freqs to 2D with shape (n_freq, 10) for later f/f_star
        freq_2D = jax.lax.broadcast_in_dim(freqs, (freqs.size, gamma.size), [0])
        f_star = jnp.where(gamma >= 0, f_high, f_low)
        summand = (freq_2D / f_star) ** gamma * jnp.sign(gamma)
        return 2 * jnp.pi * chi * jnp.sum(summand, axis=1)

    def make_binning_scheme(
        self, freqs: Float[Array, " n_freq"], n_bins: int, chi: float = 1
    ) -> tuple[Float[Array, " n_bins+1"], Float[Array, " n_bins"]]:
        """
        Make a binning scheme based on the maximum phase difference between the
        frequencies in the array.

        Parameters
        ----------
        freqs: Float[Array, "dim"]
            Array of frequencies to be binned.
        n_bins: int
            Number of bins to be used.
        chi: float = 1
            The chi parameter used in the phase difference calculation.

        Returns
        -------
        f_bins: Float[Array, "n_bins+1"]
            The bin edges.
        f_bins_center: Float[Array, "n_bins"]
            The bin centers.
        """
        phase_diff_array = self.max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)  # type: ignore
        phase_diff = jnp.linspace(phase_diff_array[0], phase_diff_array[-1], n_bins + 1)
        f_bins = interp1d(phase_diff_array, freqs)(phase_diff)
        f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
        return jnp.array(f_bins), jnp.array(f_bins_center)

    @staticmethod
    def compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
        A0_array = []
        A1_array = []
        B0_array = []
        B1_array = []

        df = freqs[1] - freqs[0]
        data_prod = jnp.array(data * h_ref.conj()) / psd
        self_prod = jnp.array(h_ref * h_ref.conj()) / psd

        # Vectorized binning using broadcasting
        freq_bins_left = f_bins[:-1]  # Shape: (len(f_bins)-1,)
        freq_bins_right = f_bins[1:]  # Shape: (len(f_bins)-1,)

        # Broadcast for vectorized comparison
        freqs_broadcast = freqs[None, :]  # Shape: (1, n_freqs)
        left_bounds = freq_bins_left[:, None]  # Shape: (len(f_bins)-1, 1)
        right_bounds = freq_bins_right[:, None]  # Shape: (len(f_bins)-1, 1)

        # Create mask matrix: True where frequency belongs to bin
        mask = (freqs_broadcast >= left_bounds) & (
            freqs_broadcast < right_bounds
        )  # Shape: (len(f_bins)-1, n_freqs)

        # Vectorized computation of frequency shifts
        f_bins_center_broadcast = f_bins_center[:, None]  # Shape: (len(f_bins)-1, 1)
        freq_shift_matrix = (
            freqs_broadcast - f_bins_center_broadcast
        ) * mask  # Shape: (len(f_bins)-1, n_freqs)

        # Vectorized computation of coefficients
        # For each bin, sum over the frequency dimension
        A0_array = (
            4 * jnp.sum(data_prod[None, :] * mask, axis=1) * df
        )  # Shape: (len(f_bins)-1,)
        A1_array = (
            4 * jnp.sum(data_prod[None, :] * freq_shift_matrix, axis=1) * df
        )  # Shape: (len(f_bins)-1,)
        B0_array = (
            4 * jnp.sum(self_prod[None, :] * mask, axis=1) * df
        )  # Shape: (len(f_bins)-1,)
        B1_array = (
            4 * jnp.sum(self_prod[None, :] * freq_shift_matrix, axis=1) * df
        )  # Shape: (len(f_bins)-1,)

        return A0_array, A1_array, B0_array, B1_array

    def maximize_likelihood(
        self,
        prior: Prior,
        likelihood_transforms: list[NtoMTransform],
        sample_transforms: list[BijectiveTransform],
        popsize: int = 100,
        n_steps: int = 2000,
    ):
        parameter_names = prior.parameter_names
        for transform in sample_transforms:
            parameter_names = transform.propagate_name(parameter_names)

        super_obj = super(HeterodynedTransientLikelihoodFD, self)

        def y(x: Float[Array, " n_dims"], data: dict) -> Float:
            named_params = dict(zip(parameter_names, x))
            for transform in reversed(sample_transforms):
                named_params = transform.backward(named_params)
            for transform in likelihood_transforms:
                named_params = transform.forward(named_params)
            return -super_obj.evaluate(named_params, data)

        print("Starting the optimizer")

        optimizer = AdamOptimization(
            logpdf=y, n_steps=n_steps, learning_rate=0.001, noise_level=1
        )

        key = jax.random.PRNGKey(0)
        initial_position = jnp.zeros((popsize, prior.n_dims)) + jnp.nan
        while not jax.tree.reduce(
            jnp.logical_and, jax.tree.map(lambda x: jnp.isfinite(x), initial_position)
        ).all():
            non_finite_index = jnp.where(
                jnp.any(
                    ~jax.tree.reduce(
                        jnp.logical_and,
                        jax.tree.map(lambda x: jnp.isfinite(x), initial_position),
                    ),
                    axis=1,
                )
            )[0]

            key, subkey = jax.random.split(key)
            guess = prior.sample(subkey, popsize)
            for transform in sample_transforms:
                guess = jax.vmap(transform.forward)(guess)
            guess = jnp.array([guess[key] for key in parameter_names]).T
            finite_guess = jnp.where(
                jnp.all(jax.tree.map(lambda x: jnp.isfinite(x), guess), axis=1)
            )[0]
            common_length = min(len(finite_guess), len(non_finite_index))
            initial_position = initial_position.at[
                non_finite_index[:common_length]
            ].set(guess[:common_length])

        _, best_fit, log_prob = optimizer.optimize(
            jax.random.PRNGKey(12094), y, initial_position, {}
        )

        named_params = dict(zip(parameter_names, best_fit[jnp.argmin(log_prob)]))
        for transform in reversed(sample_transforms):
            named_params = transform.backward(named_params)
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        return named_params


class HeterodynedPhaseMarginalizedLikelihoodFD(HeterodynedTransientLikelihoodFD):

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["phase_c"] = 0.0
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)
        log_likelihood = 0.0
        complex_d_inner_h = 0.0

        for detector in self.detectors:
            waveform_low = detector.fd_response(
                frequencies_low, waveform_sky_low, params
            )
            waveform_center = detector.fd_response(
                frequencies_center, waveform_sky_center, params
            )
            r0 = waveform_center / self.waveform_center_ref[detector.name]
            r1 = (waveform_low / self.waveform_low_ref[detector.name] - r0) / (
                frequencies_low - frequencies_center
            )
            complex_d_inner_h += jnp.sum(
                self.A0_array[detector.name] * r0.conj()
                + self.A1_array[detector.name] * r1.conj()
            )
            optimal_SNR = jnp.sum(
                self.B0_array[detector.name] * jnp.abs(r0) ** 2
                + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
            )
            log_likelihood += -optimal_SNR.real / 2

        log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

        return log_likelihood


likelihood_presets = {
    "BaseTransientLikelihoodFD": BaseTransientLikelihoodFD,
    "TimeMarginalizedLikelihoodFD": TimeMarginalizedLikelihoodFD,
    "PhaseMarginalizedLikelihoodFD": PhaseMarginalizedLikelihoodFD,
    "PhaseTimeMarginalizedLikelihoodFD": PhaseTimeMarginalizedLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "PhaseMarginalizedHeterodynedLikelihoodFD": HeterodynedPhaseMarginalizedLikelihoodFD,
}
