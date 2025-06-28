import jax
import jax.numpy as jnp
from flowMC.strategy.optimization import AdamOptimization
from jax.scipy.special import logsumexp
from jax.lax import broadcast_in_dim
from jaxtyping import Array, Float, Complex
from typing import Optional
from scipy.interpolate import interp1d

from jimgw.core.utils import log_i0, generate_initial_samples
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
from copy import deepcopy
from typing import Sequence


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform

    def __init__(self, detectors: Sequence[Detector], waveform: Waveform) -> None:
        self.detectors = detectors
        self.waveform = waveform


class ZeroLikelihood(LikelihoodBase):

    def __init__(self):
        pass

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0


class TransientLikelihoodFD(SingleEventLikelihood):
    """
    We should explain our choice of one `f_min` for all detectors here,
    and how should the user achieve data masking properly, either in the PSD or data.
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: Float = 0,
        **kwargs,
    ) -> None:
        # NOTE: having 'kwargs' here makes it very difficult to diagnose
        # errors and keep track of what's going on, would be better to list
        # explicitly what the arguments are accepted

        # Set the frequency bounds for the detectors
        _frequencies = []
        for detector in detectors:
            detector.set_frequency_bounds(f_min, f_max)
            _frequencies.append(detector.sliced_frequencies)
        assert jnp.array_equiv(
            _frequencies[0], jnp.asarray(_frequencies)
        ), "The frequency arrays are not all the same."

        self.detectors = detectors
        self.frequencies = _frequencies[0]
        self.duration = self.detectors[0].data.duration
        self.waveform = waveform
        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)
        self.kwargs = kwargs
        if "marginalization" in self.kwargs:
            marginalization = self.kwargs["marginalization"]
            assert marginalization in [
                "phase",
                "phase-time",
                "time",
            ], "Only support time, phase and phase+time marginalzation"
            self.marginalization = marginalization
            if self.marginalization == "phase-time":
                self.param_func = lambda x: {**x, "phase_c": 0.0, "t_c": 0.0}
                self.likelihood_function = phase_time_marginalized_likelihood
                logging.info("Marginalizing over phase and time")
            elif self.marginalization == "time":
                self.param_func = lambda x: {**x, "t_c": 0.0}
                self.likelihood_function = time_marginalized_likelihood
                logging.info("Marginalizing over time")
            elif self.marginalization == "phase":
                self.param_func = lambda x: {**x, "phase_c": 0.0}
                self.likelihood_function = phase_marginalized_likelihood
                logging.info("Marginalizing over phase")
        else:
            self.param_func = lambda x: x
            self.likelihood_function = original_likelihood
            self.marginalization = ""

        # the fixing_parameters is expected to be a dictionary
        # with key as parameter name and value is the fixed value
        # e.g. {'M_c': 1.1975, 't_c': 0}
        if "fixing_parameters" in self.kwargs:
            fixing_parameters = self.kwargs["fixing_parameters"]
            print(f"Parameters are fixed {fixing_parameters}")
            # check for conflict with the marginalization
            assert not (
                "t_c" in fixing_parameters and "time" in self.marginalization
            ), "Cannot have t_c fixed while having the marginalization of t_c turned on"
            assert not (
                "phase_c" in fixing_parameters and "phase" in self.marginalization
            ), "Cannot have phase_c fixed while having the marginalization of phase_c turned on"
            # if the same key exists in both dictionary,
            # the later one will overwrite the former one
            self.fixing_func = lambda x: {**x, **fixing_parameters}
        else:
            self.fixing_func = lambda x: x

    @property
    def detector_names(self):
        """The interferometers for the likelihood."""
        return [detector.name for detector in self.detectors]

    def evaluate(self, params: dict[str, Float], data: Optional[dict] = None) -> Float:
        # TODO: Test whether we need to pass data in or with class changes is fine.
        """Evaluate the likelihood for a given set of parameters."""
        _params = params.copy()
        _params["trigger_time"] = self.trigger_time
        _params["gmst"] = self.gmst
        # adjust the params due to different marginalzation scheme
        _params = self.param_func(_params)
        # adjust the params due to fixing parameters
        _params = self.fixing_func(_params)
        # evaluate the waveform as usual
        waveform_sky = self.waveform(self.frequencies, _params)
        return self.likelihood_function(
            _params,
            waveform_sky,
            self.detectors,  # type: ignore
            **self.kwargs,
        )


class HeterodynedTransientLikelihoodFD(TransientLikelihoodFD):
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
        f_min: Float = 0,
        f_max: Float = float("inf"),
        n_bins: int = 100,
        trigger_time: float = 0,
        popsize: int = 100,
        n_steps: int = 2000,
        ref_params: dict = {},
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        sample_transforms: Sequence[BijectiveTransform] = [],
        likelihood_transforms: Sequence[NtoMTransform] = [],
        **kwargs,
    ) -> None:
        super().__init__(detectors, waveform, f_min, f_max, trigger_time)

        logging.info("Initializing heterodyned likelihood..")

        # Can use another waveform to use as reference waveform, but if not provided, use the same waveform
        if reference_waveform is None:
            reference_waveform = waveform

        self.kwargs = kwargs
        if "marginalization" in self.kwargs:
            marginalization = self.kwargs["marginalization"]
            assert marginalization in [
                "phase",
            ], "Heterodyned likelihood only support phase marginalzation"
            self.marginalization = marginalization
            if self.marginalization == "phase":
                self.param_func = lambda x: {**x, "phase_c": 0.0}
                self.likelihood_function = phase_marginalized_likelihood
                self.rb_likelihood_function = (
                    phase_marginalized_relative_binning_likelihood
                )
                logging.info("Marginalizing over phase")
        else:
            self.param_func = lambda x: x
            self.likelihood_function = original_likelihood
            self.rb_likelihood_function = original_relative_binning_likelihood
            self.marginalization = ""

        # the fixing_parameters is expected to be a dictionary
        # with key as parameter name and value is the fixed value
        # e.g. {'M_c': 1.1975, 't_c': 0}
        if "fixing_parameters" in self.kwargs:
            fixing_parameters = self.kwargs["fixing_parameters"]
            logging.info(f"Parameters are fixed {fixing_parameters}")
            # check for conflict with the marginalization
            assert not (
                "t_c" in fixing_parameters and "time" in self.marginalization
            ), "Cannot have t_c fixed while marginalizing over t_c"
            assert not (
                "phase_c" in fixing_parameters and "phase" in self.marginalization
            ), "Cannot have phase_c fixed while marginalizing over phase_c"
            # if the same key exists in both dictionary,
            # the later one will overwrite the former one
            self.fixing_func = lambda x: {**x, **fixing_parameters}
        else:
            self.fixing_func = lambda x: x

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
        # adjust the params due to different marginalzation scheme
        self.ref_params = self.param_func(self.ref_params)
        # adjust the params due to fixing parameters
        self.ref_params = self.fixing_func(self.ref_params)

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

    def evaluate(self, params: dict[str, Float], data: Optional[dict] = None) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        _params = params.copy()
        _params["trigger_time"] = self.trigger_time
        _params["gmst"] = self.gmst
        # adjust the params due to different marginalzation scheme
        _params = self.param_func(_params)
        # adjust the params due to fixing parameters
        _params = self.fixing_func(_params)
        # evaluate the waveforms as usual
        waveform_sky_low = self.waveform(frequencies_low, _params)
        waveform_sky_center = self.waveform(frequencies_center, _params)
        log_likelihood = self.rb_likelihood_function(
            _params,
            self.A0_array,
            self.A1_array,
            self.B0_array,
            self.B1_array,
            waveform_sky_low,
            waveform_sky_center,
            self.waveform_low_ref,
            self.waveform_center_ref,
            self.detectors,
            frequencies_low,
            frequencies_center,
            **self.kwargs,
        )
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
        freq_2D = broadcast_in_dim(freqs, (freqs.size, gamma.size), [0])
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
        for i in range(len(f_bins) - 1):
            f_index = jnp.where((freqs >= f_bins[i]) & (freqs < f_bins[i + 1]))[0]
            freq_shift = freqs[f_index] - f_bins_center[i]
            A0_array.append(jnp.sum(data_prod[f_index]))
            A1_array.append(jnp.sum(data_prod[f_index] * freq_shift))
            B0_array.append(jnp.sum(self_prod[f_index]))
            B1_array.append(jnp.sum(self_prod[f_index] * freq_shift))

        A0_array = 4 * df * jnp.array(A0_array)
        A1_array = 4 * df * jnp.array(A1_array)
        B0_array = 4 * df * jnp.array(B0_array)
        B1_array = 4 * df * jnp.array(B1_array)
        return A0_array, A1_array, B0_array, B1_array

    def maximize_likelihood(
        self,
        prior: Prior,
        likelihood_transforms: Sequence[NtoMTransform],
        sample_transforms: Sequence[BijectiveTransform],
        popsize: int = 100,
        n_steps: int = 2000,
        return_stage1: bool = False,
    ):
        parameter_names = prior.parameter_names
        for transform in sample_transforms:
            parameter_names = transform.propagate_name(parameter_names)

        super_obj = super(HeterodynedTransientLikelihoodFD, self)
        orig_fixing_parameters = self.kwargs.get("fixing_parameters", {})

        def apply_transforms(named_params):
            for transform in reversed(sample_transforms):
                named_params = transform.backward(named_params)
            for transform in likelihood_transforms:
                named_params = transform.forward(named_params)
            return named_params

        # Generate initial samples
        initial_position, key = generate_initial_samples(
            prior,
            sample_transforms,
            popsize,
            jax.random.PRNGKey(123),
        )

        # Stage 1: Optimize subset of parameters
        # Define the subset of parameters to be optimized in Stage 1
        stage1_parameters_all = [
            "M_c",
            "m_1",
            "m_2",
            "eta",
            "ra",
            "dec",
            "azimuth",
            "zenith",
            "t_c",
            "t_det",
            "d_L",
            "d_hat",
            "iota",
            "theta_jn",
        ]
        fixed_param_keys = [
            key
            for key in parameter_names
            if not any(key.startswith(p) for p in stage1_parameters_all)
        ]

        if not len(fixed_param_keys):
            print(
                "Warning: No stage1_parameters found in parameter_names, skipping Stage 1"
            )
        else:
            print(f"Starting Stage 1 optimization")

            def y_stage1(x_array: Float[Array, " n_subset"], data: dict) -> Float:
                # Pick the fixed parameters from one random chain
                data["key"], subkey = jax.random.split(data["key"])
                idx = jax.random.randint(subkey, (), 0, popsize)
                _fixed_parameters = data.pop("fixed_parameters")
                fixed_parameters = {
                    name: row[idx] for name, row in _fixed_parameters.items()
                }
                # Update the fixing function in the likelihood
                self.fixing_func = lambda x: {
                    **x,
                    **fixed_parameters,
                    **orig_fixing_parameters,
                }
                named_params = dict(zip(parameter_names, x_array))
                named_params = apply_transforms(named_params)
                return -super_obj.evaluate(named_params, data)

            optimizer_stage1 = AdamOptimization(
                logpdf=y_stage1, n_steps=n_steps, learning_rate=0.01, noise_level=0
            )

            initial_position_dict = dict(zip(parameter_names, initial_position.T))
            initial_likelihood_parameters = apply_transforms(initial_position_dict)
            fixed_keys = set(stage1_parameters_all).intersection(
                initial_likelihood_parameters.keys()
            )
            print(f"Stage 1 parameters: {fixed_keys}")
            fixed_parameters = {
                key: initial_likelihood_parameters[key] for key in fixed_keys
            }
            original_fixing_func = deepcopy(self.fixing_func)

            key, subkey = jax.random.split(key)
            _, best_fit_stage1, log_prob_stage1 = optimizer_stage1.optimize(
                key,
                y_stage1,
                initial_position,
                {"key": subkey, "fixed_parameters": fixed_parameters},
            )

            # Update the initial position with the optimized subset
            optimised_stage1_idx = jnp.argmin(log_prob_stage1)
            optimised_stage1_params = best_fit_stage1[optimised_stage1_idx]
            optimised_stage1_dict = apply_transforms(
                dict(zip(parameter_names, optimised_stage1_params))
            )

            print("Stage 1 optimization completed.")
            print(f"Optimised parameters: {optimised_stage1_dict}")
            print(f"Optimised log likelihood: {-log_prob_stage1[optimised_stage1_idx]}")

            # Generate new initial samples for Stage 2
            initial_position, key = generate_initial_samples(
                prior,
                sample_transforms,
                popsize,
                jax.random.PRNGKey(321),
            )
            # Update the initial position with the optimized subset
            for _key in parameter_names:
                if _key in fixed_param_keys:
                    continue
                idx = parameter_names.index(_key)
                initial_position = initial_position.at[:, idx].set(
                    jnp.ones(popsize) * optimised_stage1_params[idx]
                )
            # Restore the fixing function to the original one
            self.fixing_func = original_fixing_func

        # Stage 2: Optimize all parameters
        print("Starting Stage 2 optimization")

        def y(x_array: Float[Array, " n_dims"], data: dict) -> Float:
            named_params = dict(zip(parameter_names, x_array))
            named_params = apply_transforms(named_params)
            return -super_obj.evaluate(named_params, data)

        optimizer = AdamOptimization(
            logpdf=y, n_steps=n_steps, learning_rate=0.001, noise_level=0
        )
        _, best_fit, log_prob = optimizer.optimize(key, y, initial_position, {})

        best_stage2_idx = jnp.argmin(log_prob)
        named_params = apply_transforms(
            dict(zip(parameter_names, best_fit[best_stage2_idx]))
        )

        print(f"Stage 2 optimization completed.")
        print(f"Optimised parameters: {named_params}")
        print(f"Optimised log likelihood: {-log_prob[best_stage2_idx]}")
        if return_stage1:
            return named_params, optimised_stage1_dict
        return named_params


likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
}


def original_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Complex[Array, " n_dim"]],
    detectors: list[Detector],
    **kwargs,
) -> Float:
    log_likelihood = 0.0
    for ifo in detectors:
        freqs, data, psd = ifo.sliced_frequencies, ifo.sliced_fd_data, ifo.sliced_psd
        h_dec = ifo.fd_response(freqs, h_sky, params)
        match_filter_SNR = inner_product(h_dec, data, psd, freqs)
        optimal_SNR = inner_product(h_dec, h_dec, psd, freqs)
        log_likelihood += match_filter_SNR - optimal_SNR / 2

    return log_likelihood


def phase_marginalized_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Complex[Array, " n_dim"]],
    detectors: list[Detector],
    **kwargs,
) -> Float:
    log_likelihood = 0.0
    complex_d_inner_h = 0.0 + 0.0j
    for ifo in detectors:
        freqs, data, psd = ifo.sliced_frequencies, ifo.sliced_fd_data, ifo.sliced_psd
        h_dec = ifo.fd_response(freqs, h_sky, params)
        complex_d_inner_h += complex_inner_product(h_dec, data, psd, freqs)
        optimal_SNR = inner_product(h_dec, h_dec, psd, freqs)
        log_likelihood += -optimal_SNR / 2

    log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))
    return log_likelihood


def _get_tc_array(duration: Float, sampling_rate: Float):
    return jnp.fft.fftfreq(int(duration * sampling_rate / 2), 1 / duration)


def _get_frequencies_pads(detector: Detector, fs: Float) -> tuple[Float, Float]:
    f_low, f_high = detector.frequency_bounds
    duration = detector.data.duration
    delta_f = 1 / duration

    pad_low = jnp.zeros(int(f_low * duration))

    f_Nyquist_diff = fs / 2.0 - delta_f - f_high
    if jnp.isclose(f_Nyquist_diff, 0):
        pad_high = jnp.array([])
    else:
        pad_high = jnp.zeros(int(f_Nyquist_diff * duration))
    return pad_low, pad_high


def time_marginalized_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Complex[Array, " n_dim"]],
    detectors: list[Detector],
    **kwargs,
) -> Float:
    log_likelihood = 0.0
    complex_h_inner_d = 0.0 + 0.0j
    for ifo in detectors:
        freqs, data, psd = ifo.sliced_frequencies, ifo.sliced_fd_data, ifo.sliced_psd
        h_dec = ifo.fd_response(freqs, h_sky, params)
        # using <h|d> instead of <d|h>
        complex_h_inner_d += complex_inner_product(data, h_dec, psd, freqs)
        optimal_SNR = inner_product(h_dec, h_dec, psd, freqs)
        log_likelihood += -optimal_SNR / 2
    duration = detectors[0].data.duration

    # fetch the tc range tc_array, lower padding and higher padding
    tc_range = kwargs["tc_range"]
    fs = kwargs["sampling_rate"]
    tc_array = _get_tc_array(duration, fs)
    pad_low, pad_high = _get_frequencies_pads(detectors[0], fs=fs)

    # padding the complex_h_inner_d
    # this array is the hd*/S for f in [0, fs / 2 - df]
    complex_h_inner_d_positive_f = jnp.concatenate(
        (pad_low, complex_h_inner_d, pad_high)
    )

    # make use of the fft
    # which then return the <h|d>exp(-i2pift_c)
    # w.r.t. the tc_array
    fft_h_inner_d = jnp.fft.fft(complex_h_inner_d_positive_f, norm="backward")

    # set the values to -inf when it is outside the tc range
    # so that they will disappear after the logsumexp
    fft_h_inner_d = jnp.where(
        (tc_array > tc_range[0]) & (tc_array < tc_range[1]),
        fft_h_inner_d.real,
        jnp.zeros_like(fft_h_inner_d.real) - jnp.inf,
    )

    # using the logsumexp to marginalize over the tc prior range
    log_likelihood += logsumexp(fft_h_inner_d) - jnp.log(len(tc_array))
    return log_likelihood


def phase_time_marginalized_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Complex[Array, " n_dim"]],
    detectors: list[Detector],
    **kwargs,
) -> Float:
    log_likelihood = 0.0
    complex_h_inner_d = 0.0 + 0.0j
    for ifo in detectors:
        freqs, data, psd = ifo.sliced_frequencies, ifo.sliced_fd_data, ifo.sliced_psd
        h_dec = ifo.fd_response(freqs, h_sky, params)
        # using <h|d> instead of <d|h>
        complex_h_inner_d += complex_inner_product(data, h_dec, psd, freqs)
        optimal_SNR = inner_product(h_dec, h_dec, psd, freqs)
        log_likelihood += -optimal_SNR / 2
    duration = detectors[0].data.duration

    # fetch the tc range tc_array, lower padding and higher padding
    tc_range = kwargs["tc_range"]
    fs = kwargs["sampling_rate"]
    tc_array = _get_tc_array(duration, fs)
    pad_low, pad_high = _get_frequencies_pads(detectors[0], fs=fs)

    # padding the complex_h_inner_d
    # this array is the hd*/S for f in [0, fs / 2 - df]
    complex_h_inner_d_positive_f = jnp.concatenate(
        (pad_low, complex_h_inner_d, pad_high)
    )

    # make use of the fft
    # which then return the <h|d>exp(-i2pift_c)
    # w.r.t. the tc_array
    fft_h_inner_d = jnp.fft.fft(complex_h_inner_d_positive_f, norm="backward")

    # set the values to -inf when it is outside the tc range
    # so that they will disappear after the logsumexp
    log_i0_abs_fft = jnp.where(
        (tc_array > tc_range[0]) & (tc_array < tc_range[1]),
        log_i0(jnp.absolute(fft_h_inner_d)),
        jnp.zeros_like(fft_h_inner_d.real) - jnp.inf,
    )

    # using the logsumexp to marginalize over the tc prior range
    log_likelihood += logsumexp(log_i0_abs_fft) - jnp.log(len(tc_array))
    return log_likelihood


def original_relative_binning_likelihood(
    params,
    A0_array,
    A1_array,
    B0_array,
    B1_array,
    waveform_sky_low,
    waveform_sky_center,
    waveform_low_ref,
    waveform_center_ref,
    detectors,
    frequencies_low,
    frequencies_center,
    **kwargs,
):
    log_likelihood = 0.0

    for detector in detectors:
        waveform_low = detector.fd_response(frequencies_low, waveform_sky_low, params)
        waveform_center = detector.fd_response(
            frequencies_center, waveform_sky_center, params
        )

        r0 = waveform_center / waveform_center_ref[detector.name]
        r1 = (waveform_low / waveform_low_ref[detector.name] - r0) / (
            frequencies_low - frequencies_center
        )
        match_filter_SNR = jnp.sum(
            A0_array[detector.name] * r0.conj() + A1_array[detector.name] * r1.conj()
        )
        optimal_SNR = jnp.sum(
            B0_array[detector.name] * jnp.abs(r0) ** 2
            + 2 * B1_array[detector.name] * (r0 * r1.conj()).real
        )
        log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

    return log_likelihood


def phase_marginalized_relative_binning_likelihood(
    params,
    A0_array,
    A1_array,
    B0_array,
    B1_array,
    waveform_sky_low,
    waveform_sky_center,
    waveform_low_ref,
    waveform_center_ref,
    detectors,
    frequencies_low,
    frequencies_center,
    **kwargs,
):
    log_likelihood = 0.0
    complex_d_inner_h = 0.0

    for detector in detectors:
        waveform_low = detector.fd_response(frequencies_low, waveform_sky_low, params)
        waveform_center = detector.fd_response(
            frequencies_center, waveform_sky_center, params
        )

        r0 = waveform_center / waveform_center_ref[detector.name]
        r1 = (waveform_low / waveform_low_ref[detector.name] - r0) / (
            frequencies_low - frequencies_center
        )
        complex_d_inner_h += jnp.sum(
            A0_array[detector.name] * r0.conj() + A1_array[detector.name] * r1.conj()
        )
        optimal_SNR = jnp.sum(
            B0_array[detector.name] * jnp.abs(r0) ** 2
            + 2 * B1_array[detector.name] * (r0 * r1.conj()).real
        )
        log_likelihood += -optimal_SNR.real / 2

    log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

    return log_likelihood
