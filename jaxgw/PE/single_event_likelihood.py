from jax import jit
from jaxgw.gw.likelihood.detector_projection import get_detector_response
from jaxgw.gw.likelihood.utils import inner_product

def single_detector_likelihood(waveform_model, params, data, data_f, PSD, detector):
    waveform = waveform_model(data_f, params)
    waveform = get_detector_response(waveform, params, detector)
    match_filter_SNR = inner_product(waveform, data, data_f, PSD)
    optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
    return (-2*match_filter_SNR + optimal_SNR)/2



