from gwosc.datasets import event_gps
gps = event_gps("GW150914")
start = int(gps) - 16
end = int(gps) + 16

from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data('L1', start, end)

from scipy.signal.windows import tukey

data_window = data * tukey(len(data), alpha=0.2)
data_fft = data_window.fft()
f = data_fft.frequencies.value

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar

params = jnp.array([30,0.249, 0.1,0.1, 400, 0.0, 0.0, 0.1, 0.0])
