import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
from jimgw.generate_noise import generate_fd_noise, generate_LVK_PSDdict
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time

from tap import Tap
import yaml
from tqdm import tqdm
from functools import partialmethod

class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = None
    f_sampling: int  = None
    duration: int = None
    fmin: float = None
    ifos: list[str]  = None

    # Injection parameters
    m1: float = None
    m2: float = None
    chi1: float = None
    chi2: float = None
    dist_mpc: float = None
    tc: float = None
    phic: float = None
    inclination: float = None
    polarization_angle: float = None
    ra: float = None
    dec: float = None

    # Sampler parameters
    n_dim: int = None
    n_chains: int = None
    n_loop_training: int = None
    n_loop_production: int = None
    n_local_steps: int = None
    n_global_steps: int = None
    learning_rate: float = None
    max_samples: int = None
    momentum: float = None
    num_epochs: int = None
    batch_size: int = None
    stepsize: float = None

    # Output parameters
    output_path: str = None
    downsample_factor: int = None


args = InjectionRecoveryParser().parse_args()

# opt = vars(args)
# yaml_var = yaml.load(open(opt['config'], 'r'), Loader=yaml.FullLoader)
# opt.update(yaml_var)

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

psd_dict = generate_LVK_PSDdict(args.ifos)
freqs, psd_dict, noise_dict = generate_fd_noise(args.seed, args.f_sampling, args.duration, args.fmin, psd_dict)

#Fetch injection parameters and inject signal

print("Injection signals")

Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
f_ref = 30.0
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad


# def gen_waveform_H1(f, theta):
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
#     return H1_response(f, hp, hc, ra, dec, gmst , theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

# def gen_waveform_L1(f, theta):
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
#     return L1_response(f, hp, hc, ra, dec, gmst, theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

# def gen_waveform_V1(f, theta):
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
#     return V1_response(f, hp, hc, ra, dec, gmst, theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

waveform = RippleIMRPhenomD(f_ref=f_ref)
prior = Uniform(
    xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": lambda q: q/(1+q)**2,
                 "iota": lambda iota: jnp.arccos(jnp.arcsin(jnp.sin(iota/2*jnp.pi))*2/jnp.pi),
                 "dec": lambda dec: jnp.arcsin(jnp.arcsin(jnp.sin(dec/2*jnp.pi))*2/jnp.pi)} # sin and arcsin are periodize cos_iota and sin_dec
)
true_param = jnp.array([Mc, eta, args.chi1, args.chi2, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])


from scipy.interpolate import interp1d
q_axis = np.linspace(0.1, 1.0, 10000)
eta_axis = q_axis/(1+q_axis)**2
true_q = interp1d(eta_axis, q_axis)(eta)
cos_inclination = np.cos(inclination)
sin_dec = np.sin(dec)
true_param_trans = jnp.array([Mc, true_q, chi1, chi2, dist_mpc, tc, phic, cos_inclination, polarization_angle, ra, sin_dec])

f_list = freqs[freqs>fmin]
H1_signal = gen_waveform_H1(f_list, true_param)
H1_noise_psd = noise_dict['H1'][freqs>fmin]
H1_psd = psd_dict['H1'][freqs>fmin]
H1_data = H1_noise_psd + H1_signal

L1_signal = gen_waveform_L1(f_list, true_param)
L1_noise_psd = noise_dict['L1'][freqs>fmin]
L1_psd = psd_dict['L1'][freqs>fmin]
L1_data = L1_noise_psd + L1_signal

V1_signal = gen_waveform_V1(f_list, true_param)
V1_noise_psd = noise_dict['V1'][freqs>fmin]
V1_psd = psd_dict['V1'][freqs>fmin]
V1_data = V1_noise_psd + V1_signal

# ref_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])

# data_list = [H1_data, L1_data, V1_data]
# psd_list = [H1_psd, L1_psd, V1_psd]
# response_list = [H1_response, L1_response, V1_response]

# def LogLikelihood(theta):
#     theta = jnp.array(theta)
#     # theta = theta.at[1].set(theta[1]/(1+theta[1])**2) # convert q to eta
#     # theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
#     # theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     hp_test, hc_test = gen_IMRPhenomD_polar(f_list, theta_waveform, f_ref)
#     align_time = jnp.exp(-1j*2*jnp.pi*f_list*(epoch+theta[5]))
#     h_test_H1 = H1_response(f_list, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
#     h_test_L1 = L1_response(f_list, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
#     h_test_V1 = V1_response(f_list, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
#     df = f_list[1] - f_list[0]
#     match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*H1_data)/H1_psd*df).real
#     match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*L1_data)/L1_psd*df).real
#     match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*V1_data)/V1_psd*df).real
#     optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
#     optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
#     optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).real

#     return (match_filter_SNR_H1-optimal_SNR_H1/2) + (match_filter_SNR_L1-optimal_SNR_L1/2) + (match_filter_SNR_V1-optimal_SNR_V1/2)


# logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, f_list, gmst, epoch, f_ref, heterodyne_bins)

# # Fetch sampler parameters, construct sampler and initial guess

# print("Making sampler")

# n_dim = args['n_dim']
# n_chains = args['n_chains']
# n_loop_training = args['n_loop_training']
# n_loop_production = args['n_loop_production']
# n_local_steps = args['n_local_steps']
# n_global_steps = args['n_global_steps']
# learning_rate = args['learning_rate']
# max_samples = args['max_samples']
# momentum = args['momentum']
# num_epochs = args['num_epochs']
# batch_size = args['batch_size']
# stepsize = args['stepsize']


# guess_param = np.array(jnp.repeat(true_param_trans[None,:],int(n_chains),axis=0)*(1+0.1*jax.random.normal(jax.random.PRNGKey(seed+98127),shape=(int(n_chains),n_dim))))
# guess_param[guess_param[:,1]>1,1] = 1

# print("Preparing RNG keys")
# rng_key_set = initialize_rng_keys(n_chains, seed=seed)

# print("Initializing MCMC model and normalizing flow model.")

# prior_range = jnp.array([[10,50],[0.5,1.0],[-0.5,0.5],[-0.5,0.5],[300,2000],[-0.5,0.5],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])


# initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
# for i in range(n_dim):
#     initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

# from ripple import Mc_eta_to_ms
# m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
# q = m2/m1
# initial_position = initial_position.at[:,0].set(guess_param[:,0])
# initial_position = initial_position.at[:,5].set(guess_param[:,5])

# from astropy.cosmology import Planck18 as cosmo

# z = np.linspace(0.01,0.4,10000)
# dL = cosmo.luminosity_distance(z).value
# dVdz = cosmo.differential_comoving_volume(z).value

# def top_hat(x):
#     output = 0.
#     for i in range(n_dim):
#         output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
#         output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
#     return output#+jnp.log(jnp.interp(x[4],dL,dVdz))

# def posterior(theta):
#     q = theta[1]
#     iota = jnp.arccos(theta[7])
#     dec = jnp.arcsin(theta[10])
#     prior = top_hat(theta)
#     theta = theta.at[1].set(q/(1+q)**2) # convert q to eta
#     theta = theta.at[7].set(iota) # convert cos iota to iota
#     theta = theta.at[10].set(dec) # convert cos dec to dec
#     return logL(theta) + prior


# model = RQSpline(n_dim, 10, [128,128], 8)


# print("Initializing sampler class")

# posterior = posterior
# dposterior = jax.grad(posterior)


# mass_matrix = np.eye(n_dim)
# mass_matrix = np.abs(1./(jax.grad(logL)(true_param)+jax.grad(top_hat)(true_param)))*mass_matrix
# mass_matrix = jnp.array(mass_matrix)

# local_sampler = MALA(posterior, True, {"step_size": mass_matrix*3e-3})
# print("Running sampler")

# nf_sampler = Sampler(
#     n_dim,
#     rng_key_set,
#     local_sampler,
#     posterior,
#     model,
#     n_loop_training=n_loop_training,
#     n_loop_production = n_loop_production,
#     n_local_steps=n_local_steps,
#     n_global_steps=n_global_steps,
#     n_chains=n_chains,
#     n_epochs=num_epochs,
#     learning_rate=learning_rate,
#     momentum=momentum,
#     batch_size=batch_size,
#     use_global=True,
#     keep_quantile=0.,
#     train_thinning = 40,
#     local_autotune=mala_sampler_autotune
# )

# nf_sampler.sample(initial_position)

# labels = ['Mc', 'eta', 'chi1', 'chi2', 'dist_mpc', 'tc', 'phic', 'cos_inclination', 'polarization_angle', 'ra', 'sin_dec']

# print("Saving to output")

# chains, log_prob, local_accs, global_accs, loss_vals = nf_sampler.get_sampler_state(training=True).values()
# chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()

# # Fetch output parameters

# output_path = args['output_path']
# downsample_factor = args['downsample_factor']

# np.savez(args['output_path'], chains=chains[:,::downsample_factor], log_prob=log_prob[:,::downsample_factor], local_accs=local_accs[:,::downsample_factor], global_accs=global_accs[:,::downsample_factor], loss_vals=loss_vals, labels=labels, true_param=true_param, true_log_prob=LogLikelihood(true_param))
