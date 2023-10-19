import numpy as np
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import TransientLikelihoodFD
from astropy.time import Time
from jimgw.waveform import RippleIMRPhenomD
import jax.numpy as jnp
import jax
from jimgw.prior import Uniform
from jax import config
config.update("jax_enable_x64", True)

def Mc_eta_to_ms(m):
    Mchirp, eta = m
    M = Mchirp / (eta ** (3 / 5))
    m2 = (M - np.sqrt(M ** 2 - 4 * M ** 2 * eta)) / 2
    m1 = M - m2
    return m1, m2

prior_range = np.array(
    [[20,80], # mc
     [0.5,1], # q
     [0,1], # s1_mag,
     [0,np.pi], #s1_theta,
     [0, 2* np.pi], # s1_phi,
     [0,1], # s2_mag,
     [0,np.pi], #s2_theta,
     [0, 2* np.pi], # s2_phi,
     [300,1500], # dist_mpc
     [-0.04,0.04], # tc
     [0,2*np.pi], # phic
     [-1,1], # cos_incl
     [0,np.pi], # polarization angle
     [0,2*np.pi], # ra
     [-1,1]] # sin_dec
     )

N_config = 1

duration = 4

gate_keeping_likelihood = 12
waveform = RippleIMRPhenomD(f_ref=30)
freqs = jnp.linspace(30, 2048/2, duration*2048//2)

def test_log_likelihood(true_param,seed):

    #tc_low = true_param["t_c"] - 0.01
    #tc_up = true_param["t_c"] + 0.01
    #print(tc_up, tc_low)    
    #prior = Uniform(
    #xmin = [20, 0.125, -1., -1., 100., tc_low, 0., -1, 0., 0.,-1.],
    #xmax = [80., 1., 1., 1., 1600., tc_up, 2*jnp.pi, 1., jnp.pi, 2*jnp.#pi, 1.],
    #naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", #"cos_iota", "psi", "ra", "sin_dec"],
    #transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 #"cos_iota": ("iota",lambda cos_iota: jnp.arccos(cos_iota)),
                 #"sin_dec": ("dec",lambda sin_dec: jnp.arcsin(sin_dec))}
    #            "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
    #             "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
    #)
    #true_param_dict = prior.add_name(true_param, transform_name=True, transform_value = True)
    post_trigger_duration = 2
    trigger_time = 1126259462.4
    epoch = duration - post_trigger_duration
    gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
    detector_param = {"ra": ra, "dec": dec, "gmst": gmst, "psi": polarization_angle, "epoch": epoch, "t_c": tc}
    h_sky = waveform(freqs, true_param)
    key, subkey = jax.random.split(jax.random.PRNGKey(seed+1234))
    H1.inject_signal(subkey, freqs, h_sky, detector_param)
    key, subkey = jax.random.split(key)
    L1.inject_signal(subkey, freqs, h_sky, detector_param)
    #key, subkey = jax.random.split(key)
    #V1.inject_signal(subkey, freqs, h_sky, detector_param)
    likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, duration, post_trigger_duration)
    log_likelihood = likelihood.evaluate(true_param,{})
    print("log_likelihood:", log_likelihood)
    return (log_likelihood > gate_keeping_likelihood), log_likelihood

m1_out, m2_out, s1_mag_out, s1_phi_out, s1_theta_out, s2_mag_out, s2_phi_out, s2_theta_out, dist_mpc_out, tc_out, phic_out, inclination_out, polarization_angle_out, ra_out, dec_out = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
seed_out, log_likelihood_out = [],[]
count = 0
while count < N_config:
    mc = np.random.uniform(prior_range[0,0],prior_range[0,1])
    q = np.random.uniform(prior_range[1,0],prior_range[1,1])
    eta = q/(1+q)**2
    m1, m2 = Mc_eta_to_ms(np.stack([mc,eta]))
    s1_mag = np.random.uniform(prior_range[2,0],prior_range[2,1])
    s1_theta = np.random.uniform(prior_range[3,0],prior_range[3,1])
    s1_phi = np.random.uniform(prior_range[4,0],prior_range[4,1])
    s2_mag = np.random.uniform(prior_range[5,0],prior_range[5,1])
    s2_theta = np.random.uniform(prior_range[6,0],prior_range[6,1])
    s2_phi = np.random.uniform(prior_range[7,0],prior_range[7,1])
    dist_mpc = np.random.uniform(prior_range[8,0],prior_range[8,1])
    tc = np.random.uniform(prior_range[9,0],prior_range[9,1])
    phic = np.random.uniform(prior_range[10,0],prior_range[10,1])
    cos_inclination = np.random.uniform(prior_range[11,0],prior_range[11,1])
    inclination = np.arccos(np.arcsin(np.sin(cos_inclination/2*np.pi))*2/np.pi)
    polarization_angle = np.random.uniform(prior_range[12,0],prior_range[12,1])
    ra = np.random.uniform(prior_range[13,0],prior_range[13,1])
    sin_dec = np.random.uniform(prior_range[14,0],prior_range[14,1])
    dec = np.arcsin(np.arcsin(np.sin(sin_dec/2*np.pi))*2/np.pi)
    seed = np.random.randint(low=0,high=10000)
    params_dict = {"M_c":mc, "eta":eta, "s1_z": s1_mag, "s2_z": s2_mag, "d_L":dist_mpc, "t_c":tc, "phase_c":phic, "iota":inclination, "psi":polarization_angle, "ra":ra, "dec":dec}
    gate, log_likelihood = test_log_likelihood(params_dict, seed)
    if gate:
        m1_out.append(m1)
        m2_out.append(m2)
        s1_mag_out.append(s1_mag)
        s1_theta_out.append(s1_theta)
        s1_phi_out.append(s1_phi)
        s2_mag_out.append(s2_mag)
        s2_theta_out.append(s2_theta)
        s2_phi_out.append(s2_phi)
        dist_mpc_out.append(dist_mpc)
        tc_out.append(tc)
        phic_out.append(phic)
        inclination_out.append(inclination)
        polarization_angle_out.append(polarization_angle)
        ra_out.append(ra)
        dec_out.append(dec)
        seed_out.append(seed)
        log_likelihood_out.append(log_likelihood)
        count +=1
    print(count)
    pass


directory = '/home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/configs/'




for i in range(N_config):
    f = open(directory+"injection_config_"+str(i)+".yaml","w")
    
    f.write('output_path: /home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/zw_test_batch_out/injection_'+str(i)+'\n')
    f.write('downsample_factor: 10\n')
    f.write('seed: '+str(seed_out[i])+'\n')
    f.write('f_sampling: 2048\n')
    f.write('duration: ' + str(duration) + '\n')
    f.write('fmin: 30\n')
    f.write('ifos:\n')
    f.write('  - H1\n')
    f.write('  - L1\n')
    f.write('  - V1\n')

    f.write("m1: "+str(m1_out[i])+"\n")
    f.write("m2: "+str(m2_out[i])+"\n")
    f.write("s1_mag: "+str(s1_mag_out[i])+"\n")
    f.write("s1_theta: "+str(s1_theta_out[i])+"\n")
    f.write("s1_phi: "+str(s1_phi_out[i])+"\n")
    f.write("s2_mag: "+str(s2_mag_out[i])+"\n")
    f.write("s2_theta: "+str(s2_theta_out[i])+"\n")
    f.write("s2_phi: "+str(s2_phi_out[i])+"\n")
    f.write("dist_mpc: "+str(dist_mpc_out[i])+"\n")
    f.write("tc: "+str(tc_out[i])+"\n")
    f.write("phic: "+str(phic_out[i])+"\n")
    f.write("inclination: "+str(inclination_out[i])+"\n")
    f.write("polarization_angle: "+str(polarization_angle_out[i])+"\n")
    f.write("ra: "+str(ra_out[i])+"\n")
    f.write("dec: "+str(dec_out[i])+"\n")
    f.write("heterodyne_bins: 1001\n")
    f.write("log_likelihood: "+str(log_likelihood_out[i])+"\n")

    #f.write("n_dim: 11\n")
    #f.write("n_chains: 1000\n")
    #f.write("n_loop_training: 20\n")
    #f.write("n_loop_production: 20\n")
    #f.write("n_local_steps: 200\n")
    #f.write("n_global_steps: 200\n")
    #f.write("learning_rate: 0.001\n")
    #f.write("max_samples: 50000\n")
    #f.write("momentum: 0.9\n")
    #f.write("num_epochs: 240\n")
    #f.write("batch_size: 50000\n")
    #f.write("stepsize: 0.01\n")

    f.close()