import jax.numpy as jnp 
import numpy as np
import jax
import pandas as pd
import time
from jimgw.generate_noise import *
from jimgw.detector import H1
from tqdm import tqdm

theta_extrinsic = jnp.array([440.0,0,0])
param = {'ra':0,'dec':0, 'psi':0, 'gmst': 1126259462}


ifos = 'H1'
catalog_list = []
for i in range(1):
    for j in range(0, 1):
        for k in range(0, 10): 
            for l in range(1, 10):
                catalog_list.append(str(i)+str(j)+str(k)+str(l))

f = jax.jit(H1.fd_response)

iota = 0
with tqdm(total=len(catalog_list)) as pbar:
    for item in catalog_list:

        path = '/mnt/home/averhaeghe/ceph/NR_waveforms/NR_'+str(item)+'.txt'
        data = np.genfromtxt(path)
        freq = data[:,0]
        NR = {}
        NR['p'] = data[:,1] * (1 / 2 * (1 + jnp.cos(iota) ** 2)) * jnp.exp(4j*freq)
        NR['c'] =  data[:,2] * jnp.cos(iota)  * jnp.exp(4j*freq)
        noise = generate_noise_freq(0, freqs = freq, psd_funcs = generate_LVK_PSDdict())
        noise_fd_H1 = noise[2]['H1']

        signal_fd_H1 = f(freq, NR, param) + noise_fd_H1

        df = pd.DataFrame([])
        df.insert(len(df.columns), "frequency", freq)
        df.insert(len(df.columns), "NR_signal+noise", signal_fd_H1)
        df.insert(len(df.columns), "NR_noise", noise_fd_H1)
        np.savetxt('/mnt/home/averhaeghe/ceph/NR_waveforms_noisy/NR_'+str(item)+'.txt', df.values)
        #time.sleep(5)
        pbar.update(1)