import numpy as np

prior_range = np.array([[20,50],[20,50],[-0.5,0.5],[-0.5,0.5],[400,1000],[-2,2],[-np.pi/2,np.pi/2],[-np.pi/2,np.pi/2],[0,2*np.pi],[0,2*np.pi],[0,np.pi]])

N_config = 10

m1 = np.random.uniform(prior_range[0,0],prior_range[0,1],N_config)
m2 = np.random.uniform(prior_range[1,0],prior_range[1,1],N_config)
m2,m1 = np.sort([m1,m2],axis=0)
chi1 = np.random.uniform(prior_range[2,0],prior_range[2,1],N_config)
chi2 = np.random.uniform(prior_range[3,0],prior_range[3,1],N_config)
dist_mpc = np.random.uniform(prior_range[4,0],prior_range[4,1],N_config)
tc = np.random.uniform(prior_range[5,0],prior_range[5,1],N_config)
phic = np.random.uniform(prior_range[6,0],prior_range[6,1],N_config)
inclination = np.random.uniform(prior_range[7,0],prior_range[7,1],N_config)
polarization_angle = np.random.uniform(prior_range[8,0],prior_range[8,1],N_config)
ra = np.random.uniform(prior_range[9,0],prior_range[9,1],N_config)
dec = np.random.uniform(prior_range[10,0],prior_range[10,1],N_config)

directory = '/mnt/home/wwong/ceph/GWProject/JaxGW/RealtimePE/ppPlots/configs/'

for i in range(N_config):
    f = open(directory+"injection_config_"+str(i)+".yaml","w")
    
    f.write('output_path: /mnt/home/wwong/ceph/GWProject/JaxGW/RealtimePE/ppPlots/injection_'+str(i)+'\n')
    f.write('downsample_factor: 10\n')
    f.write('seed: '+str(np.random.randint(low=0,high=10000))+'\n')
    f.write('f_sampling: 2048\n')
    f.write('duration: 4\n')
    f.write('fmin: 30\n')
    f.write('ifos:\n')
    f.write('  - H1\n')
    f.write('  - L1\n')

    f.write("m1: "+str(m1[i])+"\n")
    f.write("m2: "+str(m2[i])+"\n")
    f.write("chi1: "+str(chi1[i])+"\n")
    f.write("chi2: "+str(chi2[i])+"\n")
    f.write("dist_mpc: "+str(dist_mpc[i])+"\n")
    f.write("tc: "+str(tc[i])+"\n")
    f.write("phic: "+str(phic[i])+"\n")
    f.write("inclination: "+str(inclination[i])+"\n")
    f.write("polarization_angle: "+str(polarization_angle[i])+"\n")
    f.write("ra: "+str(ra[i])+"\n")
    f.write("dec: "+str(dec[i])+"\n")

    f.write("n_dim: 11\n")
    f.write("n_chains: 1000\n")
    f.write("n_loop: 10\n")
    f.write("n_local_steps: 1000\n")
    f.write("n_global_steps: 1000\n")
    f.write("learning_rate: 0.001\n")
    f.write("max_samples: 50000\n")
    f.write("momentum: 0.9\n")
    f.write("num_epochs: 300\n")
    f.write("batch_size: 50000\n")
    f.write("stepsize: 0.01\n")

    f.close()