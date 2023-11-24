import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.interpolate import interp1d

n_dim = 15

index = 0
data = np.load("./zw_test_batch_out/injection_"+str(index)+".npz",  allow_pickle=True)
#data = np.load("../../../data_storage/10_pv2_1109/injection_"+str(index)+".npz",  allow_pickle=True)
#data = np.load("./experiment_out/injection_"+str(index)+".npz",  allow_pickle=True)
#data = np.load("./GW150914_Pv2"+".npz")
chains = data["chains"]
log_prob = data["log_prob"]
global_accs = data["global_accs"]
print(global_accs)
print("mean:", np.mean(global_accs))
#print(chains.shape)
#mask = (log_prob >-5000.)
#mask = (log_prob >-5000.)
#print(sum(mask))
#chains = chains[mask]
#print(data.keys())
#print(chains[0])
#print(chains[1])
#print(chains[88])



chains = chains.reshape(-1, n_dim)
#print(chains.shape)
# log_prob_raw = data["log_prob"].flatten()
# nancheck = np.isnan(log_prob_raw)
# for i in range(len(log_prob_raw)):
#     if nancheck[i] == True:
#         log_prob_raw[i] = 0
# log_prob = log_prob = np.array([log_prob_raw]).T

#s1_theta = chains[:,2]
#s1_phi = chains[:,3]
#s1_mag = chains[:,4]
#s2_theta = chains[:,5]
#s2_phi = chains[:,6]
#s2_mag = chains[:,7]



# s1x, s1y, and s1z
#chains[:, 2] = s1_mag * np.sin(s1_theta) * np.cos(s1_phi)
#chains[:, 3] = s1_mag * np.sin(s1_theta) * np.sin(s1_phi)
#chains[:, 4] = s1_mag * np.cos(s1_theta) 


# s2x, s2y, and s2z
#chains[:, 5] = s2_mag * np.sin(s2_theta) * np.cos(s2_phi)
#chains[:, 6] = s2_mag * np.sin(s2_theta) * np.sin(s2_phi)
#chains[:, 7] = s2_mag * np.cos(s2_theta) 

#chains[:, 11] = np.arccos(np.arcsin(np.sin(chains[:, 11]/2*np.pi))*2/np.pi)
#chains[:, 14] = np.arcsin(np.arcsin(np.sin(chains[:, 14]/2*np.pi))*2/np.pi)
#########################
# chains = np.concatenate((chains, log_prob), axis=1)
chains = chains[::100]
truths = data["true_param"]
truths[11] = np.cos(truths[11])
truths[14] = np.sin(truths[14])
#Mc = truths_dict["Mc"]
#eta = truths_dict["eta"]
#s1x = truths_dict["s1x"]
#s1y = truths_dict["s1y"]
#s1z = truths_dict["s1z"]
#s2x = truths_dict["s2x"]
#s2y = truths_dict["s2y"]
#s2z = truths_dict["s2z"]
#d_L = truths_dict["d_L"]
#t_c = truths_dict["t_c"]
#phase_c = truths_dict["phase_c"]
#ota = truths_dict["iota"]
#psi = truths_dict["psi"]
#ra = truths_dict["ra"]
#dec = truths_dict["dec"]
#truth_array = [truths_dict['']]
#q_axis = np.linspace(0.1,1,100)
#eta = q_axis/(1+q_axis)**2
#q_interp = interp1d(eta,q_axis)
#truths[1] = q_interp(truths[1])

#print("truths: ", truths)
figure = corner.corner(
    chains,
    truths = truths,
    
    labels=[
        "$M_c$",
        "$q$",
        r"$s_{1\theta}$",
        r"$s_{1\phi}$",
        "$s_{1mag}$",
        r"$s_{2\theta}$",
        r"$s_{2\phi}$",
        "$s_{2mag}$",
        "$D$",
        "$t_c$",
        "$\phi_c$",
        "$\iota$",
        "$\Psi$",
        "RA",
        "DEC",
        "ppe",
        "log L",
        "m1",
        "m2",
        "Lambda_eff",
        "chi_eff",
#         "Log likelihood"
    ],
    smooth=True,
    show_titles=True,
)
#figure.savefig("./phenomp_corner.png")
#figure.savefig("./Injection_plots/injection_"+str(index)+'_experiment_corner.png')
figure.savefig("./injection_"+str(index)+'_experiment_corner.png')
#figure.savefig("./GW150914_Pv2"+'_corner.png')