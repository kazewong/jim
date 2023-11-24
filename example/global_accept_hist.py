import numpy as np
import matplotlib.pyplot as plt
import yaml

def get_rate(filename):
    data = np.load(filename)
    mean_global_accs = data['global_accs'].mean()
    return mean_global_accs

def get_likelihood(filename):
    with open(filename) as f:
        my_dict = yaml.safe_load(f)
    return my_dict['log_likelihood']

batch_size = 10
directory = '/home/zwang264/scr4_berti/zipeng/PPE_with_Jax/jim/example/zw_test_batch_out/'
#directory = '../../../data_storage/10_pv2_1109/'
mean_global_accs = []
for i in range(batch_size):
    name = directory+'injection_'+str(i)+'.npz'
    mean_global_accs.append(get_rate(name))
    print(mean_global_accs[-1])
directory = './configs/zw_test_batch/'
#directory = '../../../data_storage/100_pv2_1027/'
log_likelihoods=[]
for i in range(batch_size):
    name = directory+'injection_config_'+str(i)+'.yaml'
    log_likelihoods.append(get_likelihood(name))
print(np.argmin(mean_global_accs), np.min(mean_global_accs))
plt.figure()
bins_likelihood = np.linspace(0, 1500, 20)
bins_accs = np.linspace(0, 0.1, 10)
plt.scatter(log_likelihoods, mean_global_accs)#, bins=[bins_likelihood,bins_accs])
#plt.legend()
plt.savefig("./hist_log_likelihood.png",dpi=600)


