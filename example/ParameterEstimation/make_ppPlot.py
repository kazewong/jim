import numpy as np
from scipy.optimize import minimize

def get_all_quantile(filename):
    data = np.load(filename)
    
    chains = data['chains']
    true_param = data['true_param']
    chains[:,:,1] = chains[:,:,1]/(1+chains[:,:,1])**2
    chains[:,:,7] = np.arccos(chains[:,:,7])
    chains[:,:,10] = np.arcsin(chains[:,:,10])
    
    median = np.log10(0.5)
    def compute_percentile(value,data):
        f = lambda x : np.abs(np.quantile(data, 10**x) - value)
        result = minimize(f,median,method="Nelder-Mead",bounds=[[-4,0]])
        return np.abs(10**result.x[0]-0.5)*2
    
    result = [] 
    for i in range(11):
        result.append(compute_percentile(true_param[i],chains[:,:,i]))

    mean_local_accs = data['local_accs'].mean()
    mean_global_accs = data['global_accs'].mean()

    return np.array(result), true_param, mean_global_accs, mean_local_accs

directory = '/mnt/home/wwong/ceph/GWProject/JaxGW/RealtimePE/ppPlots/balance_1001/'
result = []
true_param = []
mean_global_accs = []
mean_local_accs = []
for i in range(960):
    name = directory+'injection_'+str(i)+'.npz'
    local_result = get_all_quantile(name)
    result.append(local_result[0])
    true_param.append(local_result[1])
    mean_global_accs.append(local_result[2])
    mean_local_accs.append(local_result[3])

result = np.stack(result)
true_param = np.stack(true_param)
mean_global_accs = np.stack(mean_global_accs)
mean_local_accs = np.stack(mean_local_accs)

np.savez('/mnt/home/wwong/ceph/GWProject/JaxGW/RealtimePE/ppPlots/combined_quantile_balance_1001',result=result, true_param=true_param, mean_global_accs=mean_global_accs, mean_local_accs= mean_local_accs)
