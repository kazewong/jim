import numpy as np
import matplotlib.pyplot as plt
import yaml
def get_rate(filename):
    data = np.load(filename)
    mean_global_accs = data['global_accs'].mean()
    return mean_global_accs

def get_loss(filename):
    data = np.load(filename)
    NF_loss = data['loss_vals'].reshape(-1)
    return NF_loss

directory = '../../../data_storage/10_pv2_1109/'
#mean_global_accs = []
#names = ["","_morechains","_moreepoch","_morelayers","_moretraining"]
#for i in range(5):

#    name = directory+'injection_4'+str(names[i])+'.npz'
    #mean_global_accs.append(get_rate(name))
#    print(names[i], get_rate(name))
name = directory+'injection_2'+'.npz'
NF_loss = get_loss(name)
plt.plot(NF_loss)
plt.yscale("log")   
#plt.xlim(120000,140000)
plt.savefig("loss.png", dpi=600)
