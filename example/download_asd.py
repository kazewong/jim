import numpy as np
import requests

psd_file_dict= {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}
for name in psd_file_dict.keys():
    print(name)
    print("Grabbing GWTC-2 PSD for "+name)
    url = psd_file_dict[name]
    data = requests.get(url)
    print(data)
    open(name+".txt", "wb").write(data.content)
    f_ads_vals = np.loadtxt(name+".txt", unpack=True)
    print(f_ads_vals)
