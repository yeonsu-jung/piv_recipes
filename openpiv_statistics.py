# %%
import os
import pandas as pd
import numpy as np
import time
from scipy import optimize
from matplotlib import pyplot as plt
# %%
plt.style.use('default')
plt.rcParams.update({'font.family': 'sans-serif','text.usetex': 'true'})
plt.rcParams["figure.figsize"] = (2.2,1.8)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = 'false'
#plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 2})

# %%
def piv_statistics(u_array, v_array):
    mean_u = np.mean(u_array)
    mean_v = np.mean(v_array)
    mean_U = np.sqrt(mean_u**2 + mean_v**2)
    
    deficit_u = u_array - mean_u
    deficit_v = v_array - mean_v

    deficit_vector = np.array([deficit_u, deficit_v])
    unit_streamwise = np.array([mean_u/mean_U,mean_v/mean_U])
    rotation_matrix = np.array([[0,-1],[1,0]])
    unit_normal = np.matmul(rotation_matrix,unit_streamwise.T)

    u_prime = np.zeros(len(u_array))
    v_prime = np.zeros(len(u_array))

    for i in range(len(u_array)):
        deficit_vector = np.array([deficit_u[i], deficit_v[i]])
        u_prime[i] = np.sum(deficit_vector*unit_streamwise)
        v_prime[i] = np.sum(deficit_vector*unit_normal)

    return u_prime, v_prime

# %%
folder_path = 'Tables/'
path_list = [f for f in os.listdir(folder_path) if not f.startswith('.')]
j = 0
mean_U_array = np.zeros(len(path_list))
std_U_array = np.zeros(len(path_list))

for exp_id in path_list:
    df = pd.read_csv('Tables/' + exp_id,names=['x','y','u','v'],delimiter=' ')
    if df.isnull().values.any():
        print(df)
        continue

    u_array = df.u.to_numpy()
    v_array = df.v.to_numpy()
    U_array = np.sqrt(u_array**2 + v_array**2)

    array_size = u_array.size
    column_length = int(array_size/100)    

    uu = np.mean(u_array.reshape(100,column_length),axis=1)
    mean_U_array[j] = np.mean(uu)
    std_U_array[j] = np.std(uu)
    j = j + 1

    plt.plot(uu)
    plt.plot([0,100],[np.mean(uu),np.mean(uu)])
    plt.xlabel('Image pair')
    plt.ylabel('U (mm/s)')
    plt.title('U = %.2f mm/s' %np.mean(uu))
    plt.savefig('Figures/U_fluc_%.2f.png' %np.mean(uu))
    plt.show()
# %%
galmin_to_m3s = 6.30902e-5
area = 0.2*0.193

def linfunc(x,a,b):
    return a*x+b

flow_rate_array = np.array([52,108,163,218,274,330,385,443,493,537,581,623])
avg_U_array = flow_rate_array*galmin_to_m3s/area*1000
plt.errorbar(avg_U_array,np.sort(mean_U_array),std_U_array[np.argsort(mean_U_array)],fmt='o',capsize=5)
popt, pcov = optimize.curve_fit(linfunc,avg_U_array[0:-3],np.sort(mean_U_array)[0:-3],p0=(0.6,-1.76))


xx = np.linspace(0,1000,100)
plt.plot(xx,linfunc(xx,*popt),'k-')
plt.xlabel('$U$ from flowmeter, mm/s')
plt.ylabel('$U$ from PIV, mm/s')
plt.title('$y = %.2f x %.2f$' %(popt[0],popt[1]))
plt.savefig('Figures/Fitting.png',bbox_inches = 'tight',pad_inches = 0.1)

# %%
ti_array_pt1 = np.zeros(100)
ti_array_pt2 = np.zeros(100)

path_list = [f for f in os.listdir(folder_path) if not f.startswith('.')]
path_list.remove('27_500_44.12.txt')
path_list.remove('30_500_44.12.txt')
path_list.remove('36_500_44.12.txt')

for exp_id in path_list:
    df = pd.read_csv('Tables/' + exp_id,names=['x','y','u','v'],delimiter=' ')
    if df.isnull().values.any():
        print(df)
        continue

    array_size = u_array.size

    column_length = int(array_size/100)

    u_array = df.u.to_numpy().reshape(column_length,100)
    v_array = df.v.to_numpy().reshape(column_length,100)
    U = np.mean(np.sqrt(u_array**2 + v_array**2))

    u_prime, v_prime = piv_statistics(u_array[:,1],v_array[:,1])
    for i in range(99):
        temp1, temp2 = piv_statistics(u_array[:,i+1],v_array[:,i+1])
        u_prime = np.hstack((u_prime,temp1))
        v_prime = np.hstack((v_prime,temp2))

    u_prime = u_prime.reshape(column_length,100)
    v_prime = v_prime.reshape(column_length,100)

    ti_array_pt1 = np.sqrt(0.5*(u_prime[254,:]**2 + v_prime[254,:]**2))/U
    ti_array_pt2 = np.sqrt(0.5*(u_prime[3,:]**2 + v_prime[3,:]**2))/U

    # plt.plot(ti_array_pt1)
    plt.plot(ti_array_pt2)
    plt.axis([0,100,0,0.2])
    plt.xlabel('Image pair number')
    plt.ylabel('Turbulence intensity')

    # (n_u,bins,patches) = plt.hist(u_prime,bins=100,alpha = 0.3,label='$u\'$')
    # (n_v,bins,patches) = plt.hist(v_prime,bins=100,alpha = 0.3,label='$v\'$')    
    # plt.title('U = %.2f mm/s' %U)
    # plt.xlabel('Velocity fluctuation ($\mathbf{u} - \\bar{\\mathbf{u}}$), mm/s')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.savefig('Figures/vel_fluc_hist_%s.png' %exp_id,bbox_inches = 'tight',pad_inches = 0.1)
    # plt.show()