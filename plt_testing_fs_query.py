# %%
import falkner_skan
reload(falkner_skan)
from falkner_skan import falkner_skan


# %%
eta,f0,f1,f2= falkner_skan(-0.01,7)
plt.plot(eta,f1)

eta,f0,f1,f2= falkner_skan(0.01,7)
plt.plot(eta,f1)

eta,f0,f1,f2= falkner_skan(0.05,7)
plt.plot(eta,f1)

eta,f0,f1,f2= falkner_skan(0.2,7)
plt.plot(eta,f1)

# %%
def fs_query(eta_q,delta_eta,m):
    eta,f0,f1,f2 = falkner_skan(m)
    return np.interp(eta_q,eta-delta_eta,f1)
# %%

# %%
xtmp = np.linspace(0,7,15)
ytmp = 1 - np.exp(-xtmp)

plt.plot(xtmp,ytmp,'o-')
# %%
from scipy.optimize import curve_fit
popt,pcov = curve_fit(fs_query,xtmp,ytmp)

# %%
plt.plot(xtmp,ytmp,'o-')
plt.plot(xtmp,fs_query(xtmp,*popt))

# %%
popt,pcov = curve_fit(fs_query,xtmp-0.2,ytmp)
# %%
plt.plot(xtmp-0.2,ytmp,'o-')
plt.plot(xtmp-0.2,fs_query(xtmp-0.2,*popt),'o-')

# %%
popt