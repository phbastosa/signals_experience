import numpy as np
import matplotlib.pyplot as plt

nt = 5001
dt = 1e-3

nx = 161
dx = 25.0

vi = 1000
vf = 5000
dv = 100

filename = f"cmp_gather_{nt}x{nx}_{dt*1e6:.0f}us.bin"

seismic = np.fromfile(filename, dtype = np.float32, count = nt*nx)
seismic = np.reshape(seismic, [nt,nx], order = "F")

vrms = np.arange(vi, vf + dv, dv)

offset = np.arange(nx)*dx

nv = len(vrms)

semblance = np.zeros((nt, nv))


for t in range(nt):
    for v in range(nv):
        
        target = np.array(np.sqrt((t*dt)**2 + (offset/vrms[v])**2) / dt, dtype = int) 
    
        mask = target < nt
        
        # get better correlation equation here 
        semblance[t, v] = np.sum(seismic[target[mask],:]**2) / nx


xloc = np.linspace(0, nv-1, 9)
xlab = np.linspace(vi, vf, 9)

scale = 5.0*np.std(semblance)

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (6,8))

ax.imshow(semblance, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

fig.tight_layout()
plt.show()