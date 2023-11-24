import numpy as np
import matplotlib.pyplot as plt

def analytical_reflections(v, z, x):
    Tint = 2.0 * z / v[:-1]
    Vrms = np.zeros(len(z))
    reflections = np.zeros((len(z), len(x)))
    for i in range(len(z)):
        Vrms[i] = np.sqrt(np.sum(v[:i+1]**2 * Tint[:i+1]) / np.sum(Tint[:i+1]))
        reflections[i] = np.sqrt(x**2.0 + 4.0*np.sum(z[:i+1])**2) / Vrms[i]
    return Vrms, reflections


total_time = 3.0         # seconds
n_receivers = 161        # units
spread_length = 4000     # meters 
max_model_depth = 2000   # meters

zint = np.array([500, 1000])         # depth [m]
vint = np.array([1500, 2000, 3000])  # vp [m/s]

x = np.linspace(0, spread_length, n_receivers)

direct_wave = x / vint[0]
vrms, reflections = analytical_reflections(vint, zint, x)

print(vrms)

depth = np.arange(max_model_depth)

Vint = vint[0]*np.ones(len(depth))
for i in range(len(zint)):
    Vint[int(np.sum(zint[:i+1])):] = vint[i+1]

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 8))

ax[0].plot(Vint, depth)

ax[0].set_ylim([0,max_model_depth])
ax[0].set_xticks(np.linspace(np.min(vint), np.max(vint), 5))
ax[0].set_xticks(np.linspace(np.min(vint), np.max(vint), 5, dtype = int))

ax[0].invert_yaxis()

ax[1].plot(x, direct_wave, ".", label = "Direct wave")    

for layer in range(len(zint)):        
    ax[1].plot(x, reflections[layer], ".", label = f"reflection of layer {layer+1}")

ax[1].set_title("CMP Gather", fontsize = 18)
ax[1].set_xlabel("Offset [m]", fontsize = 15)
ax[1].set_ylabel("TWT [s]", fontsize = 15)

ax[1].set_xticks(np.linspace(0, spread_length, 5))
ax[1].set_xticklabels(np.linspace(0, spread_length, 5, dtype = int))

ax[1].set_xlim([0, spread_length])
ax[1].set_ylim([0, total_time])
ax[1].invert_yaxis()

ax[1].legend(fontsize = 12, loc = "lower right")

plt.tight_layout()
plt.show()
