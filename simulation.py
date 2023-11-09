import numpy as np
import matplotlib.pyplot as plt

def analytical_refractions(v, z, x):
    refractions = np.zeros((len(z), len(x)))
    for n in range(len(z)):
        refractions[n] += x / v[n+1]
        for i in range(n+1):
            angle = np.arcsin(v[i] / v[n+1])            
            refractions[n] += 2.0*z[i]*np.cos(angle) / v[i]
    return refractions

def analytical_reflections(v, z, x):
    Tint = 2.0 * z / v[:-1]
    Vrms = np.zeros(len(z))
    reflections = np.zeros((len(z), len(x)))
    for i in range(len(z)):
        Vrms[i] = np.sqrt(np.sum(v[:i+1]**2 * Tint[:i+1]) / np.sum(Tint[:i+1]))
        reflections[i] = np.sqrt(x**2.0 + 4.0*np.sum(z[:i+1])**2) / Vrms[i]
    return reflections

n_receivers = 101
spread_length = 1000
source_position = 0

total_time = 1.0

z = np.array([20, 80, 200, 100])
v = np.array([700, 2500, 3000, 3500, 4000])

nx = 1601
nz = 241
dh = 2.5

vp = v[0]*np.ones((nz, nx))
for i in range(len(z)):
    vp[int(np.sum(z[:i+1]) / dh):] = v[i+1]

offset_min = np.abs(source_position) - 0.5*spread_length 
offset_max = np.abs(source_position) + 0.5*spread_length

if source_position > 0:
    offset_aux = offset_min
    offset_min = offset_max
    offset_max = offset_aux

x = np.linspace(offset_min, offset_max, n_receivers)

x_rec = np.linspace(0.5*((nx-1)*dh - spread_length), 0.5*((nx-1)*dh + spread_length), n_receivers) / dh 
z_rec = np.zeros_like(x)

x_src = (0.5*((nx-1)*dh) + source_position) / dh
z_src = 0

xloc = np.linspace(0, (nx-1), 5)
xlab = np.linspace(0, (nx-1)*dh, 5)

zloc = np.linspace(0, (nz-1), 5)
zlab = np.linspace(0, (nz-1)*dh, 5)

fig, ax = plt.subplots(figsize = (13, 2))
ax.imshow(vp, aspect = "auto", cmap = "Greys")
ax.plot(x_rec, z_rec, "v")
ax.plot(x_src, z_src, "*")

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

ax.set_yticks(zloc)
ax.set_yticklabels(zlab)

plt.tight_layout()

direct_wave = np.abs(x) / v[0]
reflections = analytical_reflections(v, z, np.abs(x))
refractions = analytical_refractions(v, z, np.abs(x))

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 6))

xticks = np.linspace(offset_min, offset_max, 5)

ax1.plot(x, direct_wave, ".")    
ax2.plot(x, direct_wave, ".")    

ax1.set_title("Reflections", fontsize = 18)
ax2.set_title("Refractions", fontsize = 18)

for layer in range(len(z)):    
    
    show_reflections = reflections[layer] > direct_wave
    show_refractions = refractions[layer] < direct_wave

    ax1.plot(x, reflections[layer], ".", label = f"reflection of layer {layer+1}")
    ax2.plot(x[show_refractions], refractions[layer, show_refractions], ".", label = f"refraction of layer {layer+1}")

ax1.set_xticks(xticks)
ax1.set_xticklabels(xticks)

ax2.set_xticks(xticks)
ax2.set_xticklabels(xticks)

ax1.set_xlabel("Offset [m]", fontsize = 15)
ax1.set_ylabel("TWT [s]", fontsize = 15)

ax2.set_xlabel("Offset [m]", fontsize = 15)
ax2.set_ylabel("TWT [s]", fontsize = 15)

ax1.set_xlim([offset_min, offset_max])
ax1.set_ylim([0, total_time])
ax1.invert_yaxis()

ax2.set_xlim([offset_min, offset_max])
ax2.set_ylim([0, total_time])
ax2.invert_yaxis()

ax1.legend(fontsize = 12, loc = "lower right")
ax2.legend(fontsize = 12, loc = "lower right")

plt.tight_layout()
plt.show()
