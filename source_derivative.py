import numpy as np
import matplotlib.pyplot as plt

nx = 501
nz = 501

dh = 10.

a = 0.5
vi = 2000 

V = np.zeros((nz,nx))
T = np.zeros((nz,nx))
D = np.zeros((nz,nx))

z = np.arange(nz)*dh
x = np.arange(nx)*dh

xi = 0
zi = 0

for i in range(nz):
    V[i] = vi + a*z[i]

for i in range(nz):
    if z[i] != zi:
        for j in range(nx):
            T[i,j] = (1/a)*np.arccosh(((a*(z[i]-zi))**2*(((x[j] - xi)/(z[i]-zi))**2 + 1))/(2*vi*(a*(z[i]-zi) + vi)) + 1)        

for i in range(nz):
    if z[i] != zi:
        for j in range(nx):
            D[i,j] = -(a*(z[i]-zi) + 2*vi) * np.sqrt((a**2*((x[j]-xi)**2 + (z[i]-zi)**2)) / (a**2*(x[j]-xi)**2 + (a*(z[i]-zi) + 2*vi)**2)) / (vi*(a*(z[i]-zi) + vi))

plt.figure(figsize = (10,5))
plt.imshow(T, aspect = "auto", cmap = "jet")
plt.colorbar()
plt.tight_layout()
plt.show()




