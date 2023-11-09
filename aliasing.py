import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
     
tmin = 0
tmax = 1
nTime = int(1e6)

contTime = np.linspace(tmin,tmax,nTime)

sineFreqs = [5, 15, 35]

contSine = np.zeros(nTime)
for sine in range(len(sineFreqs)):
    contSine += np.sin(2.0 * np.pi * sineFreqs[sine] * contTime)  

amsFreq = 120
deltaFilter = np.arange(tmin, tmax, 1.0/amsFreq)  
ams = np.array(deltaFilter/(contTime[1] - contTime[0]), dtype=int)

discTime = np.arange(len(ams)) * 1.0/amsFreq
discSine = contSine[ams]

specDisctSine = np.fft.fft(discSine)
discSineFreqs = np.fft.fftfreq(len(discSine),1.0/amsFreq)

#-------------- Plot do sinal ------------------------------#

plt.figure(1,figsize=(15,10))
plt.subplot(311)
plt.plot(contTime,contSine)
plt.stem(deltaFilter,contSine[ams],'k')
plt.xlabel("Tempo [s]")
plt.title("Sinal de entrada")

plt.subplot(312)
plt.plot(discTime,discSine)
plt.xlabel("Tempo [s]")
plt.title("Sinal amostrado")

plt.subplot(313)
plt.stem(discSineFreqs,np.abs(specDisctSine))
plt.xlim([0,amsFreq/2])
plt.xlabel("Frequencia [Hz]")
plt.title("Espectro do sinal amostrado")

plt.tight_layout()
plt.show()
