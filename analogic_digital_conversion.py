import numpy as np
import matplotlib.pyplot as plt

sine_frequencies = [5, 20, 35]

sampling_frequency = 80
sampling_period = 1.0/sampling_frequency

analogic_time = np.linspace(0, 1, int(1e6))
analogic_sine = np.zeros_like(analogic_time)

for frequency in sine_frequencies:
    analogic_sine += np.sin(2.0 * np.pi * frequency * analogic_time)  

comb_function = np.arange(0, 1.0, sampling_period)

samples = np.array(comb_function / (analogic_time[1] - analogic_time[0]), dtype=int)

digital_time = np.arange(len(samples)) * sampling_period
digital_sine = analogic_sine[samples]

digital_sine_spectra = np.fft.fft(digital_sine)

frequency = np.fft.fftfreq(len(digital_sine), sampling_period)

fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize = (15, 9))

ax[0].plot(analogic_time, analogic_sine)
ax[0].stem(comb_function, analogic_sine[samples], markerfmt = "k", linefmt = "k--", basefmt = "k")
ax[0].set_title("Analogic - Digital Conversion", fontsize = 18)
ax[0].set_xlabel("Time [s]", fontsize = 15)
ax[0].set_ylabel(r"$x(t)$", fontsize = 15)
ax[0].set_xlim([0,1])

ax[1].plot(digital_sine)
ax[1].set_xlim([0,len(digital_time)])
ax[1].set_title("Discrete Signal", fontsize = 18)
ax[1].set_xlabel(r"$n$", fontsize = 15)
ax[1].set_ylabel(r"$x[n]$", fontsize = 15)

ax[2].stem(frequency, np.abs(digital_sine_spectra))
ax[2].set_xlim([0,len(digital_sine_spectra)/2 - 1])
ax[2].set_xticks(np.linspace(0, 0.5*sampling_frequency, 21))
ax[2].set_xticklabels(np.linspace(0, 0.5*sampling_frequency, 21, dtype = int))
ax[2].set_title("Discrete Fourier Transform - FFT", fontsize = 18)
ax[2].set_xlabel("Frequency [Hz]", fontsize = 15)
ax[2].set_ylabel(r"$X(jw)$", fontsize = 15)

plt.tight_layout()

plt.save("analogic_digital_conversion.png", dpi = 200)
plt.show()