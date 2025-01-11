# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-11-30 19:26:35
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-01 19:11:47
import numpy as np
import matplotlib.pyplot as plt

fs = 1000 # Sampling frequency
t = np.arange(0, 1.0, 1/fs) # Time vector
x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t) # Example non-periodic signal

window = np.hanning(len(x))
windowed_signal = x * window

Y = np.fft.fft(windowed_signal)
freqs = np.fft.fftfreq(len(Y), 1/fs)

plt.loglog(freqs, np.abs(Y))
plt.xlim(0, fs/2) # Show only positive frequencies
plt.title('FFT of Windowed Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()