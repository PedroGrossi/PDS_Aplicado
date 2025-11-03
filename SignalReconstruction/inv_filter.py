import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

N = 256
n= np.arange(N)
b = [1, -.5]
a = [1, +.5]
# M = 3
# b = np.ones(M + 1) / ( M + 1)
# a = 1
x = np.random.rand(N)
sigma_noise = .1
noise = np.sqrt(sigma_noise) * np.random.rand(N)
y = ss.lfilter(b, a, x) + noise
xh = ss.lfilter(a, b, y)


w, H = ss.freqz(b, a)
w, Hi = ss.freqz(a, b)

plt.figure()
plt.plot(n, x, label="input")
plt.plot(n, y, label="output")
plt.plot(n, xh, label="inverted input")
print(np.mean(np.abs(x - xh)**2))
plt.legend()

plt.figure()
plt.plot(w / np.pi, np.abs(H), label="Forward Filter")
plt.plot(w / np.pi, np.abs(Hi), label="Inverse Filter")
plt.legend()
plt.show(block=True)