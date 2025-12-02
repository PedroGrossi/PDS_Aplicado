import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.linalg as la

N = 256
n= np.arange(N)
b = [1, -.5]
a = [1, +.5]
x = np.random.randn(N)
sigma_noise = .1
noise = np.sqrt(sigma_noise) * np.random.rand(N)
y = ss.lfilter(b, a, x)

delta = np.zeros(N)
delta[0] = 1
h = ss.lfilter(b, a, delta)

A = la.convolution_matrix(h, N)[:N, :]
ym = A @ x

z1 = A.T @ x
z2 = ss.lfilter(b, a, x[::-1])[::-1]
print(np.mean(np.abs(z1 - z2)**2))

xh = ss.lfilter(a, b, y)
Ainv = la.inv(A)
xhm = Ainv @ ym

w, H = ss.freqz(b, a)
w, Hi = ss.freqz(a, b)

plt.figure()
plt.plot(n, x, label="input")
plt.plot(n, y, label="output")
plt.plot(n, xh, label="inverted input")
# plt.plot(n, xhm, label="inverted input via matrix")
# plt.plot(n, ym, label="output via matrix")
# print(np.mean(np.abs(x - xh)**2))
# print(np.mean(np.abs(y - ym)**2))
# print(np.mean(np.abs(x - xhm)**2))
plt.legend()

plt.figure()
plt.plot(w / np.pi, np.abs(H), label="Forward Filter")
plt.plot(w / np.pi, np.abs(Hi), label="Inverse Filter")
plt.legend()
plt.show(block=True)