import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.sparse.linalg as la

f = plt.imread("images/cameraman.pgm") / 255

Mf, Nf = f.shape
Nb = 9
b = np.ones([Nb, Nb]) / Nb ** 2
def B(f):
    return ndi.convolve(f, b)

def BT(f):
    return ndi.correlate(f, b)

def BTB(f):
    return BT(B(f.reshape(Mf, Nf))).reshape(Mf*Nf, 1)

SNR = 60
# Pf = np.mean(f ** 2)
# sw = np.sqrt(Pf / (10 ** (SNR/10)))
# w = sw * np.random.randn(Mf, Nf)

# SNR_ = 10*np.log10(Pf / np.mean(w ** 2))
# print(SNR_)

g_clean = B(f)
Pf = np.mean(g_clean ** 2)
sw = np.sqrt(Pf / (10 ** (SNR/10)))
w = sw * np.random.randn(Mf, Nf)
g = g_clean + w # blured

A = la.LinearOperator((Mf*Nf, Mf*Nf), matvec=BTB, rmatvec=BTB)
fh = la.cg(A, BT(g, ).reshape(Mf*Nf, 1), x0=g.reshape(Mf*Nf, 1))[0].reshape(Mf, Nf)

plt.figure()
plt.imshow(f + w, cmap="gray")

plt.figure()
plt.imshow(g, cmap="gray")

plt.figure()
plt.imshow(fh, cmap="gray")
plt.show()