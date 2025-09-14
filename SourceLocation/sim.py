import numpy as np
import scipy.signal as ss
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from tqdm import tqdm
import sounddevice as sd

Lx = 10 # [m] Width
Ly = 10 # [m] Depth
dx = .05 # Discretização
dy = .05
Nx = round(Lx/dx)
Ny = round(Ly/dy)

Lt = .05 # Simulation time [s]
dt = 1e-4
Nt = round(Lt/dt)
fs = 1 / dt

# Source signal
f0 = 400 # [Hz]
if f0 > fs / 2:
    raise ValueError("f0 > fs / 2")

t = np.arange(Nt)*dt
bw = .5
t0 = 1.5/(bw*f0)
s = ss.gausspulse(t - t0, f0, bw)
# sd.play(s, fs)
# s *= dt**2 / (dx*dy) 

# plt.figure()
# plt.plot(t,s)
# plt.show()

m = np.zeros((Nt, Nx), dtype=np.float32)

p_0 = np.zeros((Ny, Nx), dtype=np.float32)
p_1 = np.zeros((Ny, Nx), dtype=np.float32)
p_2 = np.zeros((Ny, Nx), dtype=np.float32)

alpha = 10
a1 = 4 / (2 + alpha * dt)
a2 = - (2 - alpha * dt) / (2 + alpha * dt)

c0 = 340
c2 = c0**2 * (dt**2 / dx**2) * np.ones((Ny, Nx), dtype=np.float32)

C = np.sqrt(np.max(c2))
print(f"Courant number = {C}")
if C > 1:
    raise ValueError("C > 1")


h = np.array([[0 ,1, 0], [1, -4, 1], [0, 1, 0]])

view = 1
if view:
    vmax = .1
    vmin = -vmax
    im = plt.imshow(p_0, vmin=vmin, vmax=vmax, cmap="bwr")

xs = np.random.randint(1, Nx-2)
ys = np.random.randint(1, Ny-2)

for nt in tqdm(range(Nt)):
    p_1, p_2 = p_0, p_1
    p_0 = a1 * p_1 + a2 * p_2 + c2 * ndi.convolve(p_1,h)
    p_0[xs, ys] += s[nt]
    m[nt] = p_0[1] + .001*np.random.randn()
    
    if view:
        if not nt % view:
            im.set_data(p_0)
            plt.pause(1e-12)

#%%
nm = 52
# sd.play(m[:, -nm], fs, blocking=True)
plt.figure()
plt.plot(t, s, t, m[:, -1], t, m[:, -nm])
plt.show()

plt.figure()
plt.imshow(m, aspect="auto")
plt.show()