import numpy as np
import scipy.signal as ss
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from tqdm import tqdm
import sounddevice as sd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

Lx = 10 # [m] Width
Ly = 10 # [m] Depth
dx = .05 # Discretization
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
    # im = plt.imshow(p_0, vmin=vmin, vmax=vmax, cmap="bwr", extent=[0, Nx, 0, Ny], origin="lower")

xs1 = np.random.randint(1, Nx-2)
ys1 = np.random.randint(1, Ny-2)

while True:
    xs2 = np.random.randint(1, Nx-2)
    ys2 = np.random.randint(1, Ny-2)
    if (xs2 != xs1 or ys2 != ys1):
        break

for nt in tqdm(range(Nt)):
    p_1, p_2 = p_0, p_1
    p_0 = a1 * p_1 + a2 * p_2 + c2 * ndi.convolve(p_1,h)
    p_0[ys1, xs1] += s[nt]
    p_0[ys2, xs2] += s[nt]
    m[nt] = p_0[1] + .001*np.random.randn()
    
    # if view:
    #     if not nt % view:
            # im.set_data(p_0)
            # plt.pause(1e-12)

#%%
print("#########################################################")
# Source Founding

view = False

from scipy.optimize import differential_evolution

def emulate_source(xs1_emulated, ys1_emulated, xs2_emulated, ys2_emulated, s, Nt, Nx, Ny, a1, a2, c2, h, view=False):
    p_0 = np.zeros((Ny, Nx), dtype=np.float32)
    p_1 = np.zeros((Ny, Nx), dtype=np.float32)
    p_2 = np.zeros((Ny, Nx), dtype=np.float32)
    
    emulated_source = np.zeros((Nt, Nx), dtype=np.float32)

    if view:
        vmax = .1
        vmin = -vmax
        im = plt.imshow(p_0, vmin=vmin, vmax=vmax, cmap="bwr", extent=[0, Nx, 0, Ny], origin="lower")

    for nt in range(Nt):
        p_1, p_2 = p_0, p_1
        p_0 = a1 * p_1 + a2 * p_2 + c2 * ndi.convolve(p_1, h)
        p_0[ys1_emulated, xs1_emulated] += s[nt]
        p_0[ys2_emulated, xs2_emulated] += s[nt]
        emulated_source[nt] = p_0[1]

        if view and not nt % view:
            im.set_data(p_0)
            plt.pause(1e-12)

    return emulated_source

def emulate_source_single_sensor(xs1, ys1, xs2, ys2, Nt, Nx, Ny, a1, a2, c2, h, sensor_positions):
    p_0 = np.zeros((Ny, Nx), dtype=np.float32)
    p_1 = np.zeros((Ny, Nx), dtype=np.float32)
    p_2 = np.zeros((Ny, Nx), dtype=np.float32)
    
    Ns = len(sensor_positions)
    sig = np.zeros((Nt, Ns), dtype=np.float32)
    
    for nt in range(Nt):
        p_1, p_2 = p_0, p_1
        p_0 = a1 * p_1 + a2 * p_2 + c2 * ndi.convolve(p_1, h)
        p_0[ys1, xs1] += s[nt]
        p_0[ys2, xs2] += s[nt]
        
        # samples only on representative sensors
        for si, (y_s, x_s) in enumerate(sensor_positions):
            sig[nt, si] = p_0[y_s, x_s]
    return sig

# Normalization
def normalize_signals(sig):
    sig = sig.astype(np.float32)
    sig = sig - np.mean(sig, axis=0)
    norms = np.linalg.norm(sig, axis=0)
    norms[norms == 0] = 1.0
    sig = sig / norms
    return sig

def cost_function(pos):
    x1 = int(np.clip(int(round(pos[0])), 1, Nx-2))
    y1 = int(np.clip(int(round(pos[1])), 1, Ny-2))
    x2 = int(np.clip(int(round(pos[2])), 1, Nx-2))
    y2 = int(np.clip(int(round(pos[3])), 1, Ny-2))
    
    sinais_sim = emulate_source_single_sensor(x1, y1, x2, y2, Nt, Nx, Ny, a1, a2, c2, h, sensor_positions)
    sinais_sim_norm = normalize_signals(sinais_sim)
    
    error = np.linalg.norm(sinais_sim_norm - m_sensors_norm)
    return error

# Representative sensors to optimize simulation processing in the cost_function
sensor_positions = [(1, 1), (1, Nx//2), (1, Nx-2)]

# Select only the sensors of interest from the original vector
Ns = len(sensor_positions)
m_sensors = np.zeros((Nt, Ns), dtype=np.float32)
for si, (y_s, x_s) in enumerate(sensor_positions):
    m_sensors[:, si] = m[:, x_s]

m_sensors_norm = normalize_signals(m_sensors)

bounds = [(1, Nx-2), (1, Ny-2), (1, Nx-2), (1, Ny-2)]

it = 1
final_error = 100
error_threshold = 1

while final_error >= error_threshold:
    print("Running differential_evolution...")
    de_output = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=30, popsize=12, disp=view, workers=-1, polish=True)
    final_error = de_output.fun

    if final_error > error_threshold:
        print("Error too large, attempting to find a new solution! Iteration number:", it)
        it += 1
        final_error = 100
        error_threshold *= 1.2
        error_threshold = round(error_threshold, 3)
        print(f"Updating error threshold: {error_threshold}")

xs1_est, ys1_est = int(round(de_output.x[0])), int(round(de_output.x[1]))
xs2_est, ys2_est = int(round(de_output.x[2])), int(round(de_output.x[3]))

print("#########################################################")
print(f"Source 1 position: xs={xs1} ys={ys1}")
print(f"Source 2 position: xs={xs2} ys={ys2}")
print(f"DE.x result = {de_output.x} rounding -> xs1_est={xs1_est}, ys1_est={ys1_est}, xs2_est={xs2_est}, ys2_est={ys2_est}")
print(f"Final error: {de_output.fun}")

# fonte_encontrada = emulate_source(xs1_est, ys1_est, xs2_est, ys2_est, s, Nt, Nx, Ny, a1, a2, c2, h, view=True)

plt.figure()
plt.imshow(np.zeros((Ny, Nx)), cmap='gray', extent=[0, Nx, 0, Ny], origin='lower')
plt.plot(xs1, ys1, 'ro', label='Source 1')
plt.plot(xs2, ys2, 'ro', label='Source 2')
plt.plot(xs1_est, ys1_est, 'bx', label='Estimated Source 1')
plt.plot(xs2_est, ys2_est, 'bx', label='Estimated Source 2')
plt.legend()
plt.title("Source Location")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
