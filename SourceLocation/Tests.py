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
    # im = plt.imshow(p_0, vmin=vmin, vmax=vmax, cmap="bwr")

xs = np.random.randint(1, Nx-2)
ys = np.random.randint(1, Ny-2)

for nt in tqdm(range(Nt)):
    p_1, p_2 = p_0, p_1
    p_0 = a1 * p_1 + a2 * p_2 + c2 * ndi.convolve(p_1,h)
    p_0[xs, ys] += s[nt]
    m[nt] = p_0[1] + .001*np.random.randn()
    
    # if view:
    #     if not nt % view:
            # im.set_data(p_0)
            # plt.pause(1e-12)

# plt.close()

#%%
# nm = 52
# # sd.play(m[:, -nm], fs, blocking=True)
# plt.figure()
# plt.plot(t, s, t, m[:, -1], t, m[:, -nm])
# plt.show()

# plt.figure()
# plt.imshow(m, aspect="auto")
# plt.show()

#%%
print("#########################################################")
# Encontrando a Fonte

# Simulador da fonte
def simular_fonte(xs_emulado, ys_emulado, s, Nt, Nx, Ny, a1, a2, c2, h, view=False):
    p_0 = np.zeros((Ny, Nx), dtype=np.float32)
    p_1 = np.zeros((Ny, Nx), dtype=np.float32)
    p_2 = np.zeros((Ny, Nx), dtype=np.float32)
    
    fonte_emulada = np.zeros((Nt, Nx), dtype=np.float32)

    if view:
        vmax = .1
        vmin = -vmax
        im = plt.imshow(p_0, vmin=vmin, vmax=vmax, cmap="bwr")

    for nt in range(Nt):
        p_1, p_2 = p_0, p_1
        p_0 = a1 * p_1 + a2 * p_2 + c2 * ndi.convolve(p_1, h)
        p_0[xs_emulado, ys_emulado] += s[nt]
        fonte_emulada[nt] = p_0[1] # Sem ruido

        if view and not nt % view:
            im.set_data(p_0)
            plt.pause(1e-12)

    return fonte_emulada

# Custo = Diferença da fonte original para a fonte simulada
def custo(pos):
    xs, ys = int(pos[0]), int(pos[1])
    
    # Verifica se está dentro dos limites válidos
    if xs < 1 or xs >= Nx-2 or ys < 1 or ys >= Ny-2:
        return np.inf  # penaliza posições inválidas
    
    fonte_emulada = simular_fonte(xs, ys, s, Nt, Nx, Ny, a1, a2, c2, h, view=False)
    erro = np.linalg.norm(fonte_emulada - m)
    return erro

from scipy.optimize import differential_evolution

bounds = [(1, Nx-2), (1, Ny-2)] # limites válidos

# otimização global baseada em populações -> tenta minimizar (ou maximizar) uma função explorando o espaço de soluções com uma população de candidatos
#   Cria uma população de vetores (soluções candidatas)
#   O algoritmo escolhe três outros vetores aleatórios e combina eles para criar um novo vetor “mutado”
#   O vetor mutado é misturado com o vetor original 
#   Avalia o vetor filho usando a função de custo.
#   Se o filho tiver um erro menor que o pai, ele substitui o pai na próxima geração.
#   Repete os passos acima por várias gerações (maxiter), até que a população converge ou o erro pare de melhorar.

melhor_erro = np.inf
melhor_pos = None

for i in range(5):
    if melhor_erro < 0.5:
        break
    print("Iteração num:", i)
    resultado = differential_evolution(custo, bounds, strategy='best1bin', maxiter=30, popsize=20, disp=False, workers=-1)
    print("Solução encontrada:",resultado.x)
    if resultado.fun < melhor_erro:
        melhor_erro = resultado.fun
        melhor_pos = resultado.x

print("#########################################################")
print(f"Posição correta: [{xs} {ys}]")
print(f"Melhor posição encontrada: {melhor_pos}")
print("Erro:", melhor_erro)
print("#########################################################")
# fonte_encontrada = simular_fonte(int(melhor_pos[0]), int(melhor_pos[1]), s, Nt, Nx, Ny, a1, a2, c2, h, view=True)

plt.figure()
plt.imshow(np.zeros((Ny, Nx)), cmap='gray', extent=[0, Lx, 0, Ly])
plt.plot(xs * dx, ys * dy, 'ro', label='Fonte Real')
plt.plot(melhor_pos[0] * dx, melhor_pos[1] * dy, 'bx', label='Fonte Estimada')
plt.legend()
plt.title("Localização da Fonte")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.show()

