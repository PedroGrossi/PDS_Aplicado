import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.transform as trans
import skimage as ski

#######################################################################################
# Dependências e utilitários
#######################################################################################

def to_float01(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img

def add_gaussian_noise(img, sigma):
    return img + np.random.normal(0, sigma, size=img.shape).astype(np.float32)

def generate_random_motions(n=20, max_shift=3):
    motions = []
    for _ in range(n):
        # deslocamentos aleatórios
        tx = np.random.uniform(-max_shift, max_shift)
        ty = np.random.uniform(-max_shift, max_shift)

        # matriz identidade
        A = np.eye(2)

        motions.append({'A': A, 't': (tx, ty)})
    return motions

#######################################################################################
# Operadores do modelo de observação
#######################################################################################

# tentar com shift fracionario (ex: 1.3)
def warp_image(x, matrix, offset=(0,0), order=1):
    return ndi.affine_transform(x, matrix=matrix, offset=offset, order=order, mode='reflect')

# Kernel de média NxN
Nb = 5
b = np.ones([Nb, Nb], dtype=np.float32) / (Nb ** 2)

def blur_image(x):
    return ndi.convolve(x, b, mode="reflect")

def downsample(x, L1=2, L2=2):
    return x[::L2, ::L1]

def upsample(y, shape, L1=2, L2=2, order=3):
    return trans.resize(y, shape, order=order, anti_aliasing=False, preserve_range=True)

def H_forward(x, motion_params, L1=2, L2=2, noise_sigma=0.0):
    xk = warp_image(x, matrix=motion_params['A'], offset=motion_params['t'], order=motion_params.get('order',1))
    xb = blur_image(xk)  # blur uniforme
    y = downsample(xb, L1=L1, L2=L2)
    if noise_sigma > 0:
        y = add_gaussian_noise(y, noise_sigma)
    return y

#######################################################################################
# Simulação de LR a partir de uma HR
#######################################################################################

def simulate_LR_set(x_hr, motions, L1=2, L2=2, noise_sigma=0.01):
    x_hr = to_float01(x_hr)
    ys = []
    for k in range(len(motions)):
        yk = H_forward(x_hr, motions[k], L1, L2, noise_sigma)
        ys.append(yk.astype(np.float32))
    return ys

#######################################################################################
# HR image
#######################################################################################

x_hr = to_float01(ski.io.imread('images/cameraman.png', as_gray=True))

# movimentos (translação pura e leve rotação)
# motions = [
#     {'A': np.eye(2), 't': (0.0, 0.0)}, # sem movimento
#     {'A': np.eye(2), 't': (1.0, 1.0)}, # desloca 1 pixel para direita e 1 para baixo
#     {'A': np.eye(2), 't': (-2.0, 2.0)}, # desloca 2 pixel para esquerda e 2 para baixo
#     # {'A': np.array([[np.cos(np.deg2rad(1)), -np.sin(np.deg2rad(1))],
#     #                 [np.sin(np.deg2rad(1)),  np.cos(np.deg2rad(1))]]), 't': (-1.0, 2.0)}, # rotação de 1 grau e deslocando 1 pixel para esquerda e 2 para baixo
#     {'A': np.eye(2), 't': (2.0, -1.0)}, # desloca 2 pixels para a direita e 1 pixel para cima.
# ]

motions = generate_random_motions(n=20, max_shift=3)
print(motions) 

L1, L2 = 2, 2
noise_sigma = 0.01

# Simula LRs
ys = simulate_LR_set(x_hr, motions, L1=L1, L2=L2, noise_sigma=noise_sigma)
print(f"{len(ys)} LR images.")

# Salvar cada LR
for i, y in enumerate(ys, start=1):
    filename = f"images/LR{i}.png"
    y_uint8 = (np.clip(y, 0, 1) * 255).astype(np.uint8)
    ski.io.imsave(filename, y_uint8)
    print(f"Saving image: {filename}")

#######################################################################################
# Visualização
#######################################################################################

plt.figure(figsize=(10, 4))
plt.subplot(1, len(ys)+1, 1)
plt.imshow(x_hr, cmap="gray")
plt.title("HR original")
plt.axis("off")

for i in range(min(len(ys), 4)):
    plt.subplot(1, len(ys)+1, i+2)
    plt.imshow(ys[i], cmap="gray")
    plt.title(f"LR {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
