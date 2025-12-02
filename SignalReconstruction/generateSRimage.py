import numpy as np
import scipy.ndimage as ndi
import skimage.transform as trans
import skimage as ski
import matplotlib.pyplot as plt
import os
import time

#######################################################################################
# Utilitarios
#######################################################################################
def to_float01(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img

def mse(a, b):
    return np.mean((a - b) ** 2)

#######################################################################################
# Operadores diretos
#######################################################################################
def warp_image(x, matrix, offset=(0,0), order=1):
    return ndi.affine_transform(x, matrix=matrix, offset=offset, order=order, mode='reflect')

# Kernel de média NxN
Nb = 5
b = np.ones([Nb, Nb], dtype=np.float32) / (Nb ** 2)

def blur_image(x):
    return ndi.convolve(x, b, mode='reflect')

def downsample(x, L1=2, L2=2):
    return x[::L2, ::L1]

# W_k: warp -> blur -> downsample
def W_forward(x_hr, motion_params, L1=2, L2=2):
    xw = warp_image(x_hr, matrix=motion_params['A'], offset=motion_params['t'], order=motion_params.get('order',1))
    xb = blur_image(xw)
    y = downsample(xb, L1=L1, L2=L2)
    return y

#######################################################################################
# Operadores adjuntos
#######################################################################################
def upsample(y, out_shape, L1=2, L2=2):
    out = np.zeros(out_shape, dtype=np.float32)
    out[::L2, ::L1] = y
    return out

def blur_adjoint(x):
    b_flip = np.flip(np.flip(b, axis=0), axis=1)
    return ndi.convolve(x, b_flip, mode='reflect')

def warp_adjoint(x, matrix, offset=(0,0), order=1):
    A = np.array(matrix, dtype=np.float64)
    A_inv = np.linalg.inv(A)
    t = np.array(offset, dtype=np.float64)
    offset_adj = -A_inv.dot(t)
    return ndi.affine_transform(x, matrix=A_inv, offset=offset_adj, order=order, mode='reflect')

# W_k^T: upsample -> blur^T -> warp^T
def W_adjoint(y_lr, motion_params, out_shape, L1=2, L2=2):
    up = upsample(y_lr, out_shape, L1=L1, L2=L2)
    b_up = blur_adjoint(up)
    x_adj = warp_adjoint(b_up, matrix=motion_params['A'], offset=motion_params['t'], order=motion_params.get('order',1))
    return x_adj

#######################################################################################
# Algoritmo iterativo
#######################################################################################
def super_resolve(x0, ys, motions, L1=2, L2=2, beta=0.5, n_iters=50, verbose=True, x_hr_ref=None, save_interval=500, save_dir='output_20_LR_images'):
    x = x0.copy().astype(np.float32)
    out_shape = x.shape
    p = len(ys)

    os.makedirs(save_dir, exist_ok=True)

    # salva o chute inicial
    fname_init = os.path.join(save_dir, f"20_LR_x_iter_{1:05d}.png")
    img_init = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    ski.io.imsave(fname_init, img_init)

    for n in range(n_iters):
        grad_data = np.zeros_like(x, dtype=np.float32)
        for k in range(p):
            W_x = W_forward(x, motions[k], L1=L1, L2=L2)
            residual = ys[k] - W_x
            grad_k = W_adjoint(residual, motions[k], out_shape, L1=L1, L2=L2)
            grad_data += grad_k

        x = x + beta * grad_data
        x = np.clip(x, 0.0, 1.0)

        if verbose and (n % max(1, n_iters//10) == 0 or n == n_iters-1):
            mse_lr = np.mean([np.mean((ys[k] - W_forward(x, motions[k], L1=L1, L2=L2))**2) for k in range(p)])
            msg = f"Iter {n+1}/{n_iters} MSE_LR {mse_lr:.6f}"

            if x_hr_ref is not None:
                mse_hr_val = mse(x, x_hr_ref) 
                msg += f" | MSE_HR {mse_hr_val:.6f}"

            print(msg)
        
        # salvar a cada save_interval iterações
        if (n + 1) % save_interval == 0 or n == n_iters - 1:
            fname = os.path.join(save_dir, f"20_LR_x_iter_{n+1:05d}.png")
            # converte para uint8 antes de salvar
            img_to_save = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
            ski.io.imsave(fname, img_to_save)

    return x

#######################################################################################
# Leitura das LR
#######################################################################################
def load_LR_images(prefix='LR', n=4):
    ys = []
    for i in range(1, n+1):
        fname = f"images/{prefix}{i}.png"
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Arquivo não encontrado: {fname}")
        y = to_float01(ski.io.imread(fname, as_gray=True))
        ys.append(y.astype(np.float32))
    return ys

#######################################################################################
if __name__ == "__main__": 
    # parâmetros de amostragem 
    L1, L2 = 2, 2 
    # carregar LRs 
    ys = load_LR_images(prefix='LR', n=20) 
    # carregar HR de referência 
    hr_ref = to_float01(ski.io.imread('images/cameraman.png', as_gray=True)).astype(np.float32) 
    
    # parâmetros de movimentação das LR 
    # motions = [ {'A': np.eye(2), 't': (0.0, 0.0)}, # sem movimento 
    #             {'A': np.eye(2), 't': (1.0, 1.0)}, # desloca 1 pixel para direita e 1 para baixo 
    #             {'A': np.eye(2), 't': (-2.0, 2.0)}, # desloca 2 pixel para esquerda e 2 para baixo 
    #             # {'A': np.array([[np.cos(np.deg2rad(1)), -np.sin(np.deg2rad(1))], 
    #                             # [np.sin(np.deg2rad(1)), np.cos(np.deg2rad(1))]]), 't': (-1.0, 2.0)}, # rotação de 1 grau e deslocando 1 pixel para esquerda e 2 para baixo 
    #             {'A': np.eye(2), 't': (2.0, -1.0)}, # desloca 2 pixels para a direita e 1 pixel para cima.
    #             ] 
    
    motions = [{'A': np.eye(2), 't': (1.5866981588957492, 0.6772378947843385)}, 
               {'A': np.eye(2), 't': (-0.6165884896034761, 0.3868285488290013)}, 
               {'A': np.eye(2), 't': (-2.7167675978626127, -1.874450584288147)}, 
               {'A': np.eye(2), 't': (2.484104892643632, -1.5122864577914967)}, 
               {'A': np.eye(2), 't': (-2.944317948043914, 1.5965161527785554)}, 
               {'A': np.eye(2), 't': (-0.8697867502064431, 1.0464327037124654)}, 
               {'A': np.eye(2), 't': (-1.2083898008762586, 1.5212949902403583)}, 
               {'A': np.eye(2), 't': (1.3822814141350026, 2.9966313480568108)}, 
               {'A': np.eye(2), 't': (1.2302041904272834, 0.49401061388531087)}, 
               {'A': np.eye(2), 't': (-0.5075529438593662, -1.4539516575166607)}, 
               {'A': np.eye(2), 't': (-0.779325950447507, -2.8558511453228803)}, 
               {'A': np.eye(2), 't': (-1.5510624572875082, -0.3770716372092391)}, 
               {'A': np.eye(2), 't': (1.0708350759540401, -0.6358966794954108)}, 
               {'A': np.eye(2), 't': (1.766519247799999, -2.8268881819115306)}, 
               {'A': np.eye(2), 't': (0.7354916421367848, 2.661874405351763)}, 
               {'A': np.eye(2), 't': (-2.484366385059067, -0.5000526268925674)}, 
               {'A': np.eye(2), 't': (-2.7224015090526352, -2.517181944321856)}, 
               {'A': np.eye(2), 't': (-2.990268374473191, -1.9123496081185996)}, 
               {'A': np.eye(2), 't': (1.2077561080193258, -0.9776394235464574)}, 
               {'A': np.eye(2), 't': (2.52069550856781, -0.9276116961871947)}
               ]
    
    # tamanho HR esperado
    hr_shape = (ys[0].shape[0]*L2, ys[0].shape[1]*L1) 
    # x0: upsample bicúbico da LR1 (ys[0]) 
    x0 = trans.resize(ys[0], hr_shape, order=3, anti_aliasing=False, preserve_range=True).astype(np.float32) 
    x0 = np.clip(x0, 0.0, 1.0) 
    
    # SR iterativo 
    beta = .005
    n_iters = 4000
    x_rec = super_resolve(x0, ys, motions, L1=L1, L2=L2, beta=beta, n_iters=n_iters, verbose=True, x_hr_ref=hr_ref) 
    
    plt.figure(figsize=(12,4)) 
    plt.subplot(1,3,1); 
    plt.imshow(hr_ref, cmap='gray'); 
    plt.title('HR original'); 
    plt.axis('off') 
    plt.subplot(1,3,2); 
    plt.imshow(x0, cmap='gray'); 
    plt.title('Inicial x0 (LR1 upsampled)'); 
    plt.axis('off') 
    plt.subplot(1,3,3); 
    plt.imshow(x_rec, cmap='gray'); 
    plt.title('Reconstruída'); 
    plt.axis('off') 
    plt.show()
