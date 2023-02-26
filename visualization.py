import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators

import scipy
from scipy import io

import skimage
from skimage import transform

# Noise is added by noise level
def get_gaussian_noise(y, noise_level=0.01):
    noise = np.random.normal(size=y.shape) # random gaussian distribution
    noise /= np.linalg.norm(noise.flatten(), 2) # norma di frobenius
    return noise * noise_level * np.linalg.norm(y.flatten(), 2)


# True image
idx = 0
x_true = plt.imread(f'./data/COULE_test/{idx}.png')[:, :, 0]
x_true = transform.resize(x_true, (256, 256))
M, N = x_true.shape

# Forward problem
A = operators.MatrixOperator(io.loadmat("./data/A.mat")['A'])

# Noiseless y
y = A(x_true)

# Add noise
np.random.seed(42)
e = get_gaussian_noise(y, noise_level=0.01)
y_delta = y + e

# Load the reconstruction
idx = 0
algorithm = 'CP_TV'

rec_path = f'./results/COULE_{algorithm}_{idx}.npy'
x_sol = np.load(rec_path)

iCP_TV_recon = (algorithm == 'iCP_TV')
if iCP_TV_recon:
    h = 9

    # Visualize.
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x_true.reshape((256, 256)), cmap='gray')
    plt.title(r'$x^{gt}$')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_sol[:, h].reshape((256, 256)), cmap='gray')
    plt.axis('off')
    plt.title(f'h={h}, RE:{np.linalg.norm(x_sol[:, h].flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())}')
    plt.tight_layout()
    plt.savefig(f"./results/{idx}/iCP_TV_h_{h}.png", dpi=400)

CP_TV_recon = (algorithm == 'CP_TV')
if CP_TV_recon:
    # Visualize.
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x_true.reshape((256, 256)), cmap='gray')
    plt.title(r'$x^{gt}$')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_sol[:, -1].reshape((256, 256)), cmap='gray')
    plt.axis('off')
    plt.title(f'RE:{np.linalg.norm(x_sol[:, -1].flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())}')
    plt.tight_layout()
    plt.savefig(f"./results/{idx}/CP_TV.png", dpi=400)
    

CGLS_recon = (algorithm == 'CGLS')
if CGLS_recon:
    # Visualize.
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x_true.reshape((256, 256)), cmap='gray')
    plt.title(r'$x^{gt}$')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_sol.reshape((256, 256)), cmap='gray')
    plt.axis('off')
    plt.title(f'RE:{np.linalg.norm(x_sol.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())}')
    plt.tight_layout()
    plt.savefig(f"./results/{idx}/CGLS.png", dpi=400)
    