import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators, solvers
from iCP_TV import iCP_TV

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

# Solvers
solver = solvers.CGLS(A)

# Solution
x_sol = solver(y_delta, np.zeros((M*N, )), x_true.flatten())

# Save the solution
np.save(f'./results/COULE_CGLS_{idx}.npy', x_sol)