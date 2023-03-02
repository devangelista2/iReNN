import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators, solvers
from iCP_TV import iCP_TV
import settings

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
x_true = np.load(f'./data/COULE_test/{idx}.npy')
x_true = transform.resize(x_true, (256, 256))
M, N = x_true.shape

# Forward problem
A = operators.CTProjector(256, 256, np.linspace(0, np.pi, 120), det_size=367, geometry='fanflat')

# Noiseless y
y = A(x_true)

# Add noise
np.random.seed(42)
e = get_gaussian_noise(y, noise_level=settings.forward_problem['noise_level'])
y_delta = y + e

# Choose the parameters
algorithm = 'iCP_TV'
match algorithm:
    case 'iCP_TV':
        parameters = settings.iCP_TV_algorithm
    case 'CP_TV':
        parameters = settings.CP_TV_algorithm

# Starting points
p = parameters['p']
lmbda = parameters['lmbda']
epsilon = 1e-5 * np.max(y) * np.sqrt(len(y))

K = parameters['K']
H = parameters['H']

# Solver
solver = iCP_TV(A, (M, N))

# Solution
x_sol = solver(y_delta, epsilon, lmbda, H=H, K_schedule=K, x_true=x_true, p=p)

# Save the solution
np.save(f'./results/COULE_{algorithm}_{idx}.npy', x_sol)