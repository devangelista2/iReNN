import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators, solvers
from iCP_TV import iCP_TV, iReNN
import settings

import scipy
from scipy import io

import skimage
from skimage import transform

# Disable Tf Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
algorithm = 'iReNN'
parameters = settings.iReNN_algorithm

# Starting points
p = parameters['p']
lmbda = parameters['lmbda']
epsilon = 1e-5 * np.max(y) * np.sqrt(len(y))

K = parameters['K']
H = parameters['H']
iterative_schedule = parameters['iterative_schedule']

# Solver
solver = iReNN(A, (M, N), './model_weights')

# Solution
x_sol = solver(y_delta, epsilon, lmbda, iterative_schedule, H=H, K_schedule=K, x_true=x_true, p=p)

plt.subplot(1, 2, 1)
plt.imshow(x_true.reshape((256, 256)), cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(x_sol[:, 5].reshape((256, 256)), cmap='gray')
plt.show()

# Save the solution
np.save(f'./results/COULE_{algorithm}_{idx}.npy', x_sol)