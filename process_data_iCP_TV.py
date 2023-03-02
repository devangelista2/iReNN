import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators, solvers
from iCP_TV import iCP_TV
import settings

import scipy
from scipy import io

import skimage
from skimage import transform

import os

# Noise is added by noise level
def get_gaussian_noise(y, noise_level=0.01):
    noise = np.random.normal(size=y.shape) # random gaussian distribution
    noise /= np.linalg.norm(noise.flatten(), 2) # norma di frobenius
    return noise * noise_level * np.linalg.norm(y.flatten(), 2)

# Paths
train_path = './data/COULE_train'
test_path = './data/COULE_test'

# Forward problem
M, N = 256, 256
A = operators.CTProjector(M, N, np.linspace(0, np.pi, 120), det_size=367, geometry='fanflat')

# Solver
solver = iCP_TV(A, (M, N), info=False)

# Choose the parameters
algorithm = 'iCP_TV'
match algorithm:
    case 'iCP_TV':
        parameters = settings.iCP_TV_algorithm
    case 'CP_TV':
        parameters = settings.CP_TV_algorithm

# Initialization (Train set)
print("Creating training set...")
x_irenn = np.zeros((len(os.listdir(train_path)), parameters['H'], M, N))
for idx in range(len(os.listdir(train_path))):
    # True image
    x_true = np.load(f'{train_path}/{idx}.npy')
    M, N = x_true.shape

    # Noiseless y
    y = A(x_true)

    # Add noise
    np.random.seed(42)
    e = get_gaussian_noise(y, noise_level=settings.forward_problem['noise_level'])
    y_delta = y + e

    # Starting points
    p = parameters['p']
    lmbda = parameters['lmbda']
    epsilon = 1e-5 * np.max(y) * np.sqrt(len(y))

    K = parameters['K']
    H = parameters['H']

    # Solution
    x_sol = solver(y_delta, epsilon, lmbda, H=H, K_schedule=K, x_true=x_true, p=p).T

    # Reshape
    x_sol = np.reshape(x_sol, (H, M, N))

    # Save
    x_irenn[idx] = x_sol
    
    # Info
    rel_err = np.linalg.norm(x_sol[-1].flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())
    print(f"Image number {idx+1}/{len(os.listdir(train_path))} done. Relative error: {rel_err:0.4f}.")

# Save the dataset
np.save(f"./data/iReNN_train.npy", x_irenn)

# Initialization (Test set)
print("Creating test set...")
x_irenn = np.zeros((len(os.listdir(test_path)), parameters['H'], M, N))
for idx in range(len(os.listdir(test_path))):
    # True image
    x_true = np.load(f'{test_path}/{idx}.npy')
    M, N = x_true.shape

    # Noiseless y
    y = A(x_true)

    # Add noise
    np.random.seed(42)
    e = get_gaussian_noise(y, noise_level=settings.forward_problem['noise_level'])
    y_delta = y + e

    # Starting points
    p = parameters['p']
    lmbda = parameters['lmbda']
    epsilon = 1e-5 * np.max(y) * np.sqrt(len(y))

    K = parameters['K']
    H = parameters['H']

    # Solution
    x_sol = solver(y_delta, epsilon, lmbda, H=H, K_schedule=K, x_true=x_true, p=p).T

    # Reshape
    x_sol = np.reshape(x_sol, (H, M, N))

    # Save
    x_irenn[idx] = x_sol
    
    # Info
    rel_err = np.linalg.norm(x_sol[-1].flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())
    print(f"Image number {idx+1}/{len(os.listdir(test_path))} done. Relative error: {rel_err:0.4f}.")

# Save the dataset
np.save(f"./data/iReNN_test.npy", x_irenn)