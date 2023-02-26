import numpy as np
import matplotlib.pyplot as plt

from tomography.data import generate_phantom, generate_COULE
import operators, solvers

import scipy
from scipy import io

import skimage
from skimage import transform

x = np.load('0.npy') # generate_phantom.phantom(p_type='modified shepp-logan')
x = transform.resize(x, (256, 256))
M, N = x.shape

A = operators.CTProjector(M, N, np.linspace(0, 180, 180), 367)

y = A(x)
np.random.seed(42)
e = np.random.normal(0, 1, y.shape)
y = y + e

# Starting points
p = 1
lmbda = 1e-3
epsilon = 1e-5 * np.max(y) * np.sqrt(len(y))

x_h = np.zeros((M*N, 1))

x_vec = np.zeros((M*N, 5))
for h in range(5):
    CP_TV = solvers.ChambollePockTpV(A)
    x_h= CP_TV(y, epsilon=epsilon, lmbda=lmbda, maxiter=20, x_true=x.flatten(), starting_point=x_h, p=p)

    # Update parameters
    p = p / 2
    lmbda = lmbda / 2

    # Save the result
    x_vec[:, h] = x_h[:, 0]

plt.subplot(2, 3, 1)
plt.imshow(x, cmap='gray')
plt.axis('off')
plt.title('GT')

for h in range(5):
    plt.subplot(2, 3, h+2)
    plt.imshow(x_vec[:, h].reshape((M, N)), cmap='gray')
    plt.axis('off')
    plt.title(f'h = {h+1}')
plt.show()