import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators, solvers
from IPPy import metrics

import tensorflow as tf
from tensorflow import keras as ks

class iCP_TV:
    def __init__(self, A, img_shape, info=True) -> None:
        self.A = A
        self.m, self.n = A.shape

        # Shape of x
        self.M, self.N = img_shape

        # Solver
        self.CP_TV = solvers.ChambollePockTpV(self.A)

        # Generate Gradient operators
        self.grad = operators.myGradient(1, (self.M, self.N))

        # Continuation parameter
        self.alpha = 0.75

        # Other
        self.info = info

    def __call__(self, y, epsilon, lmbda, H=None, K_schedule=50, x_true=None, eta=2e-3, p=1):
        # Scheduling
        if H is None and type(K_schedule) is int:
            print("Define define at least H or K.")
            return None
        elif H is not None and type(K_schedule) is int:
            K_schedule = K_schedule * np.ones((H, ), dtype=int)
        else:
            H = len(K_schedule)
        
        assert len(K_schedule) == H

        # Flattening
        y = y.flatten()

        # Initialization
        rel_err = np.linalg.norm(np.zeros((self.n, )) - x_true.flatten()) / np.linalg.norm(x_true.flatten())
        
        if self.info:
            print(f"Relative error at outer iteration 0: {rel_err}.")

        obj_vec = np.zeros((H, ))

        x_vec = np.zeros((self.n, H))
        x_vec[:, 0] = np.zeros((self.n, ))
        for h in range(1, H):
            # Inner step
            x0 = np.expand_dims(x_vec[:, h-1], -1)
            x_h = self.CP_TV(y, epsilon=epsilon, lmbda=lmbda, maxiter=K_schedule[h], x_true=x_true, starting_point=x0, p=p, eta=2e-3)

            # Compute the value of the objective function TpV by reweighting
            obj_vec[h] = self.compute_obj_value(x_h, y, lmbda, p, eta)

            # Update parameters
            p = p * self.alpha
            if h==1:
                lmbda = lmbda / 2
            else:
                lmbda = lmbda * (obj_vec[h] / obj_vec[h-1])

            # Print relative error after the h-th iteration
            rel_err = np.linalg.norm(x_h.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())

            if self.info:
                print(f"Relative error at outer iteration {h}: {rel_err}.")

            # Save the result
            x_vec[:, h] = x_h[:, 0]
        
        return x_vec
    
    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2)**2 + lmbda * ftpv
    

class iReNN:
    def __init__(self, A, img_shape, weights_path, info=True) -> None:
        self.A = A
        self.m, self.n = A.shape

        # Shape of x
        self.M, self.N = img_shape

        # Solver
        self.CP_TV = solvers.ChambollePockTpV(self.A)

        # Generate Gradient operators
        self.grad = operators.myGradient(1, (self.M, self.N))

        # Continuation parameter
        self.alpha = 0.5

        # Other
        self.weights_path = weights_path
        self.info = info

    def __call__(self, y, epsilon, lmbda, iterative_schedule=None, H=None, K_schedule=50, x_true=None, eta=2e-3, p=1):
        # Scheduling
        if H is None and type(K_schedule) is int:
            print("Define define at least H or K.")
            return None
        elif H is not None and type(K_schedule) is int:
            K_schedule = K_schedule * np.ones((H, ), dtype=int)
        else:
            H = len(K_schedule)
        
        assert len(K_schedule) == H

        # Iterative schedule
        if iterative_schedule is None:
            iterative_schedule = np.zeros((H, ))

        # Flattening
        y = y.flatten()

        # Initialization
        rel_err = np.linalg.norm(np.zeros((self.n, )) - x_true.flatten()) / np.linalg.norm(x_true.flatten())
        
        if self.info:
            print(f"Relative error at outer iteration 0: {rel_err}.")

        obj_vec = np.zeros((H, ))

        x_vec = np.zeros((self.n, H))
        x_vec[:, 0] = np.zeros((self.n, ))
        for h in range(1, H):
            # Inner step
            x0 = np.expand_dims(x_vec[:, h-1], -1)
            if iterative_schedule[h] == 0:
                x_h = self.CP_TV(y, epsilon=epsilon, lmbda=lmbda, maxiter=K_schedule[h], x_true=x_true, starting_point=x0, p=p, eta=2e-3)
            else:
                model = ks.models.load_model(f"{self.weights_path}/iReNN_{h-1}to{h}.h5", custom_objects={'SSIM': metrics.SSIM})
                x_h = self.predict(model, x0)

            # Compute the value of the objective function TpV by reweighting
            obj_vec[h] = self.compute_obj_value(x_h, y, lmbda, p, eta)

            # Update parameters
            p = p * self.alpha
            if h==1:
                lmbda = lmbda / 2
            else:
                lmbda = lmbda * (obj_vec[h] / obj_vec[h-1])

            # Print relative error after the h-th iteration
            rel_err = np.linalg.norm(x_h.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())

            if self.info:
                print(f"Relative error at outer iteration {h}: {rel_err}.")

            # Save the result
            x_vec[:, h] = x_h[:, 0]
        
        return x_vec
    
    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2)**2 + lmbda * ftpv

    def predict(self, model, x):
        x = np.reshape(x, (1, self.M, self.N, 1))
        y = model.predict(x, verbose=0)
        return np.expand_dims(y.flatten(), -1)