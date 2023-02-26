import numpy as np
import matplotlib.pyplot as plt

from IPPy import operators, solvers

class iCP_TV:
    def __init__(self, A, img_shape) -> None:
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

    def __call__(self, y, epsilon, lmbda, H=5, K=50, x_true=None, eta=2e-3, p=1):
        # Flattening
        y = y.flatten()

        # Initialization
        rel_err = np.linalg.norm(np.zeros((self.n, )) - x_true.flatten()) / np.linalg.norm(x_true.flatten())
        print(f"Relative error at outer iteration 0: {rel_err}.")

        obj_vec = np.zeros((H, ))

        x_vec = np.zeros((self.n, H))
        x_vec[:, 0] = np.zeros((self.n, ))
        for h in range(1, H):
            # Inner step
            x0 = np.expand_dims(x_vec[:, h-1], -1)
            x_h = self.CP_TV(y, epsilon=epsilon, lmbda=lmbda, maxiter=K, x_true=x_true, starting_point=x0, p=p, eta=2e-3)

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