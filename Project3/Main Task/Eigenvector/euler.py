import numpy as np

class Explicit_scheme:
    def __init__(self, A, x0, N= 10000, T= 1):
        # Problem formulation for Euler
        self.N = N # number of time points
        self.T = T # Period
        self.A = A #np.array([[3., 2., 4.], [2., 0., 2.], [4., 2., 3.]])
        self.x0 = x0 / np.linalg.norm(x0)
        self.t = np.linspace(0, T, N)

    # Define Rayleigh quotient
    def ray_quo(self, A, x):
        return np.einsum('ij,jk,ik->i', x, A, x) / np.einsum('ij,ij->i', x, x)

    # Define Euler's method
    def euler_ray_quo(self):
        A = self.A; x0 = self.x0;
        dt = self.T / self.N
        x = [x0]
        for i in range(self.N - 1):
            x.append(x[-1] + dt * ((x[-1].T @ x[-1]) * A @ x[-1] - (x[-1].T @ A) @ x[-1] * x[-1]))
    
        x = np.array(x)
        ray_quo = np.einsum('ij,jk,ik->i', x, A, x) / np.einsum('ij,ij->i', x, x)
    
        return x, ray_quo