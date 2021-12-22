import sys
import tensorflow as tf
import numpy as np

import NN as nn
from euler import *
from analysis import *
sys.path.append('../Diffusion')
from functions import save_fig

def find_maximum():
    print("Finding the maximum eigenvalue by determining its corresponding maximum eigenvector")
    # Finding the maximum eigenvalue
    # Initial model and optimizer (NN)
    model = nn.DNModel(A, x0, NHN, Nt= Nt, T= T)
    model.solver(model, num_epochs= num_epochs)

    # Forward Euler
    scheme = Explicit_scheme(A, x0, N= N, T= T)
    x_euler, eig_euler = scheme.euler_ray_quo()

    # Generating final results --> analysis
    t = scheme.t
    s = t.reshape(-1, 1)
    g = nn.trial_solution(model, model.x0, s)
    eig_nn = nn.ray_quo(A, g)

    max_eigenvalue(model, A, x0, t, x_euler, eig_euler, N, g, eig_nn, Nt) # Used when solving for the max eigenvalue

def find_minimum():
    print("Finding the minimum eigenvalue by determining its corresponding minimum eigenvector")
    # Finding the minimum eigenvalue
    # Initial model and optimizer (NN)
    model = nn.DNModel(-A, x0, NHN, Nt= Nt, T= T)
    model.solver(model, num_epochs= num_epochs)

    # Forward Euler
    scheme = Explicit_scheme(-A, x0, N= N, T= T)
    x_euler, eig_euler = scheme.euler_ray_quo()
    eig_euler = -eig_euler

    # Generating final results --> analysis
    t = scheme.t
    s = t.reshape(-1, 1)
    g = nn.trial_solution(model, model.x0, s)
    eig_nn = nn.ray_quo(A, g)

    min_eigenvalue(model, A, x0, t, x_euler, eig_euler, N, g, eig_nn, Nt) # Used when solving for the min eigenvalue


#Preset
s = 30
np.random.seed(s)
# tf.random.set_seed(s)
# NHN = [100]
NHN = [100, 10]
# NHN = [100, 50, 25]
num_epochs = 2000
T = 10 # Final time
Nt = 101 # Timestep NN
N= 1001 # Timesetp Euler


# Define problem
n = 6 # Creating an n x n symmetric matrix
A = np.random.normal(0, 1, (n, n))
A = (A.T + A) * 0.5 
x0 = np.random.rand(n) # Start vector/Initial guess

# To be used instead of the seeded x0 for testing
# Normal distribution np.random.normal(0,1, n)
# x0 = np.array([ 0.8774262 ,  1.00256029, -1.04376454,  1.26802778, -0.28202225, 0.33168039])
# x0 = np.array([ 0.73873294, -0.40967783,  0.57738073,  0.41657998,  0.52145173, -1.47365648])
# x0 = np.array([ 1.2418555 ,  1.70774688,  0.3231534 , -0.99183439, -1.16328237, -0.0213351 ])
# uniform distribution [0, 1) np.random.rand(n)
# x0 = np.array([0.95546832, 0.68291385, 0.05312869, 0.30885268, 0.59259469, 0.23512041])
# x0 = np.array([0.41488598, 0.00142688, 0.09226235, 0.70939439, 0.5243456 , 0.69616046])
# x0 = np.array([0.31343675, 0.46253679, 0.81968765, 0.68449103, 0.32403382, 0.576812  ])

# Run one at a time
find_maximum()

# Run one at a time
# find_minimum()
