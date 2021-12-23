from functions import timeFunction, plot_NN

# =============================================================================
# 
# =============================================================================
import sys
import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
# from matplotlib import cm
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

## Set up the network

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def deep_neural_network(deep_params, x):
    # x is now a point and a 1D numpy array; make it a column vector
    num_coordinates = np.size(x,0)
    x = x.reshape(num_coordinates,-1)

    num_points = np.size(x,1)

    # N_hidden is the number of hidden layers
    N_hidden = np.size(deep_params) - 1 # -1 since params consist of parameters to all the hidden layers AND the output layer

    # Assume that the input layer does nothing to the input x
    x_input = x
    x_prev = x_input

    ## Hidden layers:

    for l in range(N_hidden):
        # From the list of parameters P; find the correct weigths and bias for this layer
        w_hidden = deep_params[l]

        # Add a row of ones to include bias
        x_prev = np.concatenate((np.ones((1,num_points)), x_prev ), axis = 0)

        z_hidden = np.matmul(w_hidden, x_prev)
        x_hidden = sigmoid(z_hidden)

        # Update x_prev such that next layer can use the output from this layer
        x_prev = x_hidden

    ## Output layer:

    # Get the weights and bias for this layer
    w_output = deep_params[-1]

    # Include bias:
    x_prev = np.concatenate((np.ones((1,num_points)), x_prev), axis = 0)

    z_output = np.matmul(w_output, x_prev)
    x_output = z_output

    return x_output[0][0]

## Define the trial solution and cost function
def u(x):
    return np.sin(np.pi*x)

def g_trial(point,P):
    x,t = point
    return (1-t)*u(x) + x*(1-x)*t*deep_neural_network(P,point)

# The right side of the ODE:
def function(point):
    return 0.

# The cost function:
def cost_function(P, x, t):
    cost_sum = 0

    g_t_jacobian_func = jacobian(g_trial)
    g_t_hessian_func = hessian(g_trial)

    for x_ in x:
        for t_ in t:
            point = np.array([x_,t_])

            # g_t = g_trial(point,P)
            g_t_jacobian = g_t_jacobian_func(point,P)
            g_t_hessian = g_t_hessian_func(point,P)

            g_t_dt = g_t_jacobian[1]
            g_t_d2x = g_t_hessian[0][0]

            func = function(point)

            err_sqr = ( (g_t_dt - g_t_d2x) - func)**2
            cost_sum += err_sqr

    return cost_sum /( np.size(x)*np.size(t) )

## For comparison, define the analytical solution
def g_analytic(point):
    x,t = point
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

## Set up a function for training the network to solve for the equation
@timeFunction
def solve_pde_deep_neural_network(x,t, num_neurons, num_iter, lmb):
    ## Set up initial weigths and biases
    N_hidden = np.size(num_neurons)

    ## Set up initial weigths and biases

    # Initialize the list of parameters:
    P = [None]*(N_hidden + 1) # + 1 to include the output layer

    P[0] = npr.randn(num_neurons[0], 2 + 1 ) # 2 since we have two points, +1 to include bias
    for l in range(1,N_hidden):
        P[l] = npr.randn(num_neurons[l], num_neurons[l-1] + 1) # +1 to include bias

    # For the output layer
    P[-1] = npr.randn(1, num_neurons[-1] + 1 ) # +1 since bias is included

    print('\n\nInitial cost: ',cost_function(P, x, t))

    cost_function_grad = grad(cost_function,0)

    # Let the update be done num_iter times
    for i in range(num_iter):
        cost_grad =  cost_function_grad(P, x , t)

        for l in range(N_hidden+1):
            P[l] = P[l] - lmb * cost_grad[l]

    print('Final cost: ', cost_function(P, x, t))

    return P

if __name__ == '__main__':
    ### Use the neural network:
    npr.seed(15)
    
    """
    ## Extracting the parameters use for the explicit scheme in order to do
    ## the testing with the sane preset
    Dir = "files/"
    filename= "explicit_scheme.txt"
    file = Dir + filename
    
    f_= open(file, "r")
    # parameters = class_objfetch_parameters(f)
    parameters = fetch_parameters(f_)
    print(f"\nparameters: {parameters}\n")
    
    Nt = int(parameters["Nt"])
    dt = float(parameters["dt"])
    t_idx = np.arange(Nt+1)
    t = t_idx*dt
    
    Nx = int(parameters["Nx"])
    dx = float(parameters["dx"])
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    """
    ## Decide the vales of arguments to the function to solve
    # alpha = alpha # Fourier number: alpha = D*dt/(dx)^2; D (diffusion coefficient) stability <= 1/2
    xstart = 0; xstop = 1
    L = xstop - xstart    
    Nx = 10
    dx = L/Nx
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    # print(f"index x: {x}")
    
    
    T = 1 # period [s] 
    Nt = 20
    dt = T/Nt
    t_idx = np.arange(Nt+1)
    t = t_idx*dt
    # print(f"index t: {t}")
    
    ## Set up the parameters for the network
    # num_hidden_neurons = [100, 25] # orginal
    # num_hidden_neurons = [Nt, Nx]
    # num_hidden_neurons = [Nx, Nt]
    num_hidden_neurons = [60, 60, 60]
    num_iter = 1000
    lmb = 0.01
    
    print(f"Number of iterations: {num_iter}")
    print(f"NHL: {len(num_hidden_neurons)}, NHN: {num_hidden_neurons}")
    print(f"L: {L} and Nx: {Nx} --> dx: {dx}")
    print(f"T: {T} and Nt: {Nt} --> dt: {dt}")
    
    
    Dir = "../files/"
    filename= f"NN_Nx_{Nx}_Nt_{Nt}_NHL_{len(num_hidden_neurons)}_{sum(num_hidden_neurons)}.txt"
    file = Dir + filename 
    f= open(file, "w")
    f.write(f"NHL: {len(num_hidden_neurons)}, NHN: ")
    f.writelines(str(i) + "|" for i in num_hidden_neurons)
    f.write(f"\nL: {L}, Nx: {Nx}, dx: {dx} \n")
    f.write(f"T: {T}, Nt: {Nt}, dt: {dt} \n")
    # f= open(file, "a")
    

    

    P = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb)
    
    ## Store the results
    g_dnn_ag = np.zeros((Nt+1, Nx+1))
    G_analytical = np.zeros((Nt+1, Nx+1))
    for i, t_ in enumerate(t):
        for j,x_ in enumerate(x):
            point = np.array([x_, t_])
            g_dnn_ag[i,j] = g_trial(point,P)

            G_analytical[i,j] = g_analytic(point)
    
        np.savetxt(f, g_dnn_ag[i], newline= " ", fmt= "%.8e")
        f.write('\n')
    
    f.close()
    
    # Find the map difference between the analytical and the computed solution
    diff_ag = np.abs(g_dnn_ag - G_analytical)
    print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag), "\n\n")
