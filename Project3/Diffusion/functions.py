import time
import numpy as np

## Analytical solution
def u_exact(*args):
    if len(args) == 1 and isinstance(args[0], tuple):
        # print("first")
        x,t = args[0]
    elif len(args) == 2:
        # print("second")
        x= args[0]; t= args[1]
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


def g_analytic(x,t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

# Timing Decorator

def timeFunction(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s Function Took: \t %0.3f s' % (f.__name__, (time2-time1)))
        return ret
    return wrap


# Reading from file

def extract_line(line, parameters):
    List = []
    
    for part in line.split():
        if ":" in part:
            part = part.replace(":", "")
        elif "," in part:
            part = part.replace(",", "")
        List.append(part)        
    
    for i in range(0, len(List), 2):
        parameters[List[i]] = List[i+1]
        
    return parameters

def fetch_parameters(f):
    """
    Reads the first 3 lines to extract the parameters

    Input
    ----------
    f : file object type <class '_io.TextIOWrapper'> 

    Returns
    -------
    parameters : Dictionary containing all parameters
                    in the first 3 lines.

    """
    parameters = {}
    for i in range(3):
        line = f.readline()
        parameters = extract_line(line, parameters)
    
    return parameters

def read_file(filename= "explicit_scheme.txt", number_of_lines= 10):
    """
    Reads the file and extracts the parameters from
    the first 3 lines. Then loops through every remnaing
    line in file to extract a number of results N for the
    entire spatial domain at a given time step t.

    Parameters
    ----------
    filename : str
        The name of the file. The default is "explicit_scheme.txt".
    number_of_lines : int or 'str', optional
        Takes in the number of times the user wants to extract a line in file.
        It can take in a str= "extract_all" or "all" to extract every value of u for every
        time step t. The default is 10.

    Returns
    -------
    u : numpy.ndarray
        contains the solution u for multiple time steps t
    time_step : numpy.ndarray
        the time step correspinding to each result contained in u
    parameters : dict
        The parameters/preset of the PDE

    """
    
    Dir = "files/"
    file = Dir + filename
    print("read_file: ", file)
    f= open(file, "r")
    # parameters = class_objfetch_parameters(f)
    parameters = fetch_parameters(f)
    print(f"\nparameters: {parameters}\n")
    
    
    Nt = int(parameters["Nt"])
    Nx = int(parameters["Nx"])
    dt = float(parameters["dt"])
    
    if type(number_of_lines) == int:
        N  = number_of_lines
        print(f"Extracting {N+1} results that are uniformely spaced in time from each other.")
    else:
        print(f"Extracting all results from file! {Nt + 1} lines. NOT RECOMMENDED!!")
        N  = Nt
    
    u = np.empty((N+1, Nx+1))
    time_step = np.empty(N+1)
    
    if type(number_of_lines) == int:
        s = 0
        count = 0 
        for t in range(0, Nt+1): # from dt to Nt*dt = T
            line = f.readline()
            if s%(Nt/N) == 0: # extract N lines uniformly spaced from each other
                print(f"s: {s}, count: {count}, Nt: {Nt}, N: {N}")
                u_i = np.array(line.split(), dtype= "float64")
                u[count,:] = u_i
                time_step[count] = dt*s
                count +=1
            s += 1
    else:
        for t in range(0, Nt+1): # from dt to Nt*dt = T
            line = f.readline()
            u_i = np.array(line.split(), dtype= "float64")
            u[t,:] = u_i
            time_step[t] = dt*t
    
    f.close()
    return u, time_step, parameters

# Plots

def plot_2D(u, time_step, parameters):
    import matplotlib.pyplot as plt
    
    N  = u.shape[0]
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    plt.figure()
    for i in range(N):
        if not i == 0:
            plt.plot(x, u_exact(x, time_step[i]), "k--")
            plt.plot(x, u[i], label= f"Numerical t: {time_step[i]:.2f}")
        elif i == 0:
            plt.plot(x, u_exact(x, time_step[i]), "k--", label= f"Analytical")
            plt.plot(x, u[i], label= f"Numerical t: {time_step[i]:.2f}")
    
    plt.xlabel("Position $x$"); #plt.ylabel("Numerical Solution for $u(x,t)$")
    plt.ylabel("$u(x,t)$")
    plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
    # plt.legend()
    plt.show()

def plot_line(u, time_step, parameters, idx):
    import matplotlib.pyplot as plt
    
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    plt.figure()
    plt.plot(x, u_exact(x, time_step[idx]), "k--", label= "Analytical")
    plt.plot(x, u[idx], label= f"Numerical t: {time_step[idx]:.2f}")
    
    plt.xlabel("Position $x$"); #plt.ylabel("Numerical Solution for $u(x,t)$")
    plt.ylabel("$u(x,t)$")
    plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
    # plt.legend()
    plt.show()

def plot_NN(t, x, num_hidden_neurons, g_dnn_ag, G_analytical, diff_ag):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    ## Plot the solutions in two dimensions, that being in position and time
    
    Nt  = len(t)
    # T,X = np.meshgrid(t,x)

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')
    # ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
    # s = ax.plot_surface(T,X,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    # ax.set_xlabel('Time $t$')
    # ax.set_ylabel('Position $x$');


    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')
    # ax.set_title('Analytical solution')
    # s = ax.plot_surface(T,X,G_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
    # ax.set_xlabel('Time $t$')
    # ax.set_ylabel('Position $x$');

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')
    # ax.set_title('Difference')
    # s = ax.plot_surface(T,X,diff_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    # ax.set_xlabel('Time $t$')
    # ax.set_ylabel('Position $x$');

    ## Take some slices of the 3D plots just to see the solutions at particular times
    indx1 = 0
    indx2 = int(Nt/2)
    indx3 = Nt-1

    t1 = t[indx1]
    t2 = t[indx2]
    t3 = t[indx3]

    # Slice the results from the DNN
    # res1 = g_dnn_ag[:,indx1]
    # res2 = g_dnn_ag[:,indx2]
    # res3 = g_dnn_ag[:,indx3]
    res1 = g_dnn_ag[indx1,:]
    res2 = g_dnn_ag[indx2,:]
    res3 = g_dnn_ag[indx3,:]

    # Slice the analytical results
    res_analytical1 = G_analytical[indx1,:]
    res_analytical2 = G_analytical[indx2,:]
    res_analytical3 = G_analytical[indx3,:]

    # Plot the slices
    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t1)
    plt.plot(x, res1)
    plt.plot(x,res_analytical1)
    plt.legend(['dnn','analytical'])

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t2)
    plt.plot(x, res2)
    plt.plot(x,res_analytical2)
    plt.legend(['dnn','analytical'])

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t3)
    plt.plot(x, res3)
    plt.plot(x,res_analytical3)
    plt.legend(['dnn','analytical'])

    plt.show()
    
def save_fig(name):
    import matplotlib.pyplot as plt
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')
    