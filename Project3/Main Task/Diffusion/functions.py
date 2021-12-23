import time
import numpy as np

## Analytical solution
def u_exact(*args):
    if len(args) == 1 and isinstance(args[0], tuple):
        # print("first")
        x, t = args[0]
    elif len(args) == 2:
        # print("second")
        x= args[0]; t= args[1]
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
    
    Dir = "../files/"
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
        print(f"Extracting all results from file! {Nt + 1} lines. NOT RECOMMENDED if the file is big!!")
        N  = Nt
    
    u = np.empty((N+1, Nx+1))
    time_step = np.empty(N+1)
    
    if type(number_of_lines) == int:
        s = 0
        count = 0 
        for t in range(0, Nt+1): # from dt to Nt*dt = T
            line = f.readline()
            if s%(Nt/N) == 0: # extract N lines uniformly spaced from each other
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

# Plotting

def plot_line(u, time_step, parameters, idx, NHL= None, NHN= None):
    import matplotlib.pyplot as plt
    print(f"{plot_line.__name__}")
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    Nt = int(parameters["Nt"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    u_e = u_exact(x, time_step[idx])
    u_num = u[idx]
    mse = ((u_num[1:-1] - u_e[1:-1])**2).mean()
    print(f"MSE at time step {time_step[idx]:.2f}: {mse:.4f}")
    
    plt.figure()
    plt.text(x[round(Nx/2)-1], np.max([u_e.max(), u_num.max()]) , f"MSE: {mse:.4f}", fontsize=12)
    plt.plot(x, u_e, "k--", label= "Analytical")
    plt.plot(x, u_num, label= f"Numerical t: {time_step[idx]:.2f}")
    
    plt.xlabel("Position $x$"); #plt.ylabel("Numerical Solution for $u(x,t)$")
    plt.ylabel("$u(x,t)$")
    plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
    # plt.legend()
    save_fig(f"../Figures/plot_line_{NHL}_{NHN}_{Nx}_{Nt}_{idx}.png")
    plt.show()

def plot_lines(u, time_step, parameters, NHL= None, NHN= None, number_of_lines= None):
    import matplotlib.pyplot as plt
    print(f"{plot_lines.__name__}")
    N  = u.shape[0]
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    Nt = int(parameters["Nt"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    s = 0
    count = 0 
    for i in range(N):
        if s%int(N/number_of_lines) == 0: # extract N lines uniformly spaced from each other
            u_e = u_exact(x, time_step[i])
            u_num = u[i]
            mse = ((u_num[1:-1] - u_e[1:-1])**2).mean()
            print(f"MSE at time step {time_step[i]:.2f}: {mse:.4f}")
            
            plt.figure()
            plt.text(x[round(Nx/2)-1], np.max([u_e.max(), u_num.max()]) , f"MSE: {mse:.4f}", fontsize=12)
            plt.plot(x, u_e, "k--", label= "Analytical")
            plt.plot(x, u_num, label= f"Numerical t: {time_step[i]:.2f}")
            
            plt.xlabel("Position $x$"); #plt.ylabel("Numerical Solution for $u(x,t)$")
            plt.ylabel("$u(x,t)$")
            plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
            # plt.legend()
            save_fig(f"../Figures/plot_lines_{NHL}_{NHN}_{Nx}_{Nt}_{count}.png")
            plt.show()
            count +=1
        s += 1
    

def sub_plots(u, time_step, parameters, NHL= None, NHN= None, number_of_lines= None):
    import matplotlib.pyplot as plt
    print(f"{sub_plots.__name__}")
    N  = u.shape[0]
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    Nt = int(parameters["Nt"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    half = int(number_of_lines/2)
    fig, axes = plt.subplots(2, half, sharex="col", sharey="row")
    s = 0
    count = 0 
    for i in range(N):
        s += 1
        if s%int(N/number_of_lines) == 0: # extract N lines uniformly spaced from each other
            row = round(count/number_of_lines + 0.0001)
            col = count - row*half
            
            u_e = u_exact(x, time_step[i])
            u_num = u[i]
            mse = ((u_num[1:-1] - u_e[1:-1])**2).mean()
            
            axes[row, col].set_title(f"t: {time_step[i]:.2f}", fontsize=12)
            axes[row,col].text(x[Nx-round(Nx/2)+2], np.max([u_e.max(), u_num.max()]), f"MSE: {mse:.4f}", fontsize=10)
            # axes[row,col].text(x[0], np.max([u_e.max(), u_num.max()]) + 0.001 , f"MSE: {mse:.4f}", fontsize=10)
            axes[row,col].plot(x, u_e, "k--")
            axes[row,col].plot(x, u_num)
            count +=1
        # s += 1
        
    for ax in axes.flat:
        ax.set(xlabel= "Position $x$", ylabel= r"$u(x,t)$")
    for ax in axes.flat:
        ax.label_outer()
    
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2, rect=(-0.05, -0.05, 1, 1))
    plt.subplots_adjust(hspace=0.2, wspace=0.07)
    save_fig(f"../Figures/sub_plots_{NHL}_{NHN}_{Nx}_{Nt}_{count}.png")
    plt.show()



def plot_2D(u, time_step, parameters, NHL= None, NHN= None):
    import matplotlib.pyplot as plt
    print(f"{plot_2D.__name__}")
    N  = u.shape[0]
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    Nt = int(parameters["Nt"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    avg_mse = []
    
    plt.figure()
    for i in range(N):
        u_e = u_exact(x, time_step[i])
        u_num = u[i]
        mse = ((u_num[1:-1] - u_e[1:-1])**2).mean()
        avg_mse.append(mse)
        # print(f"MSE at time step {time_step[i]:.2f}: {mse:.4f}")
        if not i == 0:
            plt.plot(x, u_e, "k--")
            plt.plot(x, u_num, label= f"Numerical t: {time_step[i]:.2f}")
        elif i == 0:
            plt.plot(x, u_e, "k--", label= f"Analytical")
            plt.plot(x, u_num, label= f"Numerical t: {time_step[i]:.2f}")
    
    avg_mse = np.mean(avg_mse)
    print(f"Avg MSE across the whole time domain: {avg_mse:.4f}")
    
    plt.xlabel("Position $x$"); #plt.ylabel("Numerical Solution for $u(x,t)$")
    plt.ylabel("$u(x,t)$")
    plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
    # plt.legend()
    save_fig(f"../Figures/plot_2D_{NHL}_{NHN}_{Nx}_{Nt}.png")
    plt.show()
    return avg_mse


def simple_plot_2D(u, time_step, parameters):
    import matplotlib.pyplot as plt
    
    N  = u.shape[0]
    dx = float(parameters["dx"])
    Nx = int(parameters["Nx"])
    
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    
    plt.figure()
    for i in range(N):
        u_e = u_exact(x, time_step[i])
        u_num = u[i]
        if not i == 0:
            plt.plot(x, u_e, "k--")
            plt.plot(x, u_num, label= f"Numerical t: {time_step[i]:.2f}")
        elif i == 0:
            plt.plot(x, u_e, "k--", label= f"Analytical")
            plt.plot(x, u_num, label= f"Numerical t: {time_step[i]:.2f}")
    
    plt.xlabel("Position $x$"); #plt.ylabel("Numerical Solution for $u(x,t)$")
    plt.ylabel("$u(x,t)$")
    plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
    # plt.legend()
    plt.show()
    
def save_fig(name):
    import matplotlib.pyplot as plt
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')
    