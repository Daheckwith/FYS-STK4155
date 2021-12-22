import numpy as np
import matplotlib.pyplot as plt

from functions import timeFunction, u_exact, g_analytic

class Explicit_scheme:
    def __init__(self, N = 100, xstart = 0, xstop = 1, alpha = 1/2, T = 1):
        self.xstart = xstart; self.xstop = xstop
        self.L = xstop - xstart
        self.alpha = alpha # Fourier number: alpha = D*dt/(dx)^2; D (diffusion coefficient) stability <= 1/2
        
        step = self.L/N

        self.Nx = N
        self.dx = step
        self.x_idx = np.arange(self.Nx+1)
        self.x = self.x_idx*self.dx
        
        self.T = T # period [s] 
        self.dt = (self.dx**2)*self.alpha
        self.dt = round(self.dt, 18)
        self.Nt = round(self.T/self.dt)
        # self.t_idx = np.arange(self.Nt+1)
        # t = t_idx*dt
        
        
        print(f"alpha: {self.alpha}, xstart: {self.xstart}, xstop: {self.xstop}")        
        print(f"L: {self.L} and Nx: {self.Nx} --> dx: {self.dx}")
        print(f"T: {self.T} and Nt: {self.Nt} --> dt: {self.dt}")
        
        self.u = np.zeros(self.Nx+1)
        self.unew = self.u.copy()
        self.initialize(self.u)
        
    def initial_condition(self, x):
        return np.sin(np.pi*x)

    def initialize(self, u, method= "vectorized"):
        Nx = self.Nx; x = self.x; dx = self.dx
        
        if method == "vectorized":
            #vectorized
            u[1:Nx] = self.initial_condition(x[1:Nx]) 
        else:
            for i in range(1, Nx): # BCs u(0,t) = u(L,t) = 0 for t >= 0
                u[i] = self.initial_condition(i*dx)
                
    # @timeFunction
    def solver(self, number_of_lines= 10, write= False, filename= "explicit_scheme.txt"):
        
        N  = number_of_lines
        
        alpha = self.alpha; u = self.u.copy(); unew = self.unew.copy()
        Nt = self.Nt; dt = self.dt
        Nx = self.Nx; x = self.x
        
        u_ = unew.copy()
        
        if not write:
            # plt.close("all")
            plt.figure()
            s = 0
            for t in range(1, Nt+1): # from dt to Nt*dt = T
                for i in range(1, Nx): # from dx to (Nx-1)*dx = L-dx
                    unew[i] = alpha*u[i-1] + (1-2*alpha)*u[i] + alpha*u[i+1]
                
                u = unew # updating "u" at the end of each time iteration "t"
                if s%(Nt/N) == 0: # plot N plots
                    # print(f"time: {t*dt}")
                    u_ = u_exact(x, t*dt)
                    if s == 0:
                        plt.plot(x, u_, "k--", label= "Analytical")
                    else:
                        plt.plot(x, u_, "k--")
                    plt.plot(x, u, label= f"t: {t*dt:.2f} s")
                    
                s += 1
            
            plt.xlabel("Position $x$"); plt.ylabel("Numerical Solution for $u(x,t)$")
            plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
            # plt.legend()
            plt.show()
        
        elif write: # Writes to file
            if filename == None:
                self.filename= f"explicit_scheme_Nx_{Nx}_Nt_{Nt}.txt"
            else:
                self.filename = filename
            
            Dir = "files/"
            file = Dir + self.filename 
            f= open(file, "w")
            f.write(f"alpha: {self.alpha}, xstart: {self.xstart}, xstop: {self.xstop} \n")
            f.write(f"L: {self.L}, Nx: {self.Nx}, dx: {self.dx} \n")
            f.write(f"T: {self.T}, Nt: {self.Nt}, dt: {self.dt} \n")
            np.savetxt(f, u, newline=" ", fmt= "%.8e")
            f.write('\n')
            
            for t in range(1, Nt+1): # from dt to Nt*dt = T
                for i in range(1, Nx): # from dx to (Nx-1)*dx = L-dx
                    unew[i] = alpha*u[i-1] + (1-2*alpha)*u[i] + alpha*u[i+1]
                    
                u = unew # updating "u" at the end of each time iteration "t"
                
                np.savetxt(f, u, newline= " ", fmt= "%.8e")
                f.write('\n')
                    
            f.close()