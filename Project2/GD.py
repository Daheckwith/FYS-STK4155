import numpy as np
from sys import exit
from sklearn.metrics import mean_squared_error



class Gradient_descent:
    def __init__(self, X, Z, reg_obj):
        print("Class Gradient_descent initialized")
        
        self.reg = reg_obj # regression
        
        self.N = X.shape[0]
        self.p = X.shape[1]
        print("N: ", self.N, " p: " , self.p)   
        
        self.epsilon = 1e-08
        
        if len(Z.shape) == 1:
            Z = Z[:,np.newaxis]
        
        self.lmb = 0
        
        self.start_beta = np.random.randn(self.p, 1) # change 1 to response.shape[1]
        
        self.X = X
        self.Z = Z
        
        print("X:", X.shape, "Z:", Z.shape)
        N= self.N; X_T = X.T;
        
        # Hessian matrix (should probably remove)
        H = (2/N) * X_T @ X
        EigValues, EigVectors = np.linalg.eig(H)
        
        learning_rate = 1.0/np.max(EigValues)
        if learning_rate > 1e-04:
            self.eta = learning_rate
        else:
            self.eta = 1e-04
        # print("eta: ", learning_rate)
    
    def Regression_gradient(self, X, Z, beta, method= "OLS"):
        X_T = X.T; N = X.shape[0]
        
        if method in ["ols", "OLS", "Ols"]:
            gradient = (2.0/N)*X_T @ (X @ beta - Z)
        elif method in ["ridge", "Ridge", "RIDGE"]:
            lmb = self.lmb
            gradient = (2.0/N)*X_T @ (X @ beta - Z) + 2*lmb*beta
        
        return gradient
    
    def Sigmoid(self, t):
        return 1/(1 + np.exp(-t))
    
    def logistic_prediction(self, weights, inputs):
        # print("dim weights: ", weights.shape, " dim inputs: ", inputs.shape)
        return self.Sigmoid(np.matmul(inputs, weights))
    
    def cross_entropy_grad(self, X, Z, beta, method= "Logistic Regression"):
        X_T = X.T
        
        # beta is the same as weights
        p = self.logistic_prediction(beta, X)
        return X_T @ (p - Z)
    
    def Gradient(self, method= "OLS"):
        if method in ["ols", "OLS", "Ols", "ridge", "Ridge", "RIDGE"]:
            return self.Regression_gradient
        else:
            # print("cross_entorpy_grad")
            return self.cross_entropy_grad
    
    def Simple_GD(self, initial_guess, method= "OLS", Niterations= 300,\
                  learning_rate= None, plotting = False):
        beta = initial_guess.copy()
        X = self.X; Z = self.Z;
        
        if learning_rate == None:
            eta = self.eta
        else:
            eta = learning_rate
        
        
        print(f"\nSimple GD {method}")
        print(f"Learning rate: {eta}")
        
        MSE_cache = []
        
        n = 10000 # Default number of iterations not adjustable by user
        
        if method in ["ridge", "Ridge", "RIDGE"]:
            print("lmb: ", self.lmb)
        
        gradient_func = self.Gradient(method= method)
        
        if not plotting:
            for i in range(n):
                gradient = gradient_func(X, Z, beta, method= method)
                beta -= eta*gradient
                
        elif plotting:
            print("Plotting preset initiated")
            n = 0
                
            
        
        if Niterations > n:
            # print("I'm bigger than you thought!")
            Z_tilde = self.reg.predict(X, beta)
            MSE = mean_squared_error(Z, Z_tilde)
            MSE_cache.append(MSE)
            
            for i in range(Niterations - n):
                gradient = gradient_func(X, Z, beta, method= method)
                beta -= eta*gradient
                
                # Convergence test
                Z_tilde = self.reg.predict(X, beta)
                MSE = mean_squared_error(Z, Z_tilde)
                MSE_cache.append(MSE)
                
                diff = abs(MSE_cache[i+1] - MSE_cache[i])
                
                if diff < self.epsilon:
                    break
                
                # if np.abs(gradient).max() < 1e-08:
                #     break
                
            print(f"Stopped GD {method} at iteration i: {i + n} \n")
        
        return beta, MSE_cache, i+n
    
    def Stochastic_GD(self, initial_guess, method= "OSL", batch_size= 5,\
                      epochs= 30, learning_rate= None, plotting= False):
        beta = initial_guess.copy()
        X = self.X; Z = self.Z
        N = self.N; p = self.p; eta = self.eta;
        # print("N: ", N, "p: ", p)
        
        if learning_rate == None:
            eta = 1e-04
        else:
            eta = learning_rate
        
        
        n = 30 # Default number of epochs not adjustable by user
        gamma = 0.2
        running_avg = np.zeros((p,1))
        
        print(f"\nStochastic GD {method}")
        print(f"Learning rate: {eta}")
        
        MSE_cache = []
        
        M = batch_size
        m = int(N/M) # number of mini-batches
        
        print(f"Data points: {N}, Batch size: {M}, number of min-batches: {m}")
        
        
        
        if method in ["ridge", "Ridge", "RIDGE"]:
            print("lmb: ", self.lmb)
        
        if N%M != 0:
            print(f"Size of mini-batches M = {M} is not divisble by N = {N}")
            exit()
            
        gradient_func = self.Gradient(method= method)
        
        indices = np.arange(N)
        if not plotting:
            for epoch in range(n):    
                rnd_indices = np.copy(indices)
                np.random.shuffle(rnd_indices)
                for i in range(m):
                    # k = np.random.randint(m)
                    # batch = rnd_indices[k*M: k*M + M]
                    batch = rnd_indices[i*M: i*M + M]
                    
                    X_ = X[batch]
                    Z_ = Z[batch]
                    
                    # print(batch.shape, X_.shape, Z_.shape)
                    
                    # gradient = gradient_func(X_, Z_, beta, method= method)
                    # beta -= eta*gradient
                    
                    # momentum
                    gradient = gradient_func(X_, Z_, beta, method= method)
                    running_avg = gamma*running_avg + eta*gradient
                    beta -= running_avg
                    
        elif plotting:
            print("Plotting preset initiated")
            n = 0
        
        if epochs > n:
            # print("I'm bigger than you thought!")
            Z_tilde = self.reg.predict(X, beta)
            MSE = mean_squared_error(Z, Z_tilde)
            MSE_cache.append(MSE)
            for epoch in range(epochs - n):  
                rnd_indices = np.copy(indices)
                np.random.shuffle(rnd_indices)
                for i in range(m):
                    # k = np.random.randint(m)
                    # batch = rnd_indices[k*M: k*M + M]
                    batch = rnd_indices[i*M: i*M + M]
                    
                    X_ = X[batch]
                    Z_ = Z[batch]
                    
                    # gradient = gradient_func(X_, Z_, beta, method= method)
                    # beta -= eta*gradient
                    
                    # momentum
                    gradient = gradient_func(X_, Z_, beta, method= method)
                    running_avg = gamma*running_avg + eta*gradient
                    beta -= running_avg
                    
                    
                # Convergence test
                Z_tilde = self.reg.predict(X, beta)
                MSE = mean_squared_error(Z, Z_tilde)
                MSE_cache.append(MSE)
                
                diff = abs(MSE_cache[epoch+1] - MSE_cache[epoch])
                
                if diff < self.epsilon:
                    break
                
                # if np.abs(gradient).max() < 1e-08:
                    # print(f"Stopped SGD {method} at epoch: {n + epoch}")
                    # break
            
            print(f"Stopped SGD {method} at epoch: {n + epoch} \n")
        return beta, MSE_cache, n+epoch
            
            
    
    def Scikit_SGD(self, X, Z, max_iter= 50):
        # change x y to X Z look how this is done
        from sklearn.linear_model import SGDRegressor
        sgdreg = SGDRegressor(max_iter= max_iter, penalty=None, eta0= self.eta)
        sgdreg.fit(X, Z.ravel())
        return sgdreg
        # return sgdreg, sgdreg.intercept_, sgdreg.coef_