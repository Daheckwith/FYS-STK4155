import numpy as np

#Scikitlearn
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split

class Regression:
    def __init__(self, x, y, deg):
        """
        
        reg.Regression(x,y) activates __init__
        but not when __new__ is added   
        
        """
        self.x = x
        self.y = y
        
        # self.X = create_design_matrix(x, y, deg)
        # self.X_T = self.X.T
        # print("__init__ Regression initialized")
        # print("__init__ x: ", self.x, "y: ", self.y, "X: ", self.X)
        
        
    def create_design_matrix(x, y, deg):
        """
        Creates a design matrix with columns:
        [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        p = int((deg + 1)*(deg + 2)/2)
        X = np.ones((N,p))

        for i in range(1, deg + 1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = x**(i-k) * y**k
        
        return X
    
    def scikit_design_matrix(x, y, deg):
        """
        Creates a design matrix with columns, using scikit:
        [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        # X = np.vstack((x,y)).T
        X = np.stack((x,y), axis = 1)
        # X = np.stack((x,y), axis = -1)
        poly = PolynomialFeatures(deg)
        X = poly.fit_transform(X)
        
        return X
        
                
    def numpy_scaler(X):
        """
        Scaling a polynolmial matrix using NumPy
        
        Parameters
        ----------
        X : Matrix to be scaled

        Returns
        -------
        X : Scaled matrix

        """
        X_ = X[:,1:] # Ommitting the intercept
        
        # print("----------------------Numpy--------------------------------")
        mean = np.mean(X_, axis=0 ); std = np.std(X_, axis=0); # var = np.var(X_, axis=0)
        X_scaled = (X_ - mean)/std
        # print("Dim X: ", X.shape, " Dim X_:", X_scaled.shape)
        # print("np.mean: ", mean, " mean.shape: ", mean.shape, "np.var: ", var, " var.shape: ", var.shape )
        # print("X_np_scaled[6]: ", X_scaled[6])
        # print("------------------------------------------------------------\n")
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) # Reconstructing X after ommiting the intercept
        
        return X
        
    def scikit_scaler(X):
        """
        Scaling a polynolmial matrix using StandardScaler from Scikit-learn
        
        Parameters
        ----------
        X : Matrix to be scaled

        Returns
        -------
        X : Scaled matrix

        """
        from sklearn.preprocessing import StandardScaler
        
        X_ = X[:,1:] # Ommitting the intercept
        
        # print("---------------------Scikit---------------------------------")
        scaler = StandardScaler()
        # scaler.fit(X_)
        # X_scaled = scaler.transform(X_)
        # print("X.shape: ", X[:,1:].shape, " mean: ", scaler.mean_, " shape: ", scaler.mean_.shape, "var:", scaler.var_ )
        # print("X_scaled[6]: ", X_scaled[6])
        X_scaled = scaler.fit_transform(X_)
        # print("X_scaled[6] back to back: ", X_scaled[6])
        # print("------------------------------------------------------------\n")
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) #Reconstructing X after ommiting the intercept
        
        return X
        
            
    # def __new__(self, x, y):
        """
        
        reg.OLS(x,y) activates __new__

        """
        # self.x = x
        # self.y = y
        # self.X = create_design_matrix(self)
        # print("__new__ Regression initialized")
        # print("__new__ x: ", self.x, "y: ", self.y, "X: ", self.X)
    

class Manual_OLS(Regression):
    def __init__(self, x, y, deg = 3):
        Regression.__init__(self, x, y, deg)
        self.beta = None;
        self.X = Regression.create_design_matrix(x, y, deg)
    
    def find_beta(self, X, z):
        X_T = X.T
        return np.linalg.pinv(X_T @ X) @ X_T @ z
        
    def OLS_Regression(self, z, test_size):
        import error as err
        from sklearn.model_selection import train_test_split
        
        X = Regression.numpy_scaler(self.X)
        # X2 = Regression.scikit_scaler(self.X)
            
        
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size= test_size)
        self.beta = self.find_beta(X_train, z_train)
        z_tilde = X_train @ self.beta
        z_pred = X_test @ self.beta
        
        # print("beta: ", self.beta.shape, " X_train: ", X_train.shape, " z_tilde: ", z_tilde.shape, " size: ", np.size(z_tilde))
        # print("z_tilde size:", z_tilde.size)
        
        # Error analysis
        Err_tilde = err.Error(z_tilde)
        
        MSE_tilde = Err_tilde.MSE(z_train, z_tilde)
        MAE_tilde = Err_tilde.MAE(z_train, z_tilde)
        R2_tilde  = Err_tilde.R2_Score(z_train, z_tilde)
        
        print("-----------Tilde-----------")
        print("R2 Score:", R2_tilde, "MSE:", MSE_tilde, "MSA:", MAE_tilde)
        print("---------------------------\n")
        
        Err_pred  = err.Error(z_pred)
        
        MSE_pred = Err_pred.MSE(z_test, z_pred)
        MAE_pred = Err_pred.MAE(z_test, z_pred)
        R2_pred  = Err_pred.R2_Score(z_test, z_pred)
        
        print("-----------Predict----------")
        print("R2 Score:", R2_pred, "MSE:", MSE_pred, "MSA:", MAE_pred)
        print("---------------------------\n")
        
        #Plotting
        import frankefunction as fr
        # Fr = fr.Franke()
        # Fr.plot_franke()
        
class Scikit_OLS(Regression):
    def __init__(self, x, y, deg = 3):
        Regression.__init__(self, x, y, deg)
        self.beta = None;
        self.X = Regression.scikit_design_matrix(x, y, deg)
        
    def find_beta(self, X, z):
        print("I'm here")
        
    