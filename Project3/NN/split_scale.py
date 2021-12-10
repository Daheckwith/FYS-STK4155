import numpy as np
from sklearn.model_selection import train_test_split


class Numpy_split_scale:
    """
    Creates a design matrix using NumPy. Splits the data into a training-set 
    and a testing-set, using Scikit. Then scales it using NumPy.
    It also scales the data without splitting for cross-validation.
    
    """
    def __init__(self, x, y, Z, deg= 3, test_size= 0.2):
        
        if len(x.shape) > 1:
            print("Class Numpy_split_scale initializer reshape")
            # x = np.ravel(x)
            # y = np.ravel(y)
            # Z = np.ravel(Z)
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            Z = Z.reshape(-1,1)
        
        print("Scaler x, y, z: ", x.shape, y.shape, Z.shape)
        self.x = x; self.y = y; self.Z = Z; self.deg = deg
        # print("Deg:", deg)
        
        self.X = self.create_design_matrix(x, y, deg)
        self.X_scaled = self.numpy_scaler(self.X) #Scales X for Cross-validation later on
        
        X_train, X_test, Z_train, Z_test = train_test_split(self.X, self.Z, test_size= test_size)
        
        self.X_train_scaled = self.numpy_scaler(X_train)
        self.X_test_scaled = self.numpy_scaler(X_test)
        
        self.Z_train = Z_train; self.Z_test = Z_test
        
    def create_design_matrix(self, x, y, deg):
        """
        Creates a design matrix with columns:
        [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
        """
        
        if len(x.shape) > 1:
            print("Numpy_split_scale create_design_matrix ravel")
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
    
    def numpy_scaler(self, X):
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
        
        mean = np.mean(X_, axis=0); std = np.std(X_, axis=0); # var = np.var(X_, axis=0)
        X_scaled = (X_ - mean)/std
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) # Reconstructing X after ommiting the intercept
        
        return X
    
class Scikit_split_scale:
    """
    Creates a design matrix using Scikit. Splits the data into a training-set 
    and a testing-set then scales it using Scikit. It also scales the data 
    without splitting for cross-validation.
    
    """
    def __init__(self, x, y, Z, deg= 3, test_size= 0.2):
        print("Class Scikit_split_scale initializer")
        self.x = x; self.y = y; self.Z = Z; self.deg = deg
        # print("Deg:", deg)
        
        self.X = self.scikit_design_matrix(x, y, deg)
        self.X_scaled = self.scikit_scaler(self.X) #Scales X for Cross-validation later on
        
        X_train, X_test, Z_train, Z_test = train_test_split(self.X, self.Z, test_size= test_size)
        
        self.X_train_scaled = self.scikit_scaler(X_train) 
        self.X_test_scaled = self.scikit_scaler(X_test)
        
        self.Z_train = Z_train; self.Z_test = Z_test
    
    def scikit_design_matrix(self, x, y, deg):
        """
        Creates a design matrix with columns, using scikit:
        [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)
        
        # X = np.vstack((x,y)).T
        X = np.stack((x,y), axis = 1)
        # X = np.stack((x,y), axis = -1)
        poly = PolynomialFeatures(deg)
        X = poly.fit_transform(X)
        
        return X
    
    def scikit_scaler(self, X):
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
        
        
        scaler = StandardScaler()
        # scaler.fit(X_)
        # X_scaled = scaler.transform(X_)
        X_scaled = scaler.fit_transform(X_)
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) #Reconstructing X after ommiting the intercept
        
        return X