import numpy as np
from sklearn.model_selection import train_test_split

class Scaler_Splitter:
    def __init__(self, x, y, Z, deg= 3, test_size= 0.2):
        
        print("Class Scaler_Splitter initializer")
        self.x = x; self.y = y; self.Z = Z;
        
        self.X = self.scikit_design_matrix(x, y, deg)
        
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
        
class Scikit_OLS(Scaler_Splitter):
    def __init__(self, scaler_obj):
        print("Class Scikit_OLS initializer")
        self.beta = None
        self.scaler = scaler_obj
        
    def fit(self, X, Z):
        from sklearn.linear_model import LinearRegression 
        linreg = LinearRegression()
        return linreg.fit(X, Z)
        
    
    def OLS_Regression(self):
        from sklearn.metrics import mean_squared_error, r2_score,\
            mean_squared_log_error, mean_absolute_error
        
        X_train_scaled = self.scaler.X_train_scaled; Z_train = self.scaler.Z_train
        X_test_scaled  = self.scaler.X_test_scaled; Z_test = self.scaler.Z_test
        
        
        linreg = self.fit(X_train_scaled, Z_train)
        
        # print('The intercept alpha: \n', linreg.intercept_.shape)
        # print('Coefficient beta : \n', linreg.coef_.shape)
        
        Z_tilde = linreg.predict(X_train_scaled)
        Z_pred  = linreg.predict(X_test_scaled)
        
        # Error analysis
        
        MSE_tilde = mean_squared_error(Z_train, Z_tilde)
        MAE_tilde = mean_absolute_error(Z_train, Z_tilde)
        R2_tilde = r2_score(Z_train, Z_tilde)
        
        print("-----------Tilde-----------")
        print("R2 Score:", R2_tilde, "MSE:", MSE_tilde, "MSA:", MAE_tilde)
        print("---------------------------\n")
        
        MSE_pred = mean_squared_error(Z_test, Z_pred)
        MAE_pred = mean_absolute_error(Z_test, Z_pred)
        R2_pred  = r2_score(Z_test, Z_pred)
        
        print("-----------Predict----------")
        print("R2 Score:", R2_pred, "MSE:", MSE_pred, "MSA:", MAE_pred)
        print("---------------------------\n")