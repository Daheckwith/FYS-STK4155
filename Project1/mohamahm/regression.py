import numpy as np
from sklearn.model_selection import train_test_split

class Scaler_Splitter:
    def __init__(self, x, y, Z, deg= 3, test_size= 0.2, method = None):
        # super(Scaler_Splitter, self).__init__()
        print("Class Scaler_Splitter initializer")
        self.x = x; self.y = y; self.Z = Z; self.method = method; self.deg = deg
        
        if self.method == "Manual":
            print("Class Scaler_Splitter Manual")
            self.X = self.create_design_matrix(x, y, deg)
        else:
            print("Class Scaler_Splitter Scikit")
            self.X = self.scikit_design_matrix(x, y, deg)
            
        
        X_train, X_test, Z_train, Z_test = train_test_split(self.X, self.Z, test_size= test_size)
        
        if self.method == "Manual":
            print("Class Scaler_Splitter Manual")
            self.X_train_scaled = self.numpy_scaler(X_train)
            self.X_test_scaled = self.numpy_scaler(X_test)
        else:
            print("Class Scaler_Splitter Scikit")
            self.X_train_scaled = self.scikit_scaler(X_train)
            self.X_test_scaled = self.scikit_scaler(X_test)
        
        self.Z_train = Z_train; self.Z_test = Z_test
        
    def create_design_matrix(self, x, y, deg):
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
        
        # print("----------------------Numpy--------------------------------")
        mean = np.mean(X_, axis=0 ); std = np.std(X_, axis=0); # var = np.var(X_, axis=0)
        X_scaled = (X_ - mean)/std
        # print("Dim X: ", X.shape, " Dim X_:", X_scaled.shape)
        # print("np.mean: ", mean, " mean.shape: ", mean.shape, "np.var: ", var, " var.shape: ", var.shape )
        # print("X_np_scaled[6]: ", X_scaled[6])
        # print("------------------------------------------------------------\n")
        
        X = np.insert( X_scaled, 0, X[:,0], axis= 1 ) # Reconstructing X after ommiting the intercept
        
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
    
class Manual_OLS(Scaler_Splitter):
    def __init__(self, scaler_obj):
        print("Class Manual_OLS initializer")
        self.beta = None
        self.scaler = scaler_obj
        
        
    def fit(self, X, Z):
        X_T = X.T
        return np.linalg.pinv(X_T @ X) @ (X_T @ Z)
        # return np.dot(np.linalg.pinv(np.dot(X_T,X)),np.dot(X_T,Z))
        
    
    def OLS_Regression(self):
        print("Manual regression train-test split")
        import error as err
        from sklearn.metrics import mean_squared_error, r2_score,\
            mean_squared_log_error, mean_absolute_error
        
        X_train_scaled = self.scaler.X_train_scaled; Z_train = self.scaler.Z_train
        X_test_scaled  = self.scaler.X_test_scaled; Z_test = self.scaler.Z_test
        self.beta = self.fit( X_train_scaled, Z_train)
        Z_tilde   = X_train_scaled @ self.beta
        Z_pred    = X_test_scaled @ self.beta
        
        # print("self.beta: ", self.beta.shape)
        
        # Error analysis
        Err_tilde = err.Error(Z_tilde)
        
        MSE_tilde = Err_tilde.MSE(Z_train, Z_tilde)
        MAE_tilde = Err_tilde.MAE(Z_train, Z_tilde)
        R2_tilde  = Err_tilde.R2_Score(Z_train, Z_tilde)
        
        print("-----------Tilde-----------")
        print("R2 Score:", R2_tilde, "MSE:", MSE_tilde, "MSA:", MAE_tilde)
        print("---------------------------\n")
        
        # MSE_tilde_sci = mean_squared_error(Z_train, Z_tilde)
        # MAE_tilde_sci = mean_absolute_error(Z_train, Z_tilde)
        # R2_tilde_sci  = r2_score(Z_train, Z_tilde)
        
        # print("-----------Sci-Tilde-----------")
        # print("R2 Score:", R2_tilde_sci, "MSE:", MSE_tilde_sci, "MSA:", MAE_tilde_sci)
        # print("---------------------------\n")
        
        Err_pred  = err.Error(Z_pred)
        
        MSE_pred = Err_pred.MSE(Z_test, Z_pred)
        MAE_pred = Err_pred.MAE(Z_test, Z_pred)
        R2_pred  = Err_pred.R2_Score(Z_test, Z_pred)
        
        print("-----------Predict----------")
        print("R2 Score:", R2_pred, "MSE:", MSE_pred, "MSA:", MAE_pred)
        print("---------------------------\n")
        
        
        # MSE_pred_sci = mean_squared_error(Z_test, Z_pred)
        # MAE_pred_sci = mean_absolute_error(Z_test, Z_pred)
        # R2_pred_sci  = r2_score(Z_test, Z_pred)
        
        # print("-----------Sci-Predict-----------")
        # print("R2 Score:", R2_pred_sci, "MSE:", MSE_pred_sci, "MSA:", MAE_pred_sci)
        # print("---------------------------\n")
        
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
        print("Scikit regression train-test split")
        from sklearn.metrics import mean_squared_error, r2_score,\
            mean_squared_log_error, mean_absolute_error
        
        X_train_scaled = self.scaler.X_train_scaled; Z_train = self.scaler.Z_train
        X_test_scaled  = self.scaler.X_test_scaled; Z_test = self.scaler.Z_test
        
        
        linreg = self.fit(X_train_scaled, Z_train)
        
        # print('The intercept alpha: \n', linreg.intercept_.shape)
        # print('Coefficient beta : \n', linreg.coef_.shape)
        
        Z_tilde = linreg.predict(X_train_scaled)
        Z_pred  = linreg.predict(X_test_scaled)
        
        # print(Z_regfit[0:4, :], "\n...", Z_tilde[0:4, :])
        
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
        
        