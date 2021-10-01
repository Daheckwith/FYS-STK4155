import numpy as np

class Manual_regression:
    def __init__(self, scaler_obj):
        """
        Takes in an object split_scale. That's to insure that the user first
        scales the data before doing any analysis. The scaled data is saved by 
        and accessed through the self.scaler instance variable.

        Parameters
        ----------
        scaler_obj : A split_scale instance.

        Returns
        -------
        None.

        """
        self.scaler = scaler_obj;
        
    def OLS(self, X, Z):
        X_T = X.T
        return np.linalg.pinv(X_T @ X) @ (X_T @ Z)
    
    def Ridge(self, X, Z):
        print("Nothing for now!")
        
    def fit(self, X, Z, method= "OLS"):
        if method in ["ols", "OLS", "Ols"]:
            return self.OLS(X, Z)
        elif method in ["ridge", "Ridge", "RIDGE"]:
            return self.Ridge(X, Z)
            
class Scikit_regression:
    def __init__(self, scaler_obj):
        self.scaler = scaler_obj;
        
    def OLS(self, X, Z):
        from sklearn.linear_model import LinearRegression 
        linreg = LinearRegression()
        return linreg.fit(X, Z)
        
    def Ridge(self, X, Z):
        print("Nothing for now!")
        
    def Lasso(self, X, Z):
        print("Nothing for now!")
        
    def fit(self, X, Z, method= "OLS"):
        if method in ["ols", "OLS", "Ols"]:
            return self.OLS(X, Z)
        elif method in ["ridge", "Ridge", "RIDGE"]:
            return self.Ridge(X, Z)