import regression as reg
# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score,\
    mean_squared_log_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

class Resampling():
    def __init__(self, scaler_obj):
        print("Class Resample initializer")
        self.scaler = scaler_obj
        self.deg =  self.scaler.deg
        
        self.ManualDict = None
        self.ScikitDict = None
        self.ErrorDitct = None
        
        self.X_train_scaled = self.scaler.X_train_scaled; self.Z_train = self.scaler.Z_train
        self.X_test_scaled  = self.scaler.X_test_scaled; self.Z_test = self.scaler.Z_test
        
    def Bootstrap(self, regression_class = "Scikit", fit_method = "OLS",\
                  n_bootstraps = 20, printToScreen : bool = False):
        
        print("Bootstrap")
        X_train_scaled = self.X_train_scaled; Z_train = self.Z_train
        X_test_scaled  = self.X_test_scaled; Z_test = self.Z_test
        
        boot_beta = np.zeros((X_train_scaled.shape[1], Z_train.shape[1], n_bootstraps))
        boot_Z_pred = np.empty((Z_test.shape[0], Z_test.shape[1], n_bootstraps))
        
        if regression_class == "Scikit":
            Scikit_REG = reg.Scikit_regression(self.scaler)
            print(Scikit_REG.__class__.__name__)
            
            for i in range(n_bootstraps):
                # X_, Z_ = resample(X_train_scaled, Z_train)
                X_, Z_ = resample(X_train_scaled, Z_train, random_state= 1) # seeded
                
                # beta = Scikit_OLS.fit(X_, Z_)
                beta = Scikit_REG.fit(X_, Z_, fit_method)
                
                Z_pred = beta.predict(X_test_scaled)
                boot_Z_pred[:, :, i] = Z_pred
        else:
            Manual_REG = reg.Manual_regression(self.scaler)
            print(Manual_REG.__class__.__name__)
            for i in range(n_bootstraps):
                # X_, Z_ = resample(X_train_scaled, Z_train)
                X_, Z_ = resample(X_train_scaled, Z_train, random_state= 1) # seeded
                
                beta = Manual_REG.fit(X_, Z_, fit_method)
                
                # boot_beta[:, :, i] = beta 
                
                Z_pred = X_test_scaled @ beta
                boot_Z_pred[:, :, i] = Z_pred
            
            # beta_ = np.mean(boot_beta, axis = 2)
            # Z_pred = X_test_scaled @ beta_
            
            # MSE = mean_squared_error(Z_test, Z_pred)
            # MAE = mean_absolute_error(Z_test, Z_pred)
            # R2 = r2_score(Z_test, Z_pred)
            
            # print("-----------Boot over beta -----------")
            # print("R2 Score:", R2, "MSE:", MSE, "MSA:", MAE)
            # print("---------------------------\n")
        
        Z_pred_ = np.mean(boot_Z_pred, axis = 2)
        
        MSE = mean_squared_error(Z_test, Z_pred_)
        MAE = mean_absolute_error(Z_test, Z_pred_)
        R2  = r2_score(Z_test, Z_pred_)
        
        self.ErrorDict = {"R2 Score" : R2,\
                           "MSE" : MSE, "MAE" : MAE}
            
        if printToScreen:
            self.printout()
        
        # print("-----------Boot over Z_pred----------")
        # print("R2 Score:", R2, "MSE:", MSE, "MSA:", MAE)
        # print("---------------------------\n")
    
    def Scikit_CrossValidation(self, fit_method, k = 5):
        print("Scikit_CrossValidation")
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score
        
        X = self.scaler.X; Z = self.scaler.Z
        X_scaled = self.scaler.X_scaled
        
        
        # Scikit_OLS = reg.Scikit_OLS(self.scaler)
        Scikit_REG = reg.Scikit_regression(self.scaler)
        
        # Initialize a KFold instance
        kfold = KFold(n_splits = k)  
        scores_KFold = np.zeros(k)
        
        l = 0
        for train_idx, test_idx in kfold.split(X):
            X_train = X_scaled[train_idx]; Z_train = Z[train_idx]
            X_test = X_scaled[test_idx]; Z_test = Z[test_idx]
            
            # beta = Scikit_OLS.fit(X_train, Z_train)
            beta = Scikit_REG.fit(X_train, Z_train, fit_method)
            
            Z_pred = beta.predict(X_test)
            
            scores_KFold[l] = np.sum((Z_pred - Z_test)**2)/np.size(Z_pred)
            
            l += 1
        
        Scikit_CV = cross_val_score(beta, X, Z, scoring='neg_mean_squared_error', cv=kfold)
        
        print("Scikit_CV: ", -Scikit_CV, "scores_KFold: ", scores_KFold)
        # print(scores_KFold, np.mean(scores_KFold))
        
    def printout(self):
        ErrorDict = self.ErrorDict
        print("-----------Bootstrap-Predict-----------")
        print("R2 Score:", ErrorDict["R2 Score"],\
              "MSE:", ErrorDict["MSE"], "MAE:", ErrorDict["MAE"])
        print("---------------------------\n")
        

"""    
def Manual_Bootstrap(self, fit_method, n_bootstraps):
    print("Manual_Bootstrap")
    
    Manual_REG = reg.Manual_regression(self.scaler)
    
    X_train_scaled = self.X_train_scaled; Z_train = self.Z_train
    X_test_scaled  = self.X_test_scaled; Z_test = self.Z_test
    
    boot_beta = np.zeros((X_train_scaled.shape[1], Z_train.shape[1], n_bootstraps))
    boot_Z_pred = np.empty((Z_test.shape[0], Z_test.shape[1], n_bootstraps))
    
    for i in range(n_bootstraps):
        # X_, Z_ = resample(X_train_scaled, Z_train)
        X_, Z_ = resample(X_train_scaled, Z_train, random_state= 1) # seeded
        
        # beta = Manual_OLS.fit(X_, Z_)
        beta = Manual_REG.fit(X_, Z_, fit_method)
        
        boot_beta[:, :, i] = beta 
        
        Z_pred = X_test_scaled @ beta
        boot_Z_pred[:, :, i] = Z_pred
    
    
    beta_ = np.mean(boot_beta, axis = 2)
    Z_pred = X_test_scaled @ beta_
    
    MSE = mean_squared_error(Z_test, Z_pred)
    MAE = mean_absolute_error(Z_test, Z_pred)
    R2 = r2_score(Z_test, Z_pred)
    
    print("-----------Boot over beta -----------")
    print("R2 Score:", R2, "MSE:", MSE, "MSA:", MAE)
    print("---------------------------\n")
    
    Z_pred_ = np.mean(boot_Z_pred, axis = 2)
    
    MSE = mean_squared_error(Z_test, Z_pred_)
    MAE = mean_absolute_error(Z_test, Z_pred_)
    R2  = r2_score(Z_test, Z_pred_)
    
    print("-----------Boot over Z_pred----------")
    print("R2 Score:", R2, "MSE:", MSE, "MSA:", MAE)
    print("---------------------------\n")
    
def Scikit_Bootstrap(self, fit_method, n_bootstraps):
    print("Scikit_Bootstrap")
    
    Scikit_REG = reg.Scikit_regression(self.scaler)
    
    X_train_scaled = self.X_train_scaled; Z_train = self.Z_train
    X_test_scaled  = self.X_test_scaled; Z_test = self.Z_test
    
    boot_Z_pred = np.empty((Z_test.shape[0], Z_test.shape[1], n_bootstraps))
    
    for i in range(n_bootstraps):
        # X_, Z_ = resample(X_train_scaled, Z_train)
        X_, Z_ = resample(X_train_scaled, Z_train, random_state= 1) # seeded
        
        # beta = Scikit_OLS.fit(X_, Z_)
        beta = Scikit_REG.fit(X_, Z_, fit_method)
        
        Z_pred = beta.predict(X_test_scaled)
        boot_Z_pred[:, :, i] = Z_pred
        
    Z_pred_ = np.mean(boot_Z_pred, axis = 2)
    
    MSE = mean_squared_error(Z_test, Z_pred_)
    MAE = mean_absolute_error(Z_test, Z_pred_)
    R2  = r2_score(Z_test, Z_pred_)
    
    print("-----------Boot over Z_pred----------")
    print("R2 Score:", R2, "MSE:", MSE, "MSA:", MAE)
    print("---------------------------\n")
"""    