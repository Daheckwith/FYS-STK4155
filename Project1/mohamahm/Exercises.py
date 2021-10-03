import frankefunction as fr
import error as err
import  split_scale as split
import regression as reg
import resampling as res
import output as out

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


def Confidence_interval(scaler_obj, regression_class, var, method= "OLS"):
    X = scaler_obj.X_train_scaled; X_T = X.T
    var_beta_numpy = var*np.diag(np.linalg.pinv(X_T @ X))
    std_beta = np.sqrt(var_beta_numpy)
    CI = 1.96*std_beta
    
    if regression_class == "Scikit":
        Scikit_REG = reg.Scikit_regression(scaler_obj)
        beta = Scikit_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        beta_array = np.copy(beta.coef_) 
        beta_array[0] += beta.intercept_
        beta = beta_array
        x = np.arange(len(beta))
        plt.figure()
        plt.title("Scikit")
    else:
        Manual_REG = reg.Manual_regression(scaler_obj)
        beta = Manual_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        x = np.arange(len(beta))
        plt.figure()
        plt.title("Manual")
    
    
    # plt.style.use("seaborn-whitegrid")
    plt.style.use("seaborn-darkgrid")
    plt.xticks(x)
    
    plt.errorbar(x, beta, CI, fmt=".", capsize= 3, label=r'$\beta_j \pm 1.96 \sigma$')
    # plt.errorbar(x, beta, CI, fmt=".", label=r'$\beta_j \pm 1.96 \sigma$')
    # plt.legend(labelcolor = "k", fancybox= True, facecolor= "white", shadow= True, edgecolor= "black")
    plt.legend(edgecolor= "black")
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.show()
        
def train_test_error(scaler_obj, regression_class, method= "OLS"):
    if regression_class == "Scikit":
        Scikit_REG = reg.Scikit_regression(scaler_obj)
        beta = Scikit_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
    else:
        Manual_REG = reg.Manual_regression(scaler_obj)
        beta = Manual_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        
    error_printout = out.PrintError(scaler_obj)
    error_printout.ManualError(beta, printToScreen= True)
    error_printout.ScikitError(beta, printToScreen= True)
    
    
def Exercise2(x, y, z, n_bootstraps = 100):
    #Bias-Variance Trade-off
    print("\n \nModel Complexity")
    
    maxdegree = 11; start = 1; degrees = np.arange(start, maxdegree)
    MSE_train = np.zeros(maxdegree - start); MSE_test = np.zeros(maxdegree - start)
    Error_list = ["R2 Score", "MSE", "MAE"]
    Error_list2 = ["Error", "Bias", "Variance"]
    degree_analysis = np.zeros((maxdegree - start, 3, 2)) # [degree, R2[0]/MSE[1]/MSA[2], Tilde[0]/Predict[1]]
    degree_analysis2 = np.zeros((maxdegree - start, 3, 2)) # [degree, Error[0]/Bias[1]/Variance[2], Predict[0]]
    l, m = 0, 0
    
    for deg in degrees:
        print(f"----------------DEGREE: {deg}-----------------")
        
        #Numpy scaling
        # scale_split = split.Numpy_split_scale(x, y, z_noisey, deg= deg)
        # scale_split = split.Numpy_split_scale(x, y, z_flat, deg= deg)
        
        #Scikit scaling
        # scale_split = split.Scikit_split_scale(x, y, z_noisey, deg= deg)
        scale_split = split.Scikit_split_scale(x, y, z, deg= deg)
        
        Res = res.Resampling(scale_split)
        Res.Bootstrap(regression_class= "Scikit", fit_method = "OLS",\
                        n_bootstraps = n_bootstraps, printToScreen= False)
        # Res.Bootstrap(regression_class= "Manual", fit_method = "OLS",\
                       # n_bootstraps = n_bootstraps, printToScreen= False)
        
        while l < 3:
            while m < 2:
                degree_analysis2[deg - start, l, m]  = Res.ErrorDict2[Error_list2[l]][m]
                print("l: ", l, "m: ", m, "array: ", degree_analysis2[deg - start, l, m])
                m += 1
            l += 1
            m = 0
        l = 0
    
    
    plt.close()
    plt.figure()
    plt.xticks(np.arange(maxdegree))
    
    plt.plot(degrees, degree_analysis2[:, 0, 0], "o-", label= "Tilde Error")
    # plt.plot(degrees, degree_analysis2[:, 1, 0], "o-", label= "Tilde Bias")
    # plt.plot(degrees, degree_analysis2[:, 2, 0], "o-", label= "Tilde Variance")
    
    plt.plot(degrees, degree_analysis2[:, 0, 1], "o-", label= "Test Error")
    # plt.plot(degrees, degree_analysis2[:, 1, 1], "o-", label= "Test Bias")
    # plt.plot(degrees, degree_analysis2[:, 2, 1], "o-", label= "Test Variance")
    
    plt.legend()
    plt.show()
    
        
    """
        print("X shape: ", scale_split.X.shape, x.shape, y.shape, z.shape)
        
        #Manual Regression
        # Manual_REG = reg.Manual_regression(scale_split)
        # ols_beta = Manual_REG.fit(scale_split.X_train_scaled, scale_split.Z_train, method= "OLS")
        
        #Scikit Regression
        Scikit_REG = reg.Scikit_regression(scale_split)
        ols_beta = Scikit_REG.fit(scale_split.X_train_scaled, scale_split.Z_train, method= "OLS")
        
        error_printout = out.PrintError(scale_split)
        # error_printout.ManualError(ols_beta, printToScreen= True) # Calculates the error using error.py
        error_printout.ScikitError(ols_beta, printToScreen= True) # Calculates the error using Scikit
        
        
        while l < 3:
            while m < 2:
                degree_analysis[deg - start, l, m]  = error_printout.ScikitDict[Error_list[l]][m]
                m += 1
            l += 1
            m = 0
        l = 0
        

    plt.close()
    plt.figure()
    # plt.plot(degree, degree_analysis[:, 0, 0], label = "R2 Tilde")
    # plt.plot(degree, degree_analysis[:, 0, 1], label = "R2 Predict")
    plt.xticks(np.arange(maxdegree))
    plt.plot(degrees, degree_analysis[:, 1, 0], "o-", label = "Train1 MSE")
    plt.plot(degrees, degree_analysis[:, 1, 1], "o-", label = "Test1 MSE")
    plt.legend()
    plt.show()
    """
    return degree_analysis