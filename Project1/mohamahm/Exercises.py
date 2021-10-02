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

def Exercise1(x, y, z_noisey, var):
    # Exercise 1
    """
    print("MANUAL")
    scale_split = split.Numpy_split_scale(x, y, z_noisey)
    REG = reg.Manual_regression(scale_split)
    ols_beta = REG.fit(scale_split.X_train_scaled, scale_split.Z_train, method= "OLS")
    error_printout = out.PrintError(scale_split)
    Manualprint = error_printout.ManualError(ols_beta, printToScreen= True)
    Scikitprint = error_printout.ScikitError(ols_beta, printToScreen = True)

    # print("Manualprint self: ", error_printout.ManualDict, "\n")

    print("Scikit")
    scale_split = split.Scikit_split_scale(x, y, z_noisey)
    REG = reg.Scikit_regression(scale_split)
    ols_beta = REG.fit(scale_split.X_train_scaled, scale_split.Z_train, method= "OLS")
    ols_beta.predict(scale_split.X_train_scaled)
    error_printout.ManualError(ols_beta, printToScreen= True)
    error_printout.ScikitError(ols_beta, printToScreen= True)
    """
    #Comfidence interval
    scale_split = split.Numpy_split_scale(x, y, z_noisey, deg= 5)
    REG = reg.Manual_regression(scale_split)

    # scale_split = split.Scikit_split_scale(x, y, z_noisey, deg= 5)
    # REG = reg.Scikit_regression(scale_split)

    ols_beta = REG.fit(scale_split.X_train_scaled, scale_split.Z_train, method= "OLS")
    # ols_beta = ols_beta.coef_
    X = scale_split.X_train_scaled; X_T = X.T
    var_beta = var*np.diag(np.linalg.pinv(X_T @ X))
    std_beta = np.sqrt(var_beta)
    CI = 1.96*std_beta

    plt.close()
    plt.figure()
    # plt.grid()
    # plt.style.use("seaborn-whitegrid") # plt.style.use("seaborn-darkgrid")
    x = range(len(std_beta))
    plt.xticks(x)
    print("len:", len(x), len(np.diag(ols_beta)), len(CI))
    plt.errorbar(x, np.diag(ols_beta), CI, fmt=".", capsize= 3, label=r'$\beta_j \pm 1.96 \sigma$')
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.show()
    
def Exercise2(x, y, z_noisey):
    #Bias-Variance Trade-off
    print("\n \nModel Complexity")

    maxdegree = 11; start = 1; degrees = np.arange(start, maxdegree)
    MSE_train = np.zeros(maxdegree - start); MSE_test = np.zeros(maxdegree - start)
    Error_list = ["R2 Score", "MSE", "MAE"]
    degree_analysis = np.zeros((maxdegree - start, 3, 2)) # [degree, R2[0]/MSE[1]/MSA[2], Tilde[0]/Predict[1]]
    l, m = 0, 0
    
    for deg in degrees:
        print(f"----------------DEGREE: {deg}-----------------")
        #Numpy scaling
        # scale_split = split.Numpy_split_scale(x, y, z_noisey, deg= deg)
        
        #Scikit scaling
        scale_split = split.Scikit_split_scale(x, y, z_noisey, deg= deg)
        
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


    # plt.close()
    plt.figure()
    # plt.plot(degree, degree_analysis[:, 0, 0], label = "R2 Tilde")
    # plt.plot(degree, degree_analysis[:, 0, 1], label = "R2 Predict")
    plt.xticks(np.arange(maxdegree))
    plt.plot(degrees, degree_analysis[:, 1, 0], "o-", label = "Train1 MSE")
    plt.plot(degrees, degree_analysis[:, 1, 1], "o-", label = "Test1 MSE")
    plt.legend()
    plt.show()
    
    return degree_analysis