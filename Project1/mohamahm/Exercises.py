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
import time

def train_test_error(scaler_obj, regression_class, method= "OLS", lmb = 0, printToScreen = False, defaultError = "Scikit"):
    print(method, lmb, printToScreen, defaultError)
    if regression_class == "Scikit":
        Scikit_REG = reg.Scikit_regression(scaler_obj)
        Scikit_REG.lmb = lmb
        beta = Scikit_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
    else:
        Manual_REG = reg.Manual_regression(scaler_obj)
        Manual_REG.lmb = lmb
        beta = Manual_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        
    
    error_printout = out.PrintError(scaler_obj)
    if defaultError == "Scikit":
        error_printout.ScikitError(beta, printToScreen= printToScreen)
    else:
        error_printout.ManualError(beta, printToScreen= printToScreen)
        
    return error_printout.ErrorDict

def confidence_interval(scaler_obj, regression_class, var, method= "OLS", lmb = 0):
    X = scaler_obj.X_train_scaled; X_T = X.T
    var_beta_numpy = var*np.diag(np.linalg.pinv(X_T @ X))
    std_beta = np.sqrt(var_beta_numpy)
    CI = 1.96*std_beta
    
    if regression_class == "Scikit":
        Scikit_REG = reg.Scikit_regression(scaler_obj)
        Scikit_REG.lmb = lmb
        beta = Scikit_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        beta_array = np.copy(beta.coef_) 
        beta_array[0] += beta.intercept_
        beta = beta_array
        x = np.arange(len(beta))
        plt.figure()
        plt.title("Scikit Regression")
    else:
        Manual_REG = reg.Manual_regression(scaler_obj)
        Manual_REG.lmb = lmb
        beta = Manual_REG.fit(scaler_obj.X_train_scaled, scaler_obj.Z_train, method= method)
        x = np.arange(len(beta))
        plt.figure()
        plt.title("Manual Regression")
    
    
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
    
def train_vs_test_MSE(xx, yy, z_flat, regression_class, maxdegree= 10, method= "OLS", lmb= 0, cv= False):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score,\
        mean_squared_log_error, mean_absolute_error
    
    maxdegree = maxdegree + 1; start = 1; degrees = np.arange(start, maxdegree)
    # Error_list = ["R2 Score", "MSE", "MAE"]
    # Error_list = ["Error", "Bias", "Variance"]
    # degree_analysis = np.zeros((maxdegree - start, 3, 2)) # [degree, Error[0]/Bias[1]/Variance[2], Predict[0]]
    n_bootstraps = 100
    
    trainerror = np.zeros(maxdegree - start)
    testerror = np.zeros(maxdegree - start)
    CV_MSE = np.zeros(maxdegree - start)
    time_CV = np.zeros(maxdegree - start)
    time_Boot = np.zeros(maxdegree - start)
    
    if regression_class == "Scikit":
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
            
            if cv:
                t0 = time.time()
                cross_validation(scale_split, method = method, k = 5, lmb= lmb)
                time_CV[deg - start] = time.time() - t0
            
            
            #Bootstrap
            t1 = time.time()
            
            for boot in range(n_bootstraps):
                X_train, X_test, Z_train, Z_test = train_test_split(scale_split.X, scale_split.Z, test_size= 0.2)
                X_train_scaled = scale_split.scikit_scaler(X_train); X_test_scaled = scale_split.scikit_scaler(X_test)
                
                Scikit_REG = reg.Scikit_regression(scale_split)
                Scikit_REG.lmb = lmb
                
                beta = Scikit_REG.fit(X_train_scaled, Z_train, method= method)
                Z_tilde = beta.predict(X_train_scaled)
                Z_pred  = beta.predict(X_test_scaled)
                
                trainerror[deg - start] += mean_squared_error(Z_train, Z_tilde) 
                testerror[deg - start] += mean_squared_error(Z_test, Z_pred)
            
            time_Boot[deg - start] = time.time() - t1
        
        plt.figure()
        plt.title("Scikit Regression")
    
    else:
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
            
            if cv:
                t0 = time.time()
                cross_validation(scale_split, method = method, k = 5, lmb= lmb)
                time_CV[deg - start] = time.time() - t0
            
            #Bootstrap
            t1 = time.time()
            for boot in range(n_bootstraps):
                X_train, X_test, Z_train, Z_test = train_test_split(scale_split.X, scale_split.Z, test_size= 0.2)
                X_train_scaled = scale_split.numpy_scaler(X_train); X_test_scaled = scale_split.numpy_scaler(X_test)
                
                Manual_REG = reg.Manual_regression(scale_split)
                Manual_REG.lmb = lmb
                
                beta = Manual_REG.fit(X_train_scaled, Z_train, method= method)
                Z_tilde   = X_train_scaled @ beta
                Z_pred    = X_test_scaled @ beta
                
                trainerror[deg - start] += mean_squared_error(Z_train, Z_tilde) 
                testerror[deg - start] += mean_squared_error(Z_test, Z_pred)
            
            time_Boot[deg - start] = time.time() - t1
        plt.figure()
        plt.title("Manual Regression")
    
    trainerror /= n_bootstraps
    testerror /= n_bootstraps
    
    for deg in degrees:
        print(f"----------------DEGREE: {deg}-----------------")
        print("Degree of polynomial: %3d" %deg)
        print("Mean squared error on training data: %.8f" % trainerror[deg - start])
        print("Mean squared error on test data: %.8f" % testerror[deg - start])
    
    if cv == False:
        plt.plot(degrees, np.log10(trainerror), "o-", label='Training Error')
        plt.plot(degrees, np.log10(testerror), "o-", label='Test Error')
    else:
        plt.plot(degrees, np.log10(testerror), "o-", label='Test Error')
        plt.plot(degrees, np.log10(CV_MSE), "o-", label= "CV Test Error")
    plt.xlabel('Polynomial degree')
    plt.ylabel('log10[MSE]')
    plt.legend()
    plt.show()
    
    if cv:
        # print(time_Boot, time_CV)
        plt.figure()
        plt.title("Measured time: Boot vs. Cross-validation")
        plt.plot(degrees, time_Boot, label= "Boot time")
        plt.plot(degrees, time_CV, label= "CV time")
        plt.xlabel('Polynomial degree')
        plt.ylabel("t [s]")
        plt.legend()
        plt.show()
    
def bias_variance_tradeoff(xx, yy, z_flat, regression_class, maxdegree= 10, method= "OLS", lmb= 0):
    from sklearn.utils import resample

    maxdegree = maxdegree + 1; start = 1; degrees = np.arange(start, maxdegree)
    error = np.zeros(maxdegree - start)
    bias = np.zeros(maxdegree - start)
    variance = np.zeros(maxdegree - start)
    n_bootstraps = 100
    
    if regression_class == "Scikit":
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Scikit_split_scale(xx, yy, z_flat, deg= deg)
            
            boot_Z_tilde = np.zeros((scale_split.Z_train.shape[0], n_bootstraps))
            boot_Z_pred = np.zeros((scale_split.Z_test.shape[0], n_bootstraps))
            
            for i in range(n_bootstraps):
                X_, Z_ = resample(scale_split.X_train_scaled, scale_split.Z_train)
               
                Scikit_REG = reg.Scikit_regression(scale_split)
                Scikit_REG.lmb = lmb
                
                beta = Scikit_REG.fit(X_, Z_, method= method)
                Z_tilde = beta.predict(X_)
                Z_pred  = beta.predict(scale_split.X_test_scaled)
                   
                boot_Z_tilde[:,i]  = Z_tilde
                boot_Z_pred[:, i] = Z_pred
            
            error[deg - start] = np.mean( np.mean((scale_split.Z_test.reshape(-1,1) - boot_Z_pred)**2, axis= 1, keepdims= True) )
            bias[deg - start] = np.mean( (scale_split.Z_test.reshape(-1,1) - np.mean(boot_Z_pred, axis= 1, keepdims= True))**2 )
            variance[deg - start] = np.mean( np.var(boot_Z_pred, axis= 1, keepdims= True) )
        
        plt.figure()
        plt.title("Scikit Regression")
    else:
        for deg in degrees:
            print(f"----------------DEGREE: {deg}-----------------")
            scale_split = split.Numpy_split_scale(xx, yy, z_flat, deg= deg)
            
            boot_Z_tilde = np.zeros((scale_split.Z_train.shape[0], n_bootstraps))
            boot_Z_pred = np.zeros((scale_split.Z_test.shape[0], n_bootstraps))
            
            for i in range(n_bootstraps):
                X_, Z_ = resample(scale_split.X_train_scaled, scale_split.Z_train)
               
                Manual_REG = reg.Manual_regression(scale_split)
                Manual_REG.lmb = lmb
                
                beta = Manual_REG.fit(X_, Z_, method= method)
                Z_tilde   = X_ @ beta
                Z_pred    = scale_split.X_test_scaled @ beta
                
                boot_Z_tilde[:,i]  = Z_tilde
                boot_Z_pred[:, i] = Z_pred
                
            error[deg - start] = np.mean( np.mean((scale_split.Z_test.reshape(-1,1) - boot_Z_pred)**2, axis= 1, keepdims= True) )
            bias[deg - start] = np.mean( (scale_split.Z_test.reshape(-1,1) - np.mean(boot_Z_pred, axis= 1, keepdims= True))**2 )
            variance[deg - start] = np.mean( np.var(boot_Z_pred, axis= 1, keepdims= True) )
        
        plt.figure()
        plt.title("Manual Regression")
    
    plt.plot(degrees, error, label= "Error")
    plt.plot(degrees, bias, label= "Bias")
    plt.plot(degrees, variance, label= "Variance")
    plt.xlabel('Polynomial degree')
    plt.ylabel("Error/Bias/Variance")
    plt.legend()
    plt.show()
    
    
def cross_validation(scaler_obj, method = "OLS", k = 5, lmb= 0):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score,\
        mean_squared_log_error, mean_absolute_error
    
    X = scaler_obj.X; Z = scaler_obj.Z
    X_scaled = scaler_obj.X_scaled
    
    # Initialize a KFold instance
    kfold = KFold(n_splits = k)  
    scores_KFold = np.zeros(k)
    
    Scikit_REG = reg.Scikit_regression(scaler_obj)
    Scikit_REG.lmb = lmb
    
    l = 0
    for train_idx, test_idx in kfold.split(X):
        X_train = X_scaled[train_idx]; Z_train = Z[train_idx]
        X_test = X_scaled[test_idx]; Z_test = Z[test_idx]
        
        beta = Scikit_REG.fit(X_train, Z_train, method= method)
        
        Z_pred = beta.predict(X_test)
        
        scores_KFold[l] = np.sum((Z_pred - Z_test)**2)/np.size(Z_pred)
        
        l += 1
        
    Scikit_CV = cross_val_score(beta, X, Z, scoring='neg_mean_squared_error', cv=kfold)
    
    # print("Scikit_CV: ", -Scikit_CV, "scores_KFold: ", scores_KFold)
    return np.mean(-Scikit_CV)

    
def Ridge_Lasso_MSE(xx, yy, z_flat, scaler_opt, regression_opt):
    if scaler_opt == "Numpy":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    else:
        scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
    
    nlambdas = 25
    lambdas = np.logspace(-4, 4, nlambdas)
    MSEpred = np.zeros(nlambdas); MSEtrain = np.zeros(nlambdas)
    MSELassotrain = np.zeros(nlambdas); MSELassopred = np.zeros(nlambdas)
    for i in range(nlambdas):
        lmb = lambdas[i]
        print(f"-------train_test_error-lmb-{lmb}-------")
        
        ErrorDict = train_test_error(scaler, regression_class= regression_opt, method = "Ridge", lmb = lmb, printToScreen= False)
        train_error, test_error = ErrorDict["MSE"]
        MSEtrain[i] = train_error; MSEpred[i] = test_error 
        
        ErrorDict = train_test_error(scaler, regression_class= "Scikit", method = "Lasso", lmb = lmb, printToScreen= False)
        train_error, test_error = ErrorDict["MSE"]
        MSELassotrain[i] = train_error; MSELassopred[i] = test_error
    
    plt.title(f"Ridge regression for deg = {scaler.deg}")
    plt.plot(np.log10(lambdas), MSEtrain, "o-", label='MSE Ridge Train')
    plt.plot(np.log10(lambdas), MSEpred, "o-", label='MSE Ridge Test')
    plt.plot(np.log10(lambdas), MSELassotrain, "o-", label='MSE Lasso Train')
    plt.plot(np.log10(lambdas), MSELassopred, "o-", label='MSE Lasso Test')
    plt.xlabel('log10($\lambda$)')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    # save_fig("RidgeLassoTrainTestMSE")

def R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 10, method= "Ridge", lmb= 1):
    train_vs_test_MSE(xx, yy, z_flat, regression_opt, maxdegree= maxdegree, method= method, lmb= lmb)
    plt.title(method + f"train vs. test MSE for $\lambda=$ {lmb}")

def R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 10, method= "Ridge", lmb= 1):
    bias_variance_tradeoff(xx, yy, z_flat, regression_class = regression_class, maxdegree= maxdegree, method= method, lmb= lmb)
    plt.title(method + f"Bias-Variance tradeoff for $\lambda=$ {lmb}")


def Exercise_4_5(xx, yy, z_flat, scaler_opt, regression_opt= "Scikit", maxdegree= 10, method= "Ridge", lmb= 1):
    if scaler_opt == "Numpy":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    else:
        scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
    
    # R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 25, method= method, lmb= 1000)
    # save_fig(Ridge_model_complexity.__name__ + f"{int(np.log10(1000))}" + "maxdegree25")
    # R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 25, method= method, lmb= 1000)
    # save_fig(Ridge_bias_variance.__name__ + f"{int(np.log10(lmb))}" + "maxdegree25" )
    
    nlambdas = 9; k = 5
    lambdas = np.logspace(-4, 4, nlambdas)
    cv_lmb = np.zeros((nlambdas, k))
    for i in range(nlambdas):
        lmb = lambdas[i]
        lmb_str = str(lmb)
        
        # cv_lmb[i,:] = cross_validation(scaler, method= method, k= 5, lmb= lmb)
        
        R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 10, method= method, lmb= lmb)
        # save_fig(Ridge_model_complexity.__name__ + f"{int(np.log10(lmb))}")
        R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 10, method= method, lmb= lmb)
        # save_fig(Ridge_bias_variance.__name__ + f"{int(np.log10(lmb))}")
    
    # plt.figure()
    # folds = np.arange(1,k+1)
    # for i in range(nlambdas):
    #     lmb = lambdas[i]
    #     plt.plot(folds, cv_lmb[i], "o-", label= f"$\lambda=$ {lmb}")
    
    # plt.xticks(folds)
    # plt.title(method + ": Cross-Validaion score for each $\lambda$, deg= 5")
    # plt.xlabel("${fold}_k$")
    # plt.ylabel("CV Score")
    # plt.legend()
    # plt.show()
    # save_fig(cross_validation.__name__ + "-Ridge")

def save_fig(name):
    plt.savefig(name, dpi= 300, bbox_inches= 'tight')
    
    
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