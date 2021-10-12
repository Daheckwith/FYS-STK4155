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

# =============================================================================
# To-do's
# Exercise 1 Confidence interval use beta = sigma*(X_T*X)^-1, talk about why we scaled the data
# Exercise 2 Fig 2.11 Hastie, Tibshirani (test and training MSE), bias-variance tradeoff(using bootstrap)
# Exercise 3 Cross-validation
# =============================================================================

# Presets
mean = 0; deviation = 0.1;  #Franke
random = True; precision = 0.05 #initialize_array
noise = True #franke_function

# Generating data using the Franke function
# Fr= fr.Franke(mean = 0, deviation = 0.02, seed = 4155)
Fr= fr.Franke(mean, deviation)

x, y = Fr.initialize_array(random, precision)
# x, y = Fr.initialize_array(random = True, precision= 0.01)
xx, yy = np.meshgrid(x, y)

plt.close("all")
z = Fr.franke_function(xx, yy, noise = False)
# Fr.contour_franke(xx, yy, z)
z_noisey, var = Fr.franke_function(xx, yy, noise)
# Fr.contour_franke(xx, yy, z_noisey)
# print("Var_Z: ", np.var(z_noisey), np.sqrt(np.var(z_noisey)), np.std(z_noisey))
z_flat = np.ravel(z_noisey)
# z_flat = z_noisey.reshape((-1,1))
# -------------------------------------------------------------------------------------

# from Exercises import confidence_interval, train_test_error,\
#     train_vs_test_MSE, bias_variance_tradeoff, cross_validation, save_fig,\
#         Ridge_Lasso_MSE, Ridge_model_complexity
from Exercises import *

print("Choose exercise! Options [\"Exercise1\", \"Exercise2\", \"Exercise3\", \"Exercise4\"]")
# exercise = str(input()) # Provide an input
exercise = "Exercise3" # or change this line and comment out the one above

scaler_opt = "Numpy" 
# scaler_opt = "Scikit"

regression_opt = "Manual" 
# regression_opt = "Scikit"

method = "OLS"
# method = "Ridge"
# method = "Lasso"

if exercise in ["Exercise1", "Exercise 1"]:
    #Exercise1
    """
    
    Choose the scaler class Numpy/Scikit. Choose the regression class Manual/Scikit
    Note: Both train_test_error and confidence_interval takes an optional parameter
    "method", defaul = "OLS". 
    Manual does OLS and Ridge. Scikit does OLS, Ridge and Lasso.
    
    """
    if scaler_opt == "Numpy":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 2)
    else:
        scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 2)
    
    print("-------train_test_error--------")
    train_test_error(scaler, regression_class= regression_opt, method = "OLS", printToScreen= True, defaultError= "Scikit")
    # train_test_error(scaler, regression_class= regression_opt, method = "OLS", printToScreen= True, defaultError= "Manual")
    print("-------Confidence_interval---------")
    confidence_interval(scaler, regression_class= regression_opt, var= var, method = "OLS")
    # save_fig(confidence_interval.__name__ + regression_opt + scaler_opt)
    
elif exercise in ["Exercise2", "Exercise 2"]:
    # Exercise2 
    """
    
    Choose the regression class Manual/Scikit. If "Manual" scaling is done using
    Numpy_split_scale then manual regression, options OLS and Ridge available.
    Otherwise if "Scikit" is selected then everything is done using Scikit.
    
    """
    print("Method: ", method, exercise)
    print("---------train_vs_test_MSE----------")
    train_vs_test_MSE(xx, yy, z_flat, regression_class= regression_opt, maxdegree= 20)
    # save_fig(train_vs_test_MSE.__name__ + regression_opt)
    print("---------bias_variance_tradeoff---------") 
    bias_variance_tradeoff(xx, yy, z_flat, regression_class= regression_opt, maxdegree= 8) #0.05, 0.02, 0.01 : maxdegree = 8, 15/18, 20
    # plt.title(regression_opt + f" Regression N= {Fr.steps}" )
    # save_fig(bias_variance_tradeoff.__name__ + regression_opt + f"N= {Fr.steps}")

elif exercise in ["Exercise3", "Exercise 3"]:
    """
    Scaling con be done usinig scikit or numpy. Prints out CV
    using both a KFold-algorithm and cross_val_score from Scikit.
    """
    
    maxdegree = 20
    train_vs_test_MSE(xx, yy, z_flat, regression_class= regression_opt, maxdegree= maxdegree, method= "OLS", lmb= 0, cv= True)
    # save_fig(train_vs_test_MSE.__name__ + regression_opt + f"maxdegree={maxdegree}")

elif exercise in ["Exercise4", "Exercise 4"]:
    
    Exercise_4_5(xx, yy, z_flat, scaler_opt, regression_opt= "Scikit", maxdegree= 10, method= "Ridge")
    
    """
    if scaler_opt == "Numpy":
        scaler = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)
    else:
        scaler = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)
    
    # Ridge_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 25, lmb= 1000)
    # save_fig(Ridge_model_complexity.__name__ + f"{int(np.log10(1000))}" + "maxdegree25")
    # Ridge_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 25, method= "Ridge", lmb= 1000)
    # save_fig(Ridge_bias_variance.__name__ + f"{int(np.log10(lmb))}" + "maxdegree25" )
    
    nlambdas = 9; k = 5
    lambdas = np.logspace(-4, 4, nlambdas)
    cv_lmb = np.zeros((nlambdas, k))
    for i in range(nlambdas):
        lmb = lambdas[i]
        lmb_str = str(lmb)
        
        cv_lmb[i,:] = cross_validation(scaler, method= method, k= 5, lmb= lmb)
        
        # R_L_model_complexity(xx, yy, z_flat, regression_opt = "Scikit", maxdegree= 10, method= method, lmb= lmb)
        # save_fig(Ridge_model_complexity.__name__ + f"{int(np.log10(lmb))}")
        # R_L_bias_variance(xx, yy, z_flat, regression_class = "Scikit", maxdegree= 10, method= method, lmb= lmb)
        # save_fig(Ridge_bias_variance.__name__ + f"{int(np.log10(lmb))}")
    
    plt.figure()
    folds = np.arange(1,k+1)
    for i in range(nlambdas):
        lmb = lambdas[i]
        plt.plot(folds, cv_lmb[i], "o-", label= f"$\lambda=$ {lmb}")
    
    plt.xticks(folds)
    plt.title(method + ": Cross-Validaion score for each $\lambda$, deg= 5")
    plt.xlabel("${fold}_k$")
    plt.ylabel("CV Score")
    plt.legend()
    plt.show()
    # save_fig(cross_validation.__name__ + "-Ridge")
    """
elif exercise in ["Exercise5", "Exercise 5"]:
    
    Ridge_Lasso_MSE(xx, yy, z_flat, scaler_opt, regression_opt)
    
    Exercise_4_5(xx, yy, z_flat, scaler_opt, regression_opt= "Scikit", maxdegree= 10, method= "Lasso")
    

if __name__ == "__main__":
    print("Running main.py")







"""
from Exercises import Exercise1, Exercise2

# Exercise1(x, y, z_noisey, var)

# degree_analysis = Exercise2(x, y, z_noisey)
degree_analysis = Exercise2(xx, yy, z_flat)


#Bootstrap
# scale_split = split.Numpy_split_scale(x, y, z_noisey)
scale_split = split.Scikit_split_scale(xx, yy, z_flat)
n_bootstraps = 100
Res = res.Resampling(scale_split)
Res.Bootstrap(regression_class= "Scikit", fit_method = "OLS", n_bootstraps = n_bootstraps, printToScreen= True)
Res.Bootstrap(regression_class= "Manual", fit_method = "OLS", n_bootstraps = n_bootstraps, printToScreen= True)
Res.Scikit_CrossValidation(fit_method = "OLS")
"""
    
"""
# del F; del FF

# print(F.__class__.__name__)
# print("wowo ", F.__class__)

# print("=====================")
# print(FAM.__class__.__name__)
# print("wowo ", FAM.__class__)

"""