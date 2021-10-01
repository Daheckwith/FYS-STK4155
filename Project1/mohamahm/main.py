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
random = False; precision = 0.005#initialize_array
noise = True #franke_function

# Generating data using the Franke function
# Fr= fr.Franke(mean = 0, deviation = 0.02, seed = 4155)
Fr= fr.Franke(mean, deviation)

x, y = Fr.initialize_array(random, precision)
# x, y = Fr.initialize_array(random = True, precision= 0.01)
xx, yy = np.meshgrid(x, y)
# z = Fr.franke_function(xx, yy, noise = False)
z_noisey, var = Fr.franke_function(xx, yy, noise)
# print("Var_Z: ", np.var(z_noisey), np.sqrt(np.var(z_noisey)), np.std(z_noisey))

from Exercises import Exercise1, Exercise2

Exercise1(x, y, z_noisey, var)

Exercise2(x, y, z_noisey)

"""
#Bootstrap
# scale_split = split.Numpy_split_scale(x, y, z_noisey)
scale_split = split.Scikit_split_scale(x, y, z_noisey)
n_bootstraps = 100
Res = res.Resampling(scale_split)
Res.Bootstrap(regression_class= "Scikit", fit_method = "OLS", n_bootstraps = n_bootstraps, printToScreen= True)
Res.Bootstrap(regression_class= "Manual", fit_method = "OLS", n_bootstraps = n_bootstraps, printToScreen= True)
Res.Scikit_CrossValidation(fit_method = "OLS")
"""

if __name__ == "__main__":
    print("Running main.py")
    
    
"""
# del F; del FF

# print(F.__class__.__name__)
# print("wowo ", F.__class__)

# print("=====================")
# print(FAM.__class__.__name__)
# print("wowo ", FAM.__class__)

"""