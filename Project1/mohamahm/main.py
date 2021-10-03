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
random = False; precision = 0.05 #initialize_array
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

from Exercises import Confidence_interval, train_test_error

"""
#Exercise1

#Numpy_split_scale
Numpy = split.Numpy_split_scale(xx, yy, z_flat, deg= 5)

train_test_error(Numpy, regression_class= "Manual")
Confidence_interval(Numpy, regression_class= "Manual", var= var)

train_test_error(Numpy, regression_class= "Scikit")
Confidence_interval(Numpy, regression_class= "Scikit", var= var)

#Scikit_split_scale
Scikit = split.Scikit_split_scale(xx, yy, z_flat, deg= 5)

train_test_error(Scikit, regression_class= "Manual")
Confidence_interval(Scikit, regression_class= "Manual", var= var)

train_test_error(Scikit, regression_class= "Scikit")
Confidence_interval(Scikit, regression_class= "Scikit", var= var)
"""

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