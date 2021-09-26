import frankefunction as fr
import error as err
import regression as reg
import resampling as res
import numpy as np


# New regression.py

# Generating data using the Franke function
# Fr= fr.Franke(mean = 0, deviation = 0.02, seed = 4155)
Fr= fr.Franke(mean = 0, deviation = 0.05)

x, y = Fr.initialize_array(random = False)
# x, y = Fr.initialize_array(random = True)
xx, yy = np.meshgrid(x, y)
# z = Fr.franke(xx, yy, noise = False)
z_noisey = Fr.franke(xx, yy, noise = True)

#Scaling
Scale_Split = reg.Scaler_Splitter(x, y, z_noisey, method = "Manual")
# Scale_Split = reg.Scaler_Splitter(x, y, z_noisey)

#Bootstrap
n_bootstraps = 2
Res = res.Bootstrap(Scale_Split)
Res.Manual_Bootstrap(n_bootstraps)
Res.Scikit_Bootstrap(n_bootstraps)
Res.Scikit_CrossValidation()

# Regression of all data no bootstrap or k-folds
# Manual_OLS = reg.Manual_OLS(Scale_Split)
# Manual_OLS.OLS_Regression()

# Scickit_OLS = reg.Scikit_OLS(Scale_Split)
# Scickit_OLS.OLS_Regression()

if __name__ == "__main__":
    print("Running main.py")
    
    
"""
# Old regression.py
# Fr= fr.Franke(mean = 0, deviation = 0.02, seed = 4155)
Fr= fr.Franke(mean = 0, deviation = 0.05)

x, y = Fr.initialize_array(random = False)
# x, y = Fr.initialize_array(random = True)
# print("x= ", x.shape, "\n", "y= ", y.shape, "\n -------------------------------------------")


OLS = reg.Manual_OLS(x, y)
# OLS = reg.Manual_OLS(x, y, deg = 5)

x, y = np.meshgrid(x,y)
# print("x= ", x, "\n", "y= ", y.shape, "\n -------------------------------------------")
# z = Fr.franke(x,y, noise = False)
z_noisey = Fr.franke(x,y, noise = True)

# OLS.OLS_Regression(z, test_size= 0.2)
X, X_train, X_test= OLS.OLS_Regression(z_noisey, test_size= 0.2)

# Fr.plot_franke(x, y, z)
# Fr.plot_franke(x, y, z_noisey)
# Fr.contour_franke(x, y, z)
Fr.contour_franke(x, y, z_noisey)



# del F; del FF

# print(F.__class__.__name__)
# print("wowo ", F.__class__)

# print("=====================")
# print(FAM.__class__.__name__)
# print("wowo ", FAM.__class__)

"""