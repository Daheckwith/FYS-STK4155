import frankefunction as fr
import error as err
import regression as reg
import numpy as np


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
OLS.OLS_Regression(z_noisey, test_size= 0.2)

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


if __name__ == "__main__":
    print("Running main.py")