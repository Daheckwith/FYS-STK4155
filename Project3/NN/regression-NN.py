from functions import *
import frankefunction as fr
import split_scale as scaler
# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
# from autograd import elementwise_grad
import matplotlib.pyplot as plt
# import seaborn as sns

# =============================================================================
# Regression Franke code
# =============================================================================



# Presets
mean = 0; deviation = 0.1;  #Franke
random = True; precision = 0.05 #initialize_array
noise = True #franke_function

# Generating data using the Franke function
Fr= fr.Franke(mean, deviation)

# x, y = Fr.initialize_array(random, precision)
# xx, yy = np.meshgrid(x, y)

# plt.close("all")
# plt.style.use("seaborn-darkgrid")

# z, x, y = Fr.franke_function(random, precision, noise = False)
# Fr.plot_contour(x, y, z)
# Fr.plot_3D(xx, yy, z)

z_noisey, x, y = Fr.franke_function(random, precision, noise)
# Fr.plot_contour(x, y, z_noisey)
# Fr.plot_3D(xx, yy, z_noisey)

# z_flat = np.ravel(z_noisey)
# z_flat = z_noisey.ravel()
# z_flat = z_noisey.reshape((-1,1))
# -------------------------------------------------------------------------------------

print("before scaler: ", z_noisey.shape, x.shape, y.shape)
scaler = scaler.Numpy_split_scale(x, y, z_noisey, deg= 5)
X_train = scaler.X_train_scaled; X_test = scaler.X_test_scaled
Z_train = scaler.Z_train; Z_test = scaler.Z_test


################ OLD #################
# print("\n", "---------------------------------------------------")
# from sklearn.model_selection import train_test_split
# # The above imports numpy as np so we have to redefine:
# N = 20                # Number of points in each dimension
# deviation = 0.1       # Added noise to the z-value
# deg = 5               # Highest order of polynomial for X

# # batch_size = int(N * N * 0.8)
# x, y, z = generate_data(N, deviation)
# print("x, y, z: ", x.shape, y.shape, z.shape)
# #X = create_X(x, y, deg)
# X = design_matrix(x, y, deg)
# print(f"Design matrix: {X.shape}")
# X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
# X_train, X_test = scikit_scaler(X_train, X_test) ####

#Presets
epochs = 50
cost= "MSE"                # "cross_entropy" "MSE"
activation= "ReLU"         # "sigmoid" "ReLU" "Leaky_ReLU" "Softmax"
last_activation= "sigmoid" # None "sigmoid" "ReLU" "Leaky_ReLU" "Softmax"

lmbd = 0.
# Sigmoid
# NHL 1 3 5
# NHN 5 20 40 100
# ReLU Leaky_ReLU
# NHL 2 3 4
# NHN 3 4 5
NHL = 2 
NHN = 3

# batch size analysis
stops, error_score, etas, batch_sizes, optimal_batch_size =\
analysis(X_train, Z_train, X_test, Z_test, epochs= epochs, cost= cost, analysis= "batch_size",\
         activation= activation, last_activation= last_activation, lmbd= lmbd, nr_of_hidden_layers = NHL, nr_of_hidden_nodes = NHN)

    
plt.close("all")
print(f"Bacht size stopped at {stops.min()} CHANGE TITLE BETWEEN MSE AND R2!!!!!")
make_heat_map(error_score, batch_sizes, np.around(np.log10(etas), 2), fmt= ".3f") # np.log10(etas) np.around(np.log10(etas), 2)
title_string1 = f"Feed Forward Neural Network (NHL= {NHL} NHN= {NHN}):\n"
title_string2 = f"MSE achieved for {epochs} epochs, as a function of \n the learning rate  $\eta$ and batch size"
title_string = title_string1 + title_string2
plt.title(title_string, fontsize= 12)
plt.xlabel(r"Batch size"); plt.ylabel(r"$\log(\eta)$")
# save_fig(f"Figures/FFNN-batch-{cost}-{activation}-{NHL}-{NHN}.png")

# lambda analysis
stops, error_score, etas, lmbds =\
analysis(X_train, Z_train, X_test, Z_test, epochs= epochs, cost= cost, analysis= "hyper",\
         activation= activation, last_activation= last_activation, batch_size= optimal_batch_size, nr_of_hidden_layers = NHL, nr_of_hidden_nodes  = NHN)

print(f"\nHyper stopped at {stops.min()}")
make_heat_map(error_score, np.log10(lmbds), np.around(np.log10(etas), 2), fmt= ".3f") # np.log10(etas) np.around(np.log10(etas), 2)
title_string1 = f"Feed Forward Neural Network (batch_size = {optimal_batch_size} NHL= {NHL} NHN= {NHN}):\n"
title_string2 = f"MSE achieved for {epochs} epochs, as a function of \n the learning rate  $\eta$ and hyperparameter $\lambda$"
title_string = title_string1 + title_string2
plt.title(title_string, fontsize= 12)
plt.xlabel(r"$\log(\lambda)$"); plt.ylabel(r"$\log(\eta)$")
# save_fig(f"Figures/FFNN-hyper-{cost}-{activation}-{NHL}-{NHN}.png")