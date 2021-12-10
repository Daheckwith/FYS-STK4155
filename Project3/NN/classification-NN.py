from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functions import *

cancer = load_breast_cancer()

inputs  = cancer.data
outputs = cancer.target
labels  = cancer.feature_names[0:30]

xx = inputs
y = outputs.reshape((len(outputs), 1))

X = np.reshape(xx[:, 1], (len(xx[:, 1]), 1))
features = range(1, 30)
for i in features:
    temp = np.reshape(xx[:, i], (len(xx[:, i]), 1))
    X = np.hstack((X, temp))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test = scikit_scaler(X_train, X_test) ####

#Presets
epochs = 5
cost= "cross_entropy" # "cross_entropy" "MSE"
activation= "Sigmoid" # "sigmoid" "ReLU" "Leaky_ReLU" "Softmax"
last_activation= None # None "sigmoid" "ReLU" "Leaky_ReLU" "Softmax"

lmbd = 0.
# Sigmoid
# NHL 1 3 5
# NHN 5 20 40 100
# ReLU Leaky_ReLU
# NHL 1 3 5
# NHN 10 20 30
NHL = 3
NHN = 20

# batch size analysis
stops, accuracy_score, etas, batch_sizes, optimal_batch_size =\
analysis(X_train, y_train, X_test, y_test, epochs= epochs, cost= cost, analysis= "batch_size",\
         activation= activation, last_activation= last_activation, lmbd= lmbd, nr_of_hidden_layers = NHL, nr_of_hidden_nodes = NHN)

print(f"Bacht size stopped at {stops.min()}")
make_heat_map(accuracy_score, batch_sizes, np.around(np.log10(etas), 2), fmt= ".3f") # np.log10(etas) np.around(np.log10(etas), 2)
title_string1 = f"Feed Forward Neural Network (NHL= {NHL} NHN= {NHN}):\n"
title_string2 = f"Accuracy achieved for {epochs} epochs, as a function of \n the learning rate  $\eta$ and batch size"
title_string = title_string1 + title_string2
plt.title(title_string, fontsize= 12)
plt.xlabel(r"Batch size"); plt.ylabel(r"$\log(\eta)$")
# save_fig(f"Figures/FFNN-batch-{cost}-{activation}-{NHL}-{NHN}.png")

# lambda analysis
stops, accuracy_score, etas, lmbds =\
analysis(X_train, y_train, X_test, y_test, epochs= epochs, cost= cost, analysis= "hyper",\
         activation= activation, last_activation= last_activation, batch_size= optimal_batch_size, nr_of_hidden_layers = NHL, nr_of_hidden_nodes  = NHN)

print(f"\nHyper stopped at {stops.min()}")
make_heat_map(accuracy_score, np.log10(lmbds), np.around(np.log10(etas), 2), fmt= ".3f") # np.log10(etas) np.around(np.log10(etas), 2)
title_string1 = f"Feed Forward Neural Network (batch_size = {optimal_batch_size} NHL= {NHL} NHN= {NHN}):\n"
title_string2 = f"Accuracy achieved for {epochs} epochs, as a function of \n the learning rate  $\eta$ and hyperparameter $\lambda$"
title_string = title_string1 + title_string2
plt.title(title_string, fontsize= 12)
plt.xlabel(r"$\log(\lambda)$"); plt.ylabel(r"$\log(\eta)$")
# save_fig(f"Figures/FFNN-hyper-{cost}-{activation}-{NHL}-{NHN}.png")