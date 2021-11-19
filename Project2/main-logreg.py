import numpy as np
import autograd.numpy as np
from autograd import grad
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import GD as gd

dataset = load_breast_cancer()
data = dataset.data
targets = dataset.target

test_size = 0.2
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size= test_size)

# train_targets = train_targets[:, np.newaxis]

# if len(train_targets.shape) == 1 and len(test_targets.shape) == 1:
#     train_targets = train_targets[:, np.newaxis]
#     test_targets = test_targets[:, np.newaxis]

N= data.shape[0]
p= data.shape[1]
N_train = train_data.shape[0]
N_test = test_data.shape[0]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def logistic_prediction(weights, inputs):
    print("dim weights: ", weights.shape, " dim inputs: ", inputs.shape)
    # return sigmoid(np.dot(inputs, weights))
    return sigmoid(np.matmul(inputs, weights))

def cross_entropy_grad(X, y, p):
    X_T = X.T
    return X_T @ (p - y)



# weights = np.random.randn(p, 1) 
# tilde = logistic_prediction(weights, train_data)
# gradient = cross_entropy_grad(train_data, train_targets, tilde)


N = int(1e2) #N = int(1e6)

GD= gd.Gradient_descent(train_data, train_targets)
weights = GD.start_beta
GD_weights = GD.Simple_GD(GD.start_beta, method= "Logistic Regression", Niterations= N)

#test
pred= logistic_prediction(GD_weights, test_data)

for i in range(len(test_targets)):
    print(pred[i][0], test_targets[i])


# print(pred[])

# print(sigmoid(data))

# def sigmoid(x):
    # return 0.5 * (np.tanh(x / 2.) + 1)

# def sigmoid(x):
#     return 1/(1 + np.exp(x))

# def logistic_predictions(weights, inputs):
#     # Outputs probability of a label being true according to logistic model.
#     return sigmoid(np.dot(inputs, weights))

# def training_loss(weights):
#     # Training loss is the negative log-likelihood of the training labels.
#     preds = logistic_predictions(weights, inputs)
#     label_probabilities = preds * targets + (1 - preds) * (1 - targets)
#     return -np.sum(np.log(label_probabilities))

# # Build a toy dataset.
# inputs = data
# targets = targets

# # Define a function that returns gradients of training loss using Autograd.
# training_gradient_fun = grad(training_loss)

# # Optimize weights using gradient descent.
# weights = np.zeros(p)
# print("Initial loss:", training_loss(weights))
# for i in range(100):
#     weights -= training_gradient_fun(weights) * 0.01

# print("Trained loss:", training_loss(weights))