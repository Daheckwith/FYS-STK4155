from FeedForwardNeuralNetwork import NeuralNetwork

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
# from autograd import elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns

# mu = 0; sigma = 0.1

# # Functions copied from frankefunction.py
# def franke_function(x, y, noise = False):
#     n= len(x)

#     term1 =  0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
#     term2 =  0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
#     term3 =  0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

#     if noise:
#         normal_dist = np.random.normal(mu, sigma, (n,n))
#         return term1 + term2 + term3 + term4 + normal_dist #, np.var(normal_dist)
#     else:
#         return term1 + term2 + term3 + term4    

# Taken directly from split_scale.py
# def design_matrix(x, y, deg):

#     if len(x.shape) > 1:
#         x = np.ravel(x)
#         y = np.ravel(y)

#     N = len(x)
#     p = int((deg + 1)*(deg + 2)/2)
#     X = np.ones((N,p))

#     for i in range(1, deg + 1):
#         q = int((i)*(i+1)/2)
#         for k in range(i+1):
#             X[:,q+k] = x**(i-k) * y**k
            
#     return X

# Taken from functions.py
def make_heat_map(matrix, x_axis, y_axis, fmt= ".2f", vmin= None, vmax= None):
    plt.figure()
    
    if fmt[-1] == "f":
        np.around(matrix, int(fmt[-2]))
    
    
    if vmin == None and vmax == None:
        vmin = matrix.max()
        vmax = matrix.min()
    
    ax = sns.heatmap(matrix, xticklabels= x_axis, yticklabels= y_axis,\
                     annot= True, fmt= fmt, vmin= vmin, vmax= vmax)
    # if vmin != None and vmax != None:
    #     ax = sns.heatmap(matrix, xticklabels= x_axis, yticklabels= y_axis,\
    #                      annot= True, fmt= fmt, vmin= vmin, vmax= vmax)
    # else:
    #     ax = sns.heatmap(matrix, xticklabels= x_axis, yticklabels= y_axis,\
    #                      annot= True, fmt= fmt)    

# Taken from functions.py
def save_fig(name):
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')

# Modified version of the one in functions.py
def analysis(X_train, y_train, X_test, y_test, epochs= 5, cost= "MSE", analysis= "batch_size",\
             activation= "sigmoid", last_activation= None, batch_size= None, lmbd= 0.001, gamma= 0.2, nr_of_hidden_layers = 1, nr_of_hidden_nodes  = 20):
    etas  = np.logspace(-4, -1, 4)
#     etas  = np.logspace(-2, -1.01, 4)
    
    if batch_size == None:
        if cost in ["mse", "MSE"]:
            batch_sizes = [20, 40, 80, 160] # Franke
        else:
            batch_sizes = [13, 35, 65, 91]  # Logistic regression cancer [5, 7, 13, 35, 65, 91]
    else:
        batch_size = batch_size
    
    NN = NeuralNetwork(
                     X                   = X_train,
                     y                   = y_train,
                     nr_of_hidden_layers = nr_of_hidden_layers,
                     nr_of_hidden_nodes  = nr_of_hidden_nodes,
                     batch_size          = batch_size,
                     eta                 = None,
                     lmbd                = None,
                     gamma               = gamma,
                     activation          = activation,
                     last_activation     = last_activation,
                     cost                = cost)
    
    
    print(f"gamma for SGD momentum: {NN.gamma}")
    if cost in ["mse", "MSE"]:
        error_func = NN.MSE_score
#         error_func = NN.R2_score
    else:
        error_func = NN.accuracy_score
    
    if analysis in ["batch_size", "batch size", "Batch size"]:
        NN.lmbd = lmbd
        error_cache = np.zeros([len(etas), len(batch_sizes)])
        stops = error_cache.copy()
        for e, eta in enumerate(etas):
            for l, batch_size in enumerate(batch_sizes):
                NN.e = eta
                NN.batch_size = batch_size
#                 epochs = int(2)
                NN.train(epochs) # 50 Epochs
                stop = NN.get_nr_of_epochs()
                
                error_cache[e][l] = error_func(X_test, y_test) 
                stops[e][l] = stop
        
        col_mean = np.mean(error_cache, axis= 0)
        if cost in ["mse", "MSE"]:
            optimal_idx = np.where(col_mean == col_mean.min())[0][0] # if MSE score
#             optimal_idx = np.where(col_mean == col_mean.max())[0][0] # if R2 score
        else:
            optimal_idx = np.where(col_mean == col_mean.max())[0][0]
        
        optimal_batch_size = batch_sizes[optimal_idx]
        
        print("\n Optimal batch size:", optimal_batch_size)
#         print("\n Optimal batch size:", optimal_batch_size, nr_of_hidden_layers, nr_of_hidden_nodes)
        return stops, error_cache, etas, batch_sizes, optimal_batch_size
    elif analysis in ["hyperparameter", "hyper", "HYPER"]:
        lmbds = np.logspace(-5, -1, 5)
        #lmbds = np.logspace(-5, -4, 2)
        error_cache = np.zeros([len(etas), len(lmbds)])
        stops = error_cache.copy()
        for e, eta in enumerate(etas):
            for l, lmbd in enumerate(lmbds):
                NN.e    = eta
                NN.lmbd = lmbd
#                 epochs = int(2)
                NN.train(epochs) # 50 Epochs
                stop = NN.get_nr_of_epochs()
                error_cache[e][l] = error_func(X_test, y_test) 
                stops[e][l] = stop
        return stops, error_cache, etas, lmbds
    
# Modified version of the initialize_array in frankefunction.py
# def generate_data(N, deviation):
#     xx   = np.linspace(0, 1, N)
#     yy   = np.linspace(0, 1, N)
#     x, y = np.meshgrid(xx, yy)
    
#     z = franke_function(x, y, noise= True)
#     z = z.reshape(N**2, 1)
    
#     return x, y, z
    
# def scikit_scaler(X_train, X_test):
#     from sklearn.preprocessing import StandardScaler
    
#     scaler = StandardScaler()
    
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled  = scaler.fit_transform(X_test)
    
#     return X_train_scaled, X_test_scaled