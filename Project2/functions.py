import GD as gd
import error as err

# =============================================================================
# Python Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analysis(X_train, Z_train, X_test, Z_test, Manual_REG, method= "OLS", plot_opt = False, batch_size= None):
    GD = gd.Gradient_descent(X_train, Z_train, Manual_REG)
    learning_rates = np.logspace(-4, -1, 4)
    # learning_rates = [0.0001]
    
    if batch_size == None:
        batch_sizes = [20, 40, 80, 160]
    else:
        batch_size = batch_size
    
    epochs_num = int(2*1e5)
    epochs_num = int(2*1e2)
    
    print(X_test.shape, Z_test.shape)
    
    if method in ["ols", "OLS", "Ols"]:
        MSE_cache = np.zeros((len(learning_rates), len(batch_sizes)))
        stops = MSE_cache.copy()
        for i, learning_rate in enumerate(learning_rates):
            for j, batch_size in enumerate(batch_sizes):
                SGD_beta, MSE_train, stop = GD.Stochastic_GD(GD.start_beta, method= method, batch_size= batch_size,\
                                                        epochs= epochs_num, learning_rate= learning_rate, plotting= plot_opt)
                # print("SGD OLS: ", SGD_beta.ravel())
                stops[i][j] = stop # At which epoch did we stop
                Z_pred = Manual_REG.predict(X_test, SGD_beta)
                Err_pred = err.Error(Z_test)
                MSE_pred = Err_pred.MSE(Z_test, Z_pred)
                MSE_cache[i][j] = MSE_pred
                # print(f"eta: {learning_rate}, batch_size: {batch_size}, {MSE_pred}, {MSE_cache[i][j]}")
        
        return stops, MSE_cache, learning_rates, batch_sizes
    elif method in ["ridge", "Ridge", "RIDGE"]:
        lmbs = np.logspace(-5, -1, 5) # lambda
        MSE_cache = np.zeros((len(learning_rates), len(lmbs)))
        stops = MSE_cache.copy()
        for i, learning_rate in enumerate(learning_rates):
            for j, lmb in enumerate(lmbs):
                GD.lmb = lmb # lambda
                SGD_beta, MSE_train, stop = GD.Stochastic_GD(GD.start_beta, method= method, batch_size= batch_size,\
                                                        epochs= epochs_num, learning_rate= learning_rate, plotting= plot_opt)
                # print("SGD OLS: ", SGD_beta.ravel())
                stops[i][j] = stop # At which epoch did we stop
                Z_pred = Manual_REG.predict(X_test, SGD_beta)
                Err_pred = err.Error(Z_test)
                MSE_pred = Err_pred.MSE(Z_test, Z_pred)
                MSE_cache[i][j] = MSE_pred
        
        return stops, MSE_cache, learning_rates, lmbs
    
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
    
def save_fig(name):
    plt.savefig(name, dpi= 180, bbox_inches= 'tight')