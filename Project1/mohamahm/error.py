import numpy as np
import time

class Error:
    def __init__(self, data):
        self.N = data.size
        
    def MSE(self, y_data, y_model):
        return np.sum((y_data - y_model)**2)/self.N
        
    def MAE(self, y_data, y_model):
        return np.sum(abs(y_data - y_model))/self.N
    
    def R2_Score(self, y_data, y_model):
        
        
        # error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        # ymean = np.mean(y_data, axis = 0)
        # return 1 - np.sum((y_data - y_model)**2)/np.sum((y_data - ymean)**2)
        
        # ymean = np.mean(y_data, axis = 1)
        # diff = (y_data.transpose() - ymean).transpose()
        # return 1 - np.sum((y_data - y_model)**2)/np.sum(diff**2)
        
        return 1. - np.sum((y_data - y_model)**2)/np.sum((y_data - np.mean(y_data))**2)
        # return 1 - np.sum((y_data[:-2] - y_model[:-2])**2)/np.sum((y_data[:-2] - np.mean(y_data))**2)