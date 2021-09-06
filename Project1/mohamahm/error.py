import numpy as np
import time

class Error:
    def __init__(self, data):
        self.N = data.size
        # print("Error is initialized")
        
    def MSE(self, y_data, y_model):
        # print("Here is MSE")
        return (np.sum(y_data - y_model)**2)/self.N
        
    def MAE(self, y_data, y_model):
        # print("Here is MAE")
        return np.sum(abs(y_data - y_model))/self.N
    
    def R2_Score(self, y_data, y_model):
        # print("Here is R2 Score")
        return 1 - np.sum((y_data - y_model)**2)/np.sum((y_data - np.mean(y_data))**2)