import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
from statsmodels.tools.eval_measures import rmse


#ErrorCalculator classs
class ErrorCalculator():
    def __init__(self, y_test, Y_predicted):
        self.y_test = y_test
        self.Y_predict = Y_predicted

# dertemine residuals
    def get_residuals(self):
        return self.y_test - self.Y_predicted

# dertemine standardized residuals
    def get_standardized_residuals(self):
        return self.get_residuals() / self.get_residuals().std()

# determine Mean Squared Error
    def get_mse(self):
        return np.square(np.subtract(self.y_test, self.Y_predicted)).mean()

# detemine Root Mean Squared Error
    def get_rmse(self):
        return np.sqrt(((self.Y_predicted - self.y_test) ** 2).mean())

# error summarry
    def error_summary(self):
        return pd.DataFrame({"Standardized Residuals Average Mean" : [self.get_standardized_residuals().mean()],
                            "Standardized Residuals Average Min": [self.get_standardized_residuals().min()],
                            "Standadized Residuals Average Max" : [self.get_standardized_residuals().max()],
                            "MSE": [self.get_sme()],
                            "RMSE": [self.get_rmse()]},
                            columns ["Standardized Residuals Average Mean",
                                     "Standardized Residuals Average Mean",
                                     "MSE"
                                     "RMSE"])
