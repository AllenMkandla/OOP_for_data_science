import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
from statsmodels.tools.eval_measures import rmse

class ErrorCalculator():
    def __init__(self, y_test, Y_predicted):
        self.y_test = y_test
        self.Y_predict = Y_predicted

    def get_residuals(self):
        return self.y_test - self.Y_predicted

    def get_standardized_residuals(self):
        return self.get_residuals() / self.get_residuals().std()

    def get_mse(self):
        return np.square(np.subtract(self.y_test, self.Y_predicted)).mean()

    def get_rmse(self):
        return np.sqrt(((self.Y_predicted - self.y_test) ** 2).mean())

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

class Plotter():
    def __init__(self,y_test,y_predicted):
        self.y_test = y_test
        self.y_predict = y_predicted

    def run_calculations(self):
        return self.y_test - self.y_predicted

    def plot(self):
        plt.hist(self.y_test - self.y_predicted)
        plt.title('Residuals Plot for predictions')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        return plt.show()

class HistogramPlotter(Plotter):
    def __init__(self, y_test,y_predicted):
        Plotter.__init__(self, y_test, y_predicted)

class ScatterPlotter(Plotter):
    def __init__(self, y_test, y_predicted):
        Plotter.__init__(self, y_test, y_predicted)

    def plot(self):
        chart = pd.DataFrame({"y_test": self.y_test, "y_prediction": self.y_predicted})
        chart.plot.scatter(x="y_test", y="y_predicted" , c="Blue")
        plt.xlabel("Actual")
        plt.title("Predicted vs Actual values")
        plt.xlabel("Actual")
        plt.ylabel("Prediction")
        return plt.show()
