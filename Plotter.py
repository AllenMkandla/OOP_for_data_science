import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
from statsmodels.tools.eval_measures import rmse



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
#
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
