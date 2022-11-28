import sys
import numpy as np
from scipy.optimize import curve_fit

class LineFitting:
    def __init__(self, point_set):

        point_set = np.array(point_set).transpose()
        self.x_set = point_set[:1][0]
        self.y_set = point_set[1:][0]




    def func_param(self):
        return np.polyfit(self.y_set, self.x_set,2)

    def func(self, X, a, b, c):
        return a*(X**2) + b*(X)+c

