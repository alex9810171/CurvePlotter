import numpy as np
from scipy.optimize import curve_fit
from function import sigmoid

def get_sigmoid_curve(x, y, x_lower_bound):
    p0 = [np.median(x), 1, max(y), min(y)]    
    popt, pcov = curve_fit(sigmoid, x, y, p0, maxfev=4000)
    
    if x_lower_bound < 0: x_lower_bound = 0
    elif x_lower_bound > max(x): x_lower_bound = max(x)-1
    x = np.linspace(x_lower_bound,
                    max(x),
                    (max(x)-x_lower_bound)*2)
    y = sigmoid(x, *popt)
    
    return x, y