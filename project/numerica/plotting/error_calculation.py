import numpy as np

def MAE_calculate(error_absolute):
    return np.mean(error_absolute)


def MSE_calculate(error_absolute):
    pow = error_absolute**2
    return np.mean(pow)


def RMSE_calculate(error_absolute):
    mse = MSE_calculate(error_absolute)
    return np.sqrt(mse)


def MRE_calculate(error_relative):
    return np.mean(error_relative)


def MAPE_calculate(error_relative):
    return np.mean(error_relative)*100