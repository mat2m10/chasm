import numpy as np
import pandas as pd
import subprocess
import os


"""
All the risk functions
"""
def no_risk_fun(x, y):
    return 0 * x

def NW_risk_fun(x, y):
    return 0.25 * (y - x + 1)

def N_risk_fun(x, y):
    return 0.5 * y

def blob_risk_fun(x, y):
    return np.where(np.sqrt((x - 0.25)**2 + (y - 0.75)**2) < 0.25, 0.5, 0)

def center_risk_fun(x, y):
    return np.where(np.sqrt((x - 0.5)**2 + (y - 0.5)**2) < 0.25, 0.5, 0)

def big_square_risk_fun(x, y):
    return np.where((y > 0.1) & (y < 0.5) & (x > 0.5) & (x < 0.9), 0.4, 0)

def square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76), 1, 0)

def hi_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76), 2, 0)

def mid_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.3) & (x > 0.6) & (x < 0.7), 1, np.where((y > 0.15) & (y < 0.35) & (x > 0.54) & (x < 0.76), 0.5, 0))

def mid_mid_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 1, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 0.7, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 0.35, 0)))

def mid_mid_mid_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 0.8, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 0.6, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 0.4, np.where((y > 0.04) & (y < 0.41) & (x > 0.44) & (x < 0.81), 0.2, 0))))

def big_bad_square_risk_fun(x, y):
    return np.where((y > 0.2) & (y < 0.6) & (x > 0.35) & (x < 0.76), 1, 0)

# Re-definition of big_square_risk_fun with new logic
def big_square_risk_fun_updated(x, y):
    return 0.75 * np.where((y > 0.2) & (y < 0.3) & (x > 0.6) & (x < 0.7), 1, np.where((y > 0.15) & (y < 0.35) & (x > 0.54) & (x < 0.76), 1, 0))

def big_big_square_risk_fun(x, y):
    return 0.6 * np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 1, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 1, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 1, 0)))

def big_big_big_square_risk_fun(x, y):
    return 0.4285 * np.where((y > 0.2) & (y < 0.25) & (x > 0.6) & (x < 0.65), 1, np.where((y > 0.15) & (y < 0.3) & (x > 0.54) & (x < 0.71), 1, np.where((y > 0.09) & (y < 0.36) & (x > 0.49) & (x < 0.76), 1, np.where((y > 0.04) & (y < 0.41) & (x > 0.44) & (x < 0.81), 1, 0))))

def two_square_risk_fun(x, y):
    return np.where(((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76)) | ((y > 0.7) & (y < 0.86) & (x > 0.1) & (x < 0.26)), 1, 0)

def three_square_risk_fun(x, y):
    return np.where(((y > 0.2) & (y < 0.36) & (x > 0.6) & (x < 0.76)) | ((y > 0.7) & (y < 0.86) & (x > 0.1) & (x < 0.26)) | ((y > 0.2) & (y < 0.36) & (x > 0.1) & (x < 0.26)), 1, 0)

def four_square_risk_fun(x, y):
    return np.where(((y > 0.2) & (y < 0.36) & (x > 0.65) & (x < 0.8)) | ((y > 0.7) & (y < 0.86) & (x > 0.2) & (x < 0.36)) | ((y > 0.2) & (y < 0.36) & (x > 0.2) & (x < 0.36)) | ((y > 0.7) & (y < 0.86) & (x > 0.65) & (x < 0.8)), 1, 0)

def as_big_blob_risk_fun(x, y):
    return np.where(np.sqrt((x - 0.25)**2 + (y - 0.75)**2) < 0.5, 0.5, 0)

def six_square_risk_fun(x, y):
    return np.where((y > 0.5) & (x < 0.7), 0.5, 0)

def gauss_blob_risk_fun(x, y):
    return 1 * np.exp(-((x - 0.25)**2 + (y - 0.75)**2) / (2 * 0.25**2))

def hi_gauss_blob_risk_fun(x, y):
    return 2 * np.exp(-((x - 0.25)**2 + (y - 0.75)**2) / (2 * 0.25**2))

def hi_hyperbole_risk_fun(x, y):
    return 2 *((x-0.5)**2 + (y-0.5)**2)

def hi_tangeant_risk_fun(x, y):
    return (x-0.5)**3 + (y-0.5)**3

def sine_risk_fun(x, y):
    f = 2
    return np.sin(f * (x+y)*2 * np.pi)