import numpy as np
"""
Custom Loss functions to measure the accuracy of chronos based models on chaotic time series based on the paper
"""

# Symmetric Mean Absolute Percentage Error (SMAPE)
def sMAPE(yPred, yTrue):
    numerator = np.abs(yTrue - yPred)
    denominator = np.abs(yPred) + np.abs(yTrue)
    smape = 2 * (100 / len(yPred)) * (np.sum(numerator) / np.sum(denominator))


    return smape

# argmax 
def valid_prediction_time(yPred, yTrue, epsilon=30):
    for t in range(len(yPred)):
        if sMAPE(yPred[:t+1], yTrue[:t+1]) > epsilon:
            return t
    return len(yPred)
