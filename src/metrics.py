from sklearn.metrics import f1_score, mean_squared_error
import numpy as np

def f1_macro(y_true, y_pred):

    return f1_score(y_true, y_pred, average="macro")


def rmse(y_true, y_prob):

    return np.sqrt(mean_squared_error(y_true, y_prob))

