from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle


def data_shuffle_split(X : np.ndarray,
                       y : np.ndarray,
                       test_size : float,
                       val_size : float,
                       seed : int
                       ) -> tuple[np.ndarray, ...]:
    """
    Shuffles the data and splits into training set, validation set and test set.

    Args:
        X (np.ndarray): Design matrix with i examples and j features
        y (np.ndarray): Target array of length i
        test_size (float): Proportion of the test set (0-1)
        val_size (float): Proportion of the validation set AFTER splitting for test set (0-1)
        seed (int): random seed to be fixed

    Returns:
        tuple[np.ndarray, ...]: Returns in following order, design matrix (X) for training, validation and test set
        followed with target array (y) of training, validation and test set
    """    

    X, y = shuffle(X, y, random_state=seed)

    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size = test_size, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size = val_size, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test