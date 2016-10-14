# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np
import math


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # Set seed
    np.random.seed(seed)

    # Get size of the training set
    count_training = math.ceil(len(y) * ratio)
    
    # Get indices from 0 to N in randomized order
    indices = np.random.permutation(len(y))
    
    # The first count_training elements of indices are for training set
    # The remaining is for test set
    training_indice, test_indice = indices[:count_training], indices[count_training:]
    
    # Use the splitted indices to split the data for both x and y
    x_training, x_test = x[training_indice], x[test_indice]
    y_training, y_test = y[training_indice], y[test_indice]
    
    return x_training, y_training, x_test, y_test
