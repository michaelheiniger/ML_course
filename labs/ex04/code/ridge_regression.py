# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_loss

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # Get w solving Ax = b
    A = np.transpose(tx).dot(tx) + 2*lamb*len(y)*np.identity(tx.shape[1])
    b = np.transpose(tx).dot(y)
    w_opt = np.linalg.solve(A, b)
    
    loss = compute_loss(y, tx, w_opt)
    
    return loss, w_opt
