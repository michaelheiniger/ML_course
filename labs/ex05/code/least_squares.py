# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    #Solve as Ax = b
    A = np.transpose(tx).dot(tx)
    b = np.transpose(tx).dot(y)
    w_opt = np.linalg.solve(A,b)
    
    loss = compute_loss(y, tx, w_opt)
    
    return loss, w_opt
        
