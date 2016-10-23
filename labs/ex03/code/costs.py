# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss"""

    e = y-tx.dot(w)
    #MSE
    return 1/(2*y.shape[0])*(np.transpose(e).dot(e)).sum()
    
    #MAE
    #return 1/(2*y.shape[0])*abs(e).sum()
    