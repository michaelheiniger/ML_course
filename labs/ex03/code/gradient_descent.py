# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs


def compute_gradient(y, tx, w):
    """Compute the gradient and loss"""

    e = y-tx.dot(w)
    gradient = -1/y.shape[0] * np.transpose(tx).dot(e)
    
    # Value of the MSE loss (or cost) function
    loss = compute_loss(y, tx, w)
    
    return gradient, loss


def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        # compute gradient and loss
        gradient, loss = compute_gradient(y, tx, w)
        
        # update w by gradient
        #w(t+1) = w(t) - gamma * gradient(w(t))
        w = w - gamma * gradient
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
