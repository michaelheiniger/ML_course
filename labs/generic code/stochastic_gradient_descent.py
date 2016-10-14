# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""    
    # Implement stochastic gradient computation.It's same as the gradient descent.
    # Gradient of MSE loss (or cost) function is: 1/N transpose(X-tilde)*e
    e = y-tx.dot(w)
    gradient = -1/y.shape[0] * np.transpose(tx).dot(e)
    
    # Value of the MSE loss (or cost) function
    loss = compute_loss(y, tx, w)
    
    return gradient, loss


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""    
    # Implement stochastic gradient descent.
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    max_iters = max_epochs

    for n_iter in range(max_iters):      
        for y_n, tx_n in batch_iter(y, tx, batch_size, True):
            gradient, loss = compute_stoch_gradient(y_n, tx_n, w)
            w = w - gamma * gradient
            
            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)

            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    
    return losses, ws
