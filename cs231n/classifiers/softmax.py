import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        sample = X[i]
        scores = np.dot(sample, W)
        scores -= np.max(scores)
        prob_unnormalized = np.exp(scores)
        prob_normalized = prob_unnormalized / np.sum(prob_unnormalized)
        correct_class = y[i]
        loss += -np.log(prob_normalized[correct_class])

        for j in range(num_classes):
            dW[:, j] += -sample * (1 if j == correct_class else 0) + sample * prob_normalized[j]
        #############################################################################
        # TODO: Compute the softmax loss and its gradient using explicit loops.     #
        # Store the loss in loss and the gradient in dW. If you are not careful     #
        # here, it is easy to run into numeric instability. Don't forget the        #
        # regularization!                                                           #
        #############################################################################
        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)
    dW /= num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    scores = np.dot(X, W)
    scores -= np.max(scores)
    prob_unnormalized = np.exp(scores)
    prob_normalized = prob_unnormalized / np.sum(prob_unnormalized, axis=1)[:, np.newaxis]
    correct_prob_normalized = prob_normalized[range(num_train), y]
    loss = -np.sum(np.log(correct_prob_normalized))
    prob_normalized[range(num_train), y] -= 1
    dW = np.dot(X.T, prob_normalized)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)
    dW /= num_train
    dW += reg * W
    return loss, dW
