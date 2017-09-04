import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num = X.shape[0]
  num_class = W.shape[1]

  for i in xrange(num):
      scores = X[i].dot(W)

      idx = np.argmax(scores)

      scores -= scores[idx] # aviod big data

      softmax_val = np.exp(scores)/np.sum(np.exp(scores))

      loss += -np.log(softmax_val[y[i]])

      for j in xrange(num_class):
        if j == y[i]:
            dW[:,j] += (softmax_val[j]-1)*X[i]
        else:
            dW[:,j] += softmax_val[j] * X[i]

  loss /= num
  loss += reg*np.sum(W*W)

  dW /= num
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num = X.shape[0]

  scores = X.dot(W)

  max_data = np.max(scores,axis=1)

  scores -= max_data.reshape(num,1) # avoid big data

  softmax_val = np.exp(scores) / np.sum(np.exp(scores),axis = 1).reshape(num,1)

  class_idx = np.zeros(scores.shape)

  class_idx[xrange(num),y] = 1

  loss_val = softmax_val * class_idx

  loss_val = loss_val.max(axis = 1)

  loss_val = -np.log(loss_val)

  loss = loss_val.sum()

  softmax_val -= class_idx

  dW = X.T.dot(softmax_val)

  loss /= num
  loss += reg*np.sum(W*W)

  dW /= num
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

