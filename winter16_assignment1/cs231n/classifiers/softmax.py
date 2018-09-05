import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. 3073x10
  - X: A numpy array of shape (N, D) containing a minibatch of data. 1000 * 3073
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means 
        that X[i] has label c, where 0 <= c < C. 1000*
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) #3073 x 10
 
  num_train = X.shape[0] # 1000
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train) :
      f = X[i].dot(W)
      f -= np.max(f)
      scores = -np.log(np.exp(f[y[i]]) / np.sum(np.exp(f)))
      loss += scores
      for j in xrange(num_classes) : 
            pj = np.exp(f[j]) / sum(np.exp(f)) 
            if j == y[i] :
                dW[:,j] += (pj-1) * X[i]
            else :
                dW[:,j] += pj * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
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
  num_train = X.shape[0] # 1000
  num_classes = W.shape[1]
  #############################################################################
      # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W) 
  f-= np.max(f,axis = 1).reshape(-1,1)
  scores = np.exp(f)/ np.sum(np.exp(f),axis=1).reshape(-1,1)
  loss = np.sum(-np.log(scores[range(num_train),list(y)]))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  
  dW = scores
  dW[range(num_train),list(y)] -= 1
  dW = (X.T).dot(dW)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

