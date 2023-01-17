from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # 对每一个样本
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]] # 正确分类的值
        # 对当前样本 计算不同的类的累计损失和梯度
        for j in range(num_classes):
            if j == y[i]: # 取的第i个样本对应的标签值，不计算类相同的
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0: # 这里没有使用max函数，而是取大于0的损失考虑进去
                loss += margin
                
                # my_code, 每次计算别的类的损失都要再对样本本来对应的类进行梯度下降，为什么？
                dW[:,j] += X[i].T
                dW[:,y[i]] += - X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0] # 1000
    num_classes = W.shape[1] # 10个分类
    scores = X.dot(W) # 总的score--》(1000,10)
    
    # 对每一个样本（每一行），找出对应的train的label（0-9，一共十个类），利用这个索引找到对应的label计算出的score
    # 注意，此时是一个行向量，所以需要reshape成一个列向量
    correct_class_score = scores[range(num_train),list(y)].reshape(-1,1)
    
    # 利用广播机制计算
    margin = np.maximum(scores - correct_class_score + 1,0)
    loss = np.sum(margin)/num_train + reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 系数矩阵 coeff_mat coefficient matrix
    
    coeff_mat = np.zeros((num_train,num_classes)) # (1000,10)
    coeff_mat[margin > 0 ] = 1 # 只有margin大于1的部分是需要计算梯度的
    coeff_mat[range(num_train),list(y)] = 0 # train 对应的标签值y 的设置为0，因为不用计算
    # 上面的代码是对每一个其他类都减去，这里一次性计算出来了，所以对每一个样本（每一行）把要修改的梯度值求和
    # 就是一个样本上面所需要优化的梯度值
    coeff_mat[range(num_train),list(y)] = - np.sum(coeff_mat,axis = 1) 
    
    dW = (X.T).dot(coeff_mat)
    dW = dW/num_train + reg * W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
