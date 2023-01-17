from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        
        # 每一个样本所有类的得分
        scores = X[i].dot(W) 
        
        # 数值稳定性需要找到最大值max，防止上下溢出
        # 总得分里面再减去这个最大值max
        # argmax返回的是下标,所以这里使用max函数
        scores -= max(scores)
        
        scores = np.exp(scores)
        # 同一个训练样本所有类的得分总和
        
        # score_all = np.sum(np.exp(scores)) 这种写法并没有改变scores原来的值，所以scores里面存在0
        score_all = np.sum(scores)
        
        # 每一个样本中所有类各自的得分除以得分总和
        score_each = scores/score_all
        # 计算梯度
        for j in range(num_classes):
            if j != y[i]:
                # 把对应的那一列3073个参数全都更新了
                dW[:, j] += score_each[j] * X[i]

            else:
                # dW[:,j] += (score_each[y[i]] - 1)*X[i]
                dW[:, j] += (score_each[j] - 1) * X[i]
        # 累计损失
        loss -= np.log(score_each[y[i]])
       
    loss /= num_train
    dW /= num_train
    
    # 正则化
    loss += reg*np.sum(W*W) 
    
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    score = X.dot(W)
    
    # 找出每一个样本最大的值 （N，10）
    idx_max = np.argmax(score,axis = 1)
    
    max_val = score[range(num_train),idx_max].reshape(-1,1)
    
    # 每一个类都减去最大的值
    score = score - max_val
    # 每一项都求指数
    score = np.exp(score)
    # 每一项都除以总和得到softmax
    score = score/(np.sum(score,axis = 1).reshape(-1,1))
    
    # 初始化loss对scores的梯度
    ds = np.copy(score)
    # 求出scores的梯度,只有ground truth的部分倒数为 q - 1
    ds[range(num_train),y] -= 1
    # 求出w的梯度
    dW = np.dot(X.T,ds)
    # 计算loss
    loss = score[range(num_train),y]
    loss = -np.log(loss).sum() # 交叉熵
    
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
