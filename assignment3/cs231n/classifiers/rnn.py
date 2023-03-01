import numpy as np

from ..rnn_layers import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    输入向量大小 D（有D行向量）
    词汇大小 V（有V个词）
    长度为 T（一句话T个词）
    隐状态维度 H
    词向量维度 W
    batch 大小N
   
   
    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=np.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
            给出了词语的字典，大小为 V，把词映射到对应的向量坐标
          
        - input_dim: Dimension D of input image feature vectors.
            图像特征向量的维度
        
        - wordvec_dim: Dimension W of word vectors.
            词向量的维度
        
        - hidden_dim: Dimension H for the hidden state of the RNN.
            隐藏状态维度
        
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors 初始化词向量（总的词的数量，每一个词的长度大小）,只要能区分表示每一个idx即可
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters 初始化隐藏状态的参数，使用的是VGG fc7的features做一个affine forward
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN RNN参数初始化
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        # lstm有四个gate，但是rnn只有一个
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights hidden_dim属于两用，首先是affine传入，其次是“向上”传播。因为他们的维度都一样
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        
        # 输入的词 The first element of captions_in will be the START
        captions_in = captions[:, :-1]
        # 输出的词 the first element of captions_out will be the first word.
        captions_out = captions[:, 1:]

        # You'll need this ， NULL不参与贡献loss
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        W_embed = self.params["W_embed"]
        
        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # (1) (N,D) -> (N,H) 使用affine生出初始隐藏层的状态 h0，这个是直接从RNN卷积到VGGV的fc7直接取出来
        affline_out, affline_cache = affine_forward(features, W_proj, b_proj)
        # print("--------captions_in--------")
        # print(captions_in)
        # print(captions_in.shape)
        # (2) (N,T) -> (N,T,W) 将输入的cations出去最后一个word，转化为词向量表示
        embed_out, embed_cache = word_embedding_forward(captions_in, W_embed)
        # 这里我在纠结为什么captions不是一个个真的words，而是一串idx。别人在做数据的时候就安排好了
        # 我只需要 self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)时，能够做出区分不同的words的matrix即可    coco_utils里面的decode_captions可以帮助返回idx对用的words
        # print("------embed_out ----------")
        # print(embed_out)
        # print(embed_out.shape)
        # (3) (N,T,W),(N,H) -> (N,T,H) “向右传播”
        if self.cell_type == 'rnn':
            cell_out, cell_cache = rnn_forward(embed_out, affline_out, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            cell_out, cell_cache = lstm_forward(embed_out, affline_out, Wx, Wh, b)
        # (4) (N,T,H) -> (N,T,M) “向上传播”
        temporal_out, temporal_cache = temporal_affine_forward(cell_out, W_vocab, b_vocab)
        # (5) 计算LOSS
        loss, dout = temporal_softmax_loss(temporal_out, captions_out, mask)

        # compute grad
        dcell_out, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dout, temporal_cache)
        if self.cell_type == 'rnn':
            dembed_out, daffline_out, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dcell_out, cell_cache)
        elif self.cell_type == 'lstm':
            dembed_out, daffline_out, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dcell_out, cell_cache)
        grads['W_embed'] = word_embedding_backward(dembed_out, embed_cache)
        _, grads['W_proj'], grads['b_proj'] = affine_backward(daffline_out, affline_cache)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word选择最高得分的作为下一个词. 
        
        The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        N, D = features.shape
        # 初始化 hidden state h0
        affine_out, _ = affine_forward(features, W_proj, b_proj)
        # 每一个词开始都是START token
        prev_word_idx = [self._start]*N
        # 一开始的hidden state
        prev_h = affine_out
        prev_c = np.zeros(prev_h.shape)
        # 第一个词是START,N个批量都是START开始
        captions[:, 0] = self._start
        for i in range(1, max_length):
            # 使用index得到对应的词 self._start = word_to_idx.get("<START>", None)获得了初始的idx
            prev_word_embed = W_embed[prev_word_idx]
            # “向右”
            if self.cell_type == 'rnn':
                next_h, _ = rnn_step_forward(prev_word_embed, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c, _ = lstm_step_forward(prev_word_embed, prev_h, prev_c, Wx, Wh, b)
                prev_c = next_c
            # 计算完hidden之后再计算得分 “向上”，W_vocab-->所有的vocab库中的得分，这样就保证了每一个词都在考虑之中
            vocab_affine_out, _ = affine_forward(next_h, W_vocab, b_vocab)
            # 找出N个样本中得分最大的那一个，axis=1是因为是批量处理
            captions[:, i] = list(np.argmax(vocab_affine_out, axis=1))
            # 更新index
            prev_word_idx = captions[:, i]
            prev_h = next_h

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
