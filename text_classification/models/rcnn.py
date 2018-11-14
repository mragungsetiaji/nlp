# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class RCNN(nn.Module):
	def __init__(self, 
                 batch_size, 
                 output_size, 
                 hidden_size, 
                 vocab_size, 
                 embedding_length, 
                 weights):
		super(RCNN, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embedding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		
		self.batch_size             = batch_size
		self.output_size            = output_size
		self.hidden_size            = hidden_size
		self.vocab_size             = vocab_size
		self.embedding_length       = embedding_length
		
		self.word_embeddings        = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.dropout                = 0.8
		self.lstm                   = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
		self.W2                     = nn.Linear(2*hidden_size+embedding_length, hidden_size)
		self.label                  = nn.Linear(hidden_size, output_size)