import torch
from torch import nn
import pandas as pd
import numpy as np

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, output_size, embed_dim, embed_matrix,hidden_dim, n_layers, input_len, device, pretrain=False):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        #Uses the GloVe weights
        if pretrain:
            self.emb = nn.Embedding.from_pretrained(embed_matrix)
        #Uses randomly initialized weights
        else:
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.emb.weight.data.uniform_(-0.1, 0.1)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.pool = nn.MaxPool1d(input_len)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        hidden_cell = torch.autograd.Variable(torch.randn(self.n_layers, x.size(0), self.hidden_dim)).to(self.device), torch.autograd.Variable(torch.randn(self.n_layers, x.size(0), self.hidden_dim)).to(self.device)
        embeds = self.emb(x)
        
        lstm_out, _ = self.lstm(embeds, hidden_cell)
        lstm_out = lstm_out.permute(0,2,1)
        pool_out = self.pool(lstm_out)
        pool_out = pool_out.view(pool_out.size(0), -1)
        drop_out = self.dropout(pool_out)
        output = self.fc(drop_out)
        output = self.sigmoid(output)
        return output[:,0]
        