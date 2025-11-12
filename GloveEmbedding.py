import numpy as np
import torch

def get_embedding(glove_file, tokens2ind, dim):
    #Creates a dictionary of all the words used
    w2emb = dict()
    with open(glove_file, "r") as file:
        for line in file:
            data = line.split()
            word = data[0]
            emb = np.array(data[1:], dtype='float32')
            w2emb[word] = emb
    
    #Creates a new matrix to hold the word embeddings
    emb_matrix = np.zeros((len(tokens2ind), dim))
    
    #Fills the new matrix with the embeddings at the words index
    for w, i in tokens2ind.items():
        emb_vect = w2emb.get(w)
        if emb_vect is not None:
            emb_matrix[i] = emb_vect
    
    return torch.FloatTensor(emb_matrix)