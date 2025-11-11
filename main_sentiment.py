#!/usr/bin/env python
# coding: utf-8
'''fun: sentiment analysis'''


import torch
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
# import pandas as pd
from DataLoader import MovieDataset
from LSTM import LSTMModel
from glove_embedding import _get_embedding
import time
import wandb
import sklearn
import sklearn.metrics


'''save checkpoint'''
def _save_checkpoint(ckp_path, model, epoches, optimizer):
    checkpoint = {'epoch': epoches,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, ckp_path)


def main():
    wandb.init(
        project="Assignment 2",  # your project name
        name="200 embedding_dim Run",              # custom run name
        config={
            # Training parameters
            "batch_size": 250,
            "n_layers": 2,
            "input_len": 150,
            "embedding_dim": 100,
            "hidden_dim": 25,
            "output_size": 1,
            "num_epochs": 15,
            "learning_rate": 0.0005,
            "clip": 5,
            "load_cpt": False,
            "ckp_path": 'cpt/name.pt',
            "pretrain": False
        }
    )
	
    
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    
    ## alternative method
    # torch.cuda.set_device(device=0) ## choose gpu number
    print('device: 2', device)

    ## ---------------------------------------------------------
    ## please change the parameter settings by yourselves
    ## ---------------------------------------------------------
    mode = 'train'
    Batch_size =wandb.config.batch_size
    n_layers = wandb.config.n_layers ## choose 1-3 layers

    ## input seq length aligned with data pre-processing
    input_len = wandb.config.input_len

    ## word embedding length
    embedding_dim = wandb.config.embedding_dim

    # lstm hidden dim
    hidden_dim = wandb.config.hidden_dim
    # binary cross entropy
    output_size = wandb.config.output_size
    num_epoches = wandb.config.num_epochs
    ## please change the learning rate by youself
    learning_rate = wandb.config.learning_rate
    # gradient clipping
    clip = wandb.config.clip
    load_cpt = wandb.config.load_cpt
    ckp_path = wandb.config.ckp_path
    # embedding_matrix = None
    ## use pre-train Glove embedding or not?
    pretrain = wandb.config.pretrain

    ##-----------------------------------------------------------------------
    ## Bonus (5%): complete code to add GloVe embedding file path below.
    ## Download Glove embedding from https://nlp.stanford.edu/data/glove.6B.zip
    ## "embedding_dim" defined above shoud be aligned with the dimension of GloVe embedddings
    ## if you do not want bonus, you can skip it.
    ##-----------------------------------------------------------------------
    glove_file = 'glove.6B/glove.6B.200d.txt' ## change by yourself
    

    ## ---------------------------------------------------------
    ## step 1: create data loader in DataLoader.py
    ## complete code in DataLoader.py (not Below)
    ## ---------------------------------------------------------
    
    
    ## step 2: load training and test data from data loader [it is Done]
    training_set = MovieDataset('data/training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size,\
                                    shuffle=True,num_workers=1)
    test_set = MovieDataset('data/test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size,\
                                shuffle=False,num_workers=1)


    ## step 3: [Bonus] read tokens and load pre-train embedding [it is Done]
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(glove_file,tokens2index,embedding_dim)
    else:
        embedding_matrix = None

    ## -----------------------------------------------
    ## step 4: import model from LSTM.py
    ## complete the code in "def forward(self, x)" in LSTM.py file
    ## then import model from LSTM.py below
    ## and also load model to device
    ## -----------------------------------------------
    model = LSTMModel(vocab_size, output_size, embedding_dim, embedding_matrix,hidden_dim, n_layers, input_len, pretrain=pretrain)
    model.to(device)

    ##-----------------------------------------------------------
    ## step 5: complete code to define optimizer and loss function
    ##-----------------------------------------------------------
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)  # 
    ## define Binary Cross Entropy Loss below
    loss_fun = nn.BCELoss()
    
    ## step 6: load checkpoint
    if load_cpt:
        print("*"*10+'loading checkpoint'+'*'*10)
        ##-----------------------------------------------   
        ## complete code below to load checkpoint
        ##-----------------------------------------------
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoches = checkpoint['epoch']
        


    ## step 7: model training
    print('*'*89)
    print('start model training now')
    print('*'*89)
    if mode == 'train':
        model.train()
        global_step = 0
        for epoches in range(num_epoches):
            total_loss = 0
            total_accy = 0
            for x_batch, y_labels in training_generator:
                global_step += 1
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                ##-----------------------------------------------
                ## complete code to get predict result from model
                ##-----------------------------------------------
                y_out = model(x_batch)

                ##-----------------------------------------------
                ## complete code to get loss
                ##-----------------------------------------------
                loss = loss_fun(y_out, y_labels)
                total_loss += loss
                
                y_pred = torch.round(y_out)
                accy = sklearn.metrics.accuracy_score( y_pred.detach().cpu().numpy(), y_labels.detach().cpu().numpy())
                total_accy += accy

                ## step 8: back propagation [Done]
                optimizer.zero_grad()
                loss.backward()
                ## **clip_grad_norm helps prevent the exploding gradient problem in LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            
            avg_train_loss = total_loss/len(training_generator)
            avg_train_accy = total_accy/len(training_generator)
            wandb.log({"Epoch": epoches, "Average Training Loss": avg_train_loss, "Average Training Accuracy": avg_train_accy})
            ##-----------------------------------------------   
            ## step 9: complete code below to save checkpoint
            ##-----------------------------------------------
            print("**** save checkpoint ****")
            ckp_path = f'checkpoint/step_{epoches}.pt'
            _save_checkpoint(ckp_path, model, epoches, optimizer)
            
    
    ##------------------------------------------------------------------
    ## step 10: complete code below for model testing
    ## predict result is a single value between 0 and 1, such as 0.8, so
    ## we can use y_pred = torch.round(y_out) to predict label 1 or 0
    ##------------------------------------------------------------------
    print("----model testing now----")
    
    model.eval()
    with torch.no_grad():
        labels = []
        preds = []
        for x_batch, y_labels in test_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            y_pred = torch.round(y_out)
            #I need to do the metrics
            labels.extend(y_labels.cpu().numpy())
            preds.extend(y_pred.cpu().numpy())
   
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    
    #Code to calculate and log metrics in WandB
    test_accy = sklearn.metrics.accuracy_score(preds, labels)
    test_recall = sklearn.metrics.recall_score(preds, labels, average='macro')
    test_precision = sklearn.metrics.precision_score(preds, labels, average='macro')
    test_f1 = sklearn.metrics.f1_score(preds, labels, average='macro')
    
    wandb.log({"Test Accuracy": test_accy, "Test Recall": test_recall, "Test Precision": test_precision, "Test F1": test_f1})
       
    #Code to create a Confusion Matrix of the testing results 
    cm = sklearn.metrics.confusion_matrix(labels, preds, labels=[0,1])
    
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.title('Sentiment')
    plt.show()

    plt.savefig("confusion_matrix_100.png", dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
    


    

    
