import torch
from torch import nn
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
from DataLoader import SentimentDataset
from LSTM import LSTMModel
from GloveEmbedding import get_embedding
import time
import wandb
import sklearn
import sklearn.metrics

if __name__ == '__main__':
    #I use weights and biases. If I want to train any of the hyperparameters
    wandb.init(
        project="Assignment 2",  # your project name
        name="200 embedding_dim Run",              # custom run name
        config={
            # Training parameters
            "batch_size": 250,
            "n_layers": 2,
            "input_len": 150,
            "embedding_dim": 200,
            "hidden_dim": 25,
            "output_size": 1,
            "num_epochs": 15,
            "learning_rate": 0.0005,
            "clip": 5,
            "load_cpt": False,
            "ckp_path": 'cpt/name.pt',
            "pretrain": True,
            "glove_file": 'glove.6B/glove.6B.200d.txt' 
        }
    )
    
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    
    print("device:", device)
    
    training_set = SentimentDataset('data/training_data.csv')
    training_generator = DataLoader(training_set, batch_size=wandb.config.batch_size, shuffle=True,num_workers=1)
    test_set = SentimentDataset('data/test_data.csv')
    test_generator = DataLoader(test_set, batch_size=wandb.config.batch_size, shuffle=False,num_workers=1)
    
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)
    
    if wandb.config.pretrain:
        embed_matrix = get_embedding(wandb.config.glove_file,tokens2index,wandb.config.embedding_dim)
    else:
        embed_matrix = None
    
    model = LSTMModel(vocab_size, wandb.config.output_size, wandb.config.embedding_dim, embed_matrix, wandb.config.hidden_dim, wandb.config.n_layers, wandb.config.input_len, device, pretrain=wandb.config.pretrain)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    loss_fun = nn.BCELoss()
    
    if wandb.config.load_cpt:
        print("Loaded from Checkpoint")
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    
    print("Training")
    model.train()
    for epoch in range(wandb.config.num_epochs):
        total_loss = 0
        total_accy = 0
        for x_batch, y_labels in training_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            
            y_output = model(x_batch)
            
            loss = loss_fun(y_output, y_labels)
            total_loss += loss
            
            y_pred = torch.round(y_output)
            accy = sklearn.metrics.accuracy_score(y_pred.detach().cpu().numpy(), y_labels.detach().cpu().numpy())
            total_accy += accy
            
            optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(),wandb.config.clip)
            optimizer.step()
        avg_train_loss = total_loss/len(training_generator)
        avg_train_accy = total_accy/len(training_generator)
        wandb.log({"Epoch": epoch, "Average Training Loss": avg_train_loss, "Average Training Accuracy": avg_train_accy})
        ckp_path = f'checkpoint/step_{epoch}.pt'
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, ckp_path)
        
        print("Epoch", epoch)
        
    print("Testing")
    model.eval()
    with torch.no_grad():
        labels = []
        preds = []
        for x_batch, y_labels in test_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            y_pred = torch.round(y_out)
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
    
    print("Finished")