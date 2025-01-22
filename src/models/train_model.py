import os
import sys
sys.path.append('../src')
import time

from balance import load_balanced
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from torch import optim
from sklearn.model_selection import train_test_split
import joblib


def evaluate_model(model, test_set):
    X, y_true = test_set[:]
   
    model.eval()
    criterion = nn.CrossEntropyLoss()
    # Calcul de la fonction de perte
    
    with torch.no_grad():
        # Prédiction du modèle pour un batch donné
        y_pred = model(X)

    loss_test = criterion(y_pred, y_true)
    
    y_pred = np.argmax(y_pred.detach().numpy(),axis=1)
    y_true = y_true.numpy()
    accuracy = precision_score(y_true, y_pred,average='macro')
    recall = recall_score(y_true, y_pred,average='macro')
    return y_true, y_pred, accuracy, recall, loss_test.item()

    
def train_a_model(model,train_loader,test_set,epochs = 100,lr=1e-3,device='cpu',outmodelname=None):


  
    
    best_accuracy = 0
    optimizer = optim.Adam(model.parameters(), lr)

       # Définir le scheduler ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    factor=0.5,
                                                    patience=5,
                                                    verbose=True)
    # Définition de la fonction de perte
    criterion = nn.CrossEntropyLoss()

    loss_list = []
    loss_list_test = []

    accuracy_list = []
    start = time.time()
    
    for epoch in range(epochs):
        
        # Dans ce mode certaines couches du modèle agissent différemment
        model.train()
        loss_total = 0
        
        for batch in train_loader:
            # Batch de données
            X_batch, y_batch = batch
            
            # Device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Gradient mis 0
            model.zero_grad()
    
            # Calcul de prédiction
            y_pred = model(X_batch)
    
            # Calcul de la fonction de perte
            loss = criterion(y_pred, y_batch)
    
            # Backpropagation : calculer le gradient de la loss en fonction de chaque couche
            loss.backward()
            
            # Clipper le gradient entre 0 et 1 pour plus de stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Descente de gradient : actualisation des paramètres
            optimizer.step()
            
            loss_total += loss.item()

        scheduler.step(loss_total/len(train_loader))
        y_true, y_pred, accuracy, recall, loss = evaluate_model(model, test_set)
        if accuracy > best_accuracy and outmodelname is not None:
            joblib.dump(model, outmodelname)
            print('save model')
            best_accuracy = accuracy
        loss_list.append(loss_total/len(train_loader))
        loss_list_test.append(loss)
        accuracy_list.append(accuracy)
        print(f"Epoch : {epoch+1}/{epochs} -- Training loss {loss_total/len(train_loader)}, --- Val loss {loss}, --- accuray : {accuracy}, --- recall : {recall}")
    
    end = time.time()
    
    print("execution time: ",end - start)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(loss_list,label='loss train')
    ax.plot(loss_list_test,label='loss test')
    ax.legend()

    table = pd.crosstab(y_true,y_pred,rownames=['True'],colnames=['Predicted'])
    print(table)

    print(classification_report(y_true, y_pred))

def cnn_type_2(input_shape=187,device='cpu',dropout=0.4):
    # shape: (batch_size, n, 187)
    model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=80, kernel_size=5, stride=1, padding=0), # shape: (batch_size, n, 183)
        nn.ReLU(),
        nn.BatchNorm1d(num_features=80),
        nn.MaxPool1d(2, stride=2), # shape: (batch_size, n, 91)
        
        nn.Conv1d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=0), # shape: (batch_size, n, 89)
        nn.ReLU(),
        nn.BatchNorm1d(num_features=80),
        nn.MaxPool1d(2, stride=2), # shape: (batch_size, n, 44)
        
        nn.Conv1d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=0), # shape: (batch_size, n, 42)
        nn.ReLU(),
        nn.BatchNorm1d(num_features=80),
        nn.MaxPool1d(2, stride=2), # shape: (batch_size, n, 21)
        
        nn.Flatten() , # shape: (batch_size, 10*n)
        nn.Dropout(dropout),
        nn.Linear(21*80, 50),
        nn.ReLU(),
        nn.Linear(50, 5),
         #nn.Softmax(dim=-1)
    )
    model.to(device)
    summary(model, input_size=(input_shape,), device=device)
    return model


# main
if not os.path.isfile('../data/processed/mitbih_train_smote_perturb_50000.csv'):
    Xo,yo = load_balanced.load_balanced_data(method='smote-perturb',n_normal=50000,
                       smote_perturb_smote_ratio=0.5)
    pd.DataFrame(np.concatenate((Xo,np.expand_dims(yo,axis=1)),axis=1)).to_csv('../data/processed/mitbih_train_smote_perturb_50000.csv',index=None,header=None)

epochs = 50

X_train, y_train = load_balanced.load('../data/processed/mitbih_train_smote_perturb_50000.csv')
X_train,X_cross, y_train, y_cross = train_test_split(X_train,y_train,test_size=0.15,random_state=12)

X_train =np.expand_dims(X_train,axis=1)
X_cross =np.expand_dims(X_cross,axis=1)


train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.Tensor(torch.from_numpy(y_train).long()))
test_set = TensorDataset(torch.from_numpy(X_cross).float(), torch.Tensor(torch.from_numpy(y_cross).long()))

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)

model_2 = cnn_type_2()

train_a_model(model_2,train_loader,test_set,epochs = epochs,lr=1e-3,outmodelname='../models/heartbeat_model.pkl')



