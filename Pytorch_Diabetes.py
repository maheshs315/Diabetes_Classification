# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:27:06 2023

@author: S Mahesh Reddy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('diabetes_prediction_dataset.csv')


#Checking Nulls
dataset.isna().sum()


##Preprocessing

x_train=dataset.iloc[:,0:dataset.shape[1]-1]
y_train=dataset.iloc[:,-1]
'''
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

label_encoder=preprocessing.LabelEncoder()
x_train['gender']=label_encoder.fit_transform(x_train['gender'])
x_train['smoking_history']=label_encoder.fit_transform(x_train['gender'])

ohe=OneHotEncoder(categories=[4])
x_train=ohe.fit_transform(x_train).toarray()

print(x_train.shape)'''


x_train['gender']=x_train['gender'].map({'Female':0,'Male':1,'Other':3})
x_train['smoking_history']=x_train['smoking_history'].map({'never':0,'former':1,\
                      'not current':2,'current':3,'ever':4,'No Info':5})

print(x_train.head())


##Defining Model

class classification_model(nn.Module):
    def __init__(self,input_dim):
        super(classification_model,self).__init__()
        self.l1=nn.Linear(input_dim,1)
        #self.r1=nn.ReLU()
        #self.l2=nn.Linear(8,1)
        #self.r2=nn.ReLU()
        #self.l3=nn.Linear(8,1)
        self.r3=nn.Sigmoid()
    
    def forward(self,x):
        out=self.l1(x)
        #out=self.r1(out)
        #out=self.l2(out)
        #out=self.r2(out)
        #out=self.l3(out)
        out=self.r3(out)
        return out
    

x_numpy=x_train.to_numpy().reshape(-1,8)
y_numpy=y_train.to_numpy().reshape(-1,1)

x_tensor=torch.from_numpy(x_numpy.astype(np.float32))
y_tensor=torch.from_numpy(y_numpy.astype(np.float32))

        

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset_train = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset=dataset_train, batch_size=5000)



input_dim=x_numpy.shape[1]

model=classification_model(input_dim)

for i in model.parameters():
    print(i.is_cuda)



#loss and optim
loss1=nn.BCELoss()
optim1=torch.optim.Adam(model.parameters(),lr=1e-2)
n_epochs=100
#training loop


if(torch.cuda.is_available()):
    device='cuda'
else:
    device='cpu'

model=model.to(device)

training_losses = []
for epoch in range(n_epochs):
    batch_losses = []
    batch_acc=[]
    for nbatch, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_pred=model(x_batch)
        l=loss1(y_pred,y_batch)
        
        optim1.zero_grad()
        l.backward()
        optim1.step()
        
        #y_pred_act=y_pred>0.5
        #acc=accuracy_score(y_pred_act.cpu().detach().numpy(),y_batch)
        batch_losses.append(l.item())
        #batch_acc.append(acc)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)
    #acc_tot=np.mean(batch_acc)
    if((epoch+1)%10==0):
        print(f'epoch {epoch+1},Training loss: {training_loss:.4f}')




with torch.no_grad():
    y_pred=model(x_tensor.to(device))
    print(y_pred[8])

y_pred.shape

y_pred1=y_pred>0.5
print(y_pred1)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm=confusion_matrix(y_pred1.to('cpu'),y_train)
print(cm)
acc=accuracy_score(y_pred1.to('cpu'),y_train)
print("accuracy:",acc)

