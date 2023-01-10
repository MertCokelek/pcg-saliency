# 

import numpy as np
import matplotlib.pyplot as plt
import torch
from data_loader import Data_loader, Saliency_loader
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from classifiers import Conv_1D_classifier

from torchmetrics.classification import Accuracy, Precision, Recall
import torch.optim as optim
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-ep' , type= int, default= 100, help= 'nomber of epochs')
# parser.add_argument('-in_ch' , type= int, default= 1, help= 'number of in_channels') #we can add all the hyper parameters if needed (later task)

args = parser.parse_args()
n_epochs = args.ep


path_demo = "../../datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
path_sig = "../../datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data"
path_sal = "/userfiles/tanjary21/saliencies/output"

saliency_dataset = Saliency_loader(path_demo, path_sal)
saliency_loader = DataLoader(saliency_dataset,batch_size = 1, shuffle=True )
# train_set, test_set = torch.utils.data.random_split(saliency_dataset, [int(0.8*len(saliency_dataset)), int(0.2 * len(saliency_dataset))])
# train_data_loader = DataLoader(dataset= train_set, batch_size = 4, shuffle= True, num_workers=2)
# test_data_loader = DataLoader(dataset =test_set, batch_size = 4, shuffle = True , num_workers=2 )

wandb.init(project="Elec547")
# define the model hyperparameters
in_channels=1
n_depth = 1
output_size=5
kernel_size = 3
n_classes=3
# create the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Conv_1D_classifier(in_channels,n_depth, kernel_size, output_size, n_classes)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
precision = Precision(task='multiclass', num_classes=3)
accuracy = Accuracy(task='multiclass', num_classes=3)
recall = Recall(task='multiclass', num_classes=3)
# train the model
for epochs in range(n_epochs):
    predlabels_epoch, labels_epoch = [], []
    for subject_data in saliency_loader:
        label = subject_data['murmur']
        # subject_AV_sal, subject_TV_sal, subject_PV_sal, subject_MV_sal, labels = subject_data['saliencies']['AV'].squeeze(0), subject_data['saliencies']['TV'].squeeze(0), subject_data['saliencies']['PV'].squeeze(0), subject_data['saliencies']['MV'].squeeze(0), subject_data['murmur']
        labels_epoch.append(label)
        # forward pass
        y_pred = model(subject_data) # the 4 signals are fed to the model, check the classifiers.py\Conv1D_classifier 
        loss = criterion(y_pred, label)
        predlabels_epoch.append(y_pred)
        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predlabels_epoch, labels_epoch = torch.concat(predlabels_epoch), torch.concat(labels_epoch)
    acc, prec, rec = accuracy(predlabels_epoch, labels_epoch).item(), precision(predlabels_epoch, labels_epoch).item(), recall(predlabels_epoch, labels_epoch).item()

    wandb.log({'accuracy': acc, 'precision': prec, 'recall': rec, 'loss': loss.item()})