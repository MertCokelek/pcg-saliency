# 

import numpy as np
import matplotlib.pyplot as plt
import torch
from data_loader import Data_loader, Saliency_loader
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from classifiers import Conv_1D_classifier
import matplotlib.pyplot as plt
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
# saliency_loader = DataLoader(saliency_dataset,batch_size = 1, shuffle=True ) # to train only with the whole available data
train_set, test_set = torch.utils.data.random_split(saliency_dataset, [int(0.7*len(saliency_dataset))+1, int(0.3 * len(saliency_dataset))])
train_data_loader = DataLoader(dataset= train_set, batch_size = 1, shuffle= True ) #num_workers=2
test_data_loader = DataLoader(dataset =test_set, batch_size = 1, shuffle = True )#, num_workers=2 

wandb.init(project="Elec547")
# define the model hyperparameters
in_channels=1
n_depth = 32
output_size=128
kernel_size = 11
n_classes=3
# create the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Conv_1D_classifier(in_channels,n_depth, kernel_size, output_size, n_classes)

# define the loss function and optimizer
weights= torch.Tensor([1, 3.94, 35.0]) # weights for balancing classes [0,1,2] respectively
criterion = nn.CrossEntropyLoss(weight= weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
precision = Precision(task='multiclass', num_classes=3)
accuracy = Accuracy(task='multiclass', num_classes=3)
recall = Recall(task='multiclass', num_classes=3)
# train the model

for epochs in range(n_epochs):
    predlabels_epoch, labels_epoch = [], []
    
    for subject_data in train_data_loader: # saliency_loader
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

    wandb.log({'train_accuracy': acc, 'train_precision': prec, 'train_recall': rec, 'train_loss': loss.item()})


    test_gt_labels=[]
    test_pred_labels=[]
    test_loss=0
    with torch.no_grad():
        model.eval()
        for test_subject_data in test_data_loader:
            test_gt_label= test_subject_data["murmur"]
            test_gt_labels.append(test_gt_label)
            test_pred_label = model(test_subject_data)
            test_pred_labels.append(test_pred_label)
            t_loss = criterion(test_pred_label, test_gt_label)
            test_loss += t_loss.item()
        # print(pred_labels)
        test_pred_labels, test_gt_labels = torch.tensor(torch.concat(test_pred_labels)), torch.tensor(torch.concat(test_gt_labels))
        # pred_labels, gt_labels = torch.tensor(pred_labels), torch.tensor(gt_labels)
        acc, prec, rec = accuracy(test_pred_labels, test_gt_labels), precision(test_pred_labels, test_gt_labels), recall(test_pred_labels, test_gt_labels)

        data = [[s] for s in test_pred_labels.argmax(1)]
        table = wandb.Table(data=data, columns=["labels"])
        wandb.log({'label_histogram': wandb.plot.histogram(table, "label",
            title="Prediction label Distribution")})
        wandb.log({'val accuracy': acc, 'val precision': prec, 'val recall': rec, 'val loss':test_loss/len(test_data_loader)})
    model.train()
