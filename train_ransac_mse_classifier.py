import sys 
#sys.path.insert(0, '/home/tanjary21/Desktop/bio/project')


import numpy as np
import matplotlib.pyplot as plt

from data_loader import Data_loader, Saliency_loader
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from classifiers import Energy_Classifier

from torchmetrics.classification import Accuracy, Precision, Recall

import torch.optim as optim

import wandb

import argparse

def train_epoch(train_loader, energy_classifier, optimizer, weights, args):
    energy_classifier.train()
    for subject_data in train_loader:
        #subject_AV_sal, subject_TV_sal, subject_PV_sal, subject_MV_sal, label = subject_data['saliencies']['AV'].squeeze(0), subject_data['saliencies']['TV'].squeeze(0), subject_data['saliencies']['PV'].squeeze(0), subject_data['saliencies']['MV'].squeeze(0), subject_data['murmur']
        energies, labels = subject_data
        
        optimizer.zero_grad()

        #logits = energy_classifier([subject_AV_sal.float(), subject_TV_sal.float(), subject_PV_sal.float(), subject_MV_sal.float()])
        logits = energy_classifier(energies)
        
        loss = F.nll_loss(logits.softmax(1).log(), labels, weight=weights) #weights[labels] * F.mse_loss(logits, F.one_hot(labels, num_classes=3).float().squeeze(0)) #F.nll_loss(logits.unsqueeze(0), label)

        loss.backward()

        optimizer.step()

        probs = logits.softmax(1) #F.softmax(logits, dim=0).unsqueeze(0)

        #probs_epoch.append(probs), labels_epoch.append(label)
        if args.log:
            acc, prec, rec = accuracy(probs, labels).item(), precision(probs, labels).item(), recall(probs, labels).item()
            wandb.log({'train loss': loss.item(), 'train accuracy': acc, 'train precision': prec, 'train recall': rec})

    return energy_classifier

def val_epoch(val_loader, energy_classifier, weights, args):
    energy_classifier.eval()
    with torch.no_grad():
        for subject_data in val_loader:
            #subject_AV_sal, subject_TV_sal, subject_PV_sal, subject_MV_sal, label = subject_data['saliencies']['AV'].squeeze(0), subject_data['saliencies']['TV'].squeeze(0), subject_data['saliencies']['PV'].squeeze(0), subject_data['saliencies']['MV'].squeeze(0), subject_data['murmur']
            energies, labels = subject_data
            
            logits = energy_classifier(energies)
            
            loss = F.nll_loss(logits.softmax(1).log(), labels, weight=weights) #weights[labels] * F.mse_loss(logits, F.one_hot(labels, num_classes=3).float().squeeze(0)) #F.nll_loss(logits.unsqueeze(0), label)
            
            probs = logits.softmax(1) #F.softmax(logits, dim=0).unsqueeze(0)

            #probs_epoch.append(probs), labels_epoch.append(label)
            if args.log:
                acc, prec, rec = accuracy(probs, labels).item(), precision(probs, labels).item(), recall(probs, labels).item()
                wandb.log({'val loss': loss.item(), 'val accuracy': acc, 'val precision': prec, 'val recall': rec})

if __name__ == '__main__':

    # usage: python train_energy_classifier.py --log

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action="store_true")
    args = parser.parse_args()

    if args.log:
        wandb.init(project="biomedical")

    #instance 
    path_demo = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
    path_demo_train = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/train.csv"
    path_demo_val = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/test.csv"
    path_sig = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data"
    path_sal = "../output"
    # saliency_dataset = Saliency_loader(path_demo, path_sal, energy_loader=True)
    # train_size = int(0.7 * len(saliency_dataset))
    # val_size = len(saliency_dataset) - train_size
    train_dataset, val_dataset = Saliency_loader(path_demo_train, path_sal, energy_loader=True, return_as='ransac_mse'), Saliency_loader(path_demo_val, path_sal, energy_loader=True, return_as='ransac_mse') #torch.utils.data.random_split(saliency_dataset, [train_size, val_size])
    train_loader, val_loader = DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset)//4, num_workers=4), DataLoader(val_dataset, shuffle=True, batch_size=len(train_dataset)//4, num_workers=4)

    # data = next(iter(saliency_loader))
    # print(data['saliency']['AV'].shape)

    energy_classifier = Energy_Classifier()
    optimizer = optim.Adam(energy_classifier.parameters(), lr=0.001)

    precision = Precision(task='multiclass', num_classes=3)
    accuracy = Accuracy(task='multiclass', num_classes=3)
    recall = Recall(task='multiclass', num_classes=3)

    weights = torch.Tensor([1, 3.94, 35.0]) # weights for balancing classes [0,1,2] respectively

    for epoch in range(45):
       
       energy_classifier = train_epoch(train_loader, energy_classifier, optimizer, weights, args)
       val_epoch(val_loader, energy_classifier, weights, args)
       
    #    probs_epoch, labels_epoch = torch.concat(probs_epoch), torch.concat(labels_epoch)
    #    acc, prec, rec = accuracy(probs_epoch, labels_epoch).item(), precision(probs_epoch, labels_epoch).item(), recall(probs_epoch, labels_epoch).item()
       
    #    if args.log:
    #     wandb.log({'accuracy': acc, 'precision': prec, 'recall': rec})
    #import ipdb; ipdb.set_trace()