import sys 
sys.path.insert(0, '/home/tanjary21/Desktop/bio/project')


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

if __name__ == '__main__':
    wandb.init(project="biomedical")

    #instance 
    path_demo = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
    path_sig = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data"
    path_sal = "../output"
    saliency_dataset = Saliency_loader(path_demo, path_sal)
    saliency_loader = DataLoader(saliency_dataset, shuffle=True)
    # data = next(iter(saliency_loader))
    # print(data['saliency']['AV'].shape)

    energy_classifier = Energy_Classifier()
    optimizer = optim.Adam(energy_classifier.parameters(), lr=0.001)

    precision = Precision(task='multiclass', num_classes=3)
    accuracy = Accuracy(task='multiclass', num_classes=3)
    recall = Recall(task='multiclass', num_classes=3)

    for epoch in range(100):
        probs_epoch, labels_epoch = [], []

        for subject_data in saliency_loader:
            subject_AV_sal, subject_TV_sal, subject_PV_sal, subject_MV_sal, label = subject_data['saliencies']['AV'].squeeze(0), subject_data['saliencies']['TV'].squeeze(0), subject_data['saliencies']['PV'].squeeze(0), subject_data['saliencies']['MV'].squeeze(0), subject_data['murmur']

            optimizer.zero_grad()

            logits = energy_classifier([subject_AV_sal.float(), subject_TV_sal.float(), subject_PV_sal.float(), subject_MV_sal.float()])
            loss = F.nll_loss(logits.unsqueeze(0), label)

            loss.backward()

            optimizer.step()

            probs = F.softmax(logits, dim=0).unsqueeze(0)

            probs_epoch.append(probs), labels_epoch.append(label)
        
        probs_epoch, labels_epoch = torch.concat(probs_epoch), torch.concat(labels_epoch)
        acc, prec, rec = accuracy(probs_epoch, labels_epoch).item(), precision(probs_epoch, labels_epoch).item(), recall(probs_epoch, labels_epoch).item()

        wandb.log({'accuracy': acc, 'precision': prec, 'recall': rec, 'loss': loss.item()})
