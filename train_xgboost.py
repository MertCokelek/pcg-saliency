from data_loader import Saliency_loader
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

if __name__ == '__main__':

    path_demo = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
    path_sig = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data"
    path_sal = "../output"
    saliency_dataset = Saliency_loader(path_demo, path_sal, energy_loader=True)
    train_size = int(0.8 * len(saliency_dataset))
    val_size = len(saliency_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(saliency_dataset, [train_size, val_size])
    train_loader, val_loader = DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset)), DataLoader(val_dataset, shuffle=True, batch_size=len(val_dataset))

    train_data, train_labels = next(iter(train_loader))
    val_data, val_labels = next(iter(val_loader))

    train_data, train_labels = train_data.numpy(), train_labels.numpy()
    val_data, val_labels = val_data.numpy(), val_labels.numpy()

    D_train = xgb.DMatrix(train_data, label=train_labels)
    D_test = xgb.DMatrix(val_data, label=val_labels)

    param = {
        'eta': 0.3, 
        'max_depth': 4,  
        'objective': 'multi:softprob',  
        'num_class': 3} 

    steps = 20  # The number of training iterations

    model = xgb.train(param, D_train, steps)

    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    print("Precision = {}".format(precision_score(val_labels, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(val_labels, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(val_labels, best_preds)))

