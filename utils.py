import torch
import os
import numpy as np
import pandas as pd
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def split_dataset(path_to_csv , train_split):
    # example:
    # train_split = 0.8
    # path_to_csv = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
    csv_dir, filename = os.path.split(path_to_csv)
    data = pd.read_csv(path_to_csv)

    msk = np.random.rand(len(data)) < train_split
    train = data[msk]
    test = data[~msk]

    train.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(csv_dir, "test.csv"), index=False)