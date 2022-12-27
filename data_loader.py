import numpy as np
import pandas as pd
# import torch 
# import torchvision 
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import math
# from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import librosa
import librosa.display

class Data_loader():
    def __init__(self, path_demo, path_sig, encode=True):
        # self.subject_id = subject_id
        self.path_sig = path_sig
        self.demo_data = pd.read_csv(path_demo) # example:~/courses/bio-sig/datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv
       
       
        self.subject_ids = self.demo_data[self.demo_data['Recording locations:'] == "AV+PV+TV+MV"]['Patient ID'] # getting all the subjects that all the recordings are available
        self.demo_data = self.demo_data[["Patient ID", "Murmur", "Outcome"]]
        if encode:
            label_encoder= LabelEncoder()
            self.demo_data['Murmur']= label_encoder.fit_transform(self.demo_data['Murmur']) # Murmur is absent/present/Unknown 
            self.demo_data['Outcome']= label_encoder.fit_transform(self.demo_data['Outcome']) # Diagnosed by the medical expert: normal/abnormal


   
    
    def get_item(self, index, record_type):
        ret = {}
        ret['subject_id'] = self.subject_ids[index]
        ret['record_type'] = record_type
        path_root = f"{self.path_sig}/{self.subject_ids[index]}_{record_type}"
        ret['audio'] , ret['sr'] = librosa.load(f"{path_root}.wav", mono=True, sr=None)
        ret['murmur'], ret['outcome'] = self.demo_data[self.demo_data['Patient ID']==self.subject_ids[index]][['Murmur', 'Outcome']].iloc[0]
        return ret
        
    

#instance 
path_demo = "../../datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
path_sig = "../../datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data"
data_sample = Data_loader(path_demo, path_sig)
d_test = data_sample.get_item(0,'TV')
print(d_test)