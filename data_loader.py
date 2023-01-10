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
import scipy.io
import scipy.io as sio
from torch.utils.data import Dataset
import torch

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
        ret['anomaly_pred'] = sio.loadmat(f"{path_root}_saliency.mat")['saliency']
        return ret
        
    

##instance 
# path_demo = "../../datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
# path_sig = "../../datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data"
# data_sample = Data_loader(path_demo, path_sig)
# data = data_sample.get_item(0,'TV')
# print(data)

class Saliency_loader(Dataset):
    def __init__(self, path_demo, path_sal, encode=True, energy_loader=False):
        # self.subject_id = subject_id
        self.energy_loader=energy_loader
        self.path_sal = path_sal
        self.demo_data = pd.read_csv(path_demo) # example:~/courses/bio-sig/datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv
       
       
        self.subject_ids = self.demo_data[self.demo_data['Recording locations:'] == "AV+PV+TV+MV"]['Patient ID'].reset_index(drop=True) # getting all the subjects that all the recordings are available
        self.demo_data = self.demo_data[["Patient ID", "Murmur", "Outcome"]]
        if encode:
            label_encoder= LabelEncoder()
            self.demo_data['Murmur']= label_encoder.fit_transform(self.demo_data['Murmur']) # Murmur is absent/present/Unknown 
            self.demo_data['Outcome']= label_encoder.fit_transform(self.demo_data['Outcome']) # Diagnosed by the medical expert: normal/abnormal
 
    
    def __len__(self):
        return self.subject_ids.shape[0]

    
    def __getitem__(self, index):
        
        
        ret = {}
        ret['subject_id'] = self.subject_ids.iloc[index]
        
        ret['saliencies'] = {}
        path_root = f"{self.path_sal}/{self.subject_ids.iloc[index]}"
        subject_AV, subject_TV, subject_MV, subject_PV = f"{path_root}_AV_saliency.mat", f"{path_root}_TV_saliency.mat", f"{path_root}_MV_saliency.mat", f"{path_root}_PV_saliency.mat"
        subject_AV, subject_TV, subject_MV, subject_PV = scipy.io.loadmat(subject_AV), scipy.io.loadmat(subject_TV), scipy.io.loadmat(subject_MV), scipy.io.loadmat(subject_PV)
        ret['saliencies']['AV'], ret['saliencies']['TV'], ret['saliencies']['MV'], ret['saliencies']['PV'] = subject_AV['saliency'], subject_TV['saliency'], subject_MV['saliency'], subject_PV['saliency']
        
        ret['murmur'], ret['outcome'] = self.demo_data[self.demo_data['Patient ID']==self.subject_ids.iloc[index]][['Murmur', 'Outcome']].iloc[0]

        if not self.energy_loader:
            return ret
        else:
            saliencies = [subject_AV['saliency'], subject_TV['saliency'], subject_MV['saliency'], subject_PV['saliency']]
            saliencies = [ torch.Tensor(saliency) for saliency in saliencies]
            energies = self.compute_energies(saliencies)
            return energies, ret['murmur']

    def compute_energies(self, x):
        
        result = []
        for sig in x:
            sig_energy = self.compute_energy(sig)
            result.append(sig_energy) 
        return torch.stack(result)
    
    def compute_energy(self, signal):
        '''
            signal is one of {AV, TV, MV, PV}
        '''
        signal = signal.reshape((1,-1))
        signal = torch.nn.functional.normalize(signal)

        energy = torch.abs(signal * signal).sum() / (2 * signal.shape[0])
        return energy
# #instance
# from torch.utils.data import DataLoader
# path_demo = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
# path_sig = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data"
# path_sal = "../output"
# saliency_dataset = Saliency_loader(path_demo, path_sal)
# saliency_loader = DataLoader(saliency_dataset)
# data = next(iter(saliency_loader))
# print(data['saliency']['AV'].shape)