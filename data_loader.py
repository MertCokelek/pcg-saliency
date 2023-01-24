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
from sklearn import linear_model
import torch
import torch.nn.functional as F

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
    def __init__(self, path_demo, path_sal, encode=True, energy_loader=False, return_as=None):
        # self.subject_id = subject_id
        self.energy_loader=energy_loader # for backward compatibility, I'm leaving this here. For forward compatibility, use return_as='ransac_mse'
        self.return_as = return_as # 'ransac_mse' or None
        
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
        elif self.energy_loader and self.return_as is None:
            saliencies = [subject_AV['saliency'], subject_TV['saliency'], subject_MV['saliency'], subject_PV['saliency']]
            saliencies = [ torch.Tensor(saliency) for saliency in saliencies]
            energies = self.compute_energies(saliencies)
            return energies, ret['murmur']
        elif self.return_as == 'ransac_mse':
            saliencies = [subject_AV['saliency'], subject_TV['saliency'], subject_MV['saliency'], subject_PV['saliency']]
            ransac_mses = self.compute_ransac_mses(saliencies)
            return ransac_mses, ret['murmur']

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

    def compute_ransac_mses(self, saliences_4):
        av, tv, mv, pv = saliences_4
        av_y, tv_y, mv_y, pv_y = av.flatten(), tv.flatten(), mv.flatten(), pv.flatten()
        av_x, tv_x, mv_x, pv_x = av_y[:, np.newaxis], tv_y[:, np.newaxis], mv_y[:, np.newaxis], pv_y[:, np.newaxis]
        
        ransacs = [linear_model.RANSACRegressor(), linear_model.RANSACRegressor(), linear_model.RANSACRegressor(), linear_model.RANSACRegressor()]
        
        ransacs[0].fit(av_x, av_y)
        ransacs[1].fit(tv_x, tv_y)
        ransacs[2].fit(mv_x, mv_y)
        ransacs[3].fit(pv_x, pv_y)

        av_fit, tv_fit, mv_fit, pv_fit = ransacs[0].predict(av_x), ransacs[1].predict(tv_x), ransacs[2].predict(mv_x), ransacs[3].predict(pv_x)

        mse_av = F.mse_loss(torch.Tensor(av_x).flatten(), torch.Tensor(av_fit).flatten()).flatten()
        mse_tv = F.mse_loss(torch.Tensor(tv_x).flatten(), torch.Tensor(tv_fit).flatten()).flatten()
        mse_mv = F.mse_loss(torch.Tensor(mv_x).flatten(), torch.Tensor(mv_fit).flatten()).flatten()
        mse_pv = F.mse_loss(torch.Tensor(pv_x).flatten(), torch.Tensor(pv_fit).flatten()).flatten()

        return torch.cat([mse_av, mse_tv, mse_mv, mse_pv])


# #instance
# from torch.utils.data import DataLoader
# path_demo = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
# path_sig = "../data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data"
# path_sal = "../output"
# saliency_dataset = Saliency_loader(path_demo, path_sal)
# saliency_loader = DataLoader(saliency_dataset)
# data = next(iter(saliency_loader))
# print(data['saliency']['AV'].shape)