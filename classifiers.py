import torch
 
class Energy_Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(   torch.nn.Linear(4,10).float(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(10,10).float(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(10,3).float() )

    def forward(self, x):
        # x will be a list of 4 saliency signals
        x = self.compute_energies(x)
        logits = self.model(x)
        return logits

    
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



class Conv_1D_classifier(torch.nn.Module):
    def __init__(self, in_channels, n_depth, kernel_size,output_size, n_classes):
        super().__init__()
        self.conv1_sig1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=n_depth, kernel_size=kernel_size) # we can manipulate them and choose different hyperparameters for each one otherwise we can define only one conv1d layer
        self.conv1_sig2 = torch.nn.Conv1d(in_channels=in_channels, out_channels=n_depth, kernel_size=kernel_size)
        self.conv1_sig3 = torch.nn.Conv1d(in_channels=in_channels, out_channels=n_depth, kernel_size=kernel_size)
        self.conv1_sig4 = torch.nn.Conv1d(in_channels=in_channels, out_channels=n_depth, kernel_size=kernel_size) #
        self.relu = torch.nn.ReLU()
        self.AdaptiveMaxPool1d = torch.nn.AdaptiveMaxPool1d(output_size)
        self.fc = torch.nn.Linear(output_size*n_depth*4, n_classes)

    def forward(self, x): # x is a dict of the signals
        sig1 = self.conv1_sig1(x['saliencies']['TV'].squeeze(0).flatten().view(1,-1).float()) # b x n_depth x L
        sig1 = self.relu(sig1)
        sig1= self.AdaptiveMaxPool1d(sig1) #  b x n_depth x output_size
        sig1 = sig1.view(sig1.size(0), -1) # b x n_depth * output_size
        
        sig2 = self.conv1_sig2(x['saliencies']['AV'].squeeze(0).flatten().view(1,-1).float()) # b x n_depth x L
        sig2 = self.relu(sig2)
        sig2= self.AdaptiveMaxPool1d(sig2) #  b x n_depth x output_size
        sig2 = sig2.view(sig2.size(0), -1) # b x n_depth * output_size

        sig3 = self.conv1_sig3(x['saliencies']['MV'].squeeze(0).flatten().view(1,-1).float()) # b x n_depth x L
        sig3 = self.relu(sig3)
        sig3= self.AdaptiveMaxPool1d(sig3) #  b x n_depth x output_size
        sig3 = sig3.view(sig3.size(0), -1) # b x n_depth * output_size

        sig4 = self.conv1_sig4(x['saliencies']['PV'].squeeze(0).flatten().view(1,-1).float()) # b x n_depth x L
        sig4 = self.relu(sig4)
        sig4= self.AdaptiveMaxPool1d(sig4) #  b x n_depth x output_size
        sig4 = sig4.view(sig4.size(0), -1) # b x n_depth * output_size
        
        out = self.fc(torch.cat((sig1,sig2, sig3, sig4),1)) # b x 4* n_depth * output_size
        return out
