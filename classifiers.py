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