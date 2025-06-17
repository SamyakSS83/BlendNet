from torch.utils.data import Dataset
import numpy as np
import torch

class PocketDataset(Dataset):
    def __init__(self, PID, Pseqs, Pfeatures, maxL = 1500, inputD = 1024, Labels = None):
        self.PID = PID
        self.Pseqs = Pseqs
        self.Pfeatures = Pfeatures
        self.maxL = maxL
        self.inputD = inputD
        self.Labels = Labels

    def __len__(self):
        return len(self.PID)
        
    def __getitem__(self, idx):
        
        pid, pseq = self.PID[idx], self.Pseqs[idx]
        pfeat = self.Pfeatures[pid]
        
        ### process data
        seqlength = len(pseq)
        input_mask = [1] * seqlength
        
        prot_feat = np.zeros((self.maxL, self.inputD))
        protein_feat = np.zeros((self.maxL, self.inputD))
        
        prot_feat[:seqlength, :] = pfeat
        input_mask += [0] * (self.maxL - seqlength)

        protein_feat[:seqlength, :] = np.sum(pfeat)
        position_ids = [i for i in range(self.maxL)]
        
        # Convert to tensors and move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prot_feat = torch.tensor(prot_feat, dtype = torch.float32).to(device)
        protein_feat = torch.tensor(protein_feat, dtype = torch.float32).to(device)
        input_mask = torch.tensor(input_mask, dtype = torch.long).to(device)
        position_ids = torch.tensor(position_ids, dtype = torch.long).to(device)

        bs = self.Labels[idx]

        bs = sorted(list(map(int, bs.split(","))))
        
        #targets = np.array(bs) - 1
        targets = np.array(bs)
        one_hot_targets = np.eye(self.maxL)[targets]
        one_hot_targets = np.sum(one_hot_targets, axis = 0) 
        
        one_hot_targets = torch.tensor(one_hot_targets, dtype = torch.float32).to(device)
        
        return prot_feat, protein_feat, input_mask, position_ids, one_hot_targets
        
class PocketTestDataset(Dataset):
    def __init__(self, PID, Pseqs, Pfeatures, maxL = 1500, inputD = 1024):
        self.PID = PID
        self.Pseqs = Pseqs
        self.Pfeatures = Pfeatures
        self.maxL = maxL
        self.inputD = inputD

    def __len__(self):
        return len(self.PID)
        
    def __getitem__(self, idx):
        
        pid, pseq = self.PID[idx], self.Pseqs[idx]
        pfeat = self.Pfeatures[pid]
        
        ### process data
        seqlength = len(pseq)
        input_mask = [1] * seqlength
        
        prot_feat = np.zeros((self.maxL, self.inputD))
        protein_feat = np.zeros((self.maxL, self.inputD))
        
        prot_feat[:seqlength, :] = pfeat
        input_mask += [0] * (self.maxL - seqlength)

        protein_feat[:seqlength, :] = np.sum(pfeat)
        position_ids = [i for i in range(self.maxL)]
        
        # Convert to tensors and move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prot_feat = torch.tensor(prot_feat, dtype = torch.float32).to(device)
        protein_feat = torch.tensor(protein_feat, dtype = torch.float32).to(device)
        input_mask = torch.tensor(input_mask, dtype = torch.long).to(device)
        position_ids = torch.tensor(position_ids, dtype = torch.long).to(device)

        return prot_feat, protein_feat, input_mask, position_ids