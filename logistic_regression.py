import json
import torch
import numpy as np 
import torch.nn as nn
import lightning as L

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
0: SUPPORTS
1: NOT_ENOUGH_INFO
2: REFUTES
3: DISPUTED
'''

class ClimateClaimDataset(Dataset):
    def __init__(self, data_path, sequence_len=50):
        with open(path, 'r') as file:
            self.data =  [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = data[idx]['claim']
        y = data[idx]['claim_label']

        if y == 'SUPPORTS':
            y = 0
        elif y == 'NOT_ENOUGH_INFO':
            y = 1
        elif y == 'REFUTES':
            y = 2
        elif y == 'DISPUTED':
            y = 3
        else:
            raise Exception(f'{y} is not a defined class...')
        return x, y
    
class ClimateClaimModule(L.LightningDataModule):
    def __init__(self, train_path, dev_path, mb=4, pin_memory=False, num_workers=7):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.mb = mb
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainDataset = ClimateClaimDataset(train_path)
            self.devDataset = ClimateClaimDataset(dev_path)
    
    def train_dataloader(self):
        return DataLoader(self.trainDataset, shuffle=True, batch_size=self.mb, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.devDataset, shuffle=True, batch_size=self.mb, pin_memory=self.pin_memory, num_workers=self.num_workers)

class embeddings(nn.Module):

class logistic_regression(nn.Module):
    def __init__(self, embedding_dim):

