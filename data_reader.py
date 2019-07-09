from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn

class cellular_dataset(Dataset):
    def __init__(self, df, channel, mode='train', transforms=None):
        self.df = df
        self.channel = channel
        self.mode = mode
        self.transforms = transforms
        self.df = self.df[self.df['channel'] == self.channel].reset_index(drop=True)
        print(f"Number of data loaded is {self.df.shape}")
        print(f"Channel is {self.channel}")
        print(f"Mode is {self.mode}")
        self.data_len = len(self.df.index)
        self.filepath = np.asarray(self.df['filepath'])
        if self.mode == 'train':
            self.sirna = np.asarray(self.df['sirna'])
        else:
            self.id_code = np.asarray(self.df['id_code'])

    def __getitem__(self, index):
        img = cv2.imread(self.filepath[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_ = cv2.bitwise_not(img_)

        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == 'train':
            return img, self.sirna[index]
        else:
            return img, self.id_code[index]

    def __len__(self):
        return self.data_len

class cv_dataset(Dataset):
    def __init__(self, df, labelcol,imagepathcolumn, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        print(f"Number of data loaded is {self.df.shape}")
        self.filepath = np.asarray(self.df[imagepathcolumn])
        self.data_len = len(self.df.index)
        self.labelcol = labelcol
        self._labelcol(self)

    def _labelcol(self):
        self.returnval = np.asarray(self.df[self.labelcol])

    def __getitem__(self, index):
        img = cv2.imread(self.filepath[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.returnval[index]

    def __len__(self):
        return self.data_len

class cv_data_splitters:
    def __init__(self,df,batch_size=32,valid_size=.3,split_batch_th=0):
        self.df = df
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.split_batch_th = split_batch_th

    def cellular_load_split_train_test(self,channel, transforms=None):
        all_data = cellular_dataset(self.df, channel=channel, mode='train', transforms=transforms)
        num_train = len(all_data)
        if self.split_batch_th > 0:
            num_train = self.split_batch_th
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        trainloader = torch.utils.data.DataLoader(all_data, sampler=train_sampler, batch_size=self.batch_size, num_workers=2)
        testloader = torch.utils.data.DataLoader(all_data, sampler=test_sampler, batch_size=self.batch_size, num_workers=2)
        return trainloader, testloader