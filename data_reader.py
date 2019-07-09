from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
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