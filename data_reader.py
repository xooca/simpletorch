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
    def __init__(self, df, id_column, imagepathcolumn, targetcolumn,mode='train', transforms=None):
        self.df = df
        self.id_column = id_column
        self.targetcolumn = targetcolumn
        self.transforms = transforms
        self.mode = mode
        print(f"Number of data loaded is {self.df.shape}")
        print(f"Mode is {self.mode}")
        self.filepath = np.asarray(self.df[imagepathcolumn])
        self.data_len = len(self.df.index)

    def _create_cols(self):
        if self.mode == 'train':
            self.target = np.asarray(self.df[self.targetcolumn])
        else:
            self.id_column = np.asarray(self.df[self.id_column])

    def __getitem__(self, index):
        img = cv2.imread(self.filepath[index])
        self._create_cols()
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == 'train':
            return img, self.target[index]
        else:
            return img, self.id_column[index]

    def __len__(self):
        return self.data_len