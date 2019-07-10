import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
import torch

class resnet50(nn.Module):
    def __init__(self, pretrained=True ,modelpath=None ):
        super().__init__()
        self.pretrained = pretrained
        self.modelpath = modelpath

    def create_model(self,output_class=1000, input_channel=6,freezelonlylastlayer=False):
        if self.pretrained:
            self.model  = models.resnet50(pretrained=True)
        else:
            self.model  = models.resnet50(pretrained=False)
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint)
            del checkpoint
        if freezelonlylastlayer:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.conv1 = nn.Conv2d(input_channel, 64, 7, 2, 3)
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=2048, out_features=2048, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=2048, out_features=output_class, bias=True) ,)
        else:
            self.model.conv1 = nn.Conv2d(input_channel, 64, 7, 2, 3)
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=2048, out_features=2048, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=2048, out_features=output_class, bias=True) ,)

    def forward(self, x):
        out = self.model(x)
        # out = F.relu(out, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.model.fc(out)
        return out


class densenet201(nn.Module):
    def __init__(self, pretrained=True ,modelpath=None ):
        super().__init__()
        self.pretrained = pretrained
        self.modelpath = modelpath
        # del checkpoint

    def create_model(self, output_class=1000, input_channel=6, freezelonlylastlayer=False,last_layer=None):
        if self.pretrained:
            self.model = models.densenet201(pretrained=True)
        else:
            self.model = models.densenet201(pretrained=False)
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint)
            del checkpoint
        if freezelonlylastlayer:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.features.conv0 = nn.Conv2d(input_channel, 64, 7, 2, 3)
            if last_layer is None:
                self.model.classifier = nn.Sequential(
                    nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Dropout(p=0.25),
                    nn.Linear(in_features=1920, out_features=1920, bias=True),
                    nn.ReLU(),
                    nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=1920, out_features=output_class, bias=True), )
            else:
                self.model.classifier = last_layer
        else:
            self.model.features.conv0 = nn.Conv2d(input_channel, 64, 7, 2, 3)
            if last_layer is None:
                self.model.classifier = nn.Sequential(
                    nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Dropout(p=0.25),
                    nn.Linear(in_features=1920, out_features=1920, bias=True),
                    nn.ReLU(),
                    nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=1920, out_features=output_class, bias=True), )
            else:
                self.model.classifier = last_layer
    def forward(self, x):
        out = self.model(x)
        # out = F.relu(out, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.model.fc(out)
        return out

class densenet161(nn.Module):
    def __init__(self, pretrained=True ,modelpath=None ):
        super().__init__()
        self.pretrained = pretrained
        self.modelpath = modelpath
        # del checkpoint

    def create_model(self, output_class=1000, input_channel=6, freezelonlylastlayer=False, last_layer=None):
        if self.pretrained:
            self.model  = models.densenet161(pretrained=True)
        else:
            self.model  = models.densenet161(pretrained=False)
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint)
            del checkpoint
        if freezelonlylastlayer:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.features.conv0 = nn.Conv2d(input_channel, 96, 7, 2,3)
            if last_layer is None:
                self.model.classifier = nn.Sequential(
                              nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2208, out_features=2208, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2208, out_features=output_class, bias=True),
                             )
            else:
                self.model.classifier = last_layer
        else:
            self.model.features.conv0 = nn.Conv2d(input_channel, 96, 7, 2,3)
            if last_layer is None:
                self.model.classifier = nn.Sequential(
                              nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2208, out_features=2208, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2208, out_features=output_class, bias=True),
                             )
            else:
                self.model.classifier = last_layer

    def forward(self, x):
        out = self.model(x)
        # out = F.relu(out, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.model.fc(out)
        return out

class inceptionv3(nn.Module):
    def __init__(self, pretrained=True ,modelpath=None ):
        super().__init__()
        self.pretrained = pretrained
        self.modelpath = modelpath
        # del checkpoint

    def create_model(self, output_class=1000, input_channel=6, freezelonlylastlayer=False,last_layer=None):
        if self.pretrained:
            self.model  =models.inception_v3(pretrained=True)
        else:
            self.model  = models.inception_v3(pretrained=False)
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint)
            del checkpoint
        if freezelonlylastlayer:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.Conv2d_1a_3x3.conv = nn.Conv2d(input_channel, 32, 3, 2)
            if last_layer is None:
                self.model.fc = nn.Sequential(
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2048, out_features=2048, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2048, out_features=output_class, bias=True),
                             )
            else:
                self.model.fc = last_layer
        else:
            self.model.Conv2d_1a_3x3.conv = nn.Conv2d(input_channel, 32, 3, 2)
            if last_layer is None:
                self.model.fc = nn.Sequential(
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2048, out_features=2048, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2048, out_features=output_class, bias=True),
                             )
            else:
                self.model.fc = last_layer

    def forward(self, x):
        out = self.model(x)
        # out = F.relu(out, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.model.fc(out)
        return out

class resnet152(nn.Module):
    def __init__(self, pretrained=True ,modelpath=None ):
        super().__init__()
        self.pretrained = pretrained
        self.modelpath = modelpath
        # del checkpoint

    def create_model(self, output_class=1000, input_channel=6, freezelonlylastlayer=False,last_layer=None):
        if self.pretrained:
            self.model = models.resnet152(pretrained=True)
        else:
            self.model = models.resnet152(pretrained=False)
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint)
            del checkpoint
        if freezelonlylastlayer:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.conv1 = nn.Conv2d(input_channel, 64, 7, 2,3)
            if last_layer is None:
                self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2048, out_features=2048, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2048, out_features=output_class, bias=True),
                    )
            else:
                self.model.fc = last_layer
        else:
            self.model.conv1 = nn.Conv2d(input_channel, 64, 7, 2,3)
            if last_layer is None:
                self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2048, out_features=2048, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2048, out_features=output_class, bias=True),
                    )
            else:
                self.model.fc = last_layer

    def forward(self, x):
        out = self.model(x)
        # out = F.relu(out, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.model.fc(out)
        return out