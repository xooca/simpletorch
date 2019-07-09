import cv_models as models
import data_reader as dr
import torch.nn as nn
import torch


class modeldefinition:
    def __init__(self,pretrained=True):
        self.pretrained = pretrained
        #self.modelpath = modelpath
        #self.channel = input_channel
        #self.modelnumber = modelnumber
        #self.models = models(pretrained=modelpath ,modelpath=self.modelpath )

    def definedensenet201(self,output_class=1000, input_channel=6, freezelonlylastlayer=False,modelpath=None):
        model = models.densenet201(pretrained=self.pretrained,modelpath=modelpath)
        model.create_model(output_class=output_class, input_channel=input_channel, freezelonlylastlayer=freezelonlylastlayer)
        return model

    def train_valid_loader(self,df,batch_size=32,valid_size=0.3,channel=1,train_test_transforms=None,split_batch_th=0):
        splitter = dr.cv_data_splitters(df, batch_size=batch_size, valid_size=valid_size,split_batch_th=split_batch_th)
        training_loader, validation_loader = splitter.cellular_load_split_train_test(channel, train_test_transforms)
        return training_loader, validation_loader

    def definemodel(self,df,modelname ,input_channel=3,output_class,image_size,valsize,freezelonlylastlayer = False ,batch_size=32, valid_size=0.3, channel=1, train_test_transforms=None,split_batch_th=0,lr=0.0001):
        if modelname == 'densenet201':
            model = self.definedensenet201( output_class=output_class, input_channel=input_channel, freezelonlylastlayer=freezelonlylastlayer, modelpath=None)
            training_loader, validation_loader = self.train_valid_loader(self, df, batch_size=32, valid_size=0.3, channel=1, train_test_transforms=None,split_batch_th=0)
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            criterion = nn.CrossEntropyLoss()
            modelname = 'densenet201'
        elif modelnumber == 1:
            batch_size = 64
            image_siz e =224
            training_loader, validation_loader = load_split_train_test(data_dir ,batch_size ,valsize ,image_size)
            model = models.resnet50(pretrained=True)
            # for param in model2.parameters():
            #    param.requires_grad = False

            # criterion = nn.CrossEntropyLoss()

            if freezelonlylastlayer == 'yes':
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Sequential(nn.Linear(2048, 720),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(720, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr)
            else:
                model.fc = nn.Sequential(nn.Linear(2048, 720),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(720, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            modelname = 'resnet50'
            criterion = nn.CrossEntropyLoss()
        elif modelnumber == 2:
            batch_size = 32
            image_siz e =299
            training_loader, validation_loader = load_split_train_test(data_dir ,batch_size ,valsize ,image_size)
            model = models.inception_v3(pretrained=True)
            # for param in model5.parameters():
            #    param.requires_grad = False
            # criterion = nn.CrossEntropyLoss()
            if freezelonlylastlayer == 'yes':
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Sequential(nn.Linear(2048, 720),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(720, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr)
            else:
                model.fc = nn.Sequential(nn.Linear(2048, 720),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(720, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            modelname = 'inception'
            criterion = nn.CrossEntropyLoss()
        elif modelnumber == 3:
            batch_size = 32
            image_siz e =224
            training_loader, validation_loader = load_split_train_test(data_dir ,batch_size ,valsize ,image_size)
            model = models.densenet161(pretrained=True)
            # for param in model5.parameters():
            #    param.requires_grad = False
            # criterion = nn.CrossEntropyLoss()
            if freezelonlylastlayer == 'yes':
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier = nn.Sequential(nn.Linear(2208, 720),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(720, 256),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.4),
                                                 nn.Linear(256, 64),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.3),
                                                 nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr)
            else:
                model.classifier = nn.Sequential(nn.Linear(2208, 720),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(720, 256),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.4),
                                                 nn.Linear(256, 64),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.3),
                                                 nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            modelname = 'densenet161'
            criterion = nn.CrossEntropyLoss()
            # model.to(device)
            # model3.to(device)
        elif modelnumber == 4:
            batch_size = 64
            image_siz e =224
            training_loader, validation_loader = load_split_train_test(data_dir ,batch_size ,valsize ,image_size)
            model = models.resnet152(pretrained=True)
            # for param in model2.parameters():
            #    param.requires_grad = False

            # criterion = nn.CrossEntropyLoss()

            if freezelonlylastlayer == 'yes':
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Sequential(nn.Linear(2048, 720),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(720, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr)
            else:
                model.fc = nn.Sequential(nn.Linear(2048, 720),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(720, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(64, 5))
                optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            modelname = 'resnet152'
            criterion = nn.CrossEntropyLoss()
        return model ,criterion ,optimizer ,modelname ,training_loader, validation_loader ,image_size