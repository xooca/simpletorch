import cv_models as models
import data_reader as dr
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os

class modeldefinition:
    def __init__(self,pretrained=True):
        self.pretrained = pretrained
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.modelpath = modelpath
        #self.channel = input_channel
        #self.modelnumber = modelnumber
        #self.models = models(pretrained=modelpath ,modelpath=self.modelpath )

    def definedensenet201(self,output_class=1000, input_channel=6, freezelonlylastlayer=False,modelpath=None,
                          last_layer=None):
        model = models.densenet201(pretrained=self.pretrained,modelpath=modelpath)
        model.create_model(output_class=output_class, input_channel=input_channel,
                           freezelonlylastlayer=freezelonlylastlayer,last_layer=last_layer)
        return model

    def definedensenet161(self,output_class=1000, input_channel=6, freezelonlylastlayer=False,modelpath=None,
                          last_layer=None):
        model = models.densenet161(pretrained=self.pretrained,modelpath=modelpath)
        model.create_model(output_class=output_class, input_channel=input_channel,
                           freezelonlylastlayer=freezelonlylastlayer,last_layer=last_layer)
        return model

    def defineresnet50(self,output_class=1000, input_channel=6, freezelonlylastlayer=False,modelpath=None,
                       last_layer=None):
        model = models.resnet50(pretrained=self.pretrained,modelpath=modelpath)
        model.create_model(output_class=output_class, input_channel=input_channel,
                           freezelonlylastlayer=freezelonlylastlayer,last_layer=last_layer)
        return model

    def defineresnet152(self,output_class=1000, input_channel=6, freezelonlylastlayer=False, modelpath=None,
                        last_layer=None):
        model = models.resnet152(pretrained=self.pretrained,modelpath=modelpath)
        model.create_model(output_class=output_class, input_channel=input_channel,
                           freezelonlylastlayer=freezelonlylastlayer,last_layer=last_layer)
        return model

    def defineinceptionv3(self,output_class=1000, input_channel=6, freezelonlylastlayer=False, modelpath=None,
                          last_layer=None):
        model = models.inceptionv3(pretrained=self.pretrained,modelpath=modelpath)
        model.create_model(output_class=output_class, input_channel=input_channel,
                           freezelonlylastlayer=freezelonlylastlayer,last_layer=last_layer)
        return model

    def train_valid_loader(self,df,batch_size=32,valid_size=0.3,channel=1,train_test_transforms=None,split_batch_th=0):
        splitter = dr.cv_data_splitters(df, batch_size=batch_size, valid_size=valid_size,split_batch_th=split_batch_th)
        training_loader, validation_loader = splitter.cellular_load_split_train_test(channel, train_test_transforms)
        return training_loader, validation_loader

    def definemodel(self, df, modelname, input_channel=3, output_class=1000, batch_size=32, valid_size=0.3, channel=1,
                    train_test_transforms=None, split_batch_th=0, freezelonlylastlayer=False, lr=0.0001, optimizer=None,
                    criterion=None):

        if modelname == 'densenet201':
            model = self.definedensenet201( output_class=output_class, input_channel=input_channel,
                                            freezelonlylastlayer=freezelonlylastlayer,modelpath=None)

        elif modelname == 'densenet161':
            model = self.definedensenet161( output_class=output_class, input_channel=input_channel,
                                            freezelonlylastlayer=freezelonlylastlayer,modelpath=None)

        elif modelname == 'resnet50':
            model = self.defineresnet50( output_class=output_class, input_channel=input_channel,
                                         freezelonlylastlayer=freezelonlylastlayer,modelpath=None)

        elif modelname == 'resnet152':
            model = self.defineresnet152( output_class=output_class, input_channel=input_channel,
                                          freezelonlylastlayer=freezelonlylastlayer,modelpath=None)
        elif modelname == 'inception':
            model = self.defineinceptionv3( output_class=output_class, input_channel=input_channel,
                                          freezelonlylastlayer=freezelonlylastlayer,modelpath=None)
        else:
            print(f"-----MODEL {modelname} NOT FOUND------")

        training_loader, validation_loader = self.train_valid_loader(self, df, batch_size=batch_size,
                                                                     valid_size=valid_size, channel=channel,
                                                                     train_test_transforms=train_test_transforms,
                                                                     split_batch_th=split_batch_th)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        return model, criterion, optimizer, modelname, training_loader, validation_loader

    def trainer(self,model, criterion, optimizer, training_loader, validation_loader, epochs=10, modeltype='other'):

        running_loss_history = []
        running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []
        # batch = 0

        for e in range(epochs):
            batch = 0
            running_loss = 0.0
            running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0

            for inputs, labels in training_loader:
                # print(inputs.shape)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch = batch + len(inputs)
                # bs, ncrops, c, h, w = inputs.size()
                # outputs = model(input.view(-1, c, h, w))
                # outputs = model(inputs)

                optimizer.zero_grad()
                if modeltype == 'inception':
                    outputs = model.forward(inputs)[0]
                else:
                    outputs = model.forward(inputs)
                # optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # outputs = torch.exp(outputs)
                _, preds = torch.max(outputs, 1)
                # preds = preds + 1
                # print(preds)
                # print( labels.data)
                # print('-------------------')
                running_loss += loss.item()
                # print(running_loss,loss.item())
                running_corrects += torch.sum(preds == labels.data)
                # print(f"Epoch {e} has accuracy of ")
                # print(torch.sum(preds == labels.data),len(inputs),int(torch.sum(preds == labels.data))/len(inputs))

            else:
                valbatch = 0
                with torch.no_grad():
                    for val_inputs, val_labels in validation_loader:
                        val_inputs = val_inputs.to(self.device)
                        val_labels = val_labels.to(self.device)
                        valbatch = valbatch + len(val_inputs)
                        if modeltype == 'inception':
                            val_outputs = model(val_inputs)[0]
                        else:
                            val_outputs = model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)
                        # val_outputs = torch.exp(val_outputs)
                        _, val_preds = torch.max(val_outputs, 1)
                        # val_preds = val_preds + 1
                        val_running_loss += val_loss.item()
                        # print(val_loss.item(),val_running_loss)
                        val_running_corrects += torch.sum(val_preds == val_labels.data)
            # print(epoch_loss)
            # print(running_corrects.float())
            # print('-----------')
            epoch_loss = running_loss / len(training_loader)
            epoch_acc = running_corrects.float() / batch
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)
            val_epoch_loss = val_running_loss / len(validation_loader)
            val_epoch_acc = val_running_corrects.float() / valbatch
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
            print('epoch :', (e + 1))
            print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
            print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
        return model

    def estimator(self,modellist,):
        def addtensorcols(val):
            return val['diagnosis1'] + val['diagnosis2']

        def extractfilename(val):
            return os.path.split(val)[1]

        def maxtensorval(val):
            # ps = torch.exp(val)
            # ps = F.softmax(val,dim=1)
            top_p, top_class = val.topk(1)
            return top_class

        newtestfinal = pd.DataFrame(columns=['id_code', 'diagnosis1'])
        for modelnum in [0, 4]:
            # counter = 0
            print('---------------------------------------------------------------------------------')
            print('Defining model and creating data loaders for model number {0}'.format(modelnum))
            model, criterion, optimizer, modelname, training_loader, validation_loader, image_size = definemodel(
                modelnum,
                pretrained=False,
                freezelonlylastlayer='no',
                lr=0.0001)
            # test_transforms = transforms.Compose([transforms.Resize(256),
            #                                    transforms.CenterCrop(image_size),
            #                                  #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            #                                transforms.ToTensor(),
            #                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            #                              ])
            test_transforms = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(image_size),
                                                  # transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
            print("Training of model {0} started".format(modelname))
            model.to(device)
            model_1 = modeltrainv2(model, criterion, optimizer, training_loader, validation_loader, epochs=11,
                                   modeltype=modelname)
