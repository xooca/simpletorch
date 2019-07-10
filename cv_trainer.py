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
