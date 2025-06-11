# backend/deepfake/modelNet.py

import torch.nn as nn
from torchvision import models
import torch

class Model(nn.Module):
    def __init__(self, num_classes,model_name="resnext50_32x4d", lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        self.model_name = model_name 

        model = models.resnext50_32x4d(pretrained = True) #Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.latent_dim = 2048
        self.lstm = nn.LSTM(self.latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim,num_classes) # hidden_dim 변수로 넣어줌
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,self.latent_dim) # resnext50_32x4d, xception : 2048, efficientnet-b0 : 1280
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))
