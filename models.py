import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import numpy as np
from torch.nn.modules.module import Module

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class model(torch.nn.Module):
    def __init__(self,max_seqlen,feature_size,Vitblock_num,cross_clip,split,beta,delta):
        super(model, self).__init__()
        self.batchsize = max_seqlen
        self.Vitblock_num = Vitblock_num
        self.cross_clip = cross_clip
        self.feature_size = feature_size
        self.scorer = Scorer(n_feature=feature_size)
        #self.scorer_c2fpl = Scorer_C2FPL(n_features=feature_size)
        self.apply(weights_init)
        self.numpylist = None
        self.lab = torch.zeros(0).to('cuda')
        self.mb_size = int(self.batchsize*beta)
        self.delta = delta
        self.iter = 0

    def __del__(self):
        for i in range(10):
            torch.cuda.empty_cache()
        print("model deleted")

    def forward(self, inputs, lab,lab_type, pseudo_label,is_training=True, is_test=False):
        if is_training==False:
            scores = self.scorer(inputs, is_training=False)
            return scores

class Scorer_C2FPL(nn.Module):  # multiplication then Addition
    def __init__(self, n_features):
        super(Scorer_C2FPL, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)

        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim=1))

        self.fc2 = nn.Linear(512, 32)

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))

        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        x = x.mean(dim=1)
        return x

class Scorer(torch.nn.Module):
    def __init__(self, n_feature):
        super(Scorer, self).__init__()
        self.fc1 = nn.Linear(n_feature, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.classifier = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def __del__(self):
        print("model deleted")

    def forward(self, inputs, is_training=True):
        if is_training:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.dropout(x)
            x = self.relu(self.fc2(x))  # 2048
            x = self.dropout(x)
            x = self.classifier(x)
            score = self.sigmoid(x)
            return torch.mean(score,dim=1)
        else:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.relu(self.fc2(x))  # 2048
            x = self.classifier(x)
            score = self.sigmoid(x)
            return torch.mean(score,dim=1)
