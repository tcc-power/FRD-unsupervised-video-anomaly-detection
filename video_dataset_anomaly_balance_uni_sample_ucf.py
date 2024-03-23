import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import random
#import cupy

class train_loader(data.Dataset):
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        self.dataset_path = './dataset'
        self.cross_clip = args.cross_clip
        self.train = train
        self.max_seqlen = args.max_seqlen
        self.pseudo_label = torch.tensor(np.load('pseudo_labels.npy')).to('cuda').unsqueeze(-1)
        self.train_features = np.load(os.path.join(self.dataset_path, 'ucfcrime','concat_UCF.npy'))
        self.len = len(self.train_features)
        print(self.train_features.shape)


    def __getitem__(self,index):
        data = np.expand_dims(self.train_features[index],1)
        pseudo_label = self.pseudo_label[index]
        for k in range(1, self.cross_clip):
            if index+k >= self.len:
                data = np.concatenate((data, np.expand_dims(self.train_features[index], 1)), axis=1)
                pseudo_label = torch.cat([pseudo_label,self.pseudo_label[index]],dim=0)
            else:
                data = np.concatenate((data, np.expand_dims(self.train_features[index+k],1)), axis=1)
                pseudo_label = torch.cat([pseudo_label, self.pseudo_label[index+1]],dim=0)
        return data,index,pseudo_label.unsqueeze(-1)

    def __len__(self):
        return len(self.train_features)

class dataset(Dataset):
    def __init__(self, args, train=False, trainlist=None, testlist=None):
        """
        :param args:
        self.dataset_path: path to dir contains anomaly datasets
        self.dataset_name: name of dataset which use now
        self.feature_modal: features from different input, contain rgb, flow or combine of above type
        self.feature_pretrain_model: the model name of feature extraction
        self.feature_path: the dir contain all features, use for training and testing
        self.videoname: videonames of dataset
        self.trainlist: videonames of dataset for training
        self.testlist: videonames of dataset for testing
        self.train: boolen type, if it is True, the dataset class return training data
        self.t_max: the max of sampling in training
        """
        args.dataset_path = './dataset'
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             args.feature_pretrain_model)
        with open(os.path.join('dataset', self.dataset_name, 'train_split.txt'), 'r') as file:
            self.videoname = [i[:-1] for i in file.readlines()]
        #tcc
        self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split.txt'))
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label.pickle'))
        self.train = train
        self.pretrain_result_dict = None
        self.video_labels = None


    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train

    def __getitem__(self, index):

        data_video_name = self.testlist[index].replace('\n', '_i3d.npy')
        self.feature = np.load(file=os.path.join(self.feature_path, data_video_name))
        return self.feature, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)

        else:
            return len(self.testlist)


class dataset_train2test(Dataset):
    def __init__(self, args, trainlist=None):
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)

        if trainlist:
            self.trainlist = self.txt2list(trainlist)
        else:
            self.trainlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_occ.txt'))

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train


    def __getitem__(self, index):
            data_video_name = self.trainlist[index].replace('\n', '').replace('Ped', 'ped')
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
            return self.feature, data_video_name

    def __len__(self):
        return len(self.trainlist)


class Dataset_Con_all_feedback_XD(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        # self.modality = args.modality
        self.is_normal = is_normal

        if test_mode:

            self.con_all = Concat_list_all_crop_feedback(True)
        else:

            self.con_all = np.load("concatenated/concat_UCF.npy")
            # self.con_all = np.load("concatenated/concat_XD.npy")

            print('self.con_all shape:', self.con_all.shape)

        self.tranform = transform
        self.test_mode = test_mode

    def __getitem__(self, index):

        if self.test_mode:
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)

        else:

            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)

        if self.test_mode:

            return features
        else:

            return features, index

    def __len__(self):
        return len(self.con_all)

import torch

def Concat_list_all_crop_feedback(Test=False, create='False'): #UCF
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    if Test is True:
        con_test = np.load("Concat_test_10.npy")
        # con_test = np.load("/l/users/anas.al-lahham/concat_test_XD_5crop.npy")
        print('Testset size:', con_test.shape)
        # con_test
        return con_test

