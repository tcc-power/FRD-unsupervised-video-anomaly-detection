import pickle
from models import model
from video_dataset_anomaly_balance_uni_sample_ucf import Dataset_Con_all_feedback_XD,train_loader
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import os
from sklearn.metrics import auc, roc_curve, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='MSTR_Net')

# settings for dataset
parser.add_argument('--dataset_name', type=str, default='ucfcrime', help='')
parser.add_argument('--dataset_path', type=str, default='',
                        help='path to dir contains anomaly datasets')
parser.add_argument('--ckpt_path', type=str, default='best_ckpt/best_ckpt_0.7756.pkl',
                        help='path to best ckpt')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')

parser.add_argument('--feature_modal', type=str, default='rgb',
                        help='features from different input, options contain rgb, flow , combine')
# settings for model MSTR
parser.add_argument('--Vitblock_num', type=int, default=8, help='1-8')
parser.add_argument('--cross_clip', type=int, default=4, help='1,2,4')
parser.add_argument('--plot', type=int, default=0, help='0,1')

def scorebinary(scores=None,threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold<threshold] = 0
    scores_threshold[scores_threshold>=threshold] = 1
    return scores_threshold

def calculate_eer(y,y_score):
    fpr,tpr,thresholds = roc_curve(y,y_score,pos_label=1)
    eer = brentq(lambda  x:1. - x - interp1d(fpr,tpr)(x),0.,1.)
    return eer

def test(dataloader, model, args, device):

    with torch.no_grad():  # #
        model.eval()
        pred = torch.zeros(0, device=device)
        gt = np.load('./gt-ucf-RTFM.npy')
        gt_start = 0
        gt_end = 0
        normal_predict_np = np.zeros(0)
        normal_label_np = np.zeros(0)
        abnormal_predict_np = np.zeros(0)
        abnormal_label_np = np.zeros(0)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            # print(input.size())
            startime = time.time()
            logits = model(input.to(device), None,None,None, is_training=False)
            # print("done in {0}.".format(time.time() - startime))
            logits_new = logits.squeeze(-1).repeat(16).cpu().detach().numpy()
            gt_end += len(logits_new)
            gt_single = gt[gt_start:gt_end]
            if np.sum(gt_single) > 0:
                abnormal_predict_np = np.concatenate((abnormal_predict_np, logits_new))
                abnormal_label_np = np.concatenate((abnormal_label_np, gt_single))
            else:
                normal_predict_np = np.concatenate((normal_predict_np, logits_new))
                normal_label_np = np.concatenate((normal_label_np, gt_single))
            pred = torch.cat((pred, logits))
            gt_start += len(logits_new)

        # print(gt.shape)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        # gt = gt[:len(pred)]

        fpr_all, tpr_all, threshold = roc_curve(list(gt), pred)
        rec_auc_all = auc(fpr_all, tpr_all)
        binary_all_predict_np = scorebinary(pred, threshold=0.5)
        tn, fp, fn, tp = confusion_matrix(y_true=gt, y_pred=binary_all_predict_np).ravel()
        all_far = fp / (fp + tn)
        fpr_abn, tpr_abn, threshold = roc_curve(list(abnormal_label_np), abnormal_predict_np)
        binary_abn_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
        tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abn_predict_np).ravel()
        abn_far = fp / (fp + tn)
        eer = calculate_eer(list(gt), pred)
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        rec_auc_all = auc(fpr_all, tpr_all)
        rec_auc_abn = auc(fpr_abn, tpr_abn)
        print('auc_all: ' + str(rec_auc_all))
        print('auc_abn: ' + str(rec_auc_abn))
        print('far_all: ' + str(all_far))
        print('far_abn: ' + str(abn_far))
        print('eer: ' + str(eer))

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join('plot',args.dataset_name)
    #load ckpt
    test_loader = DataLoader(Dataset_Con_all_feedback_XD(args, test_mode=True),
                             batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=False, drop_last=False)
    model = model(0, feature_size=2048, Vitblock_num=args.Vitblock_num,
                  cross_clip=4, split=0, beta=0, delta=0)
    trained_weight = torch.load(args.ckpt_path)
    model.load_state_dict(trained_weight)
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_time = time.time()
        auc = test(test_loader, model, args, device)
        total_time = time.time()-total_time
    print('{} inf_time {}'.format(args.dataset_name,total_time))

