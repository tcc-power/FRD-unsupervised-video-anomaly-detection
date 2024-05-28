import torch
import numpy as np
import os
from sklearn.metrics import auc, roc_curve,confusion_matrix,precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
def scorebinary(scores=None,threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold<threshold] = 0
    scores_threshold[scores_threshold>=threshold] = 1
    return scores_threshold

def calculate_eer(y,y_score):
    fpr,tpr,thresholds = roc_curve(y,y_score,pos_label=1)
    eer = brentq(lambda  x:1. - x - interp1d(fpr,tpr)(x),0.,1.)
    return eer

def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path):
    test_loader = all_test_loader
    itr = 0
    if os.path.exists(os.path.join('./result', save_path)) == 0:
        os.makedirs(os.path.join('./result', save_path))
    with open(file=os.path.join('ckpt', save_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            print(key,value)
            f.write('%s:%s\n' % (key, value))
    print('model is trained from scratch')
    snapshot=args.snapshot
    weights = args.Lambda.split('_')
    best_AUC = [0, 0, 0, 0]
    for epoch in range(epochs):
        for itr, [features,idx,pseudo_label] in enumerate(train_loader):
            features = features.to(device)
            #for C2FPL as auxiliary scorer
            Lossr, Loss_LAB, Losss = model(features, args.lab, args.lab_type, is_training=True)
            #for self as auxiliary scorer
            #Lossr, Loss_LAB, Losss = model(features, args.lab,args.lab_type,pseudo_label, is_training=True)
            total_loss = float(weights[0]) * Lossr + float(weights[1])*Loss_LAB+float(weights[2]) * Losss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if itr%snapshot == 0:
                print('epoch_{}, itr_{}, Loss: {}, lossr: {}, losss: {}'.
                    format(epoch, itr, total_loss.data.cpu().detach().numpy(), Lossr.data.cpu().detach().numpy(),
            Losss.data.cpu().detach().numpy()))
                if itr != 0:
                    auc = test(test_loader, model, args, device)
                else:
                    auc = best_AUC
            if auc[0] > best_AUC[0]:
                best_AUC = auc
                best_epoch = epoch
                with open(file=os.path.join('ckpt', save_path, 'result.txt'), mode='a') as f:
                    f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, best_AUC[0]))
                    f.write('itration_{}_EER is {}\n'.format(itr, best_AUC[4]))
                    f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, best_AUC[1]))
                    f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, best_AUC[2]))
                    f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, best_AUC[3]))
                torch.save(model.state_dict(), os.path.join('./ckpt/', save_path, 'best_ckpt_{:.4f}'.format(best_AUC[0]) + '.pkl'))

    print('epoch:',best_epoch,'best auc_all:   {:.4f}'.format(best_AUC[0]),'  auc_abn:   {:.4f}'.format(best_AUC[1]),'  far_all:   {:.4f}'.format(best_AUC[2]),'  far_abn:   {:.4f}'.format(best_AUC[3]),'  eer:   {:.4f}'.format(best_AUC[4]))


def test_FRD(test_loader, model, device):
    result = {}
    for i, data in enumerate(test_loader):
        feature, data_video_name = data
        with torch.no_grad():
            # if data_video_name[0] == '01_0135':
            #     num = model(feature, False, is_training=True,is_test=True).cpu().detach().numpy()
            #     np.save('atten_map/mstr01_0135.npy',num)
            element_logits = model(feature.to(device), None,None, is_training=False)
        element_logits = element_logits.cpu().data.numpy().reshape(-1)
        result[data_video_name[0][:-8]] = element_logits
        #print(model.scorer(model.lab,is_training=False))
    return result


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
            logits = model(input.to(device), None,None, is_training=False)
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
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)

        # np.save('UCF_pred/'+'{}-pred_UCFV1_i3d.npy'.format(epoch), pred)
        return [rec_auc_all,rec_auc_abn,all_far,abn_far,eer]
