import pickle

import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import zipfile
import io
import torch

def Update_mb(total_mb, new_mb, mb_size):
    total_mb = torch.cat([total_mb, new_mb],dim=0)
    if len(total_mb) > mb_size:
        total_mb = random_choice(total_mb, mb_size)
    return total_mb

def random_choice(gallery, num):
    """Random select some elements from the gallery.

    If `gallery` is a Tensor, the returned indices will be a Tensor;
    If `gallery` is a ndarray or list, the returned indices will be a
    ndarray.

    Args:
        gallery (Tensor | ndarray | list): indices pool.
        num (int): expected sample num.

    Returns:
        Tensor or ndarray: sampled indices.
    """
    assert len(gallery) >= num

    is_tensor = isinstance(gallery, torch.Tensor)
    perm = torch.randperm(gallery.shape[0], device=gallery.device)[:num]
    rand_inds = gallery[perm]
    return rand_inds

def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max], r

def random_extract_step(feat, t_max, step):
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r


def random_perturb(feat, length):
    samples = np.arange(length) * len(feat) / length
    for i in range(length):
        if i < length - 1:
            if int(samples[i]) != int(samples[i + 1]):
                samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
            else:
                samples[i] = int(samples[i])
        else:
            if int(samples[i]) < length - 1:
                samples[i] = np.random.choice(range(int(samples[i]), length))
            else:
                samples[i] = int(samples[i])
    # feat = feat[samples]
    return feat[samples.astype('int')], samples.astype('int')



def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length, step):
    if len(feat) > length:
        if step and step > 1:
            features, r = random_extract_step(feat, length, step)
            return pad(features, length), r
        else:
            features, r = random_extract(feat, length)
            return features, r
    else:
        return pad(feat, length), 0


def process_feat_sample(feat, length):
    if len(feat) > length:
            features, samples = random_perturb(feat, length)
            return features, samples
    else:
        return pad(feat, length), 0


def scorebinary(scores=None, threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold



def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """Fill attention mask inplace for a variable length context.
    Args
    ----
    mask: Tensor of size (B, N, D)
        Tensor to fill with mask values.
    sizes: list[int]
        List giving the size of the context for each item in
        the batch. Positions beyond each size will be masked.
    v_mask: float
        Value to use for masked positions.
    v_unmask: float
        Value to use for unmasked positions.
    Returns
    -------
    mask:
        Filled with values in {v_mask, v_unmask}
    """
    mask.fill_(v_unmask)
    n_context = mask.size(2)
    for i, size in enumerate(sizes):
        if size < n_context:
            mask[i, :, size:] = v_mask
    return mask


def median(attention_logits, args):
    attention_medians = torch.zeros(0).to(args.device)
    # attention_logits_median = torch.zeros(0).to(args.device)
    batch_size = attention_logits.shape[0]
    for i in range(batch_size):
        attention_logit = attention_logits[i][attention_logits[i] > 0].unsqueeze(0)
        attention_medians = torch.cat((attention_medians, attention_logit.median(1, keepdims=True)[0]), dim=0)
    attention_medians = attention_medians.unsqueeze(1)
    attention_logits_mask = attention_logits.clone()
    attention_logits_mask[attention_logits <= attention_medians] = 0
    attention_logits_mask[attention_logits > attention_medians] = 1
    attention_logits = attention_logits * attention_logits_mask
    attention_logits_sum = attention_logits.sum(dim=2, keepdim=True)
    attention_logits = attention_logits / attention_logits_sum
    return attention_logits

#




def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False):
    """

    :param predict_dict:
    :param label_dict:
    :param save_path:
    :param itr:
    :param zip: boolen, whether save plots to a zip
    :return:
    """
    if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:
        os.makedirs(os.path.join(save_root, save_path, 'plot'))
    if zip:
        zip_file_name = os.path.join(save_root, save_path, 'plot', 'itr_{}.zip'.format(itr))
        with zipfile.ZipFile(zip_file_name, mode="w") as zf:
            for k, v in predict_dict.items():
                img_name = k + '.jpg'
                predict_np = v.repeat(16)
                label_np = label_dict[k][:len(v.repeat(16))]
                x = np.arange(len(predict_np))
                plt.plot(x, predict_np, label='Anomaly scores', color='b', linewidth=1)
                plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
                plt.yticks(np.arange(0, 1.1, step=0.1))
                plt.xlabel('Frames')
                plt.grid(False, linestyle='-.')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                zf.writestr(img_name, buf.getvalue())


    else:
        with open('best_ckpt/ucf_AE.pickle','rb') as file:
            predict_AE_dict = pickle.load(file)
        for k, v in predict_dict.items():
            # if k == '01_028' or k == '01_047' or k == '01_074' or k == '03_002' or k == '04_002' or k == '05_043' or k == '05_044' or k == '05_045' \
            #         or k == '05_043' or k == '08_001' or k == '08_006' or k == '08_007' or k == '08_010' or k == '08_032':
            #     v = v[:-1]
            predict_np = v.repeat(16)
            predict_ae = predict_AE_dict[k].repeat(16)
            label_np = label_dict[k][:len(v.repeat(16))]
            x = np.arange(len(predict_np))
            plt.figure(figsize=(9.06, 3.4))
            plt.plot(x, predict_np, color='orange', label='Ours', linewidth=5)
            plt.plot(x, predict_ae, color='royalblue', label='MemAE',linestyle='--', linewidth=5,alpha=0.8)
            #np.save('images/05_26/' + k, label_np)
            #plt.plot(x, predict_np, color='b', linewidth=1)
            #print(type(label_np))
            if type(label_np) is list:
                label_np = np.array(label_np)
            plt.fill_between(x, label_np, where=label_np > 0, facecolor="pink",alpha=0.8)
            plt.yticks(np.arange(0.1, 1.1, step=0.1),weight='bold',size=15)
            #plt.xticks(weight='bold',size=15)
            #plt.xlabel('Frames',fontsize=15,fontweight='bold')
            #plt.ylabel('Anomaly scores',fontsize=15,fontweight='bold')
            #plt.grid(True, linestyle='-.')
            #plt.legend()
            # plt.show()
            if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:
                os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
            else:
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
            plt.close()


