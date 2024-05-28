from __future__ import print_function
import os
import torch
from models import model as model
from video_dataset_anomaly_balance_uni_sample_ucf import Dataset_Con_all_feedback_XD,train_loader
from torch.utils.data import DataLoader
from train import train
import options
import torch.optim as optim
import datetime
import numpy as np


if __name__ == '__main__':
    args = options.parser.parse_args()
    args.model_name = 'MSTR'
    args.seed = 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    time = datetime.datetime.now()
    save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.dataset_name, '_Lambda_{}'.format(args.Lambda), args.feature_modal, '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))

    model = model(args.max_seqlen, feature_size=args.feature_size, Vitblock_num=args.Vitblock_num,
                  cross_clip=args.cross_clip, split=0, beta=args.beta,delta=args.delta)
    model_dict = model.scorer_c2fpl.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('UCFfinal.pkl').items()})
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    train_loader = DataLoader(dataset=train_loader(args=args, train=True),
                              batch_size=args.max_seqlen,shuffle=True,
                              num_workers=0,pin_memory=False,generator=torch.Generator(device='cpu'))
    test_loader = DataLoader(Dataset_Con_all_feedback_XD(args, test_mode=True),
                             batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=False, drop_last=False)
    optimizer = optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95))
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=test_loader, args=args, model=model,
          optimizer=optimizer, logger=None, device=device, save_path=save_path)
