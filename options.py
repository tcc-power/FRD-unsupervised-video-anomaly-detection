import argparse
import ast
parser = argparse.ArgumentParser(description='MSTR_Net')
parser.add_argument('--device', type=str, default=0, help='GPU ID')
parser.add_argument('--lr', type=float, default=0.001,help='learning rate (default: 0.0001)')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')
parser.add_argument('--batch_size',  type=int, default=10, help='number of samples in one itration')
parser.add_argument('--Lambda', type=str, default='1_1_1', help='')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
#settings for training
parser.add_argument('--max_seqlen', type=int, default=800, help='maximum sequence length during training (default: 750)')
parser.add_argument('--max_epoch', type=int, default=1000, help='maximum iteration to train (default: 50000)')
parser.add_argument(
    '--lab',
    help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval,
    dest='lab',
)
parser.add_argument('--lab_type', type=str, default='wlab', help='wlab or slab')
parser.add_argument('--beta', type=float, default=0.05, help='(default: 0.0-1.0)')
parser.add_argument('--delta', type=float, default=0, help='(default: int)')
#settings for dataset
parser.add_argument('--dataset_name', type=str, default='shanghaitech', help='')
parser.add_argument('--dataset_path', type=str, default='/home/tu-wan/windows4t/dataset', help='path to dir contains anomaly datasets')
parser.add_argument('--feature_modal', type=str, default='combine', help='features from different input, options contain rgb, flow , combine')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')
#settings for model MSTR
parser.add_argument('--Vitblock_num', type=int, default=2, help='AE:0; vit: 1-8')
parser.add_argument('--cross_clip', type=int, default=1, help='1,2,4')
parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')

parser.add_argument('--plot', type=int, default=0, help='whether plot the video anomalous map on testing')
parser.add_argument('--snapshot', type=int, default=50, help='anomaly sample threshold')
parser.add_argument('--label_type', type=str, default='unary')


