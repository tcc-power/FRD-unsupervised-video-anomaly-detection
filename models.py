import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from losses import *
from utils import Update_mb
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import losses
from torch.nn.modules.module import Module
#self attention
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(-1, 1, N, C)
        B_new, T_new, N_new, C_new = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_new, T_new, N_new, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)#3*B*T*head*N*dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B_new, T_new, N_new, -1)
        x = self.proj_drop(self.proj(x))
        x = x.view(B, T, N, C)
        return x

#cross attention
class CrossAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(-1, 2, N, C)
        B_new, T_new, N_new, C_new = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_new, T_new, N_new, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)  # 3*B*T*head*N*dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        q = torch.split(q, 1, dim=1)
        k = torch.split(k, 1, dim=1)
        v = torch.split(v, 1, dim=1)
        #q0 k0 v0 && q0 k1 v1
        attn_map1 = ((q[0]) @ k[0].transpose(-2, -1)).softmax(dim=-1)  # attention map between query1 and key2
        attn_map2 = ((q[0]) @ k[1].transpose(-2, -1)).softmax(dim=-1)  # attention map between query2 and key1
        v1_cross = attn_map1 @ v[0]  # cross attention between value2 and attention map1
        v2_cross = attn_map2 @ v[1]  # cross attention between value1 and attention map2
        v_cross = torch.cat([v1_cross, v2_cross], dim=1)  # concatenate the two cross attentions
        x = v_cross.transpose(2, 3).reshape(B_new, T_new, N_new, -1)
        x = self.proj_drop(self.proj(x))
        x = x.view(B,T,N,C)
        return x

#cross attention
class CrossAttention4(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, T, N, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)  # 3*B*T*head*N*dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        q = torch.split(q, 1, dim=1)
        k = torch.split(k, 1, dim=1)
        v = torch.split(v, 1, dim=1)
        # q0 k0 v0 && q0 k1 v1 && q1 k2 v2 && q2 k3 v3
        # attn_map0 = (q[0] @ k[0].transpose(-2, -1)).softmax(dim=-1)  # attention map between query1 and key2
        # attn_map1 = (q[0] @ k[0].transpose(-2, -1)).softmax(dim=-1)
        # attn_map2 = ((q[0]+q[1]) @ (k[0]+k[1]).transpose(-2, -1)).softmax(dim=-1)
        # attn_map3 = ((q[0]+q[1]+q[2]) @ (k[0]+k[1]+k[2]).transpose(-2, -1)).softmax(dim=-1)# attention map between query2 and key1
        attn_map0 = (q[0] @ k[0].transpose(-2, -1)).softmax(dim=-1)
        attn_map1 = (q[0] @ k[1].transpose(-2, -1)).softmax(dim=-1)
        attn_map2 = (q[1] @ k[2].transpose(-2, -1)).softmax(dim=-1)
        attn_map3 = (q[2] @ k[3].transpose(-2, -1)).softmax(dim=-1)
        array = torch.cat([attn_map0,attn_map1,attn_map2,attn_map3],dim=0).cpu().detach().numpy()
        v0_cross = attn_map0 @ v[0]
        v1_cross = attn_map1 @ v[1]
        v2_cross = attn_map2 @ v[2]  # cross attention between value2 and attention map1
        v3_cross = attn_map3 @ v[3]  # cross attention between value1 and attention map2
        v_cross = torch.cat([v0_cross, v1_cross, v2_cross, v3_cross], dim=1)  # concatenate the two cross attentions
        x = v_cross.transpose(2, 3).reshape(B, T, N, -1)
        x = self.proj_drop(self.proj(x))
        return x

class Block(nn.Module):
    def __init__(self, dim=2048, cross_clip=1, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if cross_clip == 1 or cross_clip == 4:
            self.attn = SelfAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        elif cross_clip == 2:
            self.attn = CrossAttention2(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        else:
            self.attn = CrossAttention4(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Block1(nn.Module):
    def __init__(self, dim=2048, cross_clip=1, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Block2(nn.Module):
    def __init__(self, dim=2048, cross_clip=2, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention2(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Block4(nn.Module):
    def __init__(self, dim=2048, cross_clip=4, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention4(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            #nn.ReLU(),
        )
    def forward(self, x):
        B,C,H,W = x.size()
        x = x.view(B*C, H,W)
        #x = x.permute(0,2,1)
        x = self.conv_1(x)
        #x = x.permute(0,2,1)
        x = x.view(B, C, W)
        return x

class Decoder(nn.Module):
    def __init__(self, batchsize, embed_dim, Vitblock_num, cross_clip, split, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        if split == 0:
            embed_dim = embed_dim // 16

        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.pos_embed = get_sinusoid_encoding_table(16, embed_dim)
        self.dropout = nn.Dropout(p=0.5)
        if cross_clip == 1:
            self.decoder_blocks = nn.ModuleList([
                Block1(dim=embed_dim, cross_clip=cross_clip)
                for i in range(Vitblock_num)])
        elif cross_clip == 2:
            if Vitblock_num == 8:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 6:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 2:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip)])
            elif Vitblock_num == 4:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip),Block2(dim=embed_dim, cross_clip=cross_clip)])
        elif cross_clip == 4:
            if Vitblock_num == 8:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 6:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 2:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip)])
            elif Vitblock_num == 4:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip),Block4(dim=embed_dim, cross_clip=cross_clip)])
            elif Vitblock_num == 1:
                self.decoder_blocks = nn.ModuleList([
                Block1(dim=embed_dim, cross_clip=cross_clip)
                for i in range(Vitblock_num)])
        self.norm = norm_layer(embed_dim)
        self.norm_pix_loss = norm_pix_loss
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # feature:完整未被掩码的提的特征，target:被掩码掉待恢复的部分，feature_input:被掩码了75%的FV，mask_fei:掩码图中的灰块
    def forward(self, feature):
        if len(feature.shape) == 3:
            B, T, C = feature.size()
            feature = self.dropout(feature.view(B, T, 16, C//16))
        else:
            feature = self.dropout(feature)
        B, T, N, C = feature.size()
        feature_input = self.decoder_embed(feature)
        feature_input = feature_input + self.pos_embed.expand(B, T, -1, -1).type_as(feature_input).to(feature_input.device).clone().detach()
        for blk in self.decoder_blocks:
            feature_input = blk(feature_input)
        feature_output = self.norm(feature_input)
        feature_output = feature_output.view(B, T, -1)
        return feature_output


class Autoencoder_L(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder_L, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.encoder = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.decoder = nn.Linear(1024, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.encoder(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.decoder(x))
        return x

class Autoencoder_B(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder_B, self).__init__()
        self.encoder = nn.Linear(input_dim, 512)
        self.decoder = nn.Linear(512, input_dim)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = self.decoder(x)
        return x

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
        self.decoder = Decoder(batchsize=self.batchsize,embed_dim=feature_size,Vitblock_num=Vitblock_num,cross_clip=cross_clip,split=split)
        self.AE = Autoencoder_B(feature_size)
        self.scorer = Scorer(n_feature=feature_size)
        self.scorer_c2fpl = Scorer_C2FPL(n_features=feature_size)
        #self.scorer_c2fpl = Scorer_FRD(n_feature=feature_size)
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

    def slide_score(self,scores):
        scores_new = scores.clone()
        if self.cross_clip >= 2:
            scores_new[:, 1, :] = (scores[:, 0, :] + scores[:, 1, :]) / 2
            if self.cross_clip >= 3:
                scores_new[:, 2, :] = (scores[:, 0, :] + scores[:, 1, :] + scores[:, 2, :]) / 3
                if self.cross_clip >= 4:
                    scores_new[:, 3, :] = (scores[:, 0, :] + scores[:, 1, :] + scores[:, 2, :] + scores[:, 3,
                                                                                                 :]) / 4
        return scores_new

    def forward(self, inputs, lab,lab_type, is_training=True, is_test=False):
        if is_training:
            criterion1 = torch.nn.MSELoss(reduction='none')
            # Reconstructor
            if lab and len(self.lab) != 0:
                #concat
                inputs = torch.cat([inputs, self.lab],dim=0)
            if self.Vitblock_num == 0:
                feature_output = self.AE(inputs)
            else:
                feature_output = self.decoder(inputs.reshape(-1,self.cross_clip,self.feature_size)) #[*,10,4,2048]->[*,4,2048]
                feature_output = feature_output.reshape(-1,10,self.cross_clip,self.feature_size)
            if is_test:
                return feature_output
            lossr = torch.max(torch.mean(criterion1(feature_output, inputs), dim=-1),dim=1)[0]
            mean_g = torch.mean(lossr)
            std_g = torch.std(lossr)
            Lrgth = mean_g + 1*std_g
            pseudo_reconstruction_labels = lossr.ge(Lrgth).float()
            Lossr_nor = torch.mean(lossr[:self.batchsize])
            if lab and len(self.lab) != 0:
                Loss_mb = F.relu(-torch.mean(lossr[self.batchsize:] - Lrgth))
            else:
                Loss_mb = Lossr_nor*0
            Lossr = Lossr_nor

            inputs = inputs[:self.batchsize, :, :,:]
            pseudo_reconstruction_labels = pseudo_reconstruction_labels[:self.batchsize]
            scores = self.scorer(inputs[:self.batchsize, :, :,:], is_training=True)

            scores = self.slide_score(scores)
            scorer_c2fpl = (self.scorer_c2fpl(inputs[:self.batchsize, :, :,:], is_training=True)).ge(0.5).float().squeeze(-1)
            pseudo_reconstruction_labels = pseudo_reconstruction_labels[:len(scorer_c2fpl)] * scorer_c2fpl
            Losss = binary_CE_loss(scores[:self.batchsize, :, :], pseudo_reconstruction_labels.unsqueeze(2))
            Lsgth = torch.mean(scores) + self.delta * torch.std(scores)
            if lab:
                pseudo_scorer_labels = scores.ge(Lsgth).float()*(pseudo_reconstruction_labels.unsqueeze(-1))
                if lab_type == 'wlab':
                    #weak abnormal
                    idx = (torch.sum(pseudo_scorer_labels,dim=1)>0).squeeze()
                    pseudo_abnormal_features = inputs[idx]
                elif lab_type == 'slab':
                    # strong abnormal
                    idx = (torch.sum(pseudo_scorer_labels, dim=1) == self.cross_clip).squeeze()
                    pseudo_abnormal_features = inputs[idx]

                self.lab = Update_mb(self.lab, pseudo_abnormal_features, self.mb_size)
            return Lossr, Loss_mb, Losss
        else:
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

class Scorer_FRD(torch.nn.Module):
    def __init__(self, n_feature):
        super(Scorer_FRD, self).__init__()
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
            #return torch.mean(score,dim=1)
            return score