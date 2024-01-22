# -*- coding: utf-8 -*-            
# @Time : 2022/5/16 14:37
#  :name
# @FileName: model_utils.py
# @Software: PyCharm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoTokenizer, AutoModel
torch.set_printoptions(profile='full')


class MatchingAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MatchingAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transform = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, M, utt):
        """
        :param M: (seq_len, batch_size, emb_dim)
        :param utt: (batch_size, emb_dim)
        :return:(batch_size, emb_dim)
        """
        x_ = self.transform(utt).unsqueeze(1)  # (batch_size, 1, emb_dim)
        seq_len = M.shape[-1]
        M_ = M.permute(1, 2, 0)  # (batch_size, emb_dim, seq_len)
        hidden = torch.tanh(torch.bmm(x_, M_))  # H = (batch_size, 1, seq_len)
        alpha = F.softmax(hidden, dim=2)  # (batch, 1, seq_len)  # 得到注意力分数
        # 归一化
        alpha_sum = torch.sum(alpha, dim=2, keepdim=True)
        alpha = alpha / alpha_sum
        att_score = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # (batch_size, emb_dim)

        return att_score



class attentive_node_features(nn.Module):
    '''
    Method to obtain attentive node features over the graph convoluted features
    '''

    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, lengths, nodal_att_type):
        '''
        features : (B, N, V)
        lengths : (B, )
        '''

        if nodal_att_type == None:
            return features

        batch_size = features.size(0)
        max_seq_len = features.size(1)
        padding_mask = [l * [1] + (max_seq_len - l) * [0] for l in lengths]
        padding_mask = torch.tensor(padding_mask).to(features)  # (B, N)
        causal_mask = torch.ones(max_seq_len, max_seq_len).to(features)  # (N, N)
        causal_mask = torch.tril(causal_mask).unsqueeze(0)  # (1, N, N)

        if nodal_att_type == 'global':
            mask = padding_mask.unsqueeze(1)
        elif nodal_att_type == 'past':
            mask = padding_mask.unsqueeze(1) * causal_mask

        x = self.transform(features)  # (B, N, V)
        temp = torch.bmm(x, features.permute(0, 2, 1))
        # print(temp)
        alpha = F.softmax(torch.tanh(temp), dim=2)  # (B, N, N)
        alpha_masked = alpha * mask  # (B, N, N)

        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # (B, N, 1)
        # print(alpha_sum)
        alpha = alpha_masked / alpha_sum  # (B, N, N)
        attn_pool = torch.bmm(alpha, features)  # (B, N, V)

        return attn_pool


def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    # print('alpha.size:{}, adj.size:{}'.format(alpha.size(), adj.size()))
    return alpha - (1 - adj) * 1e30


class GAT(nn.Module):
    '''
    use linear to avoid out of memory
    Hi = alpha_ij(W_rH_j) + alpha_ii * (W_ii*H_i))
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wri = nn.Linear(hidden_size, hidden_size, bias=False)  # alpha_ii * (W_ii * H_i)

    def forward(self, Q, K, V, adj, s_mask):
        B = K.size()[0]
        N = K.size()[1]
        Q0 = Q  # 原始Query
        Q = Q.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        # print('K',K.size())
        X = torch.cat((Q, K), dim=2)  # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0, 2, 1)  # (B, 1, N)
        alpha_ii = alpha
        # alpha = F.leaky_relu(alpha)
        adj = adj.unsqueeze(1)  # (B, 1, N)  
        # print('K.size:{},alpha.size:{},adj.size:{}'.format(K.size(),alpha.size(),adj.size()))
        alpha = mask_logic(alpha, adj)  # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)
        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)  # eq(4)  归一化
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V)  # (B, N, D)
        V1 = self.Wr1(V)  # (B, N, D)
        Q1 = self.Wri(Q0)  # (B, D)

        s_mask = s_mask.unsqueeze(2).float()  # (B, N, 1)
        V = V0 * s_mask + V1 * (1 - s_mask)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) + Q1  # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum


class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1),  self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz + self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz + self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz),  self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):
        hx_concat = torch.cat((ht, xt), dim=-1)
        hm_concat = torch.cat((ht, mt), dim=-1)
        hxm_concat = torch.cat((ht, xt, mt), dim=-1)

        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii, uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x)

        return ht, Ct_x, Ct_m

    def forward(self, x, m):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        for t in range(seq_sz):
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m = self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2)
        cell_seq = torch.stack(cell_seq).permute(1, 0, 2)
        return cell_seq


class myconcat(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(myconcat, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear(self.input_sz * 1,  self.hidden_sz)
        self.all2 = nn.Linear((self.input_sz + self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.input_sz + self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.input_sz * 1), self.hidden_sz)

        self.all11 = nn.Linear(self.g_sz, self.hidden_sz)
        self.all44 = nn.Linear(self.g_sz, self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, mt):
        hx_concat = xt
        hm_concat = mt
        hxm_concat = torch.cat((xt, mt), dim=1)

        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii, uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu
        ht = o * torch.tanh(Ct_x)

        return ht, Ct_x

    def forward(self, x, m):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x = self.node_forward(xt, mt)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2)  # batch_size x max_len x hidden
        cell_seq = torch.stack(cell_seq).permute(1, 0, 2)
        return cell_seq


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class context_filter(nn.Module):
    def __init__(self):
        super(context_filter, self).__init__()
        
    def utterance_selector(self, key, context):
        '''
        :param key: (batch, dim)
        :param context: (batch, utts, dim)
        :return:(batch, utts)
        '''
        # torch.norm即范数，dim=2是2范数
        s1 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1) * torch.norm(key, dim=-1, keepdim=True))
        return s1

    def forward(self, context, ent, alpha, threshold):
        '''
        :param context: (batch, utts, emb_dim)
        :param ent: (batch, utts)
        :return:
        '''
        utt_len = context.size()[1]
        batch = context.size()[0]
        score = []
        s2mk = []
        self.alpha = alpha
        for index in range(utt_len):
            key = context[:, index, :]
            s1 = self.utterance_selector(key, context)  # 当前话语上下文相似度
            s2 = ent
            s2mid = torch.where(s2 == 0, 0, 1)
            s12 = self.alpha * s2 + (1 - self.alpha * s1)
            temp_max = torch.max(s2)
            temp_min = torch.min(s2)
            score.append(s12)
            s2mk.append(s2mid)
        score_ = torch.stack(score, dim=1)  # (batch, utt, utt)
        s2mask1 = torch.stack(s2mk, dim=1)
        s2mask2 = torch.stack(s2mk, dim=-1)
        s2mask = s2mask2 * s2mask1
        ret_score = score_ * s2mask
        mask = torch.where(ret_score > threshold, 1.0, 0.0)
        # 处理s_mask  效果不好  已删除
        # 将对角线的掩码置0
        for i in range(batch):
            diag1 = torch.diag(mask[i])
            diag1 = torch.diag_embed(diag1)
            mask[i] = mask[i] - diag1
        return mask


if __name__ == '__main__':
    x = torch.randn((8, 31, 200))
    x = x.cuda()
    model = context_filter(200, 10)
    model.cuda()
    y = model(x)
