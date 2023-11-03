# -*- coding: utf-8 -*-            
# @Time : 2022/5/16 12:52
#  :name
# @FileName: model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *
from transformers import AutoModel
#
# bertcheckpoint = torch.load('./saved_models/meld_bertmodel.pth')
# mymodelcheckpoint = torch.load('./saved_models/meld_mymodel917.pth')


class myModel(nn.Module):
    def __init__(self, args, num_class):
        super().__init__()
        # self.encoder = AutoModel.from_pretrained(args.bert_path)
        self.args = args
        self.filter = context_filter()

        # gcn layer
        self.dropout = nn.Dropout(args.dropout)
        self.lstm_hidden_dim = 600
        self.activation = nn.PReLU()
        self.gnn_layers = args.gnn_layers

        gats = []  # 通过这个rgcn，得到注意力作为边的权重
        for _ in range(args.gnn_layers):
            gats += [GAT(args.hidden_dim)]
        self.gather = nn.ModuleList(gats)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        self.graph_dim = args.hidden_dim * (args.gnn_layers + 1)  # + args.emb_dim
        self.in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        # layers = [nn.Linear(self.graph_dim, args.hidden_dim), nn.ReLU()]

        # 添加的Syn-LSTM层
        self.enhance = nn.Linear(args.emb_dim, args.emb_dim, bias=True)
        self.graph_enhance = nn.Linear(self.graph_dim, self.graph_dim, bias=True)
        self.lstm_f = MyLSTM(args.emb_dim, self.lstm_hidden_dim, self.graph_dim)  # 添加的Syn-LSTM层
        # self.lstm_f = MyLSTM(self.graph_dim, self.lstm_hidden_dim, args.emb_dim)

        # output mlp layers
        layers = [nn.Linear(self.lstm_hidden_dim, args.hidden_dim), nn.PReLU()]  # 添加的Syn-LSTM层
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.PReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, s_mask, entro, alpha, threshold):
        '''
        :param features: (B, N, D)  (64,21,1024)
        :param adj: (B, N, N)
        :param entro: (B, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        adj = self.filter(features, entro, alpha, threshold)  # 邻接矩阵mask
        H0 = self.activation(self.fc1(features))  # -> (batch, seq_len, 300)
        # H0 = self.dropout(H0)
        H = [H0]
        for l in range(self.args.gnn_layers):  # H为每层节点张量堆叠
            for i in range(0, num_utter):
                # M.size(64, 300) 逐点操作
                _, M = self.gather[l](H[l][:, i, :], H[l], H[l], adj[:, i, :], s_mask[:, i, :])
                H_temp = M.unsqueeze(1)
                if i == 0:
                    H_layer = H_temp
                else:
                    H_layer = torch.cat((H_layer, H_temp), dim=1)
            H.append(H_layer)   # 上一层将会发生情感误判的节点信息将会继承到下一层的节点中，导致情感信息无法得到纠正（这一点可以试一试）
        H = torch.cat(H, dim=2)
        # H = self.activation(self.graph_enhance(H))
        features = self.activation(self.enhance(features))
        H = self.lstm_f(features, H)

        logits = self.out_mlp(H)

        return logits


class ablationModel(nn.Module):
    def __init__(self, args, num_class):
        super().__init__()
        # self.encoder = AutoModel.from_pretrained(args.bert_path)
        self.args = args
        self.filter = context_filter()

        # gcn layer
        self.dropout = nn.Dropout(args.dropout)
        self.lstm_hidden_dim = 600
        self.activation = nn.PReLU()
        self.gnn_layers = args.gnn_layers

        gats = []  # 通过这个rgcn，得到注意力作为边的权重
        for _ in range(args.gnn_layers):
            gats += [GAT_dialoggcn_v1(args.hidden_dim)]
        self.gather = nn.ModuleList(gats)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        self.graph_dim = args.hidden_dim * (args.gnn_layers + 1)  # + args.emb_dim
        self.in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        # layers = [nn.Linear(self.graph_dim, args.hidden_dim), nn.ReLU()]

        # 添加的Syn-LSTM层
        self.enhance = nn.Linear(args.emb_dim, args.emb_dim, bias=True)
        self.graph_enhance = nn.Linear(self.graph_dim, self.graph_dim, bias=True)
        self.lstm_f = MyLSTM(args.emb_dim, self.lstm_hidden_dim, self.graph_dim)  # 添加的Syn-LSTM层
        # self.lstm_f = MyLSTM(self.graph_dim, self.lstm_hidden_dim, args.emb_dim)

        # output mlp layers
        layers = [nn.Linear(args.emb_dim + self.graph_dim, args.hidden_dim), nn.PReLU()]  # 添加的Syn-LSTM层
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.PReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, s_mask, entro, alpha, threshold):
        '''
        :param features: (B, N, D)  (64,21,1024)
        :param adj: (B, N, N)
        :param entro: (B, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        adj = self.filter(features, entro, alpha, threshold)  # 邻接矩阵mask
        H0 = self.activation(self.fc1(features))  # -> (batch, seq_len, 300)
        # H0 = self.dropout(H0)
        H = [H0]
        for l in range(self.args.gnn_layers):  # H为每层节点张量堆叠
            for i in range(0, num_utter):
                _, M = self.gather[l](H[l][:, i, :], H[l], H[l], adj[:, i, :], s_mask[:, i, :])  # M.size(64, 300)
                H_temp = M.unsqueeze(1)
                if i == 0:
                    H_layer = H_temp
                else:
                    H_layer = torch.cat((H_layer, H_temp), dim=1)
            H.append(H_layer)   # 上一层将会发生情感误判的节点信息将会继承到下一层的节点中，导致情感信息无法得到纠正（这一点可以试一试）
        features = self.activation(self.enhance(features))
        H.append(features)
        H = torch.cat(H, dim=2)

        logits = self.out_mlp(H)

        return logits


class ft_model(nn.Module):
    def __init__(self, args, n_class):
        super(ft_model, self).__init__()
        self.encoder = AutoModel.from_pretrained('./roberta-large')
        self.designed = myModel(args, n_class)
        self.encoder.load_state_dict(bertcheckpoint['model_state_dict'])
        self.designed.load_state_dict(mymodelcheckpoint['model_state_dict'])
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, batch_tokens, s_mask, entro, alpha, threshold):
        '''
        :param batch_tokens: (1, N, D)
        :param s_mask: (1, N, N)
        :param entro:  (1, N)
        :return:
        '''
        batch_tokens = batch_tokens.squeeze(dim=0)  # (N, D)
        batch_output = self.encoder(batch_tokens).last_hidden_state[:, 0, :]
        batch_output = batch_output.unsqueeze(dim=0)
        logits = self.designed(batch_output, s_mask, entro, alpha, threshold)
        return logits


class singleLinear(nn.Module):
    def __init__(self, args, n_class):
        super(singleLinear, self).__init__()
        self.encoder = AutoModel.from_pretrained('./roberta-large')
        self.classifier = nn.Sequential(
            nn.Linear(args.emb_dim, 300),
            nn.ReLU(),
            nn.Linear(300, n_class)
        )

    def forward(self, batch_tokens, s_mask, entro, alpha, threshold):
        '''
        :param batch_input_ids: B,N,N
        :return:
        '''
        batch_tokens = batch_tokens.squeeze(dim=0)  # (N, D)
        batch_output = self.encoder(batch_tokens).last_hidden_state[:, 0, :]
        batch_output = batch_output.unsqueeze(dim=0)
        logits = self.classifier(batch_output)
        return logits
