import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import math
import json
import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer


class IEMOCAPDataset(Dataset):
    '''
    原始数据集
    '''
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, self.tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix  构建图结构的关键之一
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i 邻接矩阵
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):  # 此处为当前话语，向前找，直到上一句自己说的话。这个有向图太简单了！
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v2(self, speakers, max_dialog_len):
        '''
        获得过去有向图和未来的两句话作为补充信息
        最好能学习一个值来确定窗口大小
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            future_info = 4  # 未来话语信息
            for i, s in enumerate(speaker):
                cnt = 0
                count = 0
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == self.args.windowp:
                            break
                if future_info != 0:
                    for j in range(i + 1, cur_len, 1):
                        a[i, j] = 1
                        count += 1
                        if count == future_info:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:应该是说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_crfmask(self, labels):
        crfmask = []
        for batch in labels:
            res = [0 if d == -1 else 1 for d in batch]
            crfmask.append(res)
        return torch.ByteTensor(crfmask)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        adj = self.get_adj_v2([d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]
        crfmask = self.get_crfmask(labels)
        return feaures, labels, adj, s_mask, crfmask, s_mask_onehot, lengths, speakers, utterances


class IEMOCAP2Dataset(Dataset):
    '''
    该数据集用于微调测试
    '''
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append([1 if u['speaker'] == 'M' else 0])
                tokens.append(u['tokens'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token.item() for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                utt = utt.int().tolist()
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def padding(self, tokens, tokenizer):
        # robert_pad_token_id = 1
        res = []
        for dia in tokens:
            rdia = []
            for utt in dia:
                rdia.append(torch.LongTensor(utt))
            res.append(pad_sequence(rdia, batch_first=True, padding_value=tokenizer.pad_token_id))
        batch_token_output = pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)
        return batch_token_output

    def collate_fn(self, data):
        max_dialog_len = max([d[2] for d in data])
        labels = pad_sequence([d[0] for d in data], batch_first=True, padding_value=-1)
        s_mask, s_mask_onehot = self.get_s_mask([d[1] for d in data], max_dialog_len)
        tokens = [d[3] for d in data]
        tokens = self.padding(tokens, self.tokenizer)
        entro = self.get_batch_entropy(tokens)
        speakers = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True, padding_value=-1)
        return labels, tokens, speakers, s_mask, entro


class IEMOCAP3Dataset(Dataset):
    '''
    普通自制  训练使用数据集
    '''
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(1 if u['speaker'] == 'M' else 0)
                features.append(u['features'])
                tokens.append(u['tokens'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'],\
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        获得过去有向图和未来的两句话作为补充信息
        最好能学习一个值来确定窗口大小
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            future_info = 4  # 未来话语信息
            for i, s in enumerate(speaker):
                cnt = 0
                count = 0
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
                for j in range(i + 1, cur_len, 1):
                    a[i, j] = 1
                    count += 1
                    if count == future_info:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v2(self, tokens, speakers, max_dialog_len):
        '''
        :param tokens: list(batch, utts, tokens)
        :param speakers: (batch, utts)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        ent = self.get_batch_entropy(tokens)  # batch ,utt
        for batch, speaker in enumerate(speakers):  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            for i, s in enumerate(speaker):
                past_entro = ent[batch][i]
                future_entro = ent[batch][i]
                for j in range(i - 1, -1, -1):
                    past_entro += ent[batch][j]
                    a[i][j] = 1
                    if past_entro > 3:
                        break
                for j in range(i + 1, cur_len, 1):
                    future_entro += ent[batch][j]
                    a[i][j] = 1
                    if future_entro > 5:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_crfmask(self, labels):
        crfmask = []
        for batch in labels:
            res = [0 if d == -1 else 1 for d in batch]
            crfmask.append(res)
        return torch.ByteTensor(crfmask)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        adj = self.get_adj_v2(tokens, [d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]

        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro


class IEMOCAP4Dataset(Dataset):
    '''
    DAG原始特征加入tokens和entropy，原始代码
    '''
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
                input_ids = self.tokenizer(u['text'])['input_ids']
                tokens.append(input_ids)
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'],\
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_adj_v2(self, tokens, speakers, max_dialog_len):
        '''
        :param tokens: list(batch, utts, tokens)
        :param speakers: (batch, utts)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        ent = self.get_batch_entropy(tokens)  # batch ,utt
        for batch, speaker in enumerate(speakers):  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            for i, s in enumerate(speaker):
                past_entro = ent[batch][i]
                future_entro = ent[batch][i]
                for j in range(i - 1, -1, -1):
                    past_entro += ent[batch][j]
                    a[i][j] = 1
                    if past_entro > 3:
                        break
                for j in range(i + 1, cur_len, 1):
                    future_entro += ent[batch][j]
                    a[i][j] = 1
                    if future_entro > 5:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def padding(self, tokens, tokenizer):
        # robert_pad_token_id = 1
        res = []
        for dia in tokens:
            rdia = []
            for utt in dia:
                rdia.append(torch.LongTensor(utt))
            res.append(pad_sequence(rdia, batch_first=True, padding_value=tokenizer.pad_token_id))
        batch_token_output = pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)
        return batch_token_output

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        adj = self.get_adj_v2(tokens, [d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]

        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro


class IEMOCAP5Dataset(Dataset):
    '''
    DAG原始文本
    '''
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def get_adj_v2(self, tokens, speakers, max_dialog_len):
        '''
        :param tokens: list(batch, utts, tokens)
        :param speakers: (batch, utts)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        ent = self.get_batch_entropy(tokens)  # batch ,utt
        for batch, speaker in enumerate(speakers):  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            for i, s in enumerate(speaker):
                past_entro = ent[batch][i]
                future_entro = ent[batch][i]
                for j in range(i - 1, -1, -1):
                    past_entro += ent[batch][j]
                    a[i][j] = 1
                    if past_entro > 3:
                        break
                for j in range(i + 1, cur_len, 1):
                    future_entro += ent[batch][j]
                    a[i][j] = 1
                    if future_entro > 5:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def padding(self, tokens, tokenizer):
        # robert_pad_token_id = 1
        res = []
        for dia in tokens:
            rdia = []
            for utt in dia:
                rdia.append(torch.LongTensor(utt))
            res.append(pad_sequence(rdia, batch_first=True, padding_value=tokenizer.pad_token_id))
        batch_token_output = pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)
        return batch_token_output

    def collate_fn(self, data):
        max_dialog_len = max([d[2] for d in data])
        labels = pad_sequence([d[0] for d in data], batch_first=True, padding_value=-1)
        s_mask, s_mask_onehot = self.get_s_mask([d[1] for d in data], max_dialog_len)
        speakers = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True, padding_value=-1)
        utterance = [d[3] for d in data]
        return labels, speakers, s_mask, utterance


class MELDDataset(Dataset):  # 取到完整的数据集
    '''
    原文数据集+entro+tokens
    '''
    def __init__(self, dataset_name='MELD', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v9.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
                input_ids = self.tokenizer(u['text'])['input_ids']
                tokens.append(input_ids)
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'], \
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_adj_v2(self, tokens, speakers, max_dialog_len):
        '''
        :param tokens: list(batch, utts, tokens)
        :param speakers: (batch, utts)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        ent = self.get_batch_entropy(tokens)  # batch ,utt
        for batch, speaker in enumerate(speakers):  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            for i, s in enumerate(speaker):
                past_entro = ent[batch][i]
                future_entro = ent[batch][i]
                for j in range(i - 1, -1, -1):
                    past_entro += ent[batch][j]
                    a[i][j] = 1
                    if past_entro > 3:
                        break
                for j in range(i + 1, cur_len, 1):
                    future_entro += ent[batch][j]
                    a[i][j] = 1
                    if future_entro > 5:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def padding(self, tokens, tokenizer):
        # robert_pad_token_id = 1
        res = []
        for dia in tokens:
            rdia = []
            for utt in dia:
                rdia.append(torch.LongTensor(utt))
            res.append(pad_sequence(rdia, batch_first=True, padding_value=tokenizer.pad_token_id))
        batch_token_output = pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)
        return batch_token_output

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        adj = self.get_adj_v2(tokens, [d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]

        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro


class MELD2Dataset(Dataset):
    '''
    该数据集用于微调测试
    '''
    def __init__(self, dataset_name='MELD', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                tokens.append(u['tokens'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token.item() for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                utt = utt.int().tolist()
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def padding(self, tokens, tokenizer):
        # robert_pad_token_id = 1
        res = []
        for dia in tokens:
            rdia = []
            for utt in dia:
                rdia.append(torch.LongTensor(utt))
            res.append(pad_sequence(rdia, batch_first=True, padding_value=tokenizer.pad_token_id))
        batch_token_output = pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)
        return batch_token_output

    def collate_fn(self, data):
        max_dialog_len = max([d[2] for d in data])
        labels = pad_sequence([d[0] for d in data], batch_first=True, padding_value=-1)
        s_mask, s_mask_onehot = self.get_s_mask([d[1] for d in data], max_dialog_len)
        tokens = [d[3] for d in data]
        tokens = self.padding(tokens, self.tokenizer)
        entro = self.get_batch_entropy(tokens)
        speakers = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True, padding_value=-1)
        return labels, tokens, speakers, s_mask, entro


class MELD3Dataset(Dataset):
    '''
    训练使用数据集
    '''
    def __init__(self, dataset_name='MELD', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v4.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['features'])
                tokens.append(u['tokens'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'],\
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        获得过去有向图和未来的两句话作为补充信息
        最好能学习一个值来确定窗口大小
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            future_info = 4  # 未来话语信息
            for i, s in enumerate(speaker):
                cnt = 0
                count = 0
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
                for j in range(i + 1, cur_len, 1):
                    a[i, j] = 1
                    count += 1
                    if count == future_info:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v2(self, tokens, speakers, max_dialog_len):
        '''
        :param tokens: list(batch, utts, tokens)
        :param speakers: (batch, utts)
        :param max_dialog_len:
        :return:
        '''
        adj = []
        ent = self.get_batch_entropy(tokens)  # batch ,utt
        for batch, speaker in enumerate(speakers):  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            cur_len = len(speaker)  # 当前话语的最大长度
            for i, s in enumerate(speaker):
                past_entro = ent[batch][i]
                future_entro = ent[batch][i]
                for j in range(i - 1, -1, -1):
                    past_entro += ent[batch][j]
                    a[i][j] = 1
                    if past_entro > 3:
                        break
                for j in range(i + 1, cur_len, 1):
                    future_entro += ent[batch][j]
                    a[i][j] = 1
                    if future_entro > 5:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_crfmask(self, labels):
        crfmask = []
        for batch in labels:
            res = [0 if d == -1 else 1 for d in batch]
            crfmask.append(res)
        return torch.ByteTensor(crfmask)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        crfmask = self.get_crfmask(labels)
        # midlist = [torch.tensor([1] * len(d[1])) for d in data]
        # crfmask = pad_sequence(midlist, batch_first=True, padding_value=0)
        # adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        adj = self.get_adj_v2(tokens, [d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]

        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro


class EmoryNLPDataset(Dataset):
    '''
    原始特征文件+token，能到平均性能
    '''
    def __init__(self, dataset_name='EmoryNLP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v9.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
                input_ids = self.tokenizer(u['text'])['input_ids']
                tokens.append(input_ids)
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'], \
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix  构建图结构的关键之一
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i 邻接矩阵
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                # 改这里，与未来话语邻接
                for j in range(i - 1, -1, -1):  # 此处为当前话语，向前找，直到上一句自己说的话。这个有向图太简单了！
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v2(self, speakers, max_dialog_len):
        '''
        获得未来的两句话，作为当前信息的补充
        :param speakers: (B,N)
        :param max_dialog_len:
        :return: adj(:,i,:)节点i的邻接矩阵
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0  # 过去自己说的话的数量
                # 改这里，与未来话语邻接
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        功能：找相邻的说话人掩码
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]
        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro


class EmoryNLP2Dataset(Dataset):
    '''
    该数据集用于微调测试
    '''
    def __init__(self, dataset_name='EmoryNLP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v1.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                tokens.append(u['tokens'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token.item() for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                utt = utt.int().tolist()
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def padding(self, tokens, tokenizer):
        # robert_pad_token_id = 1
        res = []
        for dia in tokens:
            rdia = []
            for utt in dia:
                rdia.append(torch.LongTensor(utt))
            res.append(pad_sequence(rdia, batch_first=True, padding_value=tokenizer.pad_token_id))
        batch_token_output = pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)
        return batch_token_output

    def collate_fn(self, data):
        max_dialog_len = max([d[2] for d in data])
        labels = pad_sequence([d[0] for d in data], batch_first=True, padding_value=-1)
        s_mask, s_mask_onehot = self.get_s_mask([d[1] for d in data], max_dialog_len)
        tokens = [d[3] for d in data]
        tokens = self.padding(tokens, self.tokenizer)
        entro = self.get_batch_entropy(tokens)
        speakers = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True, padding_value=-1)
        return labels, tokens, speakers, s_mask, entro


class EmoryNLP3Dataset(Dataset):
    '''
    自带token的特征文件，训练用
    '''
    def __init__(self, dataset_name='EmoryNLP', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        for d in raw_data:
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['features'])
                tokens.append(u["tokens"])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'], \
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix  构建图结构的关键之一
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i 邻接矩阵
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                # 改这里，与未来话语邻接
                for j in range(i - 1, -1, -1):  # 此处为当前话语，向前找，直到上一句自己说的话。这个有向图太简单了！
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v2(self, speakers, max_dialog_len):
        '''
        获得未来的两句话，作为当前信息的补充
        :param speakers: (B,N)
        :param max_dialog_len:
        :return: adj(:,i,:)节点i的邻接矩阵
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0  # 过去自己说的话的数量
                # 改这里，与未来话语邻接
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        功能：找相邻的说话人掩码
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]
        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro


class DailyDialogDataset(Dataset):  # 取到完整的数据集

    def __init__(self, dataset_name='DailyDialog', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-large')
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)
        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            tokens = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)  # 词标签不存在在labellist里面
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
                input_ids = self.tokenizer(u['text'])['input_ids']
                tokens.append(input_ids)
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'tokens': tokens
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'],\
               self.data[index]['tokens']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i, j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix  构建图结构的关键之一
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i 邻接矩阵
        '''
        adj = []
        for speaker in speakers:  # 每个batch的speakers
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                # 改这里，与未来话语邻接
                for j in range(i - 1, -1, -1):  # 此处为当前话语，向前找，直到上一句自己说的话。这个有向图太简单了！
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 1:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_batch_entropy(self, tokens):
        '''
        :param tokens: (batch, utts, tokens) utts, tokens都不等长
        :return:
        '''
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]  # 对话中所有token列表
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)  # 这个对话的信息熵已添加
            dia_entro.append(torch.Tensor(entro))
        batch_entro = pad_sequence(dia_entro, batch_first=True, padding_value=0)
        return batch_entro

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:应该是说话人顺序
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        tokens = [d[5] for d in data]  # 此处tokens不等长
        entro = self.get_batch_entropy(tokens)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]
        return feaures, labels, adj, s_mask, tokens, lengths, speakers, utterances, entro