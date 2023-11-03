from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from transformers import BertTokenizer


def get_train_valid_sampler(trainset):  # 载入所有训练/验证集
    size = len(trainset)
    idx = list(range(size))  # 数据集下标的列表
    return SubsetRandomSampler(idx)


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('./data/%s/speaker_vocab_v2.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    # person_vec_dir = './data/%s/person_vect.pkl' % (dataset_name)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec


def get_IEMOCAP_loaders(dataset_name='IEMOCAP', batch_size=1, num_workers=4, pin_memory=True, args=None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    # 固定iemocap v2 为微调数据集
    trainset = IEMOCAP4Dataset(dataset_name, 'train',  speaker_vocab, label_vocab, args)
    devset = IEMOCAP4Dataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,  # 子进程装载
                              pin_memory=pin_memory)  # True为数据拷贝至GPU CUDA区

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAP4Dataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec


def get_MELD_loaders(dataset_name='MELD', batch_size=32, num_workers=0, pin_memory=False, args=None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    ## MELD2Dataset为微调使用的数据集，MELD3Dataset为训练使用的数据集
    trainset = MELD3Dataset(dataset_name, 'train',  speaker_vocab, label_vocab, args)
    devset = MELD3Dataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,  # 子进程装载
                              pin_memory=pin_memory)  # True为数据拷贝至GPU CUDA区

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELD3Dataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec


def get_EmoryNLP_loaders(dataset_name='EmoryNLP', batch_size=32, num_workers=0, pin_memory=False, args=None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    trainset = EmoryNLPDataset(dataset_name, 'train',  speaker_vocab, label_vocab, args)
    devset = EmoryNLPDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,  # 子进程装载
                              pin_memory=pin_memory)  # True为数据拷贝至GPU CUDA区

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = EmoryNLPDataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec


def get_DailyDialog_loaders(dataset_name='DailyDialog', batch_size=32, num_workers=0, pin_memory=False, args=None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    trainset = DailyDialogDataset(dataset_name, 'train',  speaker_vocab, label_vocab, args)
    devset = DailyDialogDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,  # 子进程装载
                              pin_memory=pin_memory)  # True为数据拷贝至GPU CUDA区

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = DailyDialogDataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec
