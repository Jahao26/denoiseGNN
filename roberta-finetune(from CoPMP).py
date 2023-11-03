import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup

config = {
    'bert_path': './roberta-large',
    'dataset_path': './data/MELD',
    'saved_model': './saved_models/bertmodel2.pth',  # 这里要注意一下
    'emb_dim': 1024,
    'hidden_dim': 300,
    'n_class': 7,
    'batch': 1,
    'epoch': 10,
    'lr': 1e-6,
    'max_grad_norm': 10
}

roberta_tokenizer = AutoTokenizer.from_pretrained(config['bert_path'])


class MELD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []

        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()

        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral",
                   'sadness': "sad", 'surprise': 'surprise'}
        self.sentidict = {'positive': ["joy"], 'negative': ["anger", "disgust", "fear", "sadness"],
                          'neutral': ["neutral", "surprise"]}
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if i < 2:
                continue
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker, utt, emo, senti = data.strip().split('\t')
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)

            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)

        self.emoList = sorted(self.emoSet)
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList
        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict


def encode_right_truncated(text, tokenizer, max_length=511):
    '''
    完成分词工作
    :return:
    '''
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return [tokenizer.cls_token_id] + ids


def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)

    pad_ids = []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]

        pad_ids.append(ids + add_ids)

    return torch.tensor(pad_ids)


def make_batch_roberta(sessions):
    '''
    制作batch数据
    :return:
    '''
    batch_input, batch_labels, batch_speaker_tokens = [], [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]

        context_speaker, context, emotion, sentiment = data

        inputString = ""
        speaker_turn = {}
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            if speaker not in speaker_turn.keys():  # v2统计同一说话人的if else
                speaker_turn[speaker] = 1
            else:
                speaker_turn[speaker] += 1
            # inputString += '<s>:'  # 730试试position
            inputString += utt + " "

        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind)

    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_labels


def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val


def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []

    # label arragne
    with torch.no_grad():
        for i_batch, data in tqdm(enumerate(dataloader), desc='testing is on...'):
            """Prediction"""
            batch_input_tokens, batch_labels = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()

            pred_logits = model(batch_input_tokens)  # (1, clsNum)

            """Calculation"""
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_labels.item()

            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct / len(dataloader)
    return acc, pred_list, label_list


class ft_model(nn.Module):
    def __init__(self):
        super(ft_model, self).__init__()
        self.context_model = AutoModel.from_pretrained(config['bert_path'])
        self.classifier = nn.Sequential(
            nn.Linear(config['emb_dim'], 300),
            nn.ReLU(),
            nn.Linear(300, config['n_class'])
        )

    def forward(self, batch_input_tokens):
        batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:, 0, :]
        logits = self.classifier(batch_context_output)
        return logits


if __name__ == '__main__':
    torch.cuda.empty_cache()
    make_batch = make_batch_roberta
    train_path = config['dataset_path'] + '/MELD_train.txt'
    dev_path = config['dataset_path'] + '/MELD_dev.txt'
    test_path = config['dataset_path'] + '/MELD_test.txt'
    train_dataset = MELD_loader(train_path, 'emotion')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch'], shuffle=True,
                                  num_workers=4, collate_fn=make_batch)
    dev_dataset = MELD_loader(dev_path, 'emotion')
    dev_dataloader = DataLoader(dev_dataset, batch_size=config['batch'], shuffle=False,
                                num_workers=4, collate_fn=make_batch)
    test_dataset = MELD_loader(test_path, 'emotion')
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch'], shuffle=False,
                                num_workers=4, collate_fn=make_batch)

    model = ft_model().cuda()
    model.train()

    # training process
    num_warmup_steps = len(train_dataset)
    num_training_steps = len(train_dataset) * config['epoch']
    train_sample_num = int(len(train_dataloader))
    print('warmup:{}, train_step:{}'.format(num_warmup_steps, num_training_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro,\
    best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0
    best_epoch = 0
    for epoch in range(config['epoch']):
        model.train()
        for i_batch, data in tqdm(enumerate(train_dataloader), desc='training is on ...'):
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break

            """Prediction"""
            batch_input_tokens, batch_labels = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()

            pred_logits = model(batch_input_tokens)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_labels)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config['max_grad_norm'])  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list,
                                                                         average='weighted')

        test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list,
                                                                            average='weighted')
        """Best Score & Model Save"""
        if test_fbeta > best_test_fscore:
            best_test_fscore = test_fbeta
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.context_model.state_dict(),
            }, config['saved_model'])
        print('epoch: {}, accuracy: {}, precision: {}, recall: {}, fscore: {}'.
              format(epoch + 1, test_acc, test_pre, test_rec, test_fbeta))
    print('Final Fscore ## test-fscore: {}, test_epoch: {}'.format(best_test_fscore, best_epoch))