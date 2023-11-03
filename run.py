import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, argparse, time, pickle, random
import torch
from model import *
import json
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from dataloader import *
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

seed = 100

import logging
torch.set_printoptions(profile='full')


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_or_eval_model(model, loss_function, dataloader, epoch, alpha, threshold, cuda, args, optimizer=None, train=False, scheduler=None):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    assert not train or optimizer != None
    if train:
        model.train()
        ttype = 'training'
        # dataloader = tqdm(dataloader)
    else:
        model.eval()
        ttype = 'testing'

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        features, label, adj, s_mask, tokens, lengths, speakers, utterances, entro = data
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask = s_mask.cuda()
            lengths = lengths.cuda()
            entro = entro.cuda()

        log_prob = model(features, s_mask, entro, alpha, threshold)
        # crf_layer = CRF(log_prob.size()[-1], batch_first=True)

        # loss2 = -crf_layer(log_prob, label, mask=crfmask)
        # loss1 = loss_function(log_prob.permute(0, 2, 1), label)
        # multiloss = AutomaticWeightedLoss(num=2)
        # loss = multiloss(loss1, loss2)

        loss = loss_function(log_prob.permute(0, 2, 1), label)

        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim=2).cpu().numpy().tolist()
        # pred = crf_layer.decode(log_prob)
        pred = torch.Tensor(pred).cpu().numpy().tolist()
        preds += pred
        labels += label
        losses.append(loss.item())

        if train:
            loss_val = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

    if preds != []:
        new_preds = []
        new_labels = []
        for i, label in enumerate(labels):
            for j, l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_micro_fscore, avg_macro_fscore


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    path = './saved_models/'
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_path', type=str, default='./roberta-large')
    parser.add_argument('--save_path', type=str, default='./saved_models/meld_mymodel917.pth')

    parser.add_argument('--bert_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='IEMOCAP', type=str,
                        help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=120, metavar='E', help='number of epochs')

    parser.add_argument('--finetune', action='store_true', default=False, help='do not finetune')

    args = parser.parse_args()
    print(args)

    seed_everything()

    args.cuda = torch.cuda.is_available() and not args.no_cuda

    if args.cuda:
        act_device = 'GPU'
        print("\033[1;31;40mRunning on GPU\033[0m")
    else:
        act_device = 'CPU'
        print("\033[1;31;40mRunning on CPU\033[0m")

    logger = get_logger(path + args.dataset_name + '/logging.log')
    logger.info('start training on {} {}!'.format(act_device, os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    train_loader, valid_loader, test_loader, speaker_vocab, \
    label_vocab, person_vec = get_DailyDialog_loaders(dataset_name=args.dataset_name,
                                                  batch_size=batch_size,
                                                  num_workers=0, args=args)
    n_classes = len(label_vocab['itos'])

    print('building model..')

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    # crf_layer = CRF(n_classes, batch_first=True)
    # optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = None

    best_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 0.
    best_fscore = 0.
    alpha = 0.1
    threshold = 2.0
    ret_dict = {}
    best_model = None
    model = myModel(args, n_classes)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    for e in range(n_epochs):
        start_time = time.time()
        if args.dataset_name == 'DailyDialog':
            train_loss, train_acc, _, _, train_micro_fscore, train_macro_fscore = train_or_eval_model(model,
                                                                                                    loss_function,
                                                                                                    train_loader, e,
                                                                                                    cuda,
                                                                                                    args, optimizer,
                                                                                                    True, scheduler)
            valid_loss, valid_acc, _, _, valid_micro_fscore, valid_macro_fscore = train_or_eval_model(model,
                                                                                                    loss_function,
                                                                                                    valid_loader, e,
                                                                                                    cuda, args, scheduler)
            test_loss, test_acc, test_label, test_pred, \
            test_micro_fscore, test_macro_fscore = train_or_eval_model(model, loss_function, test_loader, e, cuda, args, scheduler)

            all_acc.append([valid_acc, test_acc])
            all_fscore.append([valid_micro_fscore, test_micro_fscore, valid_macro_fscore, test_macro_fscore])

            logger.info(
                'Epoch: {}, train_loss: {}, train_acc: {}, train_micro_fscore: {}, train_macro_fscore: {}, valid_loss: {}, valid_acc: {}, valid_micro_fscore: {}, valid_macro_fscore: {}, test_loss: {}, test_acc: {}, test_micro_fscore: {}, test_macro_fscore: {}, time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_micro_fscore, train_macro_fscore, valid_loss, valid_acc,
                    valid_micro_fscore, valid_macro_fscore, test_loss, test_acc,
                    test_micro_fscore, test_macro_fscore, round(time.time() - start_time, 2)))

        else:
            train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model, loss_function,
                                                                            train_loader, e, alpha, threshold, cuda,
                                                                            args, optimizer, True, scheduler)
            valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model, loss_function,
                                                                            valid_loader, e, alpha, threshold, cuda, args, scheduler)
            test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_model(model, loss_function,
                                                                                          test_loader, e, alpha, threshold, cuda, args, scheduler)
        # save the best model
        # if best_fscore < test_fscore:
        #     best_fscore = test_fscore
        #     best_epoch = e + 1
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #     }, args.save_path)

            all_acc.append([valid_acc, test_acc])
            all_fscore.append([valid_fscore, test_fscore])

            logger.info('Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                        format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc,
                            valid_fscore, test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        e += 1
    logger.info('finish training!')

    print('Test performance..')
    all_acc = sorted(all_acc, key=lambda x: (x[0], x[1]), reverse=True)
    all_fscore = sorted(all_fscore, key=lambda x: (x[0], x[1]), reverse=True)

    if args.dataset_name == 'DailyDialog':
        logger.info('Best micro/macro F-Score based on validation:{}/{}'.format(all_fscore[0][1], all_fscore[0][3]))
        all_fscore = sorted(all_fscore, key=lambda x: x[1], reverse=True)
        logger.info('Best micro/macro F-Score based on test:{}/{}'.format(all_fscore[0][1], all_fscore[0][3]))
    else:
        logger.info('Best F-Score based on validation:{}'.format(all_fscore[0][1]))
        logger.info('Best Accuracy based on test:{}'.format(max([f[1] for f in all_acc])))
        logger.info('Best F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))
