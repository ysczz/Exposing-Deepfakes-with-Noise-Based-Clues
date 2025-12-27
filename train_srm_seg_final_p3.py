# train.py
#!/usr/bin/env	python3

import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import config as config
from dataset import Dataset_srm_seg
from sklearn import metrics
from sklearn.metrics import auc
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from facol_loss import FocalLoss
from net.mutilsource_srm_seg_final_p3 import SRM

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)
seed_it(1314)

def train(epoch, result_file, writer):
    start = time.time()
    model.train()
    train_loss = 0.0
    train_corrects = 0.0
    val_loss = 0.0
    val_corrects = 0.0
    for batch_index, (images, gts, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            labels = labels.cuda()
            images = images.cuda()
            gts = gts.cuda()

        optimizer.zero_grad()
        x, seg = model(images)

        loss = LOSS_FOCAL(x, labels) + LOSS_L1(gts, seg)
        preds = torch.max(x, dim=1)[1]
        loss.backward()
        optimizer.step()

        iter_loss = loss.data.item()
        train_loss += iter_loss
        iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
        train_corrects += iter_corrects
        if not (batch_index % 20):
            print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(batch_index, iter_loss / config.BATCH_SIZE, iter_corrects / config.BATCH_SIZE), file=result_file, flush=True)
        
        n_iter = epoch  * len(train_loader) + batch_index + 1

        #update training loss for each iteration
        writer.add_scalar('Train_iteration/loss', loss.item(), n_iter)
        writer.add_scalar('Train_iteration/acc', iter_corrects / len(labels), n_iter)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start), file=result_file, flush=True)
    epoch_loss = train_loss / len(train_loader.dataset)
    epoch_acc = train_corrects / len(train_loader.dataset)
    print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), file=result_file, flush=True)
    print('epoch train time: {}'.format(finish - start), file=result_file, flush=True)
    writer.add_scalar('Train_epoch/loss', epoch_loss, epoch)
    writer.add_scalar('Train_epoch/acc', epoch_acc, epoch)

@torch.no_grad()
def eval_training(epoch, result_file):
    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label_set = []
    score_set = []

    for (images, labels) in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            
        x, seg = model(images)
        loss = LOSS_FOCAL(x, labels)
        preds = torch.max(x, dim=1)[1]
        scores = F.softmax(x,dim=1)[:,1]
        test_loss += loss.item()
        correct += preds.eq(labels).sum()
        label_set = np.concatenate([label_set,  labels.cpu().numpy()], axis=0)
        score_set = np.concatenate([score_set,  scores.cpu().numpy()], axis=0)
    finish = time.time()
    FPR, TPR, THRESH = metrics.roc_curve(label_set, score_set, pos_label=1, drop_intermediate=False)
    roc_auc = metrics.auc(FPR, TPR)

    print('val set: Epoch: {},Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(label_set),
        correct.float() / len(label_set),
        roc_auc,
        finish - start
    ), file=result_file, flush=True)
    print('', file=result_file, flush=True)

    return correct.float() / len(label_set), roc_auc

@torch.no_grad()
def eval_generate_training(epoch, datatype, result_file, writer):
    model.eval()
    print('###############################   generation test   #######################################', file=result_file, flush=True)
    print('', file=result_file, flush=True)
    print('', file=result_file, flush=True)
    # dataset_list = [datatype, 'celeb_v1', 'celeb_v2']
    dataset_list = [datatype]
    for dataset_type in dataset_list:
        print(dataset_type, file=result_file, flush=True)
        test_dataset = Dataset_srm_seg('test', dataset_type)
        valid_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=8,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        start = time.time()
        test_loss = 0.0 # cost function error
        correct = 0.0
        label_set = []
        score_set = []
        for (images, labels) in valid_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            x, seg = model(images)
            loss = LOSS_FOCAL(x, labels)
            preds = torch.max(x, dim=1)[1]
            scores = F.softmax(x,dim=1)[:,1]
            test_loss += loss.item()
            correct += preds.eq(labels).sum()
            label_set = np.concatenate([label_set,  labels.cpu().numpy()], axis=0)
            score_set = np.concatenate([score_set,  scores.cpu().numpy()], axis=0)
        finish = time.time()
        FPR, TPR, THRESH = metrics.roc_curve(label_set, score_set, pos_label=1, drop_intermediate=False)
        roc_auc = metrics.auc(FPR, TPR)
        print('test set: Epoch: {},Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(label_set),
            correct.float() / len(label_set),
            roc_auc,
            finish - start
        ), file=result_file, flush=True)
        print('', file=result_file, flush=True)
        writer.add_scalar('test/'+ dataset_type + '/acc', correct.float() / len(label_set), epoch)
        writer.add_scalar('test/'+ dataset_type + '/auc', roc_auc, epoch)


if __name__ == '__main__':
    print('##cuda:',torch.cuda.is_available())
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_name', type=str, default='noise_srm_seg_final_p3_c23', help='what are u doning')
    parser.add_argument('--net', type=str, default='', help='network name')
    parser.add_argument('--dataset', type=str, default='ffpp_c23', help='ffpp_c0, ffpp_c23, ffpp_c40, celeb_v1...')
    parser.add_argument('--focal_weight', type=float, default=0.75, help='frozen or not')
    parser.add_argument('--ifswap', type=int, default=1, help='frozen or not')
    parser.add_argument('--ifargu', type=int, default=1, help='frozen or not')
    args = parser.parse_args()

    # todo result_path
    result_path = os.path.join(config.logDir, args.project_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_file = open(os.path.join(result_path, 'result.txt'), 'a+')
    
    model = SRM().cuda()
    print(model, file=result_file, flush=True)

    train_dataset = Dataset_srm_seg('train',args.dataset)

    eval_dataset = Dataset_srm_seg('eval',args.dataset)

    #data preprocessing:
    train_loader = DataLoader(dataset=train_dataset,  num_workers=8, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    test_loader = DataLoader(dataset=eval_dataset,  num_workers=8, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)


    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    LOSS_CSE = nn.CrossEntropyLoss().cuda()
    LOSS_L1 = nn.L1Loss().cuda()
    LOSS_FOCAL = FocalLoss(gamma=2.0, alpha=0.75, size_average=False).cuda()

    # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, args.project_name)
    # create checkpoint folder to save models
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{dataset}-{epoch}-{type}.pth')
    writer = SummaryWriter(log_dir=os.path.join(config.logDir, args.project_name, 'log'))
    best_acc = 0.0
    best_auc = 0.0
    for epoch in range(0, config.MAX_EPOCHS + 1):
        train(epoch, result_file, writer)
        acc, auc= eval_training(epoch, result_file)
        writer.add_scalar('val_epoch/acc', acc, epoch)
        writer.add_scalar('val_epoch/auc', auc, epoch)

        #start to save best performance model after learning rate decay to 0.01
        type_str= ''
        if  best_acc < acc or best_auc < auc:
            type_str = 'best'
            if best_acc < acc:
                type_str = type_str + '_acc'
                best_acc = acc
            if best_auc < auc:
                type_str = type_str + '_auc'
                best_auc = auc
            print('############  ',type_str,'  #############', file=result_file, flush=True)
            print('acc', best_acc, file=result_file, flush=True)
            print('auc', best_auc,file=result_file, flush=True)
            print('', file=result_file, flush=True)
            print('',file=result_file, flush=True)
        weights_path = checkpoint_path.format(dataset=args.dataset, epoch=epoch, type=type_str)
        print('saving weights file to {}'.format(weights_path) , file=result_file, flush=True)
        torch.save(model.state_dict(), weights_path)
        eval_generate_training(epoch, args.dataset, result_file, writer)
