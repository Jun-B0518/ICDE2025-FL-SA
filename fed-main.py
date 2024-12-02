#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
# import osmo
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
# import pdb
import datetime
import os

from helpers.datasets import partition_data
from helpers.synthesizers import SASynthesizer
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, KLCom,setup_seed, test
from models.generator import Generator
from models.nets import CNNMnist
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model, client_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)

        local_acc_list = []
        for iter in tqdm(range(self.args.local_ep)):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                # ---------------------------------------
                output = model(images)
                loss = F.cross_entropy(output, labels)
                # ---------------------------------------
                loss.backward()
                optimizer.step()
            acc, test_loss = test(model, test_loader)

            local_acc_list.append(acc)
        return model.state_dict(), np.array(local_acc_list)


import logging
def get_logger(logpath):
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    formatter1 = logging.Formatter('%(filename)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter1)

    formatter2 = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
    # os.makedirs(logpath, exist_ok=True)
    file_handler = logging.FileHandler(logpath)
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter2)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s', level=logging.DEBUG,
    #                     filename=logpath, filemode='a')

    return logger

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dense', help='dense or dense_pro')

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--kl_alpha1', default=0, type=float)
    parser.add_argument('--kl_alpha2', default=1, type=float)

    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')

    parser.add_argument('--pretrainid', type=str, default='')
    parser.add_argument('--timestr', type=str, default='')
    parser.add_argument('--gpus', default='0,')
    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e

class EnsemblePro(torch.nn.Module):
    def __init__(self, model_list, id=-1):
        super(EnsemblePro, self).__init__()
        self.models = model_list
        self.seleted_index = id   # -1 means all

    def select_client(self, id=-1):
        print(f'selected model id : {id}')
        assert -1 <= self.seleted_index <= len(model_list)
        self.seleted_index = id

    def forward(self, x):
        if self.seleted_index == -1:
            logits_total = 0
            for i in range(len(self.models)):
                logits = self.models[i](x)
                logits_total += logits
            logits_e = logits_total / len(self.models)

            return logits_e
        else:
            assert 0 <= self.seleted_index <= len(model_list)
            return self.models[self.seleted_index](x)


def kd_train(synthesizer, model, criterion, optimizer, args=None):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda()
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            args.flogger.info(f'[{idx}/{len(epochs)}: avg loss[{avg_loss}], acc[{acc}]')

            epochs.set_description(description.format(avg_loss, acc))

def sa_kd_train(synthesizer, model, criterion, optimizer, args=None):
    student, teacher = model
    student.train()
    teacher.cuda()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0

    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images, targets) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda()
            teacher.set_batch_targets(targets)
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            args.flogger.info(f'[{idx}/{len(epochs)}: avg loss[{avg_loss}], acc[{acc}]')

            epochs.set_description(description.format(avg_loss, acc))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    else:
        pass

    return global_model


if __name__ == '__main__':

    args = args_parser()

    setup_seed(args.seed)
    # pdb.set_trace()
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL
    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    if args.type == "pretrain":
        # ===============================================
        for idx in range(args.num_users):
            users.append("client_{}".format(idx))
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx)

            acc_list.append(local_acc)
            local_weights.append(copy.deepcopy(w))

        pkl_path = './pkl/'
        os.makedirs(pkl_path, exist_ok=True)
        torch.save(local_weights, os.path.join(pkl_path, f'{args.timestr}-{args.dataset}_{args.num_users}clients_{args.beta}.pkl'))
        # update global weights by FedAvg
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        test_acc, test_loss = test(global_model, test_loader)

    else:
        # ===============================================
        pkl_path = './pkl/'
        local_weights = torch.load(os.path.join(pkl_path, f'{args.timestr}-{args.dataset}_{args.num_users}clients_{args.beta}.pkl'))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        # print("avg acc:")
        test_acc, test_loss = test(global_model, test_loader)

        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)

        ensemble_model = Ensemble(model_list)

        # print("ensemble acc:")
        ens_acc, ens_loss = test(ensemble_model, test_loader)
        args.flogger.info(f"ensemble acc:{ens_acc}, ensemble_loss:{ens_loss}")
        # ===============================================
        global_model = get_model(args)
        # ===============================================

        # data generator
        nz = args.nz
        nc = 1
        img_size = 28
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        args.cur_ep = 0
        img_size2 = (1, 28, 28)
        num_class = 10
        args.save_dir = f'run/{args.dataset}/{args.timestr}/{args.algorithm}-iid({args.beta})-C{args.num_users}-Eps{args.local_ep}-KLEps{args.epochs}-gsteps{args.g_steps}-nz{args.nz}-glce{args.oh}-glbn{args.bn}-gladv{args.adv}-kla1{args.kl_alpha1}-kla2{args.kl_alpha2}'

        synthesizer = SASynthesizer(ensemble_model, model_list, global_model, generator,
                                     nz=nz, num_classes=num_class, img_size=img_size2,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size,
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir=args.save_dir, dataset=args.dataset, test_dataloader=test_loader, args=args)

        criterion = KLCom(T=args.T, alpha1=args.kl_alpha1, alpha2=args.kl_alpha2,reduction='mean')
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []
        for epoch in tqdm(range(args.epochs)):
            # 1. Data synthesis
            synthesizer.gen_data(args.cur_ep)  # g_steps
            args.cur_ep += 1

            sa_kd_train(synthesizer, [global_model, synthesizer.en_teacher], criterion, optimizer,
                     args=args)  # # kd_steps

            acc, test_loss = test(global_model, test_loader)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)
            os.makedirs('df_ckpt', exist_ok=True)
            _best_ckpt = f'df_ckpt/{args.timestr}-{args.other}.pth'
            args.flogger.info("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)

        np.save("distill_acc_{}.npy".format(args.dataset), np.array(distill_acc))



