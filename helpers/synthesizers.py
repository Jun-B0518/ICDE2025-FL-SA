import copy
from abc import ABC, abstractclassmethod
from typing import Dict

import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
# import ipdb
from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from helpers.utils import ImagePool, ImagePoolPro, DeepInversionHook, average_weights, kldiv, test, test_pro, test_pro_e
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import os

upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)


class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)

class Ensemble_A(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble_A, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e

class Ensemble_E(torch.nn.Module):
    def __init__(self, model_list, label_logit_weights, model_logit_weights):
        super(Ensemble_E, self).__init__()
        self.models = model_list
        self.label_logit_weights = label_logit_weights
        self.model_logit_weights = model_logit_weights

    def set_batch_targets(self, targets):
        self.targets = targets
        V = np.array([])
        U = np.array(self.label_logit_weights)
        for s in range(len(targets)):
            V = np.vstack((V, U[targets[s]]))
        self.V = V

    def compute_bn_loss(self, hooks):
        m_bn_loss = []
        for idx, hs in hooks.items():
            loss_bn = sum([h.r_feature for h in hs])
            m_bn_loss.append(loss_bn)

        return sum(m_bn_loss)/len(m_bn_loss)


    def forward(self, x):
        P = []
        for k in range(len(self.models)):
            logits = self.models[k](x)
            P.append(logits)

        s = x.shape[0]
        P_b = np.array([])
        for i in range(s):
            P_a = np.array([])
            for k in range(len(self.models)):
                P_a = P[k][i]
                P_b = self.V[i] @ P_a

        P_b = torch.tensor(P_b).cuda().to(torch.float32)

        return P_b

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class SASynthesizer():
    def __init__(self, teacher, model_list, student, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, save_dir, dataset, test_dataloader, args):
        super(SASynthesizer, self).__init__()
        self.student = student
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        # self.data_pool = ImagePool(root=self.save_dir)
        self.data_pool = ImagePoolPro(root=self.save_dir)
        self.data_iter = None
        self.teacher = teacher
        self.dataset = dataset
        self.test_dataloader = test_dataloader

        self.generator = generator.cuda().train()
        self.generator_list = [copy.deepcopy(self.generator) for i in range(len(model_list))]
        self.strong_class_list = [[] for i in range(len(model_list))]    # record strong class list for each client
        self.model_list = model_list
        self.label_logit_weights = [[1 for _ in range(self.num_classes)] for _ in range(len(model_list))]
        self.model_logit_weights = [[1 for _ in range(self.num_classes)] for _ in range(len(model_list))]

        self.args = args

        self.sythesis_acc_num = 0   # record the total number of correctly evaluated generated smaples

        os.makedirs('./lw', exist_ok=True)
        self.logit_weight_file = os.path.join('./lw', self.args.timestr)

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])



        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        self.access_teacher_models()

        self.en_teacher = Ensemble_E(self.model_list, self.label_logit_weights, self.model_logit_weights)

    def access_teacher_models(self):

        # if client models have been evaluated
        if not os.path.exists(self.logit_weight_file):
            for id in range(len(self.generator_list)):
                self.teacher.select_client(id)
                self.interview(self.teacher.models[id], id)

            with open(self.logit_weight_file, 'w') as file:
                for row in self.label_logit_weights:
                    file.write(' '.join(map(str, row)) + '\n')

    def gen_data(self, cur_ep):
        self.synthesize(self.en_teacher, cur_ep)


    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

    def eval_test(self, model, test_data):
        model.eval()
        test_loss = 0
        correct = 0
        predictions = []  # List to store prediction labels


        test_data = test_data.cuda()
        output = model(test_data)

        # pred = torch.max(output, 1)[1]

        probabilities = F.softmax(output, dim=-1)

        predicted_labels = torch.argmax(probabilities, dim=-1).tolist()

        # Append prediction labels to the list
        predictions.extend(predicted_labels)  # Convert to CPU and to a NumPy array

        return predictions

    def interview(self, net, userid):
        net.eval()

        label_batch_loss = {u: [] for u in range(self.num_classes)}
        label_var_degree = []

        with tqdm(range(self.num_classes)) as tl:
            for label in tl:
                best_cost = 1e6
                best_inputs = None
                z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()  #
                targets = torch.randint(low=label, high=label+1, size=(self.synthesis_batch_size,))
                targets = targets.sort()[0]
                targets = targets.cuda()
                reset_model(self.generator_list[userid])
                optimizer = torch.optim.Adam([{'params': self.generator_list[userid].parameters()}, {'params': [z]}], self.lr_g,
                                             betas=[0.5, 0.999])
                hooks = []

                for m in net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        hooks.append(DeepInversionHook(m))
                        # print(f'm: {m}')

                with tqdm(range(self.iterations)) as t:
                    for it in t:
                        optimizer.zero_grad()
                        # optimizer_mlp.zero_grad()
                        inputs = self.generator_list[userid](z)  # bs,nz
                        # global_view = inputs
                        global_view, _ = self.aug(inputs)  # crop and normalize
                        #############################################
                        t_out = net(global_view)

                        loss_oh = F.cross_entropy(t_out, targets)  # ce_loss

                        s_out = self.student(global_view)
                        mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()

                        loss = self.oh * loss_oh
                        # loss = loss_inv
                        if best_cost > loss.item() or best_inputs is None:
                            best_cost = loss.item()
                            best_inputs = inputs.data
                        # record the batch loss
                        label_batch_loss[label].append(loss.item())

                        loss.backward()
                        optimizer.step()
                        # optimizer_mlp.step()
                        t.set_description('[interview]=>label:{}, iters:{}, loss:{}, ce:{}'.format(label, it, loss.item(), loss_oh))
                    # vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)


                var_degree = (max(label_batch_loss[label])-min(label_batch_loss[label]))/min(label_batch_loss[label])
                label_var_degree.append(var_degree)

        self.label_logit_weights[userid] = label_var_degree


    def synthesize(self, net, cur_ep):
        net.cuda()
        net.eval()
        best_cost = 1e6
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))

        targets = targets.cuda()

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        #############################################
        dim_in = 500 if "cifar100" == self.dataset else 50


        hooks = {idx:[] for idx in range(len(net.models))}
        for subidx, subnet in enumerate(net.models):
            for m in subnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    hooks[subidx].append(DeepInversionHook(m))


        net.set_batch_targets(targets)

        with tqdm(range(self.iterations)) as t:
            for it in t:
                optimizer.zero_grad()
                # optimizer_mlp.zero_grad()
                inputs = self.generator(z)  # bs,nz
                global_view = inputs
                # Gate
                t_out = net(global_view)

                loss_bn = net.compute_bn_loss(hooks)
                loss_oh = F.cross_entropy(t_out, targets)

                s_out = self.student(global_view)
                mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation

                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

                loss.backward()
                optimizer.step()
                # optimizer_mlp.step()
                t.set_description('iters:{}, loss:{}, ce:{}, bn:{}， adv:{}'.format(it, loss.item(), loss_oh, loss_bn, loss_adv))
            vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)


        self.data_pool.add(best_inputs, targets.cpu().numpy())