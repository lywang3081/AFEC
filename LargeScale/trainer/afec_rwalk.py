from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer
from copy import deepcopy
import itertools

import networks


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla, model_emp=None):
        super().__init__(model, args, optimizer, evaluator, taskcla)

        self.lamb = args.lamb
        self.lamb_emp = args.lamb_emp
        self.lamb_emp_tmp = args.lamb_emp
        self.fisher_emp = None
        
        self.lamb=args.lamb
        self.alpha = 0.9
        self.s = {}
        self.s_running = {}
        self.fisher = {}
        self.fisher_running = {}
        self.p_old = {}

        self.eps = 0.01
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.s[n] = 0
                self.s_running[n] = 0
                self.fisher[n] = 0
                self.fisher_running[n] = 0
                self.p_old[n] = p.data.clone()

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def update_lr_emp(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer_emp.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("emp net: Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def setup_training_emp(self, lr):

        for param_group in self.optimizer_emp.param_groups:
            print("Setting LR to %0.4f in emp net" % lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def train(self, train_loader, test_loader, t):

        if t == 0:
            self.model_emp_tmp = deepcopy(self.model)
            self.model_emp = deepcopy(self.model)
            #self.model_emp_pt = deepcopy(self.model)
        #self.model_emp = deepcopy(self.model_emp_pt)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.decay)
        self.optimizer_emp = torch.optim.SGD(self.model_emp.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.decay)
        
        lr = self.args.lr
        
        self.setup_training(lr)
        # Do not update self.t
        if t>0:
            self.update_frozen_model()
            self.freeze_fisher_and_s()
        
        # Now, you can update self.t
        self.t = t
        kwargs = {'num_workers': 8, 'pin_memory': True}
        #kwargs = {'num_workers': 0, 'pin_memory': False}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True, **kwargs)

        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)

            #train the empty net and measure fim
            if t > -1:
                #train empty net
                for samples in tqdm(self.train_iterator):
                    data, target = samples
                    data, target = data.cuda(), target.cuda()
                    batch_size = data.shape[0]

                    output = self.model_emp(data)[t]
                    loss_CE = self.ce(output, target)

                    self.optimizer_emp.zero_grad()
                    (loss_CE).backward()
                    self.optimizer_emp.step()

                # freeze the empty net
                self.model_emp_tmp = deepcopy(self.model_emp)
                self.model_emp_tmp.eval()
                for param in self.model_emp_tmp.parameters():
                    param.requires_grad = False

                # Fisher ops
                self.fisher_emp = self.fisher_matrix_diag_emp()

            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.cuda(), target.cuda()

                output = self.model(data)[t]
                loss_CE = self.ce(output,target)

                loss = loss_CE

                if t > 0:
                    loss_fg = self.criterion_fg()

                    loss = loss_CE + loss_fg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_fisher_and_s()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
    
    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if self.t>0:
            for (n,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                loss_reg+=torch.sum((self.fisher[n] + self.s[n])*(param_old-param).pow(2))
        return self.ce(output,targets)+self.lamb*loss_reg

    def criterion_fg(self):
        # Regularization for all previous tasks
        loss_reg = 0
        loss_reg_emp = 0

        if self.t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                     self.model_fixed.named_parameters()):
                # if 'conv' in name:
                loss_reg += torch.sum((self.fisher[name] + self.s[name]) * (param_old - param).pow(2)) / 2

            for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                     self.model_emp_tmp.named_parameters()):
                # if 'conv' in name:
                loss_reg_emp += torch.sum( self.fisher_emp[name] * (param_old - param).pow(2)) / 2

        return self.lamb * loss_reg + self.lamb_emp * loss_reg_emp
    
    def update_fisher_and_s(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    # Compute running fisher
                    fisher_current = p.grad.data.pow(2)
                    self.fisher_running[n] = self.alpha*fisher_current + (1-self.alpha)*self.fisher_running[n]

                    # Compute running s
                    loss_diff = -p.grad * (p.detach() - self.p_old[n])
                    fisher_distance = (1/2) * (self.fisher_running[n]*(p.detach() - self.p_old[n])**2)
                    s = loss_diff / (fisher_distance+self.eps)
                    self.s_running[n] = self.s_running[n] + s

                self.p_old[n] = p.detach().clone()

    def freeze_fisher_and_s(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.fisher[n] = self.fisher_running[n].clone()
                    self.s[n] = (1/2) * self.s_running[n].clone()
                    self.s_running[n] = self.s[n].clone()

    def fisher_matrix_diag_emp(self):
        # Init
        fisher = {}
        for n, p in self.model_emp.named_parameters():
            fisher[n] = 0 * p.data
        # Compute
        self.model_emp.eval()
        criterion = torch.nn.CrossEntropyLoss()
        for samples in tqdm(self.fisher_iterator):
            data, target = samples
            data, target = data.cuda(), target.cuda()

            # Forward and backward
            self.model_emp.zero_grad()
            outputs = self.model_emp.forward(data)[self.t]
            loss = self.ce(outputs, target)
            loss.backward()

            # Get gradients
            for n, p in self.model_emp.named_parameters():
                if p.grad is not None:
                    fisher[n] += self.args.batch_size * p.grad.data.pow(2)
        # Mean
        with torch.no_grad():
            for n, _ in self.model_emp.named_parameters():
                fisher[n] = fisher[n] / len(self.train_iterator)
        self.model_emp.train()
        return fisher