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
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
        self.lamb = args.lamb
        self.lamb_emp = args.lamb_emp
        self.lamb_emp_tmp = args.lamb_emp
        self.lamb_emp_fo = args.lamb_emp_fo

        self.fisher_emp = None
        
    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("full net: Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
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
            print("Setting LR to %0.4f in full net"%lr)
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
            self.model_emp = deepcopy(self.model)
            self.model_emp_tmp = deepcopy(self.model)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.decay)
        self.optimizer_emp = torch.optim.SGD(self.model_emp.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.decay)

        lr = self.args.lr
        self.setup_training(lr)
        self.setup_training_emp(lr)

        # Do not update self.t
        if t>0:
            self.update_frozen_model()
            self.update_fisher()
        
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
            self.update_lr_emp(epoch, self.args.schedule)

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
                self.fisher_emp, _ = self.fisher_matrix_diag_emp()

            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.cuda(), target.cuda()
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.ce(output,target)
                loss = loss_CE

                if t > 0:
                    loss_fg = self.criterion_fg()
                    loss = loss_CE + loss_fg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.ce(output,targets)+self.lamb*loss_reg

    def criterion_fg(self):
        # Regularization for all previous tasks
        loss_reg=0
        loss_reg_emp = 0

        if self.t > 0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                if 'conv' in name:
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_emp_tmp.named_parameters()):
                if 'conv' in name:
                    loss_reg_emp+=torch.sum(self.fisher_emp[name]*(param_old-param).pow(2))/2

        return self.lamb*loss_reg + self.lamb_emp*loss_reg_emp
    
    def fisher_matrix_diag(self):
        # Init
        fisher={}
        for n,p in self.model.named_parameters():
            fisher[n]=0*p.data
        # Compute
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for samples in tqdm(self.fisher_iterator):
            data, target = samples
            data, target = data.cuda(), target.cuda()

            # Forward and backward
            self.model.zero_grad()
            outputs = self.model.forward(data)[self.t]
            loss=self.criterion(outputs, target)
            loss.backward()
            
            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n]+=self.args.batch_size*p.grad.data.pow(2)
        # Mean
        with torch.no_grad():
            for n,_ in self.model.named_parameters():
                fisher[n]=fisher[n]/len(self.train_iterator)
        return fisher

    def fisher_matrix_diag_emp(self):
        # Init
        fisher = {}
        for n, p in self.model_emp.named_parameters():
            fisher[n] = 0 * p.data

        fo = {}
        for n, p in self.model_emp.named_parameters():
            fo[n] = 0 * p.data
        # Compute
        self.model_emp.eval()

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
                    fo[n] += self.args.batch_size * p.grad.data  # first order gradient

        # Mean
        with torch.no_grad():
            for n, _ in self.model_emp.named_parameters():
                fisher[n] = fisher[n] / len(self.train_iterator)
                fo[n] = fo[n] / len(self.train_iterator)

        self.model_emp.train()
        return fisher, fo

    def update_fisher(self):
        if self.t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=self.fisher_matrix_diag()
        if self.t>0:
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*self.t)/(self.t+1)
