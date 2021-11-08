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

        self.epsilon=0.01
        self.omega = {}
        self.W = {}
        self.p_old = {}

        n=0

        # Register starting param-values (needed for “intelligent synapses”).
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr, self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def update_lr_emp(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer_emp.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("emp net: Changing learning rate from %0.4f to %0.4f" % (self.current_lr, self.current_lr * self.args.gammas[temp]))
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
        self.setup_training_emp(lr)
        
        # Do not update self.t
        if t>0:
            self.update_frozen_model()
            self.update_omega()
        
        # Now, you can update self.t
        self.W = {}
        self.p_old = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

        self.t = t
        kwargs = {'num_workers': 8, 'pin_memory': True}
        #kwargs = {'num_workers': 0, 'pin_memory': False}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True, **kwargs)
        
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.vae.train()
            self.update_lr(epoch, self.args.schedule)
            self.update_lr_emp(epoch, self.args.schedule)

            #train the empty net and measure fim
            if t > 0: #-1:
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
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                        self.p_old[n] = p.detach().clone()

                del loss, data, target

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
        
        
    
    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if self.t>0:
            loss_reg=self.surrogate_loss()

        return self.ce(output,targets)+self.lamb*loss_reg

    def criterion_fg(self):
        # Regularization for all previous tasks
        loss_reg = 0
        loss_reg_emp = 0
        if self.t > 0:
            loss_reg += self.surrogate_loss()

            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_emp_tmp.named_parameters()):
                # if 'conv' in name:
                loss_reg_emp += torch.sum(self.fisher_emp[name] * (param_old - param).pow(2)) / 2

        return self.lamb * loss_reg + self.lamb_emp * loss_reg_emp

    def update_omega(self):
        """After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)"""

        # Loop over all parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = self.W[n] / (p_change ** 2 + self.epsilon)
                try:
                    omega = getattr(self.model, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.model.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        """Calculate SI’s surrogate loss"""
        try:
            losses = []
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.model, '{}_SI_prev_task'.format(n))
                    omega = getattr(self.model, '{}_SI_omega'.format(n))
                    # Calculate SI’s surrogate loss, sum over all parameters
                    n_tmp = n.replace('__', '.')
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return 0.

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