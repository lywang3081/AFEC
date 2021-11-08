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

import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
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

        
    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def train(self, train_loader, test_loader, t):
        
        lr = self.args.lr
        
        self.setup_training(lr)
        # Do not update self.t
        if t>0:
            self.update_frozen_model()
            self.freeze_fisher_and_s()
        
        # Now, you can update self.t
        self.t = t
        #kwargs = {'num_workers': 8, 'pin_memory': True}
        kwargs = {'num_workers': 0, 'pin_memory': False}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.cuda(), target.cuda()

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target)

                self.optimizer.zero_grad()
                (loss_CE).backward()
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
    
