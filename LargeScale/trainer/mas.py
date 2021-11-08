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
        self.omega = {}
        for n,_ in self.model.named_parameters():
            self.omega[n] = 0

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
            self.omega_update()
        
        # Now, you can update self.t
        self.t = t
        #kwargs = {'num_workers': 8, 'pin_memory': True}
        kwargs = {'num_workers': 0, 'pin_memory': False}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        self.omega_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True, **kwargs)
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


            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
        
        
    
    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                loss_reg+=torch.sum(self.omega[name]*(param_old-param).pow(2))/2
            
        return self.ce(output,targets)+self.lamb*loss_reg    
        
    
    def omega_update(self):
        sbatch = 20
        
        # Compute
        self.model.train()
        for samples in tqdm(self.omega_iterator):
            data, target = samples
            data, target = data.cuda(), target.cuda()
            # Forward and backward
            self.model.zero_grad()
            outputs = self.model.forward(data)[self.t]

            # Sum of L2 norm of output scores
            loss = torch.sum(outputs.norm(2, dim = -1))
            loss.backward()

            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    self.omega[n]+= p.grad.data.abs() / len(self.train_iterator)

        return 

