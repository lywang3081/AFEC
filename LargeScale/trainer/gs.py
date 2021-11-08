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

import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
        self.lamb=args.lamb 
        self.mu = args.mu
        
        # Initialize mask dictionary
        self.mask = {}
        self.freeze = {}
        for (name,p) in self.model.named_parameters():
            if len(p.size())<2:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            self.mask[name] = torch.zeros(p.shape[0])
                
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
            self.omega_update()
            self.reinitialization()
            self.update_frozen_model()
            self.update_freeze()
        
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
                
                if t>0:
                    for name, param in self.model.named_parameters():
                        if 'bias' in name or 'last' in name or 'conv1' in name:
                            continue
                        key = name.split('.')[0]
                        param.data = param.data*self.freeze[key]


            self.proxy_grad_descent()
            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
    
    def criterion(self,output,targets):
        # Regularization for all previous tasks
        return self.ce(output,targets)
    
    def update_freeze(self):
        self.freeze = {}
        
        key = 0
        prekey = 0
        for name, param in self.model.named_parameters():
            with torch.no_grad():
                if 'bias' in name or 'last' in name:
                    continue
                key = name.split('.')[0]
                if 'conv1' not in name:
                    temp = torch.ones_like(param)
                    if 'conv' in name:
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp
                    else: 
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0) , -1))
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)
                prekey = key
                    
    def reinitialization(self):
        t = self.t
        dummy = networks.ModelFactory.get_model(self.args.dataset, self.args.trainer, self.taskcla).cuda()

        name = 0
        pre_name = 0
        
        for (name,dummy_layer),(_,layer) in zip(dummy.named_children(), self.model.named_children()):
            with torch.no_grad():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    if pre_name!=0:
                        temp = (self.omega[pre_name]>0).float()
                        if isinstance(layer, nn.Linear) and 'conv' in pre_name:
                            temp = temp.unsqueeze(0).unsqueeze(-1)
                            weight = layer.weight
                            weight = weight.view(weight.size(0), temp.size(1), -1)
                            weight = weight * temp
                            layer.weight.data = weight.view(weight.size(0), -1)
                        elif len(weight.size())>2:
                            temp = temp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            layer.weight *= temp
                        else:
                            temp = temp.unsqueeze(0)
                            layer.weight *= temp

                    weight = layer.weight.data
                    bias = layer.bias.data
                    norm = weight.view(weight.shape[0],-1).norm(2,dim=1)
                    if len(weight.size()) > 2:
                        mask = (self.omega[name]==0).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    else:
                        mask = (self.omega[name]==0).float().unsqueeze(-1)
                    
                    zero_cnt = int((mask.sum()).item())
                    indice = np.random.choice(range(zero_cnt), int(zero_cnt*(1-self.args.rho)), replace=False)
                    indice = torch.tensor(indice).long()
                    idx = torch.arange(weight.shape[0])[mask.flatten(0)==1][indice]
                    mask[idx] = 0
                    
                    layer.weight.data = (1-mask)*layer.weight.data + mask*dummy_layer.weight.data
                    mask = mask.squeeze()
                    layer.bias.data = (1-mask)*bias + mask*dummy_layer.bias.data
                    pre_name = name
                if isinstance(layer, nn.ModuleList):
                    
                    weight = layer[t].weight
                    weight[:, self.omega[pre_name] == 0] = 0
    
    def proxy_grad_descent(self):
        t = self.t
        lr = self.current_lr
        mu = self.args.mu
        lamb = self.args.lamb
        with torch.no_grad():
            for (name,module),(_,module_old) in zip(self.model.named_children(),self.model_fixed.named_children()):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue
                key = name
                weight = module.weight
                bias = module.bias
                weight_old = module_old.weight
                bias_old = module_old.bias
                norm = weight.view(weight.shape[0],-1).norm(2, dim=1)
                norm = (norm**2 + bias**2).pow(1/2)                

                aux = F.threshold(norm - mu * lr, 0, 0, False)
                alpha = aux/(aux+mu*lr)
                coeff = alpha * (1-self.mask[key])

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1) 
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0

                if t>0:
                    norm = (weight - weight_old).view(weight.shape[0],-1).norm(2,dim=1)
                    norm = (norm**2 + (bias-bias_old)**2).pow(1/2)
                    
                    aux = F.threshold(norm - self.omega[key]*lamb*lr, 0, 0, False)
                    boonmo = lr*lamb*self.omega[key] + aux
                    alpha = (aux / boonmo)
                    alpha[alpha!=alpha] = 1
                        
                    coeff_alpha = alpha * self.mask[key]
                    coeff_beta = (1-alpha) * self.mask[key]


                    if len(weight.size()) > 2:
                        penalty_weight = coeff_alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*weight.data + \
                                            coeff_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*weight_old.data
                    else:
                        penalty_weight = coeff_alpha.unsqueeze(-1)*weight.data + coeff_beta.unsqueeze(-1)*weight_old.data
                    penalty_bias = coeff_alpha*bias.data + coeff_beta*bias_old.data

                weight.data = sparse_weight + penalty_weight
                bias.data = sparse_bias + penalty_bias
            
        return

    def omega_update(self):
        temp=self.cal_omega()
#         temp=self.cal_omega_grad()
        for n in temp.keys():
            if self.t>0:
                self.omega[n] = self.args.eta * self.omega[n]+temp[n] #+ t* temp[n]*mask      # Checked: it is better than the other option
            else:
                self.omega = temp
            self.mask[n] = (self.omega[n]>0).float()
            
    def cal_omega(self):
        # Init
        param_R = {}
        for name, param in self.model.named_parameters():
            if len(param.size()) <= 1:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param = param.view(param.size(0), -1)
            param_R[name]=torch.zeros((param.size(0)))

        # Compute
        self.model.train()
        total = 0
        for samples in tqdm(self.omega_iterator,desc='Omega update'):
            data, target = samples
            data, target = data.cuda(), target.cuda()
            total += data.shape[0]

            # Forward and backward
            outputs = self.model.forward(data, True)[self.t]
            cnt = 0
            for idx, (act, key) in enumerate(zip(self.model.act, param_R.keys())):
                act = torch.mean(act, dim=0) # average N samples
                if len(act.size())>1:
                    act = torch.mean(act.view(act.size(0), -1), dim = 1).abs()
                self.model.act[idx] = act
                
            for name, param in self.model.named_parameters():
                if len(param.size()) <= 1 or 'last' in name:
                    continue
                name = name.split('.')[:-1]
                name = '.'.join(name)
                param_R[name] += self.model.act[cnt].abs().detach()*data.shape[0]
                cnt+=1 

        with torch.no_grad():
            for key in param_R.keys():
                param_R[key]=(param_R[key]/total)
        return param_R

