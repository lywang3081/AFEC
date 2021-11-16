import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
args = get_args()
import itertools


class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=2e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None,log_name = None, empty_net = None):
        self.model=model
        self.model_old=model
        self.model_emp = empty_net
        self.model_emp_tmp = empty_net
        self.model_pt = None

        self.fisher_emp = None

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min *1/3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.optimizer_emp = self._get_optimizer_emp()
        self.lamb = args.lamb
        self.lamb_emp = args.lamb_emp

        self.alpha = 0.9
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
        
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
        
        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

    def _get_optimizer_emp(self, lr=None):
        if lr is None: lr = self.lr
        optimizer = torch.optim.Adam(self.model_emp.parameters(), lr=lr)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        self.optimizer_emp = self._get_optimizer_emp(lr)
        self.add_emp = 0

        if t == 0:
            self.model_emp = deepcopy(self.model) #use the same initialization
            self.model_emp_tmp = deepcopy(self.model)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            num_batch = xtrain.size(0)

            #train the empty net and measure fim
            if t > self.add_emp-1:  #self.add_emp-1 ptemp

                self.train_emp_epoch(t, xtrain, ytrain, e)

                # freeze the empty net
                self.model_emp_tmp = deepcopy(self.model_emp)
                self.model_emp_tmp.train()
                utils.freeze_model(self.model_emp_tmp)

                # Fisher ops
                self.fisher_emp, _ = utils.fisher_matrix_diag_emp(t, xtrain, ytrain, self.model_emp, self.criterion)
            
            self.train_epoch(t,xtrain,ytrain)

            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')

            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
                    self.optimizer_emp = self._get_optimizer_emp(lr)
            print()
        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        utils.freeze_model(self.model_old) # Freeze the weights

        
        # Update fisher & s
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.fisher[n] = self.fisher_running[n].clone()
                    self.s[n] = (1/2) * self.s_running[n].clone()
                    self.s_running[n] = self.s[n].clone()

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward current model
            outputs = self.model.forward(images)[t]
            loss=self.ce(outputs,targets) #self.criterion(t,outputs,targets)

            if t > self.add_emp:
                loss_fg = self.criterion_fg(t)
                loss += loss_fg

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute Fisher & s
            self.update_fisher_and_s()
            
        return

    def train_emp_epoch(self, t, x, y, epoch):
        self.model_emp.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = x[b]
            targets = y[b]

            # train empty net
            # Forward current model
            outputs = self.model_emp.forward(images)[t]
            loss = self.ce(outputs, targets)

            # Backward
            self.optimizer_emp.zero_grad()
            loss.backward()
            self.optimizer_emp.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward
            output = self.model.forward(images)[t]
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (n,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum((self.fisher[n] + self.s[n])*(param_old-param).pow(2))
        return self.ce(output,targets)+self.lamb*loss_reg

    #fgmb
    def criterion_fg(self,t):
        # Regularization for all previous tasks
        loss_reg=0
        loss_reg_emp = 0

        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                if 'last' not in name:
                    loss_reg+=torch.sum((self.fisher[name] + self.s[name])*(param_old-param).pow(2))/2

            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_emp_tmp.named_parameters()):
                if 'last' not in name:
                    loss_reg_emp+=torch.sum(self.fisher_emp[name]*(param_old-param).pow(2))/2

        return self.lamb*loss_reg + self.lamb_emp*loss_reg_emp
    
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
                    s = loss_diff /(fisher_distance+self.eps)
                    self.s_running[n] = self.s_running[n] + s

                self.p_old[n] = p.detach().clone()
