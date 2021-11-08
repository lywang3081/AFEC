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
args = get_args()

if 'omniglot' in args.experiment:
    from networks.conv_net_omniglot import Net
elif 'mixture' in args.experiment:
    from networks.alexnet import Net
else:
    from networks.conv_net import Net


class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None, log_name = None):
        self.model=model
        self.model_old=model
        self.omega=None
        self.log_name = log_name

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=args.lamb 
        self.initail_mu = args.mu
        self.mu = args.mu
        self.freeze = {}
        self.mask = {}
        
        for (name,p) in self.model.named_parameters():
            if len(p.size())<2:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            self.mask[name] = torch.zeros(p.shape[0])

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):

        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        if t>0:
            self.freeze = {}
            for name, param in self.model.named_parameters():
                if 'bias' in name or 'last' in name:
                    continue
                key = name.split('.')[0]
                if 'conv1' not in name:
                    if 'conv' in name: #convolution layer
                        temp = torch.ones_like(param)
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp
                    else:#linear layer
                        temp = torch.ones_like(param)
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0) , -1))
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)
                prekey = key
                
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()

            # CUB 200 xtrain_cropped = crop(x_train)
            num_batch = xtrain.size(0)

            self.train_epoch(t,xtrain,ytrain,lr)

            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            
            # Adapt lr
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
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model.act = None

        temp=utils.gs_cal(t,xtrain,ytrain,self.criterion, self.model)  #calculate omega

        for n in temp.keys():
            if t>0:
                self.omega[n] = args.eta * self.omega[n] + temp[n]  #update old omega, equ 8
            else:
                self.omega = temp
            self.mask[n] = (self.omega[n]>0).float()   #mask important node for omega > 0
            
        torch.save(self.model.state_dict(), './trained_model/' + self.log_name + '_task_{}.pt'.format(t))
        
        test_loss, test_acc = self.eval(t, xvalid, yvalid)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc))
        
        dummy = Net(input_size, taskcla).cuda()

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
                    
                    if len(weight.size()) > 2:
                        norm = weight.norm(2,dim=(1,2,3))
                        mask = (self.omega[name]==0).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    else:
                        norm = weight.norm(2,dim=(1))
                        mask = (self.omega[name]==0).float().unsqueeze(-1)

                    zero_cnt = int((mask.sum()).item())
                    indice = np.random.choice(range(zero_cnt), int(zero_cnt*(1-args.rho)), replace=False)
                    indice = torch.tensor(indice).long()
                    idx = torch.arange(weight.shape[0])[mask.flatten(0)==1][indice]
                    mask[idx] = 0

                    layer.weight.data = (1-mask)*layer.weight.data + mask*dummy_layer.weight.data
                    mask = mask.squeeze()
                    layer.bias.data = (1-mask)*bias + mask*dummy_layer.bias.data

                    pre_name = name

                if isinstance(layer, nn.ModuleList):
                    
                    weight = layer[t].weight
                    if 'omniglot' in args.experiment:
                        weight = weight.view(weight.shape[0], self.omega[pre_name].shape[0], -1)
                        weight[:,self.omega[pre_name] == 0] = 0
                        weight = weight.view(weight.shape[0],-1)
                    else:
                        weight[:, self.omega[pre_name] == 0] = 0
        test_loss, test_acc = self.eval(t, xvalid, yvalid)
        
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        utils.freeze_model(self.model_old) # Freeze the weights
        return

    def train_epoch(self,t,x,y,lr):
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
            loss=self.criterion(t,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Freeze the outgoing weights
            if t>0:
                for name, param in self.model.named_parameters():
                    if 'bias' in name or 'last' in name or 'conv1' in name:
                        continue
                    key = name.split('.')[0]
                    param.data = param.data*self.freeze[key]

        self.proxy_grad_descent(t,lr)   #add regularization, that is approximated
        
        return

    def eval(self,t,x,y):
        with torch.no_grad():
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
    
    def proxy_grad_descent(self, t, lr):
        with torch.no_grad():
            for (name,module),(_,module_old) in zip(self.model.named_children(),self.model_old.named_children()):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue
                
                mu = self.mu
                
                key = name
                weight = module.weight
                bias = module.bias
                weight_old = module_old.weight
                bias_old = module_old.bias
                
                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1,2,3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm**2 + bias**2).pow(1/2)                

                aux = F.threshold(norm - mu * lr, 0, 0, False)
                alpha = aux/(aux+mu*lr)
                coeff = alpha * (1-self.mask[key])   #mask is omega

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1) 
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0

                if t>0:
                    if len(weight.size()) > 2:
                        norm = (weight - weight_old).norm(2, dim=(1,2,3))
                    else:
                        norm = (weight - weight_old).norm(2, dim=(1))

                    norm = (norm**2 + (bias-bias_old)**2).pow(1/2)

                    aux = F.threshold(norm - self.omega[key]*self.lamb*lr, 0, 0, False)  #delta_theta
                    boonmo = lr*self.lamb*self.omega[key] + aux
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

                diff_weight = (sparse_weight + penalty_weight) - weight.data
                diff_bias = sparse_bias + penalty_bias - bias.data
                

                weight.data = sparse_weight + penalty_weight
                bias.data = sparse_bias + penalty_bias

        return

    def criterion(self,t,output,targets):
        return self.ce(output,targets)
