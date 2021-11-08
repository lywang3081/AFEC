import sys
import torch
import torch.nn as nn
from utils import *

class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        
        ncha,size,_=inputsize #28
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3)
        s = compute_conv_output_size(size,3) #26
        self.conv2 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #24
        s = s//2 #12
        self.conv3 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #10
        self.conv4 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #8
        s = s//2 #4
        
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(s*s*64,n)) #4*4*64 = 1024
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.conv1(x))
        act2=self.relu(self.conv2(act1))
        h=self.MaxPool(act2)
        act3=self.relu(self.conv3(h))
        act4=self.relu(self.conv4(act3))
        h=self.MaxPool(act4)
        h=h.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        self.grads={}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
            return hook
        
        if avg_act == True:
            names = [0, 1, 2, 3]
            act = [act1, act2, act3, act4]
            
            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))
        
        return y