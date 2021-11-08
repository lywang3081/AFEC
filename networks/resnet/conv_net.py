import sys
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, droprate=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.drop = nn.Dropout(p=droprate) if droprate > 0 else None
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        x = self.bn1(x)
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn2(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, block=PreActBlock, num_blocks=[2, 2, 2, 2], num_classes=10, in_channels=3):
        super().__init__()
        #super(PreActResNet, self).__init__()
        self.in_planes = 64
        last_planes = 512*block.expansion

        ncha,size,_=inputsize
        self.taskcla = taskcla

        self.conv1 = conv3x3(in_channels, 64)
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_last = nn.BatchNorm2d(last_planes)
        #self.last = nn.Linear(last_planes, num_classes)  # last layer

        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(last_planes, n, bias=False))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return out

    def forward(self, x):
        x = self.features(x)
        x = self.bn_last(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)

        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x.view(x.size(0), -1)))
        return y


def resnet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=10)

'''
class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.conv1(x))
        act2=self.relu(self.conv2(act1))
        h=self.drop1(self.MaxPool(act2))
        act3=self.relu(self.conv3(h))
        act4=self.relu(self.conv4(act3))
        h=self.drop1(self.MaxPool(act4))
        act5=self.relu(self.conv5(h))
        act6=self.relu(self.conv6(act5))
        h=self.drop1(self.MaxPool(act6))
        h=h.view(x.shape[0],-1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))

        return y
'''