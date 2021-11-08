import torch
import torch.nn as nn
import torchvision


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, taskcla):
        super(AlexNet, self).__init__()
        self.taskcla = taskcla
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(4096,n))
            
        self.smid=6
        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),192)
        self.ec3=torch.nn.Embedding(len(self.taskcla),384)
        self.ec4=torch.nn.Embedding(len(self.taskcla),256)
        self.ec5=torch.nn.Embedding(len(self.taskcla),256)
        
        self.efc1=torch.nn.Embedding(len(self.taskcla),4096)
        self.efc2=torch.nn.Embedding(len(self.taskcla),4096)

    def forward(self,x,t,s, mask_return=False):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gc4,gc5,gfc1,gfc2=masks
        # Gated
        x = self.maxpool(self.relu(self.conv1(x)))
        x=x*gc1.view(1,-1,1,1).expand_as(x)
        x = self.maxpool(self.relu(self.conv2(x)))
        x=x*gc2.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.conv3(x))
        x=x*gc3.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.conv4(x))
        x=x*gc4.view(1,-1,1,1).expand_as(x)
        x = self.maxpool(self.relu(self.conv5(x)))
        x=x*gc5.view(1,-1,1,1).expand_as(x)
        
        x = torch.flatten(x, 1)
        x=self.dropout(self.relu(self.fc1(x)))
        x=x*gfc1.expand_as(x)
        x=self.dropout(self.relu(self.fc2(x)))
        x=x*gfc2.expand_as(x)
        
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))
        
        if mask_return:
            return y,masks
        return y
    
    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        gc5=self.gate(s*self.ec5(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gc4,gc5,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gc4,gc5,gfc1,gfc2=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc6.data.view(-1,1,1).expand((self.ec6.weight.size(1),
                                              self.smid,
                                              self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        elif n=='c4.weight':
            post=gc4.data.view(-1,1,1,1).expand_as(self.c4.weight)
            pre=gc3.data.view(1,-1,1,1).expand_as(self.c4.weight)
            return torch.min(post,pre)
        elif n=='c4.bias':
            return gc4.data.view(-1)
        elif n=='c5.weight':
            post=gc5.data.view(-1,1,1,1).expand_as(self.c5.weight)
            pre=gc4.data.view(1,-1,1,1).expand_as(self.c5.weight)
            return torch.min(post,pre)
        elif n=='c5.bias':
            return gc5.data.view(-1)
        elif n=='c6.weight':
            post=gc6.data.view(-1,1,1,1).expand_as(self.c6.weight)
            pre=gc5.data.view(1,-1,1,1).expand_as(self.c6.weight)
            return torch.min(post,pre)
        elif n=='c6.bias':
            return gc6.data.view(-1)
        return None


def alexnet(taskcla, pretrained=False):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(taskcla)
    
    if pretrained:
        pre_model = torchvision.models.alexnet(pretrained=True)
        for key in model.state_dict().keys():
            print(key)
        for key in pre_model.state_dict().keys():
            print(key)
        for key1, key2 in zip(model.state_dict().keys(), pre_model.state_dict().keys()):
            if 'last' in key1:
                break
            if model.state_dict()[key1].shape == torch.tensor(1).shape:
                model.state_dict()[key1] = pre_model.state_dict()[key2]
            else:
                model.state_dict()[key1][:] = pre_model.state_dict()[key2][:]
    
    return model

