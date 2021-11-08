import torch
import torch.nn as nn
import torchvision


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Net(nn.Module):

    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()
        self.taskcla = taskcla
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.c2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(4096,n))
            
            
        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),192)
        self.ec3=torch.nn.Embedding(len(self.taskcla),384)
        self.ec4=torch.nn.Embedding(len(self.taskcla),256)
        self.ec5=torch.nn.Embedding(len(self.taskcla),256)
        self.efc1=torch.nn.Embedding(len(self.taskcla),4096)
        self.efc2=torch.nn.Embedding(len(self.taskcla),4096)
        
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.ec4.weight.data.uniform_(lo,hi)
        self.ec5.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gc4,gc5,gfc1,gfc2=masks
        
        # Gated
        h=self.relu(self.c1(x))
        h=self.maxpool(h)
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        
        h=self.relu(self.c2(h))
        h=self.maxpool(h)
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        
        h=self.relu(self.c3(h))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c4(h))
        h=h*gc4.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c5(h))
        h=self.maxpool(h)
        h=h*gc5.view(1,-1,1,1).expand_as(h)
        
        h=h.view(x.shape[0],-1)
        
        h=self.dropout(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.dropout(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
            
        
        return y,masks
    
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
            pre=gc5.data.view(-1,1,1).expand((self.ec5.weight.size(1),
                                              1,
                                              1)).contiguous().view(1,-1).expand_as(self.fc1.weight)
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
        for key1, key2 in zip(model.state_dict().keys(), pre_model.state_dict().keys()):
            if 'last' in key1:
                break
            if model.state_dict()[key1].shape == torch.tensor(1).shape:
                model.state_dict()[key1] = pre_model.state_dict()[key2]
            else:
                model.state_dict()[key1][:] = pre_model.state_dict()[key2][:]
    
    return model


