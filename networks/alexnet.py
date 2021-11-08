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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(4096,n))

    def forward(self, x, avg_act = False):
        act1 = self.relu(self.conv1(x))
        x = self.maxpool(act1)
        act2 = self.relu(self.conv2(x))
        x = self.maxpool(act2)
        act3 = self.relu(self.conv3(x))
        act4 = self.relu(self.conv4(act3))
        act5 = self.relu(self.conv5(act4))
        x = self.maxpool(act5)
        
        x = torch.flatten(x, 1)
        act6 = self.relu(self.fc1(self.dropout(x)))
        act7 = self.relu(self.fc2(self.dropout(act6)))
        
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](act7))
            
        self.grads={}
        self.act = []
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
            return hook
        
        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]
            
            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))
        
        return y


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

