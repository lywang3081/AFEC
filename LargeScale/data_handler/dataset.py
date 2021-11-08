from torchvision import datasets, transforms
import torch
import numpy as np
from arguments import get_args
import math
import time
args = get_args()


class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name, tasknum):
        self.classes = classes
        self.name = name
        self.tasknum = tasknum
        self.train_data = None
        self.test_data = None
        self.loader = None

class CUB200(Dataset):
    def __init__(self):
        super().__init__(200, "CUB200", args.tasknum)
        
        mean = [0.485, 0.500, 0.432]
        std = [0.232, 0.227, 0.266]
        
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        
        print('Load start!')
        clock1 = time.time()
        data = datasets.ImageFolder("../dat/CUB_200_2011/images", transform=self.train_transform)
        self.loader = data.loader
        
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.data_dict = {}
        class_cnt = [0]*200
        class_num = [
            60, 60, 58, 60, 44, 41, 53, 48, 59, 60, 60, 56, 60, 60, 58, 58, 57, 45, 59, 59,
            60, 56, 59, 52, 60, 60, 60, 59, 60, 60, 60, 53, 59, 59, 60, 60, 59, 60, 59, 60,
            60, 60, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 58, 60, 59,
            60, 60, 60, 60, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 57, 60, 60, 59, 60, 60,
            60, 60, 60, 53, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 59, 60, 60, 60,
            50, 60, 60, 60, 49, 60, 59, 60, 60, 60, 60, 60, 50, 60, 59, 60, 59, 60, 59, 60,
            60, 60, 60, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 59, 60, 60, 60, 60, 60,
            58, 60, 60, 60, 60, 60, 60, 60, 59, 60, 51, 60, 59, 60, 60, 60, 59, 60, 60, 59,
            60, 60, 60, 60, 60, 59, 60, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 56, 59, 60,
            59, 60, 60, 60, 60, 60, 50, 60, 60, 58, 60, 60, 60, 60, 60, 59, 60, 60, 60, 60
        ]
        for i in range(len(data.imgs)):
            path, target = data.imgs[i]
#             self.data_dict[path] = self.loader(path)
            if class_cnt[target] < math.ceil(class_num[target]*0.8):
                self.train_data.append(path)
                self.train_labels.append(target)
            else:
                self.test_data.append(path)
                self.test_labels.append(target)
            class_cnt[target] += 1
        
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        
        self.taskcla = []
        
        clock2 = time.time()
        print('Load finished!')
        print('Time elapse: %d'%(clock2-clock1))
        
        for t in range(self.tasknum):
            self.taskcla.append((t, self.classes // self.tasknum))


class ImageNet(Dataset):
    def __init__(self):
        super().__init__(1000, "ImageNet", args.tasknum)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        traindir = '/data/LargeData/Large/ImageNet/train'
        valdir = '/data/LargeData/Large/ImageNet/val'

        self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])

        self.test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,])

        trainset = datasets.ImageFolder(traindir, self.train_transform)

        testset = datasets.ImageFolder(valdir, self.train_transform)

        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.data_dict = {}

        for i in range(len(trainset.imgs)):
            path, target = trainset.imgs[i]
            self.train_data.append(path)
            self.train_labels.append(target)

        for i in range(len(testset.imgs)):
            path, target = testset.imgs[i]
            self.test_data.append(path)
            self.test_labels.append(target)

        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)


        self.loader = trainset.loader
        self.taskcla = []

        for t in range(self.tasknum):
            self.taskcla.append((t, self.classes // self.tasknum))

        print('Load finished!')
