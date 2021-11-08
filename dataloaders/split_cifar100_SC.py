import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0,pc_valid=0.10, tasknum = 20):
    data={}
    taskcla=[]
    size=[3,32,32]
    tasknum = 20

    if not os.path.isdir('../dat/binary_split_cifar100_5_spcls/'):
        os.makedirs('../dat/binary_split_cifar100_5_spcls')

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        superclass = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        
        # CIFAR100
        dat={}
        
        dat['train']=datasets.CIFAR100('../dat/',train=True,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('../dat/',train=False,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for n in range(tasknum):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']= 5
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}

        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                task_idx = superclass[target]
                #task_idx = target.numpy()[0] // 5     #num_task
                #print("task_idx", task_idx)
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0])

        for s in ['train','test']:
            for task_idx in range(tasknum): #20 tasks
                unique_label_t = np.unique(data[task_idx][s]['y'])
                #print("unique_label_t", unique_label_t)
                for i in range(len(data[task_idx][s]['y'])):
                    for j in range(len(unique_label_t)):
                        if data[task_idx][s]['y'][i] == unique_label_t[j]:
                            #print('data[task_idx][s][y][i]', data[task_idx][s]['y'][i])
                            data[task_idx][s]['y'][i] = j
                            #print('data[task_idx][s][y][i]', data[task_idx][s]['y'][i])

        # "Unify" and save
        for t in range(tasknum):
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('../dat/binary_split_cifar100_5_spcls'),
                                                         'data'+str(t+1)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../dat/binary_split_cifar100_5_spcls'),
                                                         'data'+str(t+1)+s+'y.bin'))
    
    # Load binary files
    data={}
    data[0] = dict.fromkeys(['name','ncla','train','test'])
    ids=list(shuffle(np.arange(tasknum),random_state=seed)+1)
    print('Task order =',ids)
    for i in range(tasknum):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100_5_spcls'),
                                                    'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100_5_spcls'),
                                                    'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i-1])
            
    # Validation
    for t in range(tasknum):
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in range(tasknum):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
