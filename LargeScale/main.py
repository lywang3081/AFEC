import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import torch
from arguments import get_args
import random
import utils

import data_handler
from sklearn.utils import shuffle
import trainer
import networks

# Arguments

args = get_args()

##########################################################################################################################33
if args.trainer == 'afec_ewc' or args.trainer == 'ewc' or args.trainer == 'afec_rwalk' or args.trainer == 'rwalk' or args.trainer == 'afec_mas' or args.trainer == 'mas' or args.trainer == 'afec_si' or args.trainer == 'si':
    log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.dataset, args.trainer,args.seed, 
                                                                       args.lamb, args.lr, args.batch_size, args.nepochs)
elif args.trainer == 'gs':
    log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, 
                                                                                          args.dataset, 
                                                                                          args.trainer, 
                                                                                          args.seed, 
                                                                                          args.lamb, 
                                                                                          args.mu,
                                                                                          args.rho,
                                                                                          args.eta,
                                                                                          args.lr, 
                                                                                          args.batch_size, 
                                                                                          args.nepochs)

if args.output == '':
    args.output = './result_data/' + log_name + '.txt'
########################################################################################################################
# Seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not os.path.isdir('dat'):
    print('Make directory for dataset')
    os.makedirs('dat')

print('Load data...')
data_dict = None
dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
loader = dataset.loader
if args.dataset == 'CUB200' or 'ImageNet':
    data_dict = dataset.data_dict
taskcla = dataset.taskcla
print('\nTask info =', taskcla)

if not os.path.isdir('result_data'):
    print('Make directory for saving results')
    os.makedirs('result_data')
    
if not os.path.isdir('trained_model'):
    print('Make directory for saving trained models')
    os.makedirs('trained_model')

# Args -- Experiment
    
# Loader used for training data
shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)


train_dataset_loaders = data_handler.make_ResultLoaders(dataset.train_data,
                                                        dataset.train_labels,
                                                        taskcla,
                                                        transform=dataset.train_transform,
                                                        shuffle_idx = shuffle_idx,
                                                        loader = loader,
                                                        data_dict = data_dict,
                                                       )

test_dataset_loaders = data_handler.make_ResultLoaders(dataset.test_data,
                                                       dataset.test_labels,
                                                       taskcla,
                                                       transform=dataset.test_transform,
                                                       shuffle_idx = shuffle_idx,
                                                       loader = loader,
                                                       data_dict = data_dict,
                                                      )

# Get the required model
myModel = networks.ModelFactory.get_model(args.dataset, args.trainer, taskcla).cuda()

# Define the optimizer used in the experiment

optimizer = torch.optim.SGD(myModel.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

# Initilize the evaluators used to measure the performance of the system.
t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier")

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(myModel, args, optimizer, t_classifier, taskcla)

########################################################################################################################

utils.print_model_report(myModel)
utils.print_optimizer_config(optimizer)
print('-' * 100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
#kwargs = {'num_workers': 0, 'pin_memory': False}
kwargs = {'num_workers': 8, 'pin_memory': True}
for t, ncla in taskcla:
    print("tasknum:", t)
    # Add new classes to the train, and test iterator
    
    train_loader = train_dataset_loaders[t]
    test_loader = test_dataset_loaders[t]
    
    myTrainer.train(train_loader, test_loader, t)

    for u in range(t+1):
        
        test_loader = test_dataset_loaders[u]
        test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        test_loss, test_acc = t_classifier.evaluate(myTrainer.model, test_iterator, u)
        print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss, 100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss
    
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))
    
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')
    #torch.save(myModel.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))

    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)


print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')


