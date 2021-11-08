import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import utils
import torch
from arguments import get_args


tstart = time.time()

# Arguments

args = get_args()

##########################################################################################################################33

if args.approach == 'afec_ewc' or args.approach == 'ewc' or args.approach == 'afec_rwalk' or args.approach == 'rwalk' or args.approach == 'afec_mas' or args.approach == 'mas' or args.approach == 'afec_si' or args.approach == 'si' or args.approach == 'ft' or args.approach == 'random_init':
    log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed,
                                                                       args.lamb, args.lr, args.batch_size, args.nepochs)
elif args.approach == 'gs':
    log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment,
                                                                                          args.approach, args.seed, 
                                                                                          args.lamb, args.mu, args.rho,
                                                                                                   args.eta,
                                                                                          args.lr, args.batch_size, args.nepochs)


if args.output == '':
    args.output = './result_data/' + log_name + '.txt'
tr_output = './result_data/' + log_name + '_train' '.txt'
########################################################################################################################
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Args -- Experiment
if args.experiment == 'split_cifar100':
    from dataloaders import split_cifar100 as dataloader
if args.experiment == 'split_cifar100_SC':
    from dataloaders import split_cifar100_SC as dataloader
elif args.experiment == 'split_cifar10_100':
    from dataloaders import split_cifar10_100 as dataloader

# Args -- Approach
if args.approach == 'gs':
    from approaches import gs as approach
elif args.approach == 'afec_ewc':
    from approaches import afec_ewc as approach
elif args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'afec_si':
    from approaches import afec_si as approach
elif args.approach == 'si':
    from approaches import si as approach
elif args.approach == 'afec_rwalk':
    from approaches import afec_rwalk as approach
elif args.approach == 'rwalk':
    from approaches import rwalk as approach
elif args.approach == 'afec_mas':
    from approaches import afec_mas as approach
elif args.approach == 'mas':
    from approaches import mas as approach
elif args.approach == 'ft':
    from approaches import ft as approach

if args.experiment == 'split_cifar100' or args.experiment == 'split_cifar100_SC' or args.experiment == 'split_cifar10_100':
    from networks import conv_net as network

########################################################################################################################


# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed, tasknum=args.tasknum) # num_task is provided by dataloader
print('\nInput size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not os.path.isdir('result_data'):
    print('Make directory for saving results')
    os.makedirs('result_data')
    
if not os.path.isdir('trained_model'):
    print('Make directory for saving trained models')
    os.makedirs('trained_model')

net = network.Net(inputsize, taskcla).cuda()
net_emp = network.Net(inputsize, taskcla).cuda()
if 'afec' in args.approach:
    appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, empty_net = net_emp)
else:
    appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)

utils.print_model_report(net)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)
relevance_set = {}
# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    if t==1 and 'find_mu' in args.date:
        break
    
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    xtrain = data[t]['train']['x'].cuda()
    xvalid = data[t]['valid']['x'].cuda()

    ytrain = data[t]['train']['y'].cuda()
    yvalid = data[t]['valid']['y'].cuda()
    task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss
        
    # Save
    
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')
    #if args.approach != 'gs':
    #    torch.save(net.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))

    
# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        # save task names
        from copy import deepcopy

        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t, ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t] = deepcopy(acc[t, :])
            appr.logs['test_loss'][t] = deepcopy(lss[t, :])
        # pickle
        import gzip
        import pickle

        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################

