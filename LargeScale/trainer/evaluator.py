import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from arguments import get_args
args = get_args()


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="trainedClassifier"):
        if testType == "trainedClassifier":
            return softmax_evaluator()


class softmax_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self):
        self.ce=torch.nn.CrossEntropyLoss()
    
    def evaluate(self, model, iterator, t):
        with torch.no_grad():
            total_loss=0
            total_acc=0
            total_num=0
            model.eval()

            # Loop batches
            for data, target in iterator:
                data, target = data.cuda(), target.cuda()
                
                if args.trainer == 'hat':
                    task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)
                    output = model(data,task,args.smax)[t]
                else:
                    output = model(data)[t]
                loss=self.ce(output,target)
                _,pred=output.max(1)
                hits=(pred==target).float()

                # Log
    
                total_loss+=loss.data.cpu().numpy()*data.shape[0]
                total_acc+=hits.sum().data.cpu().numpy()
                total_num+=data.shape[0]

            return total_loss/total_num,total_acc/total_num
