import copy
from arguments import get_args
args = get_args()
import torch

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args, optimizer, evaluator, taskcla):
        
        if args.trainer == 'ewc':
            import trainer.ewc as trainer
        elif args.trainer == 'afec_ewc':
            import trainer.afec_ewc as trainer
        elif args.trainer == 'afec_mas':
            import trainer.afec_mas as trainer
        elif args.trainer == 'mas':
            import trainer.mas as trainer
        elif args.trainer == 'afec_rwalk':
            import trainer.afec_rwalk as trainer
        elif args.trainer == 'rwalk':
            import trainer.rwalk as trainer
        elif args.trainer == 'afec_si':
            import trainer.afec_si as trainer
        elif args.trainer == 'si':
            import trainer.si as trainer
        elif args.trainer == 'gs':
            import trainer.gs as trainer
        return trainer.Trainer(myModel, args, optimizer, evaluator, taskcla)
    
class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, model, args, optimizer, evaluator, taskcla):
        
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.evaluator=evaluator
        self.taskcla=taskcla
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.current_lr = args.lr
        self.ce=torch.nn.CrossEntropyLoss()
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None