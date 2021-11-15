# [AFEC: Active Forgetting of Negative Transfer in Continual Learning]() 

------
This code is the official implementation of our paper, which will be published in NeurIPS2021.

## **Execution Details**

### Requirements

- Python 3
- GPU 1080Ti / Pytorch 1.3.1+cu9.2 / CUDA 9.2

### Execution command
We provide a demo command to run AFEC on visual classification tasks. 
To reproduce other baselines and the adaptation of AFEC to representative weight regularization approaches, 
please check arguments.py for the command, and Appendix  C.1 (Table.4) for the hyperparameters.

For small-scale images:

```
# CIFAR-100-SC
$ python3 ./main.py --experiment split_cifar100_SC --approach afec_ewc --lamb 40000 --lamb_emp 1

# CIFAR-100
$ python3 ./main.py --experiment split_cifar100 --approach afec_ewc --lamb 10000 --lamb_emp 1

# CIFAR-10/100
$ python3 ./main.py --experiment split_cifar10_100 --approach afec_ewc --lamb 25000 --lamb_emp 1

```

For large-scale images:

```
$ cd LargeScale

# CUB-200
$ python3 ./main.py --dataset CUB200 --trainer afec_ewc --lamb 40 --lamb_emp 0.001

# ImageNet-100
$ python3 ./main.py --dataset ImageNet --trainer afec_ewc --lamb 80 --lamb_emp 0.001

```

## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{wang2021afec,
  title={AFEC: Active Forgetting of Negative Transfer in Continual Learning},
  author={Wang, Liyuan and Zhang, Mingtian and Jia, Zhongfan and Li, Qian and Bao, Chenglong and Ma, Kaisheng and Zhu, Jun and Zhong, Yi},
  journal={arXiv preprint arXiv:2110.12187},
  year={2021}
}
```
