# Assignment 2 : Convolutional Neural Network
 
 This repository contains project code for Machine Learning Lesson (EE369). 
 This code is based on [kuangliu's repository](https://github.com/kuangliu/pytorch-cifar) and use [pytorch-summary](https://github.com/sksq96/pytorch-summary). Here we apply some famous convolutional neural networks (CNN) and perform them on CIFAR-10 dataset. We also test the influence of [Random-Erasing-Data-Augmentation](https://arxiv.org/abs/1708.04896).

## 1. Setup

In order to run the code, I recommend Python 3.5 or higher. The code is based on PyTorch 0.4.1. I am sure that something will go wrong when the code works with different versions for PyTorch such as 1.0.0. The code does not work with Python 2.7.

### Requirements

#### General
- Python (verified on 3.6.2)
- cuda 8.0 or higher

#### Python Packages
- numpy (verified on 1.12.1)
- pandas (verified on 0.20.3)
- matplotlib (verified on 2.1.1)
- pytorch (__only verified on 0.4.1__)
- torchvision (verified on )
- sklearn (verified on 0.19.0)
- pytorch-summary ([pytorch-summary](https://github.com/sksq96/pytorch-summary))

### Setup with anaconda 

Setup a Python virtual environment (optional):

```
conda create -n m_env python=3.6.2
source activate m_env
```

Install the requirements:

```
conda install pip
pip install -r requirements.txt
git clone https://github.com/sksq96/pytorch-summary
python pytorch-summary/setup.py install
```

or

```
conda install matplolib
conda install pytorch==0.4.1 torchvision -c pytorch
git clone https://github.com/sksq96/pytorch-summary
python pytorch-summary/setup.py install
```

If everything works well, you can run `python main.py` without any other arguments for default to train a LeNet on CIFAR-10. 

## 2. General Description of Scripts

### Run a script with user-friendly command-line

```
python [script name].py -h
```

All the scripts use `argparse` module to make it easy to write user-friendly command-line interfaces. You can use option `-h` or `--help` to get a useful usage message for each script.

## 3. Detailed Documentation

Detailed documentation and examples of how to use the scripts are given below.

|  Directory or file name  |               description                   |
| ------------------------ |:-------------------------------------------:|
|    ./data/               | directory to store dataset                  |
|    ./checkpoint/         | directory to store checkpoint during training |
|    ./logs/               | directory to store runtime logs             |
|    ./results/            | directory to store result infomation (include graphs, reports, and json)       |
|    ./nets/               | contains all network constructors           | 
|    ./utils/              | contains some helper scripts                |

### main.py

This script will train a CNN on CIFAR-10 dataset. 

```
usage: main.py [-h] [-d [D]] [-l [L]] [-c [C]] [-r [R]]
               [-m [{VGG19,LeNet,PreActResNet18,MobileNetV2,WRN_28_10}]]
               [--depth [DEPTH]] [--widen-factor [WIDEN_FACTOR]] [--rt]
               [--p [P]] [--sl [SL]] [--sh [SH]] [--r1 [R1]]
               [--train-batch [TRAIN_BATCH]] [--test-batch [TEST_BATCH]]
               [--max-epoch [MAX_EPOCH]] [--early-stopping [EARLY_STOPPING]]
               [--drop [DROP]] [--lr [LR]] [--momentum [MOMENTUM]]
               [--weight-decay [WEIGHT_DECAY]]
               [--schedule [epoch [epoch ...]]] [--gamma [GAMMA]]

PyTorch CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  -d [D]                directory to store dataset. Default './data'
  -l [L]                path to save log. Default
                        'log_<model_name>_<datetime>.txt'
  -c [C]                path to save checkpoint. Default
                        'ckpt_<model_name>_<datetime>.t7'
  -r [R]                path to latest checkpoint. Default None
  -m [{VGG19,LeNet,PreActResNet18,MobileNetV2,WRN_28_10}]
                        Model name to apply. Default WRN_28_10
  --depth [DEPTH]       Model depth. Default 28
  --widen-factor [WIDEN_FACTOR]
                        Widen factor. Default 10
  --rt                  Specify to apply Random Erasing
  --p [P]               Random Erasing - probability. Default 0.5
  --sl [SL]             Random Erasing -min erasing area. Default 0.02
  --sh [SH]             Random Erasing - max erasing area. Default 0.4
  --r1 [R1]             Random Erasing - aspect of erasing area. Default 0.3
  --train-batch [TRAIN_BATCH]
                        train batchsize. Default 128
  --test-batch [TEST_BATCH]
                        test batchsize. Default 100
  --max-epoch [MAX_EPOCH]
                        max epoch, default is 300
  --early-stopping [EARLY_STOPPING]
                        early stopping patience, default is 10
  --drop [DROP]         dropout ratio. Default 0.0
  --lr [LR]             initial learning rate. Default 0.1
  --momentum [MOMENTUM]
                        momentum. Default 0.9
  --weight-decay [WEIGHT_DECAY]
                        weight decay. Default 1e-4
  --schedule [epoch [epoch ...]]
                        decrease learning rate at these epochs. Default [150,
                        225]
  --gamma [GAMMA]       LR is multiplied by gamma on schedule. Default 0.1
```

Example：

 To train a VGG19 with Random-Erasing:

 ```
python -m main -m VGG19 --rt
 ```

 To resume the training (of whose checkpoint save as filename ckpt_LeNet_20181212_173035.t7)

 ```
python -m main -m LeNet -r ckpt_LeNet_20181212_173035.t7
 ```

### svm.py

This script will run SVM classifier on CIFAR-10 dataset. 

```
usage: svm.py [-h] [-d [D]] [--pca_percent [PCA_PERCENT]] [--svm_c [SVM_C]]
              [--svm_kernel [{linear,poly,rbf}]] [--svm_gamma [SVM_GAMMA]]
              [--svm_degree [SVM_DEGREE]] [--svm_coef0 [SVM_COEF0]] [--output]
              [--outfile [OUTFILE]]

optional arguments:
  -h, --help            show this help message and exit
  -d [D]                directory to store dataset. Default './data'
  --pca_percent [PCA_PERCENT]
                        How much variance in percent to retain by setting
                        number of components in PCA. Default = 0.8
  --svm_c [SVM_C]       Penalty parameter C of the error term. Default = 5.0
  --svm_kernel [{linear,poly,rbf}]
                        Specifies the kernel type to be used in the algorithm.
                        Default = rbf
  --svm_gamma [SVM_GAMMA]
                        Kernel coefficient for ‘rbf’ and ‘poly’. Default =
                        0.025
  --svm_degree [SVM_DEGREE]
                        Degree of the polynomial kernel function (‘poly’).
                        Ignored by all other kernels. Default = 9
  --svm_coef0 [SVM_COEF0]
                        Independent term of the polynomial kernel function
                        (‘poly’). Ignored by all other kernels. Default = 1
  --output              Whether to print the result report to file.
  --outfile [OUTFILE]   File to save the result report. Default =
                        './results/svm/report.txt'
```

Example：

 Run SVM with `linear kernel`, `C = 2.0` and store the result report in `./results/test_report.txt`.

```
python -m svm --svm_kernel linear --svm_c 2.0 --output --outfile ./results/test_report.txt
```
