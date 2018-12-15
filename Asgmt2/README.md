# Assignment 1 : Classifiers on MNIST
 
 This repository contains project code for Machine Learning Lesson (EE369). Here we apply some classifiers (SVM, KNN, LR) on MNIST dataset.

## 1. Requirements
#### General
- Python (verified on 3.6.1)

#### Python Packages
- numpy (verified on 1.12.1)
- pandas (verified on 0.20.3)
- numexpr (verified on 2.6.2)
- matplotlib (verified on 2.1.1)
- sklearn (verified on 0.19.0)
- cv2 (verified on 3.1.0)

## 2. General Description of Scripts

### Install the requirements

```
pip install -r requirements.txt
```

### Run a script with user-friendly command-line

```
python [script name].py -h
```

All the scripts use `argparse` module to make it easy to write user-friendly command-line interfaces. You can use option `-h` or `--help` to get a useful usage message for each script.

## 3. Detailed Documentation

Detailed documentation and examples on each algorithm are given below.

|  Directory or file name  |               description                   |
| ------------------------ |:-------------------------------------------:|
|    ./data/               | directory to store dataset                  |
|    ./logs/               | directory to store runtime logs             |
|    ./results/            | directory to store result reports           |
|    ./figs/               | directory to store result graphs            |
|    ./classifiers/        | contains three custom classifiers           | 
|    ./util/               | contains some scripts to preprocess dataset |
|    ./test/               | contains some scripts to test custom models |
|    ./visualization/      | contains some scripts for visualization     |

### baseline.py

This script will run all classifiers -- SVM, LR and KNN on MNIST dataset. 

```
usage: baseline.py [-h] [--data_size [DATA_SIZE]] [--normalize]
                   [--pca_percent [PCA_PERCENT]] [--svm_c [SVM_C]]
                   [--svm_kernel [{linear,poly,rbf}]]
                   [--svm_gamma [SVM_GAMMA]] [--svm_degree [SVM_DEGREE]]
                   [--svm_coef0 [SVM_COEF0]] [--lr_solver [{lbfgs}]]
                   [--lr_c [LR_C]] [--knn_n [KNN_N]] [--output]
                   [--outfile [OUTFILE]]

optional arguments:
  -h, --help            show this help message and exit
  --data_size [DATA_SIZE]
                        Size of data dataset. Default = 0.1
  --normalize           Whether to normalize the features.
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
  --lr_solver [{lbfgs}]
                        Solver for logistic regression. Default = lbfgs
  --lr_c [LR_C]         Parameter C for svm classifier. Default = 1.0
  --knn_n [KNN_N]       Number of neighbors for knn. Default = 5
  --output              Whether to print the result report to file.
  --outfile [OUTFILE]   File to save the result report. Default =
                        './results/report+<current time>.txt'
```

Example：

 Run SVM with `linear kernel`, `C = 2.0`, LR with default parameters and KNN with `n_neighbors = 1` on `all` dataset without normalization, and store the result report in `./results/test_report.txt`.

```
python -m baseline --data_size 1.0 --svm_kernel linear --svm_c 2.0 --knn_n 1 --output --outfile ./results/test_report.txt
```

 Run all classifiers with default parameters on `50%` of dataset with `90%` of variance retained after applying PCA.

```
python -m baseline --data_size 0.5 --pca_percent 0.9
```

## baseline_custom.py

The difference between `baseline_custom.py` and `baseline.py` is that all the classifiers applied in `baseline_custom.py` is custom implementation while those in `baseline.py` are just imported from sklearn package. So the majority of the parameters are same except some. Here we only show the parameters list and describe the new parameters.

Note: Training custom SVM will take a long time. (On total dataset with rbf kernel, it will take about 20 minutes)

```
usage: baseline_custom.py [-h] [--data_size [DATA_SIZE]] [--normalize]
                          [--pca_percent [PCA_PERCENT]]
                          [--max_iter [MAX_ITER]] [--svm_c [SVM_C]]
                          [--svm_kernel [{linear,poly,rbf}]]
                          [--svm_gamma [SVM_GAMMA]]
                          [--svm_degree [SVM_DEGREE]]
                          [--svm_coef0 [SVM_COEF0]] [--lr_lr [LR_LR]]
                          [--knn_n [KNN_N]] [--output] [--outfile [OUTFILE]]

optional arguments:
  --max_iter [MAX_ITER] 
                        Hard limit on iterations within solver. Default = 3000
  --lr_lr [LR_LR]       Learning rate for Logistic Regression. Default = 5e-5
```
