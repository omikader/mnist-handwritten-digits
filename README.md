# mnist-handwritten-digits
Implement linear SVM and construct a CNN based on LeNet-5 using PyTorch to achieve hand-written digit recognition.

## Introduction
 
For a formal definition of the assignment, please see the project [description](docs/proj02.pdf). For a summary of the results, please see my [final report](docs/final_report.pdf).

## How to Run the Code

Create a new virtual Python3 environment so as to not affect any other projects or programs and install the binaries listed in [requirements.txt](requirements.txt) using pip.

There are three scripts in the [src](src/) directory. Each script is meant to be run in its entirety from the command line as such.

```
(venv) wireless-10-105-16-100:src omar$ python svm.py
```

* svm.py - Linear SVM classifier
* svm_lda.py - Linear SVM classifier after reducing the dataset to 9 dimensions using LDA
* deep_learning.py - Convolutional neural network based on LeNet-5

The [deep learning](https://github.com/omikader/mnist-handwritten-digits/blob/master/src/deep_learning.py#L15-L30) script allows the additional usage of several command line arguments. These include batch size and learning rate, for example. Defaults are set for all parameters in the event that none are explicitly provided.
