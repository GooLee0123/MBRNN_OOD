# MBRNN Pytorch Implementation
This repository contains a [PyTorch](https://pytorch.org/) implementation of the multi-stage training strategy of MBRNN for OOD detection introduced in "[Lee et al. 2021]()".

# Installation
This package requires Python >= 3.7.

## Library Dependencies 
- PyTorch: refer to [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch with proper version for your local setting.
- Numpy: use the below command with pip to install Numpy (Refer [here](https://github.com/numpy/numpy) for any issues installing Numpy).
```
pip install numpy
```

# How to Run The Model

## Data Preparation
```
TBP
```

## Training of The Model
Although our deploy version code includes the pre-trained network, one can train a new model from scratch using below command.
```
python main.py --train True
```

## Model Testing
Since the default setting for the train option is *False*, one may use the below command for the test of the model.

```
python main.py
```

The process will dump an array shaped [*nsamp*, *nbin*+1] into the folder *Outputs* with *npy* format, where *nsamp* and *nbin* are the number of samples and bins, respectively. The first *nbin* columns of the array are model output probabilities, and the last column is the photometric redshift.