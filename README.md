# Pytorch Implementation of the OOD detection based on MBRNN
This repository contains a [PyTorch](https://pytorch.org/) implementation of the multi-stage training strategy of MBRNN for out-of-distribution (OOD) detection introduced in "[Lee et al. 2021]()".

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
python main.py
```

## Model Testing
One may use the below commands to test the TS2 and TS3 model.

```
python main.py --test
```

The processes will dump arrays containing TS2 and TS3 model outputs each shaped [*nsamp*, *nbin*\*2+2] for in-distribution, labeled OOD, and unlabeled samples into the folder *Outputs* with *npy* format, where *nsamp* and *nbin* are the number of samples and bins, respectively. The first *nbin*\*2 columns of the array are model output probabilities from high and low entropy models. The following column contains the OOD score, and the last column is the average photometric redshift.

### Model Inference

```
python main.py --infer
```

The processes will dump arrays containing TS2 and TS3 model outputs each shaped [*nsamp*, *nbin*\*2+4] for in-distribution, labeled OOD, and unlabeled samples into the folder *Outputs* with *npy* format. The first *nbin*\*2 columns of the array are model output probabilities from high and low entropy models. Each of the following columns is the OOD score, average photometric redshift, mode redshift, and the standard deviation of the model output probability distributions.

## Option Change
We deploy the model with the best-performing configuration described in our paper, but one can adjust the model structure and other settings by modifying the options of the *config_file/config.cfg* file.
