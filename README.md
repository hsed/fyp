# FYP Source Files 

## About



## Structure
- `datasets`: includes raw dataset files plus helper functions to load and/or debug and/or display raw information from dataset. Should include at-least one class of pytorch datasets derived type with `__getitem__` function
- `data_utils`: includes all functions that manipulate data in any way after being loaded in raw forma and before being input to any model. Also includes functions to collect results during testing i.e. perform inverse transformations or other manipulation to transform outputs from models back to the same scope as the raw data. These can be used to store evaluation results in the same format as gt labels.
- `utils`: all other extra helper functions needed during the training and/or testing can include:
  - Additional losses e.g. 3D avg error not directly used for the purpose of training.
  - Visualisation of training, tensorboardX, visualisation of final losses e.g. % of frames vs mm error.
  - Logging and progress-bar etc
- `ext`: anything from externel sources not currently in use but planned for future use, files in this folder is usually temporary
- `models`: All code defining models.
- `metrics`: including losses and eval metrics for validation etc. These files are to be moved.
- `trainer`: all code defining training and validation stuff for one epoch, these are referenced by top-level `train.py`


## Progress

### Dataset Loaded & Debugging

- [x] Basic Structure of dataset
- [ ] Working debugging functions to plot etc. a random sample
- [ ] Given a sample (raw sample no transform):
    - [ ] Plot depth, keypt, com in 2D if HPE
    - [ ] Plot (first 5, last 5) depths, keypts, coms in 2D if HAR
- [x] Dataset class from pytorch
- [x] Dataset `__getitem__` function for either HPE/HAR
- [ ] Working transformer functions and util functions
- [ ] Working augmentation functions
- [ ] Working plot of augmented/transformed data e.g. crop, rotate etc


### Baseline LSTM
- [x] Model design
  - [x] Look at pad data and pack data and other rnn helper functions
  - [x] Look at pytorch rnn tutorial, also lstm sample
  - [ ] look at DLmaths rnn etc.
- [x] Train+Test

### Baseline HPE/HPG
- [ ] Better and faster image loading..
  - [x] Convert all openCV loading to PIL loading as its much faster (predicts ~20mins vs ~30mins)
  - [ ] Create new h5py saving procedure based on model type (hpe,har) and dataset type (train,test)
  - [ ] find clever way to automatically laod the correct dataset, for validation no need to worry as pytorch dataloader majically handles it, it only uses a sampler etc. underlying dataset obj is only one
  - [ ] save h5py files and load appropriately, test on init if file exits if not do the loading...
  - [ ] test reading from h5py files and see perf difference vs no file real
  - [ ] code (can also write instead on every file read but maybe slower...):
  ```py
    with open('data.h5py', 'wa') as f:
        for i,item in self.depthmaps:
            f.create_dataset('%d' % i , item)
  ```
- [ ] Baseline Experiments
  - [ ] VAE vs AE
    - [ ] Import Models from spurra/vae-hands-3d
    - [ ] Test pre-trained models, see perf on dataset
    - [ ] See quiv perf on our own dataset.
  - [ ] AE vs VAE
  - [ ] VAE vs cVAE
  - [ ] cVAE vs cGAN+cVAE 


## Acknoledgements
Where possible this document aims to acknoledges sources used for this project.

### Main Structure
Template from https://github.com/victoresque/pytorch-template. See `LICENSE-1` for details on the license.

### Dataset + Hand Model Image + Most Details
https://github.com/guiggh/hand_pose_action

### Some Helper Utils and Transformers
v2v-pytorch


---

## Experiments
### Experiment #1: Train DeepPrior HPE on FirstPersonHandAction dataset

### Experiment #1: Train DeepPrior HPE on FirstPersonHandAction dataset

#### Setup



```
tmux attach -t 0
```