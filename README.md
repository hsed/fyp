# FYP Source Files 

## About



## Structure
### Folder Structure
  ```
  root/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── data_utils/ - anything about data loading goes here
  |   ├── base_data_loader.py - abstract base class for data loaders
  │   └── data_loaders.py
  │
  ├── datasets/ - default directory for storing input data
  │
  ├── experients/ - abstract base classes
  │   ├── config.json - config file
  |   ├── config_*.json
  │   └── config_*.json - other experiments
  │   
  ├── ext/ - external files
  |
  ├── models/ - models, losses, and metrics
  |   ├── base_model.py - abstract base class for models
  │   ├── blocks.py
  │   ├── **
  │   └── model.py
  |
  ├── metrics/ - losses, and metrics
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── saved/ - default checkpoints folder
  │   └── runs/ - default logdir for tensorboardX
  │
  ├── trainer/ - trainers
  |   ├── base_trainer.py - abstract base class for trainers
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      ├── logger.py - class for train logging
      ├── visualization.py - class for tensorboardX visualization support
      └── ...
  ```

### Notes
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
- [x] Better and faster image loading using h5py.. (**~30mins** per train_set using PIL -> **~1s** per train set using h5py, gzip compression level 7; both tested on 4 workers)
  - [x] Convert all openCV loading to PIL loading as its a bit faster (predicts ~20mins vs ~30mins)
  - [x] Create new h5py saving procedure based on model type (hpe,har) and dataset type (train,test)
  - [x] find clever way to automatically laod the correct dataset, for validation no need to worry as pytorch dataloader magically handles it, it only uses a sampler etc. underlying dataset obj is only one
  - [x] save h5py files and load appropriately, test on init if file exits if not do the loading...
  - [x] test reading from h5py files and see perf difference vs no file real
  - [x] Have persistent data loader to load only once at start of experiment to memory
- [ ] Baseline Experiments
  - [ ] HPE training on FHAD (depth,keypoint, action)
    - [ ] Finalise PCA
      - [ ] Training on RAW KeyPt
      - [ ] Training with Data Aug
      - [ ] Saving/Loading Cache
      - [ ] Testing Pipeline works fine for transformation
      - [ ] Full pipeline testing like Depp-Prior as Avg 3D error as new metric
    - [ ] Train HPE on Action+Depth => KeyPt
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
Input: Depth; Output: Keypoints

#### Progress:
- [x] Cacheable procedure to load train/test flat sequences of images
- [ ] Incorporate original deep-prior preprocessing with augmentation
  - [x] X,Y pre-process+augment
  - [ ] only Y pre-process+aug for pca
  - [ ] PCA and pca learning
- [ ] Train on orig deep-prior++ with orig method i.e. augmentation
- [ ] Add procedure to validate usign custom metric (3D error) per epoch
- [ ] Perform NEW pre-processing on depth images without any augmentation
- [ ] Perform NEW pre-processing on keypoints images without any augmentation
- [ ] Train on NEW without augmentation
- [ ] Add NEW augmentation procedures to pre-processing
- [ ] Train NEW with augmentation
- [ ] ???


NEW: no depth thresholding, better method for standardisation, better method for cropping
     using 40px padded bounding box in 2D

#### Results
Validation is a split of 0.2 from trainset
Test scores are obtained by training on entire train_set and using test_set as validation
for early stopping procedures

To check all hyper-param values chosen, see the json config file in timestamp folder.

| Experiment | Timestamp | Avg_3D_error_val | Avg_3D_error_t | Notes |
| ---- | ---- | ---- | ---- | ---- |



### Experiment #1: Train DeepPrior HPE with Action on FirstPersonHandAction dataset

#### Setup

#### Compression Updates

GZIP, 7 -> 47s; ~500MB

GZIP,7, bigger cache block -> 56s

GZIP,4, bigger cache block -> 40s;1.1GB


LZF, my laptop -> ~21s; 1.9GB!
GZIP,4, my laptop -> ~24s; 0.78GB
GZIP,7, my laptor -> ~20s; 0.68 GB <---sticking with this for now

## joblib methods -- untested oon my laptop, file size is roughly LZF size
```

tmux attach -t 0

<ctrl+b>, w

<select window>
```


## new procedure

iterate through the data loader in the beginning and everyitem will be a list of two items

one is the inputs the other is the outputs 


## validate extra metrics
targets ; outputs ;

if outputs is of type tuple then...
first elem is for val_loss of network
second elem is for keypoint error

so basically condition is... 

if it is a tuple send the second output to metric_eval and first output for val_loss