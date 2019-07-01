# Final Year Project: Fusing Spatial and Temporal Models for Joint Hand Pose Estimation and Action Recognition

<!-- ![logo](results/title_teaser.pdf) -->

## Overview / Summary / About



## Structure
### Folder Structure
  ```
  root/
  │
  ├── train.py - main entry-point to start training
  |
  ├── test.py - main entry-point to evalate any trained model from checkpoint (*.pth)
  │
  ├── configs/ - yaml based configs for all reported experiments
  │   ├── combined/
  |   |   └── *.yaml - combined model configs
  │   ├── hpe/
  |   |   └── *.yaml - hpe model configs
  │   └── har/
  |       └── *.yaml - har model configs
  │   
  ├── data_utils/ - anything about data loading goes here
  |   ├── base_data_loader.py - abstract base class
  |   ├── collate_functions.py - helper functions for custom collation
  |   ├── data_augmentors.py - helper functions for data augmentation (mostly unused)
  |   ├── data_transformers.py - data transformation functions
  |   ├── data_loaders.py - data loader classes for hpe, har and combined modeldebu
  │   └── debug_plot.py - functions to general gifs and comparative plots of different methods 
  │
  ├── datasets/ - default directory for storing input data
  |   ├── base_data_types.py - base types used for data_utils
  |   └── hand_pose_action.py - pytorch extended dataset class for dataset used
  |
  ├── docs/ - misc rough notes (may be removed in future)
  │
  ├── ext/ - external files
  │
  ├── img/ - images for readme
  |
  ├── logs/ - default logdir for tensorboardX
  |
  ├── metrics/ - losses, and metrics
  │   ├── loss.py
  │   └── metric.py
  │
  ├── models/ - models, losses, and metrics
  |   ├── base_model.py - abstract base class
  │   ├── har_baseline_model.py - har model class
  │   ├── hpe_baseline_model.py - hpe model class
  │   └── combined_model.py - combined model class
  |
  ├── results/ - plot generation scripts and pdfs presented in report
  |
  ├── saved/ - default checkpoints folder
  |
  ├── tests/ - pytorch unit tests for some model variants
  │
  ├── trainer/ - trainers
  |   ├── base_trainer.py - abstract base class
  │   └── trainer.py - main trainer class
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
- `ext`: anything from externel sources that is either used directly, indirectly or kept as a reference.
- `models`: code defining models and their variants.
- `metrics`: including losses and eval metrics for validation etc. These files are to be moved.
- `trainer`: code defining training and validation stuff for one epoch, these are referenced by top-level `train.py`



## Acknoledgements
Where possible this document aims to acknoledges sources used for this project.

### Main Structure
Template from https://github.com/victoresque/pytorch-template. See `LICENSE-1` for details on the license.

### Dataset + Hand Model Image + Most Details
https://github.com/guiggh/hand_pose_action

### Some Helper Utils and Transformers
https://github.com/dragonbook/V2V-PoseNet-pytorch


---












































