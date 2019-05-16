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
  │   ├── config.yaml - config file
  |   ├── config_*.yaml
  │   └── config_*.yaml - other experiments
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
- [x] Incorporate original deep-prior preprocessing with augmentation
  - [x] X,Y pre-process+augment
  - [x] only Y pre-process+aug for pca
  - [x] PCA and pca learning
- [x] Train on orig deep-prior++ with orig method i.e. augmentation
- [x] Add procedure to validate usign custom metric (3D error) per epoch
- [x] Perform NEW pre-processing on depth images without any augmentation
- [x] Perform NEW pre-processing on keypoints images without any augmentation
- [x] Train on NEW without augmentation
- [x] Add NEW augmentation procedures to pre-processing
- [x] Train NEW with augmentation


NEW: no depth thresholding, better method for standardisation, better method for cropping
     using 40px padded bounding box in 2D

#### Results
Validation is a split of 0.2 from trainset
Test scores are obtained by training on entire train_set and using test_set as validation
for early stopping procedures

all val splits (0.8:0.2) experiment will run for 15epochs

To check all hyper-param values chosen, see the json config file in timestamp folder.

| Experiment | Timestamp | Avg_3D_error_val (test) | Avg_3D_error_train | Notes |
| -------- | -------- | ---- | ---- | :------------------------------------: |
| Baseline   | BaselineHPE/0401_084525 | ~26.6mm (Best) | 14mm (Best) | All baseline stuff, 1:1 split train:test, val as testset, training error reduces steadiliy, val error jumps around, stagnates around 20 epochs, best@ep40, tot50epochs
| Baseline+ValSplit | BaselineHPE/0401_111005 | 23.04mm | 22.24 | Here train rate of decrease in error is slow, however less effects of overfitting also val error is way better than before, but maybe cause its just a smaller set


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


## problems

- ~~pca data for training and pca training mean/std~~

- ~~ pca after training for transform of y ~~

- X or depthmaps after transform ARE DIFFERENT!!
  - Check out where the issue is, try using xy transforms from msra to test out the problem
  - or just see / compare visually which transform gives different results

---
## Meeting 8/3/19 Notes
The main problem was the camera matrix transformation etc so apparently y_val of mm2px was coming to be wrong for some 
reason ths is now changed so org deep prior methods are used

however there is still some issue with validation curves!! they stop going less that 0.05 even though final avg 3d error is 0.25 so val loss shuld actually be about 0.3!

something else might still be worng!!
have a test at val loss etc, maybe try increasing early stop epochs and see if overall scores improves, for some reason val loss is being reported higher than normal...why??


current 25mm test set train on full train set
have augmentation see what happens... 3mm less.. (scaling)
per image changed cropping ...
show visual estimation in augmentation

goal: beating the baseline experiments below atleast most of them with the cyclic architechture...

validation error remains the same average 3D error decreases... why dies this happen? Is it a bug? Doesn't seem to be, as I renamed the msra folder to ensure no msra is being used during val. we do get value decreases in val error but not as much, i think its probably just the test set.

Note for experiments exp on val set choose best their THEN retrain on full train set!

---

## TODO
 - [ ] Implement Visualisation AFTER training with test cases and predicted output along with 3D error for that frame, so you see more than just avg 3D error.
 - [x] Implement Augmentation **EXP 1**
 - [x] Implement New Crop Method see if its better or worse **EXP 2**
 - [ ] Try 'conditioning' in various ways, see experiments below. **EXP 3**


### Experiment #1:
#### Details
Test with data augmentation:
Rot only
Rot + Transl
Rot + Transl + Scale

CHOOSE: PICK BEST AUG OPTION

#### Results

### Experiment #2:
#### Details
Alternative cropping method (40mm padding) VS Original Cropping Method
Test on NO AUG: is it better than 25mm?

If So, then test on training with augmentation using mest augmentation method chosen earlier

CHOOSE: PICK BEST CROPPING OPTION

### Experiment # 3:
#### Details
After all the best choices from above, do different choices for concatenation of action label as a baseline


Baseline: Original + (OLD/NEW) CROP + (???) AUG -> 25mm

Baseline: BaselineHPE/0420_2254 ()

Epochs: 30 (BEST@EP__)

| Method of Concat | Experiment | Other Notes | Val Error (Train on 1.0 train-set, test on 0.2 val-set) | Test Error (Train on full train-set) |
| -----------------| --------- | -------------------------- | ---------------- | ----------------- |
Baseline | BaselineHPE/0509_1106 | See experiment for config, crop2, deterministic, gpuid3, no data aug, valsplit -0.2 | 22.0583 (22.0340@EP26) | - |
SoftmaxActionOut | BaselineHPE/0509_1232 | Train with action label as output rather than input so a softmax layer as oputput and gradients backpropagating, its like multi-task learning | ~32.0747 (~31.5010mm@EP28) | - |
ConcatAsImgChannel1 |  | 45x1x1dim one-hot -> 45x128x128 with only one channel ones all zeros; 45:1 concat | - | - |
ConcatAsImgChannel2 -crop2 | BaselineHPE/0509_1704 | idx->45 (learnt embed) ; 45->45x1x1->45x128x128 expansion 45:1 concat | - | - |
ConcatAsImgChannel3 | - | 45->128x128 embedding then concat 1:1 | - | - |
ConcatAsImgChannel4 | - | 45x1x1 -> 45x128x128 upsample then concat 45:1 | - | - |
ConcatAsImgChannel4 | - | 45x1x1 -> 45x128x128 upsample the conv -> 1x128x128 then concat 1:1 | - | - |
Concat+Softmax | - | Best Concat + Softmax | - | - |



### methods for concat
1. FiLM Layer:
   - First tried only after bn in resblock -- doesn't work error too high we tried one BN only or all of them, the best was last BN layer that too was not that good 
   - next tried putting it in BEFORE residual addition -- a lot better initially then in converges in val error i.e. doesnt decrease but training is still decreasing this implies it is overfitting maybe?
     - a possible solution: disable conditioning halfway through trainining? Basically after certain epochs, we set action condition of model to 0. something like epoch callback attrib in the model
   - Now we have made conditioning changes in many places we have currently last BN+ReLU+Conv block with conditioning and also the skip connection with conditining, we have alos disabled BN training for those layers with conditioning. need to ensure if this works properly?
   - NEXT: we try putting the condition AFTER residual addition also maybe having it for fewer
   - NEXT: Try the turn of conditioning setting after few epochs
   - NEXT: Try chaning the embedding to linear layer and from long index tensor to 1-hot 2d tensor as then we can use probabilities!
2. Simple use of duplicating index information and concat as 1 channel 128x128 that worked but not better than no condition
3. BEST ONE SO FAR: concat of 45 action channels with 1 channel of depth this 45 channels is expanded in width and height and converted from index to embedding of 45 dim using embedding layer.
4. next we can try to do number 3 but use embedding layer?
5. 


use rnn cell for meta learning rnn for feedbck loop see deepleanring last and final lecture

convert json configs to yaml configs

write down structure somewhere?

enhance logging by posting to tensorboard information on current experiment etc



#### New code to convert config to readable format for tensorboard

```py

import json, logger, logging
from utils.visualization import WriterTensorboardX

logger = logging.getLogger()
writer = WriterTensorboardX('saved/test', logger, True)

my_config = json.load(open('experiments/config.json', 'r'))
yaml_str = yaml.dump(my_config)

## first convert all '\n' to '<br/>' for line breaks
yaml_str_6 = yaml_str.replace('\n', '<br/>')

## next convert spaces to &nbsp; to make it non-breaking so that it doesn't collapse to no space
yaml_str_6_1 = yaml_str_6.replace(' ', '&nbsp;')

## one line version
yaml_str_7_1 = yaml_str.replace('\n','<br/>').replace(' ', '&nbsp;')

## now write to tensorboard!
writer.add_text('test_tag_info', yaml_str_6_1, 0)
```


### Errors
**DATA AUGMENTATION IS BROKEN**

Note for **BaselineHPE/0401_132505** you can clearly see that by doing pca_all_Aug and training only rot_aug error
is WAY TOO HIGH +10mm diff

also for **BaselineHPE/0401_122247** we did only pca_aug (all modes) and no aug for training and got almost same
or lower results (like with val it was slightly higher and with train it was wobbly so not so sure although not clearly better)


now for new test **BaselineHPE/0401_133754**, we use Rot+None aug for both pca and training, if we don't see any visible improvements
or if we see worse results then definitely augmentation is broken and need to see where we are wrong.
Error is still bad! starts around ~40mm instead of ~30mm!

TRY: do aug mode and train on msra vs no augmode on msra and see difference.. if we see improvmenets for msra then basically we need to do correct augmentation for new dataset

If we switch to ONLY augtype=None for both pca and training then its fine as before so atleast augtype none is not doing anything dodgy, this is experiment **BaselineHPE/0401_134257**

Note: We find PCA values to be 4 and -3 and Y_values to be 4 and -5 so that means we are not getting the 1 and -1 we expected
so that means many of our depth images are running out of crop area! so that is bad! only -1 and 1 is ok for y or near that value
we achieve these values near enough for msra but not for fhad **SUSPECT CROPPING OF KEYPOINTS AND DEPTH**

#### Test MSRA
Now we test MSRA, first we do AugType None for Both PCA+Train... This is experiment  **BaselineHPE/0401_134755** note we use same 0.8:0.2 strategy so the same amount of val set. We find that error is much beeter maybe dataset is easier or problems with pca or cropping, one thing we observe is that the y range of values is very close to -1 and 1 (before pca) so this was not the case with FHAD maybe this could be a source of error in fhad

Another thing we observed is that after testing on AugType PCA+None for PCA+Train.. experiement **BaselineHPE/0401_140338** we see worse results for MSRA AS WELL! so about 5mm worse results. 

---

### **TODO** So things to try:

- first fix the cropping problem in msra , maybe use original cropping transformation functions etc and see if we can get better avg 3d error when doing rot augmentation than when not doing it.

- port over, one by one the new cropping procedure i.e. draw 40px padded box, first without augmentation and see if for both fhad and msra u get better or same results atleast for fhad we expect better results, do it first only without augmentation

- port over new cropping procedure for augmentation as well and test effects on moth msra and fhad, does it match performance of original augmentation functions?



| Experiment | Notes | Train/ValError@5Epochs |
| ---------- | ----- | ---------------- |
| BaselineHPE/0408_000147 | NoAug New Transformers | 14.34/14.67 |
| BaselineHPE/0408_003207 | NoAug Old Transformers, a lowerbound to above slightly | 13.72/14.67 (Best: 13.91) |


### Note:
Experiments with 5 epochs are inconclusive because when you do data augmentation basically the train error is increased due to regularisation essentially and also valid error is much more to begin with then slowly valid error overtakes non-regul errors

**SCALE TRANSFORRM** seem to give poorer results BaselineHPE/0408_1901 when used for train data augmentation but with pca augmentation its fine... maybe improve crop procedure?

BaselineHPE/0408_1845 and BaselineHPE/0408_1817 are the best one basically they use orig transform but have Rot+None for train and ro+scale+none for pca transform

### TODO: 
1. get same valid curve on 3d error or atleast train curve using new transformers for the best two config, see tensorboard for details
2. try fix scale transform by maybe improving crop size in general for everything? is that bad i.e. basically it means same thing?
3. Try with trans transform e.g. rot+trans on pca and see any improvements?
4. finalise best transforms then try on fhad!! Try better cropping method on fhad which is max/min x,y,z crops according to train data, maybe ask guillermo on his crop method is it looking at target too much during inference?


```
/root/../fyp> python -m tests.tests
```




We carried out a few experiments to investigate deterministic vs non-deterministic settings

in general, while deterministic experiments are repeatable, the whole idea of adata augmentation lies on the fact that
data generation is randomised and new data is generated AT EVERY EPOCH so that there is no sign of overfitting

we tested several combinations of data augmentation and determinisic setting. from these we dfound that one with all augment gave the best lower bound (0410_1838) although it was still worse than no augmentation (0420_1720 or 1946). Other experiements in this setting start from BaselineHPE/0410_1720 till about 1900 or 2000.

Now we changed to new device to results are a bit skewed but nevertheless all 3 augmentations WITH RANDOMNESS (BaselineHPE/0411_0220
) is much better that without randomness (BaselineHPE/0410_2011). 

However when it comes to augmentation vs no augmentation, augmentation doesn't REALLY help, non-augmentation either always win or comes very close, maybe for long term epoch it might be better but atleast till 20 epoch, augmentation doesn't really help...

Maybe this implies that augmentation is 'too stochastic'?

##### Tests for different crop sizes

PCA: 200k
AUG_MODES: 0 1 2 3
PCA_AUG_MODES: 0 1 2 3
RANDOMNESS: True

| Experiment           | Crop SZ | Y Range [Min, Max]| PCA Range [Min, Max]  | Valid3DError@Ep10 (@EP20) | Notes |
| -------------------- | ------- | ----------------- | --------------------  | ------------------------ | ----- |
|BaselineHPE/0411_0220 | 200mm   | [-1.0394, 1.0775] | [-1.7956, 2.2264]     | 12.48mm | - |
|BaselineHPE/0411_0150 | 220mm   | [-0.9450, 0.9796] | [-1.6323, 2.0240]     | 13.46mm | Always an upper bound to above |
|BaselineHPE/0411_1100 | 210mm   | [-0.9900, 1.0262] | [-1.7101, 2.1204]     | 14.39mm | Much worse than both |
|BaselineHPE/0411_1130 | 190mm   | [-1.0942, 1.1342] | [-1.8901, 2.3436]     | 11.6774mm (10.8176mm) | Much better than anything tried so far! |
|**BaselineHPE/0411_1208**| 190mm   | [-1.0942, 1.1342] | [-1.8901, 2.3436]     | 11.4144mm (10.3342mm) | We tried only this without any augmentation to see results, we get beter results but its a very close call  |
|BaselineHPE/0411_1453    | 180mm   | [-1.1549, 1.1972] | [-1.9951, 2.4738]  | 13.0373mm (13.08mm) | Well at this point it turns really bad! |


BEST CHOICE: 
PCA: 200k NO AUG PCA NO AUG TRAIN RANDOMNESS TRUE CROP_SZ 190mm

```py
#we can save logging info in file and simultaneously print to console as well!

logging.basicConfig(
    format="%(asctime)s [%(name)s_%(funcName)s] [%(levelname)s]  %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
])
class abc(object):
  def __init__(self):
    pass

log = logging.getLogger(name=abc().__class__.__name__)

# not .info() works too!
log.warn("Hello there")

```


```py

#new function to define deterministic params for fhad
self.dataset.make_transform_params_static(AugType, \
                    (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
                     custom_aug_modes=train_aug_list)

```


URGENTLY NEED NEW CROP FUNCTION FOR HPE!!
FOR FHAD!

Y_FINAL value for pca is the likes of 2.5!! which means completely cropped and in the background!
either cropping doesn't work or something is really bad !
maybe croppin is buggy!
visualise your results! use jnb to check worst case scenarios!

cropping to 400m showed some improvements to this area has some potentials!!



Cropping:
- try new padding method with range of different paddings e.g. +10px, +40px, +100px -10px, -40px, -100px,, pick best
- try largest value for each axis method where after picture is centered you find the largest and smallest mm px in x,y,z seperately and from that u decide your crop sz. crop sz can also be a multiple of this e.g. 1.2x or 0.8x or 1x.
- pick best from both of above method tested on fhad, no aug is really required...


Action:
- try all methods listed above to incorporate action info into HPE...




```bash
cat /dev/zero | ssh-keygen -q -N ""
cat ~/.ssh/id_rsa.pub
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
# <after adding key to git>
git clone git@github.com:hsed/fyp.git
cd fyp

apt update && apt install -y libsm6 libxext6 libxrender-dev zip unzip curl wget nano
pip install jupyterlab tensorflow tensorboardx opencv-python

curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN -o datasets/hand_pose_action/data_train_hpe_cache.h5
curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN -o datasets/hand_pose_action/data_test_hpe_cache.h5

```

---
---

### New Cropping Methods

Tried:
- standard method
- tried the padding method notice the not ideal not exact ratio of x/y causes the distoreted? squashed inages

here device id 2 was used! (gpu 3)
new tests in DETERMIISTIC MODE FOR REPETITION

(best ound at earlier stage, mode 2 not sure how we got here exactly?)

0414_1446 -> Mode 0 (wasn't really working that well --- a bit upper bounding) -- need to do this again for full 20 epochs!!!
0414_1650 -> Mode 1 (usable future choice)
0414_1552 -> Mode 2 (note im not sure if this is with double int or single rounding, see later...; 0414_1902 is WITH DOUBLE INT ROUNDING)
                    Need one more experiment to see if 1552 is better or not basically now test single int rounding...
                    If next experiment is worse that 1902, then re-enable double rounding! int(...) ; int(...).
                    1902 is better so we stick to double rounding!!
0414_1809 -> Mode 3 (almost as good as mode 1)


mode 0: usually upper bound for most graps however a lot of variaion  
mode 1: least variations also the best lowest score in 20 steps
mode 2: generally an upper bound for most large parts where errors remain large, it does come down however in 20 steps as lowest score
mode 3: looks good as well mimic mode 1 in amny places

one thing we've noticed is that the keypoints don't scale properly i.e. unhandled by code so we can't have different x and different y ranges this is the case with method 2 the problem occurs that keypoint projection on depthmaps clearly show misalignement thus padd w.r.t CoM in both x and y dirs must be same! Otherwise aspect ratio is not presevered, we can see method 2 images appearing squashed

Method 0:
Original crop from deep-prio++ for msra dataset WITH OFFSET ADDITION REMOVED (note this is imp!) as this is not needed for FHAD and produces undesirable results, this has no impact on msra provided only meth 0, 1, 3 are used i.e. no custom aspect ratio cropping.

Method 1:
  We supply keypt values centered w.r.t to CoM i.e keypt_diff = keypt - CoM then take the max(abs(keypt_diff)) of each dir and add some extra mm padding in each direc (+ and - sperately) with equal amounts of padding for x,y with max_val(x_max, y_max) chosen.
  similar for z using redefined padding value.

method 2:
  Here all stuff is done in pixel domain, find first the min and max of, x,y values and then for each dir choose whichever is farthest from com and set that as 'amount' of buffer for xy_startend 
  zstart is based on depth_pad param and max and min keypts value so not really linked to com....

method 3:
 an improvement of method 2 based on com now, a px_diff is now computed much similar to method 1. the x and y diffs are compared and max of that is selected for both x and y for aspect ratio preserving and for zaxis as well com is used so that in end com is always in center of point for x,y and z axis. the buffer is added respectively for x, y and z for x,y, exact same values are highly recommended for aspect ratio preservation.


YOU NEED SMOOTHING TO DRAW SOME CONCLUSIONS! SEE 0.6x-0.8x smoothing! also this one after smoothing method 1 or 3 are best

it makes logical sense to have all axis CoM as center this is so that it is analogous ti 3D domain where the 3D values of skeleton is always w.r.t CoM therefore it makes sense if CoM is also in center of depth img



| Experiment           | Crop SZ | Y Range [Min, Max]| PCA Range [Min, Max]  | Valid3DError@Ep10 (@EP20) | Notes |
| -------------------- | ------- | ----------------- | --------------------  | ------------------------ | ----- |
|BaselineHPE/0414_1218 | 200mm   | ??? | ??? | 12.48mm | - |


----

#try crop_pad_3d: [90., 90., 100.] [0., 0., 100.] vs #[30., 30., 100.]
try changing crop_shape (useful for y values) to ~400!!


## next experiments
AUG MODES + DETERMINISM

-> try mode 1 -- with augmentation train_0,1 + pca_0,1,3
try mode 2 -- with augmentation train_0,1 + pca_0,1,3

try mode 3 -- with augmentation train_0,1 + pca_0,1,3


### Choosing the Validation Set

- Tried Trainset 0.8:0.2 vs 1.0+Test_Set method, validation curve do not follow test-set. Possible reasoning: PCA used entire train-set, train-set gets too small. See '' vs '' 

<INSERT PIC HERE>


- Newer method: Tried train set 1.0 + 0.2 Test Set. Actually, good upper bound on test set which is good. Also matches quite closely. See experiment BaselineHPE/0419_1744 vs BaselineHPE/0417_1634

< insert picture here >


Answer: 
1.0 Trainset + 0.2 Validation Set for Quick Testing w/ 20-30EP training depending on impact. With ~10EP early stopping. no checkpointing but maybe saving the best model and if best model != last model then saving that too for continuation.

### Choosing the Cropping Method

We applied four different cropping methods see statement above. The best we found was method #2 and we will stick with that, even though it produces worse results when visually seeing it images get squashed but overall error is low. **NEED TO VERIFY THAT AFTER PREDICTION IT LOOKS GOOD TOO**.

< insert picture here>

### Data Augmentation
This does slight improvements but after a long number of epochs, so its not very benefitial -> 

Tests : Best data aug : 0 (None) + 1 (Rot) + 3 (Trans) & PCA: 0, 1, 2 (Scale), 3.

BaselineHPE/0419_1007 (note this is full test set maybe its better to show validation partial test set as we'll use that in continuation!)


### Baseline Performance

Test-Set as shown in BaselineHPE/0417_1634

We will do a full baseline validation set performance as well....

Validation: BaselineHPE/0419_1744


### Action Test
We perform one test with having target with action and using combined loss function. loss is too high and we get a much worse value. we didnt test with alpha i.e. how much of one component to use. 

In future we will try to cleate a new class and add new data loaders that convert action to one hot and so on and then we have an embedding layer that converts 1x1x45 to 128x128x45 -> 64x64x32 or something like that basically 1/2 or 1/4 or 3/4 of the channels of input then we use the network as is and try to see if any imprvement possible we can also use VAE



# Meeting Summary 15/5/19
- Somehow improve your HPE model by incorporating temporal information. Currently your model has no temporal information. Definitely with temporal information you should perform better for HPE. The 1mm inprovement you got is actually with g.t. action information it is too less or basically its not very helpful as you won't have action during inference! So you should have a model which somehow predicts action not uses g.t. atleast
- A good step would be to use the output of the previous action prediction as input to next prediction.
- Pre-procesing stuff should not be included in the main report as its not the main thing maybe just give it a one-paragraph or one sentence even.
- Need to be very very sure of your baseline! Must do stuff that actually improves the baseline!


Full test set 60ep baseline: BaselineHPE/0417_1634
