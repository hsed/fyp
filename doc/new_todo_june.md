

### Choosing deterministic or random-ish Experiements
We carried out a few experiments to investigate deterministic vs non-deterministic settings

in general, while deterministic experiments are repeatable, the whole idea of adata augmentation lies on the fact that
data generation is randomised and new data is generated AT EVERY EPOCH so that there is no sign of overfitting

we tested several combinations of data augmentation and determinisic setting. from these we dfound that one with all augment gave the best lower bound (0410_1838) although it was still worse than no augmentation (0420_1720 or 1946). Other experiements in this setting start from BaselineHPE/0410_1720 till about 1900 or 2000.

Now we changed to new device to results are a bit skewed but nevertheless all 3 augmentations WITH RANDOMNESS (BaselineHPE/0411_0220
) is much better that without randomness (BaselineHPE/0410_2011). 

However when it comes to augmentation vs no augmentation, augmentation doesn't REALLY help, non-augmentation either always win or comes very close, maybe for long term epoch it might be better but atleast till 20 epoch, augmentation doesn't really help...

Maybe this implies that augmentation is 'too stochastic'?

### Choosing the Validation Set

- Tried Trainset 0.8:0.2 vs 1.0+Test_Set method, validation curve do not follow test-set. Possible reasoning: PCA used entire train-set, train-set gets too small. See '' vs '' 

<INSERT PIC HERE>


- Newer method: Tried train set 1.0 + 0.2 Test Set. Actually, good upper bound on test set which is good. Also matches quite closely. See experiment BaselineHPE/0419_1744 vs BaselineHPE/0417_1634

< insert picture here >


Answer: 
1.0 Trainset + 0.2 Validation Set for Quick Testing w/ 20-30EP training depending on impact. With ~10EP early stopping. no checkpointing but maybe saving the best model and if best model != last model then saving that too for continuation.

---

### Choosing the Cropping Method

We applied four different cropping methods see statement above. The best we found was method #2 and we will stick with that, even though it produces worse results when visually seeing it images get squashed but overall error is low. **NEED TO VERIFY THAT AFTER PREDICTION IT LOOKS GOOD TOO**. **SEE HPE_DEPTH_CROP PAGE**

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



## NEW: A NOTE ON BATCH_NORM:
VERY IMP:
batchnorm2d has 2 additional params during intialisation. One param is `affine` and the other is `track_running_stats`. Both are quite helpful for us to do various things for our model.

NEED TO CHANGE CODE: such that whenever a context layer is present in bn_relu_block simply set affine to FALSE such that when traning the affine stuff is not done for batch norm this is because affine transform is handled by film layers (later down the line). however the standardisation still takes place as before along with running stats. This way theoretically we should achieve very similar or exact performance as before.

Note: must re-run for 30epochs and test some score on hpe then make changes and re-run also first save on github to easily undo any changes! its important to test the effects! Also if you do this you need to consider what happens to old saves do they still work?

So for instead of hacky context layers that are not actually supplied you basically set affine=False for batch_norm


NOW FOR COMBINED MODEL DURING FINE TUNING:
during fine tuning there are newer batch sizes and many different sizes for which we DO NOT know in advance of size
for some models it can be as low as 2 or 4. In this case its best to set `track_running_stats` to FALSE this is same as setting `training` to FALSE. this way always its ensured that EXACT CURRENT batch_size is used to calc mean and variance so a large change in batch_sizes wont cause big issues so this is equivalent to train mode but then the training can still occur. nonetheless training is training of affine params so if we use some model with action condition that that too is disabled. and it degenerated to exactly as training = false

for this you should basically loop around all modules in for loop fashion and for every nn.batchnorm2d found simply set `track_running_stats` to FALSE only do this for certain version numbers..... so like maybe v4d first then v3d and then maybe v2d as v2d2

### HPE Tests

please keep note of newer models that replace older models....

| Old Model | New Model  | Descr. |
| ----------| ---------- | ------- |
| 0525_0129 | 0608_0803  | baseline, no cond,  best of 100ep |
| 0528_0756 | 0608_1050  | act_cond7.0, equiprob cond 100 ep |
| 0529_0026 | 0608_1200 |  act_cond7.12, gt_act_cond, linear++ cond, no conv cond, equiprob cond 100 ep
| 0527_1455 | 0608_1145  | baseline, no cond, cond_v0  best of 30ep |
| 0527_1508 | 0608_0047/model_best.ep30.pth  | act_cond7.0, equiprob cond, best of 30 ep (values are based on valset -1.0) |
| ??        | 0608_1954 | act_cond6.0 train for 100ep this is useful to compare lin with non-linear
| ???       | 0609_0145 | act_cond7.181 this is like cond 7.18 (100% equiprob lin=1 res=true) but now support variable equiprob

Experiment: 0608_1050 -- HPE Act Condv7 REPLICA of 0528_0756 with longer training monitor


CONFIG: combined_act_in_act_out.yaml
HPE: 0608_1145 (VAL_ACC_ON_TEST: ???mm) (Best of 30EP Trained)
HPE_wActCond: 0608_0047/model_best.ep30.pth (VAL_ACC_ON_TEST: ???mm) (ResNet Cond + Equiprob 100% "7" Stage-1) (Best of 30EP Trained)
HAR: 0531_0551 (GT_ACC: 72.3%) (Best of 100EP)
TESTED ON DIRECT VALIDATION WITHOUT ANY TRAINING (WE FIND WHAT GIVES BEST TEST TIME PERF WITHOUT ANY ARCH CHANGES)
NOTE: FULL VAL SET IS USED TO COMPARE
6,7,7.15 ALL USE ONLY RESNET COND
6->0% Equiprob, 7->100% Equiprob, 7.15->X% Equiprob
| Action Condition | Proportion of 2nd Pass | Prob of Equiprob | VAL_ACC | VAL_3D |
| ---- | ---- | ----- | ----- | ------- |
| 0    | 0%   | N/A   | 56.5% | 14.89mm |
| 7    | 50%  | 100%  | 58.8% | 14.39mm |
| 7    | 100% | 100%  | 58.1% | 14.46mm |
| 6    | 100% | N/A   | 38.1% | 18.50mm |
| 7.15 | 100% | 100%  | 58.1% | 14.46mm |
| 7.15 | 100% | 99.9% | 58.1% | 14.46mm |
| 7.15 | 100% | 99.0% | 57.9% | 14.53mm |
| 7.15 | 100% | 95.0% | 57.0% | 14.63mm |
| 7.15 | 100% | 90.0% | 54.6% | 14.91mm |
| 7.15 | 100% | 80.0% | 54.1% | 15.04mm |
| 7.15 | 100% | 60.0% | 49.2% | 16.09mm |
| 7.15 | 100% | 35.0% | 45.2% | 16.71mm |
| 7.15 | 100% |  0.0% | 38.1% | 18.50mm |


NEW TESTS TESTED ON **FULL 100EP TRAINED MODELS**: 0608_0803, 0608_1050, 0531_0551
NEW ALSO TESTED ON **0609_0145** for val2 and val3
BEST OF 100EP
NOTE: for cond.7 @ 100% its worse than baseline due to two pass model but with single pass it shuld be better..
USE: configs\combined\combined_act_in_act_out.yaml
| Action Condition | Proportion of 2nd Pass | Prob of Equiprob | VAL_ACC | VAL_3D | VAL2_ACC | VAL2_3D |
| ----       | ---- | ----- | ----- | ------- |  ----- | ------- |
| 0          | 0%   | N/A   | 59.0% | 14.43mm |  % | mm |
| 7          | 50%  | 100%  | 60.9% | 14.15mm |  % | mm |
| 7          | 50%  | 95.0% | 60.4% | 14.21mm |  % | mm |
| 0          | 50%  | N/A   | 59.0% | 14.43mm |  % | mm |
| 7          | 100% | 100%  | 58.1% | 14.46mm |  % | mm |
| 6          | 100% | N/A   | 39.4% | 18.77mm |  % | mm |
| 7.1{5\|81} | 100% | 100%  | 58.1% | 14.46mm |  % | mm |
| 7.1{5\|81} | 100% | 99.9% | 58.1% | 14.46mm |  % | mm |
| 7.1{5\|81} | 100% | 99.0% | 57.6% | 14.62mm |  % | mm |
| 7.1{5\|81} | 100% | 95.0% | 57.6% | 14.62mm |  % | mm |
| 7.1{5\|81} | 100% | 90.0% | 55.2% | 14.63mm |  % | mm |
| 7.1{5\|81} | 100% | 80.0% | 55.2% | 14.66mm |  % | mm |
| 7.1{5\|81} | 100% | 60.0% | 53.4% | 15.26mm |  % | mm |
| 7.1{5\|81} | 100% | 35.0% | 50.9% | 15.60mm |  % | mm |
| 7.1{5\|81} | 100% |  0.0% | 39.4% | 18.77mm |  % | mm |


Now testing with 0608_1200 on 2nd pass..
VAL_TOP_ACC: 54.3%
VAV_AVG_3D: 15.89mm



### New Combined Fixed Tests

Experiment: **0608_0954** -- 10dv2 --- new dataset using newer HPE model and new fixed dataset for HAR --- Action Cond Ver: 7.15 (fine tuning training) USING LABEL_FLIP_PROB: 0.99 -- act_cond_7_model_0608_0047_pre-training


Experiment: **0608_1456** -- 3d --- new dataset using newer HPE model and new fixed dataset for HAR -- act cond is 0 we only perform combined training
saved/BaselineHPE/0608_0803/model_best.pth is hpe model acc is about 0.67 and increasing


Experiment: **0608_2028** -- v4d --- we use 0608_1050 for act feedback but with .95% prob now of equiprob during fine tuning


Experiment: **0608_2137** -- v6d --- feedback+attention we use 0608_1050 for act feedback but with .95% prob now of equiprob during fine tuning






----
## STORYLINE

### Baseline

1) HAR -- 0531_0551
2) HPE -- 0608_0803 (model.best and model.best.gold should be same)

```
# to generate results
python train.py -d0 -c configs/combined/combined_trivial_baseline.yaml -nl

# HAR
python test.py -r saved/BaselineLSTM/0531_0551/model_best.pth

# HPE
python test.py -r saved/BaselineHPE/0608_0803/model_best.pth

# combined
python test.py -r saved/BaselineCombined/0610_1056/model_best.pth
```


Trivial Baseline Results: ? Use combined_trivial_Baseline.yaml -> ~59% ; ~14.43mm?

### Test-Time Improvement

Two model arch, average_alpha=0.5, both HPEwActCond come very close so we try them both for future experiments..?

1) HPE -- 0608_0803
2) HPE wActCond -- 0609_0145 (res+lin) OR 0608_1050 (lin)
3) HAR -- 0531_0551

Results: SHOW TABLE
FIRST SHOW CHAGNE IN EQUIPROB PROB -- BEST IS 80-100%

THEN PICK SAY 95% AND SHOW CHANGES IN ALPHA -- BEST IS 0.5??

### Simple Connected CombinedTraining
This is version 2d test running on nebula: 0609_2057
Score best about 68-69%

- need to talk about training issues here..
- LR needed to be lowered
- needed to add weight decay penalty due to exploding weights trianing overfitting
- needed to ensure batchnorm and dropout both are fixed just one is not enough for batchnorm its because of the runningmeans for      droppout? probably not needed at this stage as its fine tuning most gradents already converged
- 


### Connected CombinedTraining Action Feedback
This is v4d, care must be taken whether comparing **0609_0145** based models or **0608_1050** based.
Some tests in nebula other maybe in server 2 backup
Generally these performed worst we tested 100% , 99% , 95% one of them came quite close... to baseline...

SHOW FINAL VALUE WITH THIS ;; THIS IS TYPE 2 ARCH
BETTER STICK TO **0608_1050** here its easier


### Connected CombinedTraining Action Feedback WITH ATTENTION
This is version 6d you can show that its still not improved so much....



### Extension of 2-Step Model To Train-Time
This is version 11d or 11d2 care must be taken which one you are doing for this you can show incremental changes to **0608_1050** and from there select best config then also test on **0609_0145** results are all around the place!

11d: keep only hpeact trainable
11d2: keep both trainable

11d2 is slightly better but takes a long time to train, to find good params we use 11d and once finalised on some good choices we try 11d2

See notes for this!!
results in server 2 and server 3
average_alpha: act_0c_alpha
action_equiprob_chance:
BaselineCombined

| Experiment | average_alpha | action_equiprob_prob | hpe_act| 11d or d2? |
| -----------| ------------- | -------------------- | ------ | ---------- |
| 0609_1533  | 0.5 | 0.95 | 7.181 | 11d  |
| 0609_1636  | 0.5 | 0.9  | 7.181 | 11d  |
| 0609_1814  | 0.5 | 0.8  | 7.181 | 11d  |
| 0609_1929  | 0.5 | 0.99 | 7.181 | 11d  |
| 0609_2103  | 0.5 | 1.00 | 7.181 | 11d  |
| 0609_0204  | 0.5 | 0.95 | 7.15  | 11d  |
| 0609_0506  | 0.5 | 0.95 | 7.15  | 11d2 |
| 0609_1236  | 0.5 | 0.9  | 7.15  | 11d  |
| 0609_1401  | 0.5 | 0.8  | 7.15  | 11d  |
| 0609_1519  | 0.5 | 0.99 | 7.15  | 11d  |
| 0609_0547  | 0.5 | 1.00 | 7.15  | 11d  |



| Experiment | average_alpha | action_equiprob_prob | hpe_act| 11d or d2? |
| -----------| ------------- | -------------------- | ------ | ---------- |
| 0609_1533  | 0.75 | 0.95 | 7.15 | 11d  |
| 0609_1636  | 0.75 | 0.9  | 7.15 | 11d  |
| 0609_1814  | 0.75 | 0.8  | 7.15 | 11d  |
| 0609_1929  | 0.75 | 0.99 | 7.15 | 11d  |
| 0609_2103  | 0.75 | 1.00 | 7.15 | 11d  |



ALL v2D Experiements -- using val 0.2 -- 3 epochs
(last few experiments can be tested in v3d = unrolled if no diff in v2d)
| Experiment | Config |
| ---------- | ------ |
| 0610_1157 | lr=0.003;wdecay=0;alpha=0.02 |
| 0610_1202 | lr=0.0003;wdecay=0;alpha=0.02 |
| 0610_1208 | **lr=0.00003;wdecay=0;alpha=0.02** |
| 0610_1231 | lr=0.000003;wdecay=0;alpha=0.02 |

| 0610_1239 | lr=0.00003;wdecay=0.1;alpha=0.02 |
| 0610_1245  | lr=0.00003;wdecay=0.001;alpha=0.02 |
| 0610_1251  | **lr=0.00003;wdecay=0.00001;alpha=0.02** |
| 0610_1256  | lr=0.00003;wdecay=0.0000001;alpha=0.02 |
| 0610_1208 | lr=0.00003;wdecay=0;alpha=0.02 |

| 0610_1458  | lr=0.00003;wdecay=0.00001;alpha=0.2 |
| 0610_1511  | lr=0.00003;wdecay=0.00001;alpha=0.02 |
| 0610_1524  | **lr=0.00003;wdecay=0.00001;alpha=0.002** |
| 0610_1537  | lr=0.00003;wdecay=0.00001;alpha=0.0002 |

|  0610_1511  | lr=0.00003;wdecay=0.00001;alpha=0.002;bn=Train;Dropout=Train |
|  0610_1617 | lr=0.00003;wdecay=0.00001;alpha=0.002;bn=Fixed;Dropout=Train |
|   | **lr=0.00003;wdecay=0.00001;alpha=0.002;bn=Fixed;Dropout=Fixed** |



### Testing 2-step training model with attention




### new experiments
all 30 epoch tests on act in act out
har is chosen as 100ep

if nothing useful gets out here then do 100ep test but itll be harder

