repeat all hpe cond experiments

repeat all equiprob % use experiments

repeat all combined experiments

HAR is fine!!

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

