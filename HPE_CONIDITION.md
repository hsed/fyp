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




Why does hpe with equiporb cond is the best output for 2-step model?
i think it is to do with temporal smoothing or something very similar, just look at how 2-step process happens.
First whole seq is converted to y domain then all of them combined to produce a FINAL action condition vector

This conditioning vector is a weak form of bias/conditioning because the network is trained on just quiprob versions of it it is learnt to just ignore it during training but since its not seen any other values for the action vector, it behaves in an unexpected way which to our benefit is better for us. it basically reacts by WHAT WE THINK making the output more constraint or temporally smooth w.r.t not using any conditioning this doesn't really help the hpe becaue the 3D error may be the same i.e. the pose might be the same but the intra-sequence (within seq or within class) correlation of sequences is stronger due to this new 'unkown' value being enforced throughout the resnet layers, basically this causes what we think less variance in the poses (need to some how test this maybe variance in error per frame?) but nevertheless this is waht we think causes an improvement in action recognition Top1 acc because the lstm is basically slightly more confident about the action class now or something to that effect.



## SOME NEW NOTES NOW IN LATEX ##

First of all why can't we select the best HPEwActCond?

The best model in stage-1 is the worst in stage-2, this is because it is expecting, stable, one-hot encoded, clean action labels. When it instead gets dirty, softmaxed, highly varying action conditions, the keypoint estimation output is terrmible about 30-40mm error. One thing that comes in mind is that the model is relying too much on action information


How can we alleviate this?

To help alleviate this issue we divise a method of training with equiprob vectors, so that the model pays less reliance on action conditions, we could have also tried noisy labels but during first stage this isn't available and studying how the nosiy action labels in stage-2 (combined training) can be modelled using some distribution during stage-1 is outside the scope of this project. Perhaps, this can be left for future work. Just using any random noise distribution won't work because it needs to match very closely to how an LSTM network's action_layers would predict at each timestep (not just in the end).


How to find the best equiprob trained model?
**NOTE CONFIRM THIS FROM RECENT EXP**
ALL EXP FOR 30 EP
0->Baseline,7->ResNet,7.16->Lin, 7.17->Lin++, 7.18->Res+Lin, 7.19->Res+Lin++ (note for lin just show lin++ results unless lin is better)
So now we have our requirement, we wish to use an equiprob trained model, that performs ATLEAST as well as the non-cond model when supplied equiprob. With this we try several different architectures which included conditioning in only linear layers, resnet layers, or both. In conclusion we observed that the linear type architectures, that while performed best when supplied with gt act, performs generally worse (observe the large bias) than the baseline. They do not meet our requirement. The only model type that meets our requirement is cond7.0->resnet act cond. This can be explained as it is conditioning in conv layers thus supplying redundant information is not harmful to the resnet arch (maybe due to residual nature?).
**PLEASE CONFIRM THIS SO FAR v7 is the best**


Please note that we also perform some experiements with supplying some gt action for certain proportion of the time but this generally leads to divergence after 15-30 epochs, thus this wasn't explored further...

How to change the model in type-2 to accept noisy act condition?
Naturally, just directly supplying noisy act cond in stage-2 to a model which in stage-1 was **only ever shown one type of vector** i.e. the **clean equiprob** vector would be disasterous. This was proven by switching to act-cond6 in stage 2 and just validating without training with the 2-step model showed val perf: 12% acc and 30.97mm error, where the baseline with no feedback performed 0.42 and 20.821mm...

This showed that it is impossible to train such an arch with action supplied during all timesteps. Nevertheless to show, the usefulness of action information we supply it X% of time during stage 2 training. after first getting scores during *validation-without-training* we saw that about 5% of action can be given without perf degrading worse than baseline, nevertheless training experiments showed that 1% was the best choice. 

Also, using action increases accurracy at the cost of avg 3d error degrading there is a trade-off there due to gt keypoints appearing smooth in the dataset, due to such limitations of the dataset we need to keep into accound the avg3d and the action acc.

so in that case the increased bias in error is neglegible in 1% but its more noticeable in 5%

NOTE OF PLOT: 
Show a plot where no action is supplied and training is done then show a plot where 1% action supplied and training is done
then show a plot with 5% and training is done and then with 10%

We would expect 10% to be defo bad
5% maybe on margin
1% is actually better!!

**TABLE IS OUTDATED DO AGAIN WITH FIXED DATA**
CONFIG: combined_act_in_act_ou.yaml
HPE: 0527_1455 (VAL_ACC_ON_TEST: 21.97mm) (Best of 30EP Trained)
HPE_wActCond: 0527_1508 (VAL_ACC_ON_TEST: 21.86mm) (ResNet Cond + Equiprob 100% "7" Stage-1) (Best of 30EP Trained)
HAR: (GT_ACC: 72.3%) (Best of 100EP)
TESTED ON DIRECT VALIDATION WITHOUT ANY TRAINING
7,7,7.15 ALL USE ONLY RESNET COND
6->0% Equiprob, 7->100% Equiprob, 7.15->X% Equiprob
| Action Condition | Proportion of 2nd Pass | Prob of Equiprob | VAL_ACC | VAL_3D |
| ---- | ---- | ----- | ----- | ----- |
| 0    | 0%   | N/A   | 38.4% | 21.62mm
| 7    | 50%  | 100%  | 42%   | 20.82mm
| 7    | 100% | 100%  | 40.7% | 21.46mm
| 6    | 100% | N/A   | 12.2% | 30.98mm
| 7.15 | 100% | 100%  | 40.7% | 21.46mm
| 7.15 | 100% | 99.9% | 40.7% | 21.46mm
| 7.15 | 100% | 99.0% | 39.7% | 21.66mm
| 7.15 | 100% | 95.0% | 39.0% | 21.90mm
| 7.15 | 100% | 90.0% | 36.5% | 22.56mm
| 7.15 | 100% | 80.0% | 36.0% | 22.85mm
| 7.15 | 100% | 60.0% | 29.9% | 24.81mm
| 7.15 | 100% | 20.0% | 18.9% | 28.67mm
| 7.15 | 100% |  0.0% | 12.2% | 30.98mm


From here good candidates are 95% equiprob and 99% equiprob, anything below than that is worse than baseline, so it won't produce as good as imrpovements when trained compared to baseline. but we will test 90%, 95% and 99% for 10 epochs and see the trend. The best performing will be chosen for further experiments.


FIX BUG IN OTHER COMBINED METHODS!!!
NOTE: DEFAULT IS COND 7 ; WITHO THIS EVERYTHING IS SAME AS BEFORE!



what about dynamic conditioning ... increasing the prob of action conditioning every 2 epochs by 1%?


experiment on 5%
experiment on 10%

experiment using dynamic cond:
  +1% of act_cond every 2 epochs the model can handle noisy action then show it up more...


experiment the best one using attention and then using attention+smoothing...






**BaselineCombined/0607_1056**
config: combined_feedback_dual_train.yaml
action_cond: v7 (pre-training); v15 (fine-tuning); v15 with equiprob = 99%;
its better than before for most epochs until 22 ep it thne starts to decline it almost touches baseline (no act feedback)
but is actually worse from 22ep to 30ep compared to baseline_v7 (equiprob = 100%). this shows that 95% might be probably bad we dont know need to experiment!