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