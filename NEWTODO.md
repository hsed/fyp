# FYP Aims (Revision 15/5/19)
Notation I:

- HAR/HPE: Hand Action Recognizer/Hand Pose Estimator
- `a` & `b`: two different versions (e.g. prediction using two different models) of the same modality.
- `a=b`: a has similar performance compared to b w.r.t ground truth (gt)
- `a<b`: a has worse performance compared to b w.r.t gt
- `a>b`: a has better performance compared to b w.r.t gt

Notation II:
- `x`: (gt depth); `y`: (gt 3D keypoints); `z`: (gt action class)
- `y'`: keypoints predicted using HPE trained on `x,y` ALONE 
- `z'`: action predicted using HAR trained on `y,z` ALONE

- `y''`: keypoints predicted using HPE trained on `x,y` ALONE and **tested on x** (anticipated `y'' = y'`)
- `z''`: action predicted using HAR trained on `y,z` ALONE and **tested on y''** (anticipated `z'' < z'`)

- `y'''`: keypoints predicted using arbitrary joint model trained on `x,y,z` and **tested on `x`**
- `z'''`: action predicted using arbitrary joint model trained on `x,y,z` and **tested on `x`**


Aims:
- The whole project in one sentence: 
  You have **3 modalities** (`x,y,z`) at train time and only **1 modality** `x` at test time. Task is to combine near state of art baselines that deliver `x->y'` and `y->z'` such that you can produce `x->(y''',z''')`.


- The problem is non-trivial since `x->y'` is best performed on individual frames whereas `y->z'` is performed on temporal sequences. There are many possible ways to combine two such classes of models.

- The most trivial baseline is `x->y'` and `y->z'` during *training* and then combining these models linearly as `x->y''->z''` during *test* time (see Notation II). Here `y''` is undegraded i.e. `y''=y'` but `z''` is an *inferior* version of `z'` i.e. `z'' < z'`. Note that `y''` is a **trivial baseline case** of `y'''` and similarly for `z''` vs `z'''`.

- In general, theoretically we anticipate either `y'''>y'` OR `z'''>z'` but it doesn't make sense for both to be better due to the *chicken and egg problem*.

- Nevertheless, if we can show that for any arbitrary model `x->(y''',z''')` if `y''' > y'' = y'` AND `z''' > z''` (or the same argument but y and z exchanged) then we have achieved the main aim of the project. In words: atleast one out of two output modalities is enhanced by using side information and the other output modality is better with respect to the trivial baseline. Definitely both modalities should be better with respect to the trivial baseline.