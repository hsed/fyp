
## Background
MSRA Hand Gesture database is described in the paper
Cascaded Hand Pose Regression, Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang and Jian Sun, CVPR 2015.
Please cite the paper if you use this database.


## Dataset 
In total 9 subjects' right hands are captured using Intel's Creative Interactive Gesture Camera. Each subject has 17 gestures captured and there are 500 frames for each gesture. For subject X and gesture G, the depth images and ground truth 3D hand joints are under \PX\G\.

### Camera
The camera intrinsic parameters are: principle point = image center(160, 120), focal length = 241.42.


### Sample Description
While the depth image is 320x240, the valid hand region is usually much smaller. To save space, each *.bin file only stores the bounding box of the hand region.
Specifically, each bin file starts with 6 unsigned int: img_width img_height left top right bottom. [left, right) and [top, bottom) is the bounding box coordinate in this depth image.
The bin file then stores all the *depth pixel values* in the bounding box in *row scanning order*, which are  (right - left) * (bottom - top) floats. The unit is **millimeters**.
The bin file is *binary* and needs to be opened with *std::ios::binary flag*.

The corresponding *.jpg file is just for visualization of depth and ground truth joints.

*joint.txt* file stores *500 frames x 21 hand joints per frame*. Each line has *3 * 21 = 63 floats* for 21 3D points in (x, y, z) coordinates.
The 21 hand joints are: wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp, ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip, thumb_dip, thumb_tip.

For any questiones, please send email to yichenw@microsoft.com


## Data Loading
- As mentioned there are 9 subjects labelled P1,P2,...,P9

- As mentioned there are 17 guestures/poses labelled 1,2,....,9,,I,IP,L,MP,RP,T,TIP,Y for each subject
  - each gesture is a hand gesture e.g. making the '1' sign.

- First load all list of bin files these contain the 3d gt bounding box brop region for the hand plus all depth values of the 320x240 depth pixels present in the file. The closer the object, the lower the depth value. Usually it is 100mm closest and 700mm farthest. This acts or can act as our input i.e. training data. Values are in mm.

- Next, we have a file called joint.txt for each guesture/pose with exact 3D co-ords of 21 joints (in mm) for each of the 17 guestures. Within each guesture there are 500 frames or variations so this txt file has 500 lines one for each sample (out of 500). Each line has 3*21 mm values.

- bin files -> bounding box + depth values within boudning box
- joint.txt -> these are actual gt values of 3d loc of joint



###
Train_data_x:

(896, 1, 128, 128)

^^ this is our train data


x_train_data is train_data -==> self.train_data_x.container.value.shape

y_Train is y_train === this is 896.30?!!

        :param val_data: validation data
        :param val_y: validation labels

        (16128, 1, 128, 128) --> self_train_Data_xDB entire dataset....probably...


Procedure so far...
loadMiniBatch loads a sample from macro batch 'x' if macro batch 'x' is not the current loaded then that particular macro batch is loaded.


minibatch is what its trained on currently 896


training samples 16904

TODO:
- so firt load_model as theano

- then use train_X and train_y as the input and outputs respectively and pass through dataset and 


- use eval function to test back...
from datasets.msra_hand import *




### Data loading procedure..

- Open data file: `joint.txt` and read number of samples: N_SAMPLES as first line

- For i in range(N_SAMPLES):
  - Reat the corresponding ith line in `joint.txt` and store as GT_3D_COORDS_FLAT
  - Read correspoing ith `*.bin` file:
    - IMG_W; IMG_H; LEFT; TOP; RIGHT; BOTTOM; <--- first 6 values
    - {DEPTH_VAL0, DEPTH_VAL1, ..., DEPTH_VAL_N-1} <--- N depth values for N pixels where N = (RIGHT-LEFT)*(BOTTOM-TOP). Note: ALL OF THESE ARE VALUES RELATIVE AND WITHIN the TRUE GT BOUNDING BOX! Note 2: THESE ARE ONLY Z-AXIS (DEPTH) VALUES SO ONLY xVector i.e. INPUT to NN.
    - Place all these values in a 2D np.zeros matx IMGDATA with num_rows = IMG_H; num_cols = IMG_W;
    - The top:bottom and left:right region of this 2D matx indexes the hand bounding box so..
      - IMGDATA[top:bottom, left:right] = depth_values.reshape((bottom-top), (right-left))
      - At this this stage you have depth values relative to entire 'scene' (320x240)
  - Convert GT_3D_COORDS to matrix form as:
    - Originally: GT_3D_COORDS_FLAT = [joint_1_x, joint_1_y, joint_1_z, joint_2_x, ...., joint_21_z]
    - Now: GT_3D_COORDS = [[joint_1_x, joint_1_x, joint_1_x], [joint_2_x, .., ..,], ...., [.., .., joint_21_z] ]
    - Orig_shape => (63,); New_Shape => (21, 3)
    - Make sure all Z-VALS are first inverted i.e. GT_3D_COORDS[:, 2] *= -1 this can be done directly at import stage as well. See how V2V does it in `_load()` in `msrahand.py`.


- evaluation:
  - pass the x_2d coords centered through the model
  -  use ref point to transform back to REAL MM values i.e
        - mm distances of keypoints is w.r.t focal point of img
        - need to first transform pred_std_centered -> pred_mm_centered using the crop value
        - then transform pred_mm_centered -> pred_mm_not_centered by appending CoM value
        - now store this final value in 'keypoints'
        - also store gt_mm_not_centered in keypoints_gt for future error calc
        - calc error as avg_3D_error:
          - abs(R^{500, 21, 3} - R^{500, 21, 3}) => R^{500, 21, 3} abs_err between gt and pred
          - R^{500, 21, 3} *== avg_err_per_joint ==>* R^{500, 21} <-- do avg of x,y,z errors
          - R^{500, 21} *== avg_err_across_dataset ==>* R^{21}
          - R^{21} *== avg_err_across_joints ==>* R


- CURRENTLY:
  - we use only one gesture '1'
  - only 3 models 'P1', 'P2', 'P3'
  - test on 'P1', train on 'P2' & 'P3'
  - the CoM is using gt keypoint which is y_gt_keypoints[5] <-- middle finger mcp
  - ^^ we can used refinedCoM by editing `msra_hand: __getitem()__` however its wrong UNLESS we use ALL gestures due to the way it is listed in file. The com_refined file for each id is all train_data refine com for all subjects EXCEPT that test_id and in test_id its that data provided. so ~8\*17\*500 = ~ 68k train-set and 1 \* 17 \* 500 = ~ 8k in test-set totalling ~75.8k