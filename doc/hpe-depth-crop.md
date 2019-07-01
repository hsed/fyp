
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