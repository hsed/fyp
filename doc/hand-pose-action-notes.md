# Hand Pose Action


### Rough Notes on Intrinsic vs Extrinsic Params

#### Extrinsic Params
The Extrinsic Params represent a projection matrix in homogenous co-ords represented as:

$$
\bm{P} = [\bm{R} | \bm{t}]
$$

It consists of a rotation matrix $\bm{R}$ and a translation vector $\bm{t}$ and its sole purpose is to trnsform 3D world co-ordinates 3D co-ordinates w.r.t RGB camera frame.

Its needed because the 3D world co-ordinates are calibrated with the depth sensor i.e. 3D world co-ords $\equiv$ 3D depth sensor camera frame co-ords.

#### Intrinsic Params
The intrinsic params are the usual:

$$

$$