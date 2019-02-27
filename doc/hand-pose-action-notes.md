# Hand Pose Action


### Rough Notes on Intrinsic vs Extrinsic Params

#### Extrinsic Params
The Extrinsic Params represent a projection matrix in homogenous co-ords represented as:


<img src="https://latex.codecogs.com/gif.latex?\mathbm{P}=[\mathbm{R}|\mathbm{t}]" title="eqn" />


It consists of a rotation matrix $\bm{R}$ and a translation vector $\bm{t}$ and its sole purpose is to trnsform 3D world co-ordinates 3D co-ordinates w.r.t RGB camera frame.

Its needed because the 3D world co-ordinates are calibrated with the depth sensor i.e. 3D world co-ords $\equiv$ 3D depth sensor camera frame co-ords.

#### Intrinsic Params
The intrinsic params are the usual:


<img src="https://latex.codecogs.com/gif.latex?f=(f_x,f_y)&colon;\text{Focal&space;Length}" title="eqn" />

<img src="https://latex.codecogs.com/gif.latex?p=(p_x,p_y)\equiv(u_0,&space;v_0)&colon;\text{Principal&space;Point}" title="eqn" />
