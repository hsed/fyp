import os
import sys
import multiprocessing
import ctypes

from enum import IntEnum, Enum

import cv2
import torch
import numpy as np

from .data_augmentors import *
from datasets import ExtendedDataType as DT, FHADCameraIntrinsics as CAM, \
                             DepthParameters as DepParam

from .debug_plot import plotImg

from sklearn.decomposition import PCA


def standardiseImg(depth_img, com_dpt_mm, crop_dpt_mm, extrema=(-1,1), copy_arr=False):
    # create a copy to prevent issues to original array
    # standardises based on assumption that:
    # max_depth (farthest) = com_dpt_mm + com_dpt_mm/2;
    # min_depth (closest) = com_dpt_mm - com_dpt_mm/2
    # ALL 0. POINTS are assumed at INF are set to MAX_DEPTH value
    # Thus all such points then become +1 in the end.
    # TODO: Change this for new dataset and no z thresholding

    if copy_arr:
        depth_img = np.asarray(depth_img.copy())
    if extrema == (-1,1):
        depth_img[depth_img == 0] = com_dpt_mm + (crop_dpt_mm / 2.)
        depth_img -= com_dpt_mm
        depth_img /= (com_dpt_mm / 2.)
    elif extrema == (0, 1):
        depth_img[depth_img == 0] = com_dpt_mm + (crop_dpt_mm / 2.)
        depth_img -= (com_dpt_mm - (crop_dpt_mm / 2.))
        depth_img /= crop_dpt_mm
    else:
        raise NotImplementedError("Please use a valid extrema.")
    
    return depth_img

def standardiseKeyPointsCube(keypoints_mm, cube_side_mm, copy_arr=False):
    '''
        standardises based on one-side only
        Assumption: 
        All sides equal so standardises by same factor along each axis\n
        `cube_side_mm` => the z-axis or any other crop length in mm
        returns val in range [-1, 1]
    '''
    ## only one standarisation method, I reckon this is -1 -> 1 standardisation
    ## as by default max values will be -crop3D_sz_mm/2, +crop3D_sz_mm/2
    ## biggest assumption is that crop vol is a regular cube i.e. all sides equal
    ## if not then we need to scale per axis
    ## keypoints are the one relative to CoM
    ## TODO: change this for no Z_thresholding or better normalisation
    if copy_arr:
        keypoints_mm = np.asarray(keypoints_mm.copy())
    return keypoints_mm / (cube_side_mm / 2.)

def unStandardiseKeyPointsCube(keypoints_std, cube_side_mm):
    '''
        `keypoints_std` => keypoints in range [-1, 1]\n
        `cube_side_mm` => any crop length in mm (all sides assumed equal)\n
        returns val in range [-cube_side_mm/2, +cube_side_mm/2]
    '''
    return keypoints_std * (cube_side_mm / 2.)


## from deep-prior
class Pixels2MM(object):
    """
        Converts a collection of (fractional) pixel indices (w.r.t an image
        with original dimensions (un-cropped)) to a collection of mm values
        (w.r.t the original focal point i.e 0mm at focal point).
        :param sample: row vectors in (x,y,z) with x,y in original image 
        coordinates and z in mm
        :return: {(x1,y1,z1), ..., (xN, yN, zN)} in mm 
    """
    def __init__(self, cam_intrinsics = CAM):
        cam_intr = np.array([[cam_intrinsics.FX.value, 0, cam_intrinsics.UX.value],
                         [0, cam_intrinsics.FY.value, cam_intrinsics.UY.value], [0, 0, 1]])
        # store inverse for later use
        self.inv_cam_intr = np.linalg.inv(cam_intr)
    
    def __call__(self, sample):
        # rescale x,y by z-values then append orig z values
        # then transpose, then dot by inverse intrinsic then transpose
        return \
            self.inv_cam_intr\
                .dot(np.column_stack((sample[:,:2]*sample[:, 2:], sample[:, 2:])).T).T

class Pixel2MM(object):
    """
        Converts (fractional) pixel indices (w.r.t an image
        with original dimensions (un-cropped)) to mm values
        (w.r.t the original focal point i.e 0mm at focal point).
        :param sample: vector in (x,y,z) with x,y in original image 
        coordinates and z in mm
        :return: (x,y,z) in mm 
    """
    def __init__(self, cam_intrinsics = CAM):
        cam_intr = np.array([[cam_intrinsics.FX.value, 0, cam_intrinsics.UX.value],
                         [0, cam_intrinsics.FY.value, cam_intrinsics.UY.value], [0, 0, 1]])
        # store inverse for later use
        self.inv_cam_intr = np.linalg.inv(cam_intr)
    
    def __call__(self, sample):
        # rescale x,y by z-values then append orig z values
        # then transpose, then dot by inverse intrinsic then transpose
        return self.inv_cam_intr.dot(np.append(sample[:2]*sample[2], sample[2]))


class MM2Pixels(object):
    """
        Converts a collection of mm values (w.r.t the original focal point 
        i.e 0mm at focal point) to a collection of indices with px values
        (w.r.t an image of original dimensions (un-cropped)).
        :param sample: row vectors in (x,y,z) with x,y,z in mm

        needs to be float because keypoints can exist in between pixels

        :return: {(x1,y1,z1), ..., (xN, yN, zN)} with x,y in original image 
        coordinates and z in original mm form (untouched)
    """
    def __init__(self, cam_intrinsics = CAM):
        self.cam_intr = np.array([[cam_intrinsics.FX.value, 0, cam_intrinsics.UX.value],
                         [0, cam_intrinsics.FY.value, cam_intrinsics.UY.value], [0, 0, 1]])

    def __call__(self, sample):
        # simpler version -- equivalent
        # skel_hom2d = np.array(cam_intr).dot(keypt_mm_orig.transpose()).transpose()
        # keypt_px_orig = np.column_stack(((skel_hom2d[:, :2] / skel_hom2d[:, 2:]), skel_hom2d[:, 2]))
        #
        # scale down by factor of z, this will make the z axis values == 1
        # notice how in skel_hom2d the z values are exactly the same as original i.e. unaffected by transform
        # the '[:, 2:]' part actually ensure to keep the dim (21,1) rather than (21,) so its a neat shortcut 
        # to reshape, actually the dim is (21,3), if we just do (21,3)/(21,) then numpy throws error
        # for col_stack however, it works fine to stack 1d array to 2d
        #
        return np.column_stack(((self.cam_intr.dot(sample.T).T[:, :2] / sample[:, 2:]),
                              sample[:, 2]))

class MM2Pixel(object):
    """
    Denormalize each joint/keypoint from metric 3D to image coordinates
    Per joint version of MM2Pixels for 1D array
    :param sample: joints in (x,y,z) (Shape = (3,)) with x,y and z in mm
    :return: joint in (x,y,z) with x,y in image coordinates and z in mm
    """
    def __init__(self, cam_intrinsics = CAM):
        self.cam_intr = np.array([[cam_intrinsics.FX.value, 0, cam_intrinsics.UX.value],
                         [0, cam_intrinsics.FY.value, cam_intrinsics.UY.value], [0, 0, 1]])
    
    def __call__(self, sample):
        return np.append(self.cam_intr.dot(sample)[:2] / sample[2], sample[2])


class TransformerBase(object):
    '''
        Class to initialise all properties useful to any or all transformers

        
    '''


    def __init__(self, num_joints = 21, world_dim = 3, cube_side_mm = 200,
                 cam_intrinsics = CAM, dep_params = DepParam, debug_mode=False):
        self.num_joints = num_joints
        self.world_dim = world_dim

        # only supported crop shape is regular cube with all sides equal
        self.crop_shape3D_mm = (cube_side_mm, cube_side_mm, cube_side_mm)
        
        #intrinsic camera params
        self.cam_intrinsics = cam_intrinsics



        # output sz in px; only 1:1 ratio supported
        self.out_sz_px = (dep_params.OUT_PX.value, dep_params.OUT_PX.value)

        self.dpt_range_mm = dep_params.DPT_RANGE_MM.value

        self.debug_mode = debug_mode


class JointReshaper(TransformerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sample):
        sample[DT.JOINTS] = \
            sample[DT.JOINTS].reshape(self.num_joints, self.world_dim)

        return sample


class JointSeqReshaper(TransformerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sample):
        # convert frames x 63 => frames x 21 x 3
        sample[DT.JOINTS_SEQ] = \
            sample[DT.JOINTS_SEQ].reshape(-1, self.num_joints, self.world_dim)

        return sample


class JointCentererStandardiser(TransformerBase):
    '''
        
        A simple transformer for centering joints w.r.t CoM point
        For HPE, provides centered joints as one 2D matrix per sample

        Also performs standardisation to set values in range -> (-1,1)

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class
    

    def __call__(self, sample):
        
        #joints = sample[DT.JOINTS]#.reshape(self.num_joints, self.world_dim)
        #coms = sample[DT.COM]

        com = sample[DT.COM]
        dpt_range = self.crop_shape3D_mm[2]

        # only perform aug adjustments if aug actually happened
        if DT.AUG_MODE in sample:
            if sample[DT.AUG_MODE] == AugType.AUG_TRANS:
                # add augmentation offset
                com = com + sample[DT.AUG_PARAMS]
            if sample[DT.AUG_MODE] == AugType.AUG_SC:
                # scale dpt_range accordingly
                dpt_range = dpt_range*sample[DT.AUG_PARAMS]
        
        ## broadcasting is done automatically here -- centering: CoM is new origin in 3D space
        sample[DT.JOINTS] = sample[DT.JOINTS] - com

        ## standardisation -- values between -1 and 1
        sample[DT.JOINTS] = \
            standardiseKeyPointsCube(sample[DT.JOINTS], dpt_range)\
                                     .reshape(self.num_joints*self.world_dim)

        ### all other values returned as is
        return sample


class JointSeqCentererStandardiser(TransformerBase):
    '''
        A simple transformer for centering joints w.r.t CoM point
        For HAR, provides centered joints as a collection (3D matrix) per sample.

        In Joints_(seq), CoM_(seq) => Out Joints_(seq)_centered :: DIM: (Seq, 63)
        Also performs standardisation to set values in range -> (-1,1)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class
    

    def __call__(self, sample):
        '''
            sample = {
                'joints_seq': _,
                'action_seq': _,
            }
        '''

        ## TODO: adjust com and crop_shape3D_mm if data is augmented!
        ## see non-sequence version for example
        
        joints_seq = sample[DT.JOINTS_SEQ]#.reshape(-1, self.num_joints, self.world_dim)
        coms_seq = sample[DT.COM_SEQ]
        
        sample[DT.JOINTS_SEQ] = joints_seq - coms_seq[:, np.newaxis, :]

        sample[DT.JOINTS_SEQ] = \
            standardiseKeyPointsCube(sample[DT.JOINTS_SEQ], self.crop_shape3D_mm[2])\
                                     .reshape(-1, self.num_joints*self.world_dim)

        ### all other values returned as is
        return sample


class JointUnstandardiser(TransformerBase):
    '''

        Performs unstandardisation to set values in range
        (-1,1) -> (-crop_shape3D_mm/2, +crop_shape3D_mm/2)

        Note: this does NOT do decentering!
        Note 2: No fixes (special cases) for data augmentation so only use
        for testing!

        Does reshape (63,) -> (21,3)

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class
    

    def __call__(self, sample):

        ## ustandardisation -- values between -1 and 1
        sample[DT.JOINTS] = \
            unStandardiseKeyPointsCube(sample[DT.JOINTS], self.crop_shape3D_mm[2])\
                                     .reshape(self.num_joints, self.world_dim)

        ### all other values returned as is
        return sample



class ActionOneHotEncoder(TransformerBase):
    '''
        A simple transformer for encoding action idx into one-hot vectors
        For HAR, provides one encoded vector per sample

        In Action_idx => Out Action_one-hot
    '''
    def __init__(self, action_classes = 45, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class
        self.action_classes = action_classes

    def __call__(self, sample):
        action_idx = sample[DT.ACTION]
        action_one_hot = np.zeros(self.action_classes)
        action_one_hot[action_idx] = 1

        sample[DT.ACTION] = action_one_hot

        return sample



class DepthCropper(TransformerBase):
    '''
        Apply crop to depth images
        If set in init, also return transform matrices for use with 2D pts

        In Depth => Out Depth_centered_cropped + Transform_Matrix_Crop

        Default values for fx,fy (focal length) ; ux, uy (image center) are for FHAD

        Depth images are 640 x 480
    '''
    


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class
        
        self.mm2px_multi = MM2Pixels(cam_intrinsics=self.cam_intrinsics)
        self.mm2px = MM2Pixel(cam_intrinsics=self.cam_intrinsics)
        #self.px2mm_multi = Pixels2MM(cam_intrinsics=self.cam_intrinsics)

    def __call__(self, sample):
        
        ### orig data loading ###
        ## reshape already done
        dpt_orig = sample[DT.DEPTH]
        keypt_mm_orig = sample[DT.JOINTS]
        com_mm_orig = sample[DT.COM]

        ## convert joints & CoM to img coords
        ## note this is right for original image but wrong for cropped pic
        keypt_px_orig = self.mm2px_multi(keypt_mm_orig)
        com_px_orig = self.mm2px(com_mm_orig)


        #print("KeyptShape: ", keypt_px_orig.shape, "Com_shape:", com_px_orig.shape)
        #print("KeyptVals:\n", keypt_px_orig[:3,:], "\nCom_val:", com_px_orig)
        ### cropping + centering ###
        ## convert input image to cropped, centered and resized 2d img
        ## required CoM in px value
        ## convert to 128x128 for network
        ## px_transform_matx for 2D transform of keypoints in px values
        dpt_crop, crop_transf_matx = cropDepth2D(
                                                dpt_orig, com_px_orig,
                                                fx=self.cam_intrinsics.FX.value,
                                                fy=self.cam_intrinsics.FY.value,
                                                crop3D_mm=(self.dpt_range_mm,
                                                           self.dpt_range_mm,
                                                           self.dpt_range_mm),
                                                out2D_px=self.out_sz_px
                                                )
        
        #if self.debug_mode:
            #plotImg(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, crop_transf_matx)
            #quit()
        
        ### all extra types are used for augmentation or can be ignored ###
        ## exrtra datatypes
        sample[DT.DEPTH_ORIG] = dpt_orig
        sample[DT.COM_ORIG_PX] = com_px_orig
        sample[DT.JOINTS_ORIG_PX] = keypt_px_orig
        
        ## extra datatypes
        sample[DT.COM_ORIG_MM] = com_mm_orig
        sample[DT.JOINTS_ORIG_MM] = keypt_mm_orig

        ## actual useful data type and tranf matx
        ## float32 type is needed for standardisation, done automatically
        ## if augmentation is called
        if dpt_crop is not None: sample[DT.DEPTH] = dpt_crop.astype(np.float32)
        sample[DT.CROP_TRANSF_MATX] = crop_transf_matx

        return sample


class DepthSeqCropper(object):
    '''
        Apply crop to depth images
        If set in init, also return transform matrices for use with 2D pts

        In Depth_(seq) => Out Depth_(seq)_centered_cropped + Transform_Matrix_Crop_Seq
    '''


class DepthAndJointsAugmenter(TransformerBase):
    '''
        Perform Augmenting for both depth and keypoints

        For Future use see how its implemented not so important atm 
        Only mainly useful for HPE maybe this should only be for HPE

        In:
        Depth_cropped (+ DPT orig as seperate)
        keypt_px_orig
        com_px_orig

        Out:
        depth_cropped_Aug
        keypt_px_orig_aug
        com_px_orig_aug

        Now key_pt_px_aug and com_px_aug is mult by Transform_Matrix_Crop
        to get keypt_mm_aug, com_mm_aug

        Centering is done in end using these last values
    '''
    def __init__(self, scale_std=0.02, trans_std=5, abs_rot_lim_deg=180,
                 aug_mode_lst = [AugType.AUG_NONE], **kwargs):
        super().__init__(**kwargs)
        self.fx = self.cam_intrinsics.FX.value
        self.fy = self.cam_intrinsics.FY.value

        self.mm2px_multi = MM2Pixels(cam_intrinsics=self.cam_intrinsics)
        self.mm2px = MM2Pixel(cam_intrinsics=self.cam_intrinsics)

        self.px2mm_multi = Pixels2MM(cam_intrinsics=self.cam_intrinsics)
        self.px2mm = Pixel2MM(cam_intrinsics=self.cam_intrinsics)

        self.rot_lim = abs_rot_lim_deg if abs_rot_lim_deg <=180 else 180
        self.sc_std = scale_std if scale_std <= 0.02 else 0.02
        self.tr_std = trans_std if trans_std <= 5 else 5
        self.aug_mode_lst = aug_mode_lst

        # temporary TODO: change this to how its done originally basically
        # project x,y keypoitn to image cords and draw 40px bounding box enclosing
        # image
        self.crop_vol_mm = (self.dpt_range_mm, self.dpt_range_mm, self.dpt_range_mm)
    

    def __call__(self, sample):
        dpt_orig = sample[DT.DEPTH_ORIG]
        com_px_orig = sample[DT.COM_ORIG_PX]
        keypt_px_orig = sample[DT.JOINTS_ORIG_PX]
        
        ## extra datatypes
        com_mm_orig = sample[DT.COM_ORIG_MM]
        keypt_mm_orig = sample[DT.JOINTS_ORIG_MM]

        ## actual useful data type and tranf matx
        dpt_crop = sample[DT.DEPTH]
        crop_transf_matx = sample[DT.CROP_TRANSF_MATX]
        
        
        aug_mode, aug_param = getAugModeParam(self.aug_mode_lst, self.rot_lim, 
                                                self.sc_std, self.tr_std)
        
        (dpt_crop_aug, keypt_px_orig_aug, com_px_orig_aug, aug_transf_matx) = \
            rotateHand2D(dpt_crop, keypt_px_orig, com_px_orig, aug_param) \
            if aug_mode == AugType.AUG_ROT \
            else translateHand2D(dpt_crop, keypt_px_orig, com_px_orig, com_mm_orig, aug_param, 
                                    self.fx, self.fy, crop_transf_matx, self.mm2px, self.crop_vol_mm) \
            if aug_mode == AugType.AUG_TRANS \
            else scaleHand2D(dpt_crop, keypt_px_orig, com_px_orig, com_mm_orig, aug_param, 
                                    self.fx, self.fy, crop_transf_matx,
                                    self.mm2px, crop3D_mm=self.crop_vol_mm) \
            if aug_mode == AugType.AUG_SC \
            else (dpt_crop, keypt_px_orig, com_px_orig, np.eye(3, dtype=np.float32))

        ## the'centering part' is now done by another transform
        #keypt_mm_crop_aug = self.px2mm_multi(keypt_px_orig_aug) - self.px2mm_multi(com_px_orig_aug)
        
        ### debugging using plots ###
        ## final input (before std) is dpt_crop_aug
        #self.debug_mode=False
        plotImg(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, 
                crop_transf_matx=crop_transf_matx, aug_transf_matx=aug_transf_matx,
                aug_mode=aug_mode, aug_val=aug_param, dpt_crop_aug=dpt_crop_aug) \
                    if (self.debug_mode and dpt_orig is not None) else None

        # store mm_orig_aug as default value, now only standardisation is needed
        sample[DT.DEPTH] = dpt_crop_aug
        sample[DT.JOINTS] = self.px2mm_multi(keypt_px_orig_aug) # keypt_mm_orig_aug
        sample[DT.COM] = self.px2mm(com_px_orig_aug) # com_mm_orig_aug
        sample[DT.AUG_MODE] = aug_mode
        sample[DT.AUG_PARAMS] = aug_param

        return sample
        

class DepthStandardiser(TransformerBase):
    '''
        Before transformation, image is cast to float32, resulting value is float32
        This is needed when no augmentation is done
        To allow for further calc; orig type is np.int32
        A simple transformer for standardising depth values as:
        -1 (nearest) -> +1 (furthest), all 0. values are assumed
        points at inf and are set to +1.
        For HPE, provides centered joints as one 2D matrix per sample
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class
    

    def __call__(self, sample):
        
        ### Standardisation ###
        ## This must be the last step always!
        # TODO: dpt_range_mm => 200mm change to something more meaningful for FHAD
        # TODO: also when there is data aug of scale or translation this maybe
        # wrong as the cube depth is now changed or com z_val has changed
        
        com_z = sample[DT.COM_ORIG_MM][2]
        dpt_range = self.dpt_range_mm
        
        # only perform aug adjustments if aug actually happened
        if DT.AUG_MODE in sample:
            if sample[DT.AUG_MODE] == AugType.AUG_TRANS:
                # add augmentation offset
                com_z = com_z + sample[DT.AUG_PARAMS][2]
            if sample[DT.AUG_MODE] == AugType.AUG_SC:
                # scale dpt_range accordingly
                dpt_range = dpt_range*sample[DT.AUG_PARAMS]

        sample[DT.DEPTH] = \
            standardiseImg(sample[DT.DEPTH],
                           com_z, dpt_range)[np.newaxis, ...]
          
        return sample


class PCATransformer(TransformerBase):
    '''
        A simple PCA transformer, supports PCA calc from data matrix and only single sample transformations\n
        `device` & `dtype` => torch device and dtype to use.                            
        `n_components` => Final PCA_components to keep.                         
        `use_cache` => whether to load from cache if exists or save to if doesn't.
        `overwrite_cache` => whether to force calc of new PCA and save new results to disk, 
        overwriting any prev.\n
        PCA is calculated using SVD of co-var matrix using torch (can be GPU) but any subsequent calls
        transform numpy-type samples to new subspace using numpy arrays.
    '''
    def __init__(self, dtype=torch.float,
                 n_components=30, use_cache=False, overwrite_cache=False,
                 cache_dir='datasets/hand_pose_action'):
        
        ## filled on fit, after fit in cuda the np versions copy the matrix and mean vect
        self.transform_matrix_torch = None
        self.transform_matrix_np = None

        self.mean_vect_torch = None
        self.mean_vect_np = None
        
        ## filled on predict
        self.dist_matx_torch = None

        self.device = torch.device('cpu')   # this is needed so keep as in
        self.dtype = dtype
        self.out_dim = n_components

        self.use_cache = use_cache
        self.overwrite_cache = overwrite_cache
        self.cache_dir = cache_dir

        if self.use_cache and not self.overwrite_cache:
            self._load_cache()
                
                
                    

    def __call__(self, sample):
        '''
            sample is tuple of 2 np.array (x,y)
            later make this a dictionary
            single y_data sample is 1D
        '''
        if self.transform_matrix_np is None:
            raise RuntimeError("Please call fit first before calling transform.")
        
        #y_data = sample[1]
        
        # automatically do broadcasting if needed, but in this case we will only have 1 sample
        # note our matrix is U.T
        # though y_data is 1D array matmul automatically handles that and reshape y to col vector
        return (sample[0], np.matmul(self.transform_matrix_np, (sample[1] - self.mean_vect_np)))
    
    
    def _load_cache(self):
        cache_file = os.path.join(self.cache_dir, 'pca_'+str(self.out_dim)+'_cache.npz')
        if os.path.isfile(cache_file):
            npzfile = np.load(cache_file)

            matx_shape = npzfile['transform_matrix_np'].shape
            vect_shape = npzfile['mean_vect_np'].shape

            shared_array_base = multiprocessing.Array(ctypes.c_float, matx_shape[0]*matx_shape[1])
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            
            shared_array_base2 = multiprocessing.Array(ctypes.c_float, vect_shape[0])
            shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
            
            shared_array = shared_array.reshape(matx_shape[0], matx_shape[1])
            #print("SharedArrShape: ", shared_array.shape)

            shared_array[:, :] = npzfile['transform_matrix_np']
            shared_array2[:] = npzfile['mean_vect_np']

            self.transform_matrix_np = shared_array
            self.mean_vect_np = shared_array2
        else:
            # handles both no file and no cache dir
            Warning("PCA cache file not found, a new one will be created after PCA calc.")
            self.overwrite_cache = True # to ensure new pca matx is saved after calc
    
    def _save_cache(self):
        ## assert is a keyword not a function!
        assert (self.transform_matrix_np is not None), "Error: no transform matrix to save."
        assert (self.mean_vect_np is not None), "Error: no mean vect to save."

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        cache_file = os.path.join(self.cache_dir, 'pca_'+str(self.out_dim)+'_cache.npz')
        np.savez(cache_file, transform_matrix_np=self.transform_matrix_np, mean_vect_np=self.mean_vect_np)


    
    
    ## create inverse transform function?
    
    def fit(self, X, return_X_no_mean=False):
        ## assume input is of torch type
        ## can put if condition here
        if X.dtype != self.dtype or X.device != self.device:
            # make sure to switch to float32 if thats default
            # note need '=' as this is not inplace operation!
            X = X.to(device=self.device, dtype=self.dtype)
        
        if self.transform_matrix_np is not None:
            Warning("PCA transform matx already exists, refitting...")


        sklearn_pca = PCA(n_components=self.out_dim)
        sklearn_pca.fit(X.cpu().detach().numpy())

        self.transform_matrix_np = sklearn_pca.components_.copy()
        self.mean_vect_np = sklearn_pca.mean_.copy()

        del sklearn_pca # no longer needed

        shared_array_base = multiprocessing.Array(ctypes.c_float, self.transform_matrix_np.shape[0]*self.transform_matrix_np.shape[1])
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        
        shared_array_base2 = multiprocessing.Array(ctypes.c_float, self.mean_vect_np.shape[0])
        shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
        
        shared_array = shared_array.reshape(self.transform_matrix_np.shape[0], self.transform_matrix_np.shape[1])
        #print("SharedMemArr", shared_array.shape)

        shared_array[:, :] = self.transform_matrix_np
        shared_array2[:] = self.mean_vect_np

        del self.transform_matrix_np
        del self.mean_vect_np

        self.transform_matrix_np = shared_array
        self.mean_vect_np = shared_array2

        self.transform_matrix_np.setflags(write=False)
        self.mean_vect_np.setflags(write=False)

        if self.use_cache and self.overwrite_cache:
            # only saving if using cache feature and allowed to overwrite
            # if initially no file exits overwrite_cache is set to True in _load_cache
            self._save_cache()
        
        if return_X_no_mean:
            # returns the X matrix as mean removed! Also a torch tensor!
            print("TYPES: ", "X_TYPE: ", X.dtype, "OTHER: ", torch.from_numpy(self.mean_vect_np).expand_as(X).dtype)
            return X - torch.from_numpy(self.mean_vect_np).expand_as(X)
        else:
            return None
    
    
    def fit_transform(self, X):
        ## return transformed features for now
        ## by multiplying appropriate matrix with mean removed X
        # note fit if given return_X_no_mean returns X_no_mean_Torch
        return np.matmul(self.fit(X, return_X_no_mean=True).numpy(), self.transform_matrix_np.T)


class ToTuple(object):
    '''
        Final transformer to convert dictionary to tuple
        aka modality extractor

        Requires during init if its HAR or HPE
        Requires during init what modalities to output: depth: Y/N; joint: Y/N; action: Y/N

        Note:
        For Action no seq is returned in either HPE or HAR case
    '''
    class ET(Enum):
        '''
            ExtractType used internally
        '''
        JOINTS_SEQ = 'joints_seq'
        JOINTS_ACTION_SEQ = 'joints_action_seq'
        DEPTH_JOINTS_SEQ = 'depth_joints_seq'
        DEPTH_JOINTS_ACTION_SEQ = 'depth_joints_action_seq'
        JOINTS = 'joints'
        JOINTS_ACTION = 'joints_action'
        DEPTH_JOINTS = 'depth_joints'
        DEPTH_JOINTS_ACTION = 'depth_joints_action'


    def __init__(self, extract_type = 'joints_action_seq'):

        if extract_type not in ToTuple.ET._value2member_map_:
            raise RuntimeError("Invalid extract type, choose from: ",
                               ToTuple.ET._value2member_map_.keys())

        self.extract_type = ToTuple.ET._value2member_map_[extract_type]
    
    def __call__(self, sample):
        '''
            Note any 'seq' term in extract type will return seq types
            for everything except for action, action by default
            requires NO seq for either HAR or HPE
        '''

        return \
            (sample[DT.JOINTS_SEQ]) \
                if (self.extract_type == ToTuple.ET.JOINTS_SEQ) \
            else (sample[DT.JOINTS_SEQ], sample[DT.ACTION]) \
                if (self.extract_type == ToTuple.ET.JOINTS_ACTION_SEQ) \
            else (sample[DT.DEPTH_SEQ], sample[DT.JOINTS_SEQ]) \
                if (self.extract_type == ToTuple.ET.DEPTH_JOINTS_SEQ) \
            else (sample[DT.DEPTH_SEQ], sample[DT.JOINTS_SEQ], sample[DT.ACTION]) \
                if (self.extract_type == ToTuple.ET.DEPTH_JOINTS_ACTION_SEQ) \
            else (sample[DT.JOINTS]) \
                if (self.extract_type == ToTuple.ET.JOINTS) \
            else (sample[DT.JOINTS], sample[DT.ACTION]) \
                if (self.extract_type == ToTuple.ET.JOINTS_ACTION) \
            else (sample[DT.DEPTH], sample[DT.JOINTS]) \
                if (self.extract_type == ToTuple.ET.DEPTH_JOINTS) \
            else (sample[DT.DEPTH], sample[DT.JOINTS], sample[DT.ACTION]) \
                if (self.extract_type == ToTuple.ET.DEPTH_JOINTS_ACTION) \
            else None