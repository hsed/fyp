import os
import sys
import multiprocessing
import ctypes

from enum import IntEnum, Enum

import cv2
import torch
import numpy as np

from .dp_augment import *
from base import BaseDataType as DT

from sklearn.decomposition import PCA


def standardiseImg(depth_img, com_dpt_mm, crop_dpt_mm, extrema=(-1,1), copy_arr=False):
    # create a copy to prevent issues to original array
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
    if copy_arr:
        keypoints_mm = np.asarray(keypoints_mm.copy())
    return keypoints_mm / (cube_side_mm / 2.)

def unStandardiseKeyPointsCube(keypoints_std, cube_side_mm, copy_arr=False):
    '''
        `keypoints_std` => keypoints in range [-1, 1]\n
        `cube_side_mm` => any crop length in mm (all sides assumed equal)\n
        returns val in range [-cube_side_mm/2, +cube_side_mm/2]
    '''
    if copy_arr:
        keypoints_std = np.asarray(keypoints_std.copy())
    return keypoints_std * (cube_side_mm / 2.)


class TransformerBase(object):
    '''
        Class to initialise all properties useful to any or all transformers
    '''


    def __init__(self, num_joints = 21, world_dim = 3, cube_side_mm = 200):
        self.num_joints = num_joints
        self.world_dim = world_dim

        # only supported crop shape is regular cube with all sides equal
        self.crop_shape3D_mm = (cube_side_mm, cube_side_mm, cube_side_mm)
        
        #intrinsic camera params
        #self.fx = -1
        #self.fy
        #self.px
        #self.py

        self.joints_seq = 'joints_seq'



class JointCenterer(TransformerBase):
    '''
        
        A simple transformer for centering joints w.r.t CoM point
        For HPE, provides centered joints as one 2D matrix per sample

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class


class JointSeqCenterer(TransformerBase):
    '''
        A simple transformer for centering joints w.r.t CoM point
        For HAR, provides centered joints as a collection (3D matrix) per sample.

        In Joints_(seq), CoM_(seq) => Out Joints_(seq)_centered :: DIM: (Seq, 63)
        Also standardises
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
        joints_seq = sample[DT.JOINTS_SEQ].reshape(-1, self.num_joints, self.world_dim)

        coms_seq = sample[DT.COM_SEQ]
        
        sample[DT.JOINTS_SEQ] = joints_seq - coms_seq[:, np.newaxis, :]

        sample[DT.JOINTS_SEQ] = \
            standardiseKeyPointsCube(sample[DT.JOINTS_SEQ], self.crop_shape3D_mm[2])\
                                     .reshape(-1, self.num_joints*self.world_dim)

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




class DepthCropper(object):
    '''
        Apply crop to depth images
        If set in init, also return transform matrices for use with 2D pts

        In Depth => Out Depth_centered_cropped + Transform_Matrix_Crop
    '''


class DepthSeqCropper(object):
    '''
        Apply crop to depth images
        If set in init, also return transform matrices for use with 2D pts

        In Depth_(seq) => Out Depth_(seq)_centered_cropped + Transform_Matrix_Crop_Seq
    '''


class DepthAndJointsAugmenter(object):
    '''
        Perform Augmenting for both depth and keypoints

        For Future use see how its implemented not so important atm 
        Only mainly useful for HPE maybe this should only be for HPE

        In:
        Depth_cropped
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