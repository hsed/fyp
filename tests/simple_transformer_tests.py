from argparse import Namespace

import numpy as np

from ..datasets import *
from data_utils import *


def basic_test():
    trnsfrm_base_params = {
        'num_joints': 21,
        'world_dim': 3,
        'cube_side_mm': 200,
        'debug_mode': True,
        'cam_intrinsics': MSRACameraIntrinsics,
        'dep_params': DepthParameters,
        'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180)
        }

    rot_lim=trnsfrm_base_params['aug_lims'].abs_rot_lim_deg
    sc_std=trnsfrm_base_params['aug_lims'].scale_std
    tr_std=trnsfrm_base_params['aug_lims'].trans_std

    dat = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=None,
                        test_subject_id=0, randomise_params=False)


    dat.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]))


