

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

from .data_augmentors import *

'''
    TODO: MOVE TO VIZUAL...py in future!
'''


def plotImg(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, 
            crop_transf_matx, dpt_orig_aug=None, dpt_crop_aug=None, 
            aug_transf_matx=None, aug_mode=None, aug_val=None):
    '''
        All matrices supplied are homogenous projection matrix must work with
        homogenous coords so [x, y, 1]^T. for x,y pixels.
        `dpt_orig` => Original 2D depthmap
        `keypt_px_orig` => Original (21, 3) keypoint matrix
        `com_px_orig` => Shape (3,) with [0],[1] => (x_px_coord,y_px_coord) and [2] as dpt_mm
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.imshow(dpt_orig, cmap=cmap.jet)
    ax.plot(com_px_orig[0], com_px_orig[1], 'kx')
    ax.plot(keypt_px_orig[:,0], keypt_px_orig[:, 1], marker='.', linewidth=0, color='g') ## remove line...
    ax.set_title("Original")
    
    com_px_crop = affineTransform2D(crop_transf_matx, com_px_orig)
    keypt_px_crop = affineTransform2D(crop_transf_matx, keypt_px_orig)

    ax2 = fig.add_subplot(222)
    ax2.imshow(dpt_crop, cmap=cmap.jet)
    ax2.plot(com_px_crop[0], com_px_crop[1], 'kx')
    ax2.plot(keypt_px_crop[:,0], keypt_px_crop[:,1], 'g.')
    ax2.set_title("Cropped")
    

    if dpt_crop_aug is None or aug_transf_matx is None or aug_mode is None or aug_val is None:
        # simulate augmentation by calling test functions
        aug_mode = AugType.AUG_ROT
        aug_val = 90
        
        dpt_orig_aug, keypt_px_orig_aug, com_px_orig_aug, aug_transf_matx = \
            rotateHand2D(dpt_orig, keypt_px_orig, com_px_orig, aug_val)
        
        dpt_crop_aug, keypt_px_crop_aug, com_px_crop_aug, _ = \
            rotateHand2D(dpt_crop, keypt_px_crop, com_px_crop, aug_val)
    else:
        com_px_orig_aug = affineTransform2D(aug_transf_matx, com_px_orig)
        keypt_px_orig_aug = affineTransform2D(aug_transf_matx, keypt_px_orig)
        com_px_crop_aug = affineTransform2D(aug_transf_matx, com_px_crop)
        keypt_px_crop_aug = affineTransform2D(aug_transf_matx, keypt_px_crop)

        if dpt_orig_aug is None:
            dpt_orig_aug = affineTransformImg(aug_transf_matx, dpt_orig)

    if isinstance(aug_val, np.ndarray) == True:
        aug_val = "(%0.2f, %0.2f, %0.2f)" % (aug_val[0], aug_val[1], aug_val[2])
    elif isinstance(aug_val, float) == True:
        aug_val = "%0.2f" % aug_val

    ax3 = fig.add_subplot(223)
    ax3.imshow(dpt_orig_aug, cmap=cmap.jet)
    ax3.plot(com_px_orig_aug[0], com_px_orig_aug[1], 'kx')
    ax3.plot(keypt_px_orig_aug[:,0], keypt_px_orig_aug[:,1], 'g.')
    ax3.set_title("Orig + " + aug_mode.name + "\n(Val: %s)" % aug_val)

    ax4 = fig.add_subplot(224)
    ax4.imshow(dpt_crop_aug, cmap=cmap.jet)
    ax4.plot(com_px_crop_aug[0], com_px_crop_aug[1], 'kx')
    ax4.plot(keypt_px_crop_aug[:,0], keypt_px_crop_aug[:,1], 'g.')
    ax4.set_title("Cropped + " + aug_mode.name + "\n(Val: %s) = Final" % aug_val)
    
    print("\nCom_Orig: ", com_px_orig, "\nCom_Crop: ", com_px_crop,
          "\nCom_Crop_Aug: ", com_px_crop_aug,
          "\nKeyPt_Crop_Aug_XY: Max => %0.2f, Min => %0.2f" % 
              ((keypt_px_crop_aug[:,:2]).max(), (keypt_px_crop_aug[:,:2]).min()))

    # if (keypt_px_crop_aug > dpt_crop_aug.shape[0]).any() or (keypt_px_crop_aug < 0).any():
    #     print("Warning: Keypoints trasformed to outside image region!")

    plt.tight_layout()
    plt.show()