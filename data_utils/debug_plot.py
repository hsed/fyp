

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

    visualize_joints_2d(ax, keypt_px_orig, joint_idxs=False)
    
    com_px_crop = affineTransform2D(crop_transf_matx, com_px_orig)
    keypt_px_crop = affineTransform2D(crop_transf_matx, keypt_px_orig)

    ax2 = fig.add_subplot(222)
    ax2.imshow(dpt_crop, cmap=cmap.jet)
    ax2.plot(com_px_crop[0], com_px_crop[1], 'kx')
    ax2.plot(keypt_px_crop[:,0], keypt_px_crop[:,1], 'g.')
    ax2.set_title("Cropped")

    visualize_joints_2d(ax2, keypt_px_crop, joint_idxs=False)
    

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




def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
                 (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)