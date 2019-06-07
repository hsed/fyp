

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

from sklearn.metrics import confusion_matrix

import seaborn as sns

## for use with latex
# doesn't work with jupyter disabling this for now...
#plt.rcParams["font.family"] = "Times New Roman"

from .data_augmentors import affineTransform2D, rotateHand2D, affineTransformImg, AugType

'''
    TODO: MOVE TO VIZUAL...py in future!
'''

def plotImgV2(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, crop_transf_matx, action_gt,
              final_action_pred_probs,
              keypt_pred_px_1, pred_err_1, action_pred_probs_1,
              keypt_pred_px_2=None, pred_err_2=-1, action_pred_probs_2=None,
              show_aug_plots=False, return_fig=False, final_action_pred_probs_2=None,
              frame_curr=0, frame_max=0):
    '''
        All matrices supplied are homogenous projection matrix must work with
        homogenous coords so [x, y, 1]^T. for x,y pixels.
        `dpt_orig` => Original 2D depthmap
        `keypt_px_orig` => Original (21, 3) keypoint matrix
        `com_px_orig` => Shape (3,) with [0],[1] => (x_px_coord,y_px_coord) and [2] as dpt_mm
        `show_aug_plots` => Now must be false, depreciated command
    '''
    if show_aug_plots:
        raise NotImplementedError("Aug Plots Not Implemented Yet.")
    
    fig = plt.figure(dpi=250) #150 #, constrained_layout=True #dpi=150 # dpi=200

    # no ax3
    row_ids = [1]
    keypt_preds = [keypt_pred_px_1]
    reg_errs = [pred_err_1]
    action_probs = [action_pred_probs_1]
    n_rows = 2

    if ((keypt_pred_px_2 is not None) and (pred_err_2 != -1) and (action_pred_probs_2 is not None)):
        row_ids += [2]
        keypt_preds += [keypt_pred_px_2]
        reg_errs += [pred_err_2]
        action_probs += [action_pred_probs_2]
        n_rows = 3
    
    ### first row ###
    ax1 = plt.subplot2grid((n_rows, 10), (0, 0), colspan=4, fig=fig)
    ax1.imshow(dpt_orig, cmap=cmap.jet)
    ax1.plot(keypt_px_orig[:,0], keypt_px_orig[:, 1], 'g.') ## linewidth=0 remove line...
    visualize_joints_2d(ax1, keypt_px_orig, joint_idxs=False)
    ax1.plot(com_px_orig[0], com_px_orig[1], 'rx', markersize=5)
    title_str = "Final 3DErr1: %0.1fmm" % pred_err_1
    title_str += " 3DErr2: %0.1fmm" % pred_err_2 if pred_err_2 > -2 else ""
    ax1.set_title(title_str, fontsize=8)

    ax2 = plt.subplot2grid((n_rows, 10), (0, 4), colspan=6, fig=fig)
    #lw  = 2
    w   = 0.5
    x   = np.arange(final_action_pred_probs.shape[0]).astype(np.float32)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x.astype(np.uint32), fontsize=4.5, rotation=45)
    ax2.bar(x, action_gt, width=w+0.1, align='center', color=cmap.get_cmap('viridis')(0.4), alpha=0.4, label='GT') #cmap.get_cmap('jet')(0.9) #red
    buff = 0.0 if final_action_pred_probs_2 is None else w/2
    ax2.bar(x-buff, final_action_pred_probs, width=w, align='center', color=cmap.get_cmap('viridis')(0.1), label='HPE Estimate (1)') # viridis
    
    if final_action_pred_probs_2 is not None: 
        ax2.bar(x+buff, final_action_pred_probs_2, width=w, align='center', color=cmap.get_cmap('viridis')(0.8), 
                label='HPE_wActCond Estimate (2)')
        ax2.set_title("Actual: %d, Pred1: %d, Pred2: %d" % (np.argmax(action_gt),
                                                                            np.argmax(final_action_pred_probs),
                                                                            np.argmax(final_action_pred_probs_2)), fontsize=8)
    else:
        ax2.set_title("Actual: %d, Pred: %d" % (np.argmax(action_gt), np.argmax(final_action_pred_probs)), fontsize=8)
    ax2.legend(loc=1, prop={'size': 4}) # upper right

    ### second row ###
    for (row_id, keypt_pred_px, pred_err, action_probs) in zip(row_ids, keypt_preds, reg_errs, action_probs):
        ax4 = plt.subplot2grid((n_rows, 10), (row_id, 0), colspan=4, fig=fig) #fig.add_subplot(223)
        keypt_pred_px_crop = affineTransform2D(crop_transf_matx, keypt_pred_px)
        ax4.imshow(dpt_orig, cmap=cmap.jet)
        ax4.plot(keypt_pred_px[:,0], keypt_pred_px[:,1], 'b.')
        visualize_joints_2d(ax4, keypt_pred_px, joint_idxs=False, linestyle='--')
        ax4.plot(com_px_orig[0], com_px_orig[1], 'rx', markersize=5)
        ax4.set_title("GTDep. & PredJ", fontsize=8)

        ax5 = plt.subplot2grid((n_rows, 10), (row_id, 4), colspan=3, fig=fig)
        com_px_crop = affineTransform2D(crop_transf_matx, com_px_orig)
        keypt_px_crop = affineTransform2D(crop_transf_matx, keypt_px_orig)
        ax5.imshow(dpt_crop, cmap=cmap.jet)
        ax5.plot(keypt_px_crop[:,0], keypt_px_crop[:,1], 'g.')
        ax5.plot(keypt_pred_px_crop[:,0], keypt_pred_px_crop[:,1], 'b.')
        visualize_joints_2d(ax5, keypt_px_crop, joint_idxs=False)
        visualize_joints_2d(ax5, keypt_pred_px_crop, joint_idxs=False, linestyle='--')
        ax5.plot(com_px_crop[0], com_px_crop[1], 'rx', markersize=3)
        ax5.set_title("CropDep. 3DError: %0.2fmm" % pred_err,fontsize=8)

        ax6 = plt.subplot2grid((n_rows, 10), (row_id, 7), colspan=3, fig=fig)
        x   = np.arange(action_probs.shape[0])
        ax6.bar(x, action_probs, width=w, align='center', color=cmap.get_cmap('viridis')(0.7)) # viridis
        ax6.bar(x, action_gt, width=w+0.1, align='center', color=cmap.get_cmap('viridis')(0.9), alpha=0.4) #cmap.get_cmap('jet')(0.9) #red
        ax6.set_title("Hist. frame: %d of %d" % (frame_curr, frame_max),fontsize=8)



            

    plt.suptitle("Plot") #fontsize=14
    if not return_fig:
        plt.tight_layout(pad=0.2, rect=[0,0,1,0.95]) 
        #plt.tight_layout(pad=2.0, rect=[0,0.05,1,1])
        #plt.tight_layout() #(pad=2.0, rect=[0,0.05,1,1])
        plt.show()
    else:
        plt.tight_layout(pad=0.2, rect=[0,0,1,0.95]) #0.95 #(pad=2.0, rect=[0,0.05,1,1])
        #plt.show()
        # plt.gcf().savefig('test.pdf',
        #             format='pdf',
        #             dpi=300,
        #             transparent=True,
        #             bbox_inches='tight',
        #             pad_inches=0.01)
        #plt.show()
        return plt.gcf()


def plotImg(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, 
            crop_transf_matx, dpt_orig_aug=None, dpt_crop_aug=None, 
            aug_transf_matx=None, aug_mode=None, aug_val=None,
            show_aug_plots=True, return_fig=False, keypt_pred_px=None, pred_err=-1):
    '''
        All matrices supplied are homogenous projection matrix must work with
        homogenous coords so [x, y, 1]^T. for x,y pixels.
        `dpt_orig` => Original 2D depthmap
        `keypt_px_orig` => Original (21, 3) keypoint matrix
        `com_px_orig` => Shape (3,) with [0],[1] => (x_px_coord,y_px_coord) and [2] as dpt_mm
    '''
    
    fig = plt.figure(dpi=150) # dpi=200
    ax = fig.add_subplot(221) if show_aug_plots else fig.add_subplot(221) if keypt_pred_px is not None else fig.add_subplot(121)
    ax.imshow(dpt_orig, cmap=cmap.jet)
    ax.plot(com_px_orig[0], com_px_orig[1], 'rx', markersize=5)
    ax.plot(keypt_px_orig[:,0], keypt_px_orig[:, 1], marker='.', linewidth=0, color='g') ## remove line...
    visualize_joints_2d(ax, keypt_px_orig, joint_idxs=False)

    ax.set_title("Original Depth & Joints")

    
    
    com_px_crop = affineTransform2D(crop_transf_matx, com_px_orig)
    keypt_px_crop = affineTransform2D(crop_transf_matx, keypt_px_orig)

    

    ax2 = fig.add_subplot(222) if show_aug_plots else fig.add_subplot(122)
    ax2.imshow(dpt_crop, cmap=cmap.jet)
    ax2.plot(com_px_crop[0], com_px_crop[1], 'rx')
    ax2.plot(keypt_px_crop[:,0], keypt_px_crop[:,1], 'g.')
    ax2.set_title("Cropped Depth")

    visualize_joints_2d(ax2, keypt_px_crop, joint_idxs=False)

    if keypt_pred_px is not None:
        keypt_pred_px_crop = affineTransform2D(crop_transf_matx, keypt_pred_px)
        ax_ = fig.add_subplot(223)
        ax_.imshow(dpt_orig, cmap=cmap.jet)
        ax_.plot(keypt_pred_px[:,0], keypt_pred_px[:,1], 'b.')
        visualize_joints_2d(ax_, keypt_pred_px, joint_idxs=False, linestyle='--')
        ax_.set_title("Orig Depth & Pred Joints")

        ax2.plot(keypt_pred_px_crop[:,0], keypt_pred_px_crop[:,1], 'b.')
        visualize_joints_2d(ax2, keypt_pred_px_crop, joint_idxs=False, linestyle='--')
    
    if show_aug_plots:
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

    if pred_err != -1:
        plt.suptitle("Average 3D Error: %dmm" % pred_err)

    
    if not return_fig:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout(pad=2.0, rect=[0,0,1,1])
        #plt.show()
        # plt.gcf().savefig('test.pdf',
        #             format='pdf',
        #             dpi=300,
        #             transparent=True,
        #             bbox_inches='tight',
        #             pad_inches=0.01)
        #plt.show()
        return plt.gcf()




def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1, linestyle='-'):
    """Draw 2d skeleton on matplotlib axis, only correct for FHAD"""
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
    _draw2djoints(ax, joints, links, alpha=alpha, linestyle=linestyle)


def _draw2djoints(ax, annots, links, alpha=1, linestyle='-'):
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
                alpha=alpha, linestyle=linestyle)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1, linestyle='-'):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linestyle=linestyle)



# if __name__ == "__main__":
# import numpy as np
# plotImgV2(np.random.randn(480,640), 
#             np.random.randn(128,128),
#             np.random.randn(21,3),
#             np.random.randn(3),np.eye(3),
#             keypt_pred_px=np.random.randn(21,3), pred_err=2)
import numpy as np; from data_utils.debug_plot import plotImgV2;
# plotImgV2(np.random.randn(480,640), np.random.randn(128,128),np.random.randn(21,3),np.random.randn(3),np.eye(3),
#           np.eye(45)[5], np.arange(45)/45, np.random.randn(21,3), 2, np.arange(45)/45