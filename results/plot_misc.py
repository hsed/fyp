'''
    functions taken from mlcv-cw/plot_utils.py!
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, LogLocator, AutoLocator
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import matplotlib.cm as cmap

from sklearn.metrics import confusion_matrix

import seaborn as sns

import argparse
from argparse import RawTextHelpFormatter
from os import path
from os import listdir

# from data_utils.debug_plot import visualize_joints_2d
# from data_utils.data_augmentors import affineTransform2D

import yaml

## for use with latex
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "14"
plt.rcParams["text.usetex"] = True


# from stackoverflow
def ord(n):
    return str(n)+("th" if 4<=n%100<=20 else {1:"st",2:"nd",3:"rd"}.get(n%10, "th"))


## recopied here doue to import errors
def affineTransform2D(M, X):
    '''
        M must be a (m, 3) matx with
            m => {2, 3} only
        X can be a 
            (N,) 1D matx (vector) OR
            (N,k) 2D matx with
                k => {2, 3} only

    '''
    ## in order to perform transl we need a "homogenous vector" which is a vector
    ## we can actually transl using (2x3) matrix but then the operation is non-linear
    ## as simply there is no inverse for (2x3) matx (maybe psuedo inverse...)
    ## written in homogenous coord as opposed to cartesian coords
    ## its for projective geometry and requires a 2D vector as [x,y,1]^T
    ## here we transform com using how many 'pixels' it must move left and down
    ## along the 2D plane to now be the center of the image.
    ## we can imagine a crop area around the hand and then that being 
    ## translated towards the center of orig img area
    ## where the crop area < orig img area
    
    if M.shape == (2, 3):
        # special case if matx comes from opencv
        M = np.vstack((M, np.array([0,0,1], dtype=np.float32)))
    elif M.shape != (3, 3):
        raise AssertionError("Error: Matrix M must be 2x3 or 3x3") 

    if len(X.shape) == 2:
        ## X is 2D matx, assume each sample is row vector in this matx
        if X.shape[1] == 2 or X.shape[1] == 3:
            # for affine transform must use homogeneous coords
            # this condition works for 2D row vector or 3D
            # Y tries to keep orig shape
            Y = np.matmul(np.column_stack((X[:,:2], np.ones(X.shape[0], dtype=np.float32))), M.T)

            ## after multiply Y will always be of shape (m, 3)
            ## if each row in X is 2-elem then this function would make Y (m, 2)
            ## otherwise it ensures that the third elem is orig 'z' value
            ## in X, untransformed.
            Y = np.column_stack((Y[:,:2], X[:, 2])) if X.shape[1] == 3 else Y[:,:2]
        else:
            raise AssertionError("Error: X.shape[1] must be 2 or 3")
    elif len(X.shape) == 1:
        ## X is a 1D vector
        if X.shape[0] == 2 or X.shape[0] == 3:
            Y = np.matmul(M, np.array([X[0], X[1], 1], dtype=np.float32))
            Y = np.array([Y[0], Y[1], X[2]]) if X.shape[0] == 3 else Y[:2]
        else:
            raise AssertionError("Error: X.shape[0] must be 2 or 3")
    else:
        raise AssertionError("Error: X.shape[1] must be 1 or 2")
    
    return Y


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

    

def surfacePlotGridSearch(pltDatArr, xLbl, yLbl, zLbl='Accuracy Score', title='3D Plot', 
                          xStrFmt='%0.1f', yStrFmt='%0.1f', zStrFmt='%0.1f', zCustomFmt=None, azim=-130, elev=35,
                          xTicks=4, yTicks=6, zTicks=None, pad=0.05, plotContour=False, pltdatArr2=None,
                          savefig=False, figfilename='fig3d.pdf', base2logscale=False,
                          tight_rect=None, subplt1Title='', subplt2Title='', z3DLogScale=False,
                          figSize3D=(8,6), showplt=True):
    '''
        Output: Surface plot of first two params in model vs accuracy
    '''
    fig = None#plt.figure()
    if not plotContour:
        fig = plt.figure(figsize=figSize3D)
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_trisurf(pltDatArr[:,0],pltDatArr[:,1],pltDatArr[:,2], cmap=plt.cm.viridis,
                           linewidth=0, antialiased=True)
        # Customize the z axis.
        ax.set_zlim(np.min(pltDatArr[:,2]), np.max(pltDatArr[:,2]))
        
        if zCustomFmt is not None:
            ax.zaxis.set_major_formatter(zCustomFmt)
        else:
            ax.zaxis.set_major_formatter(FormatStrFormatter(zStrFmt))
        ax.set_zlabel(zLbl)
        
        if zTicks is not None: ax.zaxis.set_major_locator(LinearLocator(numticks=zTicks))

        fig.colorbar(surf, shrink=0.4, aspect=5, format=zCustomFmt, pad=pad)

        ax.azim = azim
        ax.elev = elev

        if z3DLogScale: ax.zaxis._set_scale('log')
        #ax.zaxis.set_major_locator(LogLocator(base=10.0,numticks=6))
        #ax.set_zlim(None,4e7)
        
    else:
        if pltdatArr2 is None:
            fig = plt.figure()
            ax = fig.gca()
            tricont = ax.tricontourf(pltDatArr[:,0],pltDatArr[:,1],pltDatArr[:,2], cmap=plt.cm.viridis)
            cbar = fig.colorbar(tricont, shrink=0.4, aspect=5, format=zCustomFmt, pad=pad, use_gridspec=True)
            if zLbl is not None: cbar.set_label(zLbl, rotation=270, labelpad=11, y=0.45)
            #plt.title(title, y=1.1)
        else:
            fig = plt.figure(figsize=(10,4))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            tricont = ax1.tricontourf(pltDatArr[:,0],pltDatArr[:,1],pltDatArr[:,2], cmap=plt.cm.viridis)
            cbar = fig.colorbar(tricont, shrink=0.5, aspect=5, format=zCustomFmt, pad=pad, ax=ax1, use_gridspec=True)
            if zLbl is not None: cbar.set_label(zLbl, rotation=270, labelpad=11, y=0.45)

            tricont2 = ax2.tricontourf(pltdatArr2[:,0],pltdatArr2[:,1],pltdatArr2[:,2], cmap=plt.cm.viridis)
            cbar2 = fig.colorbar(tricont2, shrink=0.5, aspect=5, format=zCustomFmt, pad=pad, ax=ax2, use_gridspec=True)
            if zLbl is not None: cbar2.set_label(zLbl, rotation=270, labelpad=11, y=0.45)

            ax1.set_title(subplt1Title)
            ax2.set_title(subplt2Title)

            # ax2.yaxis.set_label_coords(-0.10, 0.5) # this is not really good...
        
        # can do this for contours only
        if tight_rect is not None:
            plt.tight_layout(rect=[0.03, 0.01, 1.01, 0.94])


    for ax in ([ax1,ax2] if pltdatArr2 is not None else [ax]):
        ax.set_xlabel(xLbl)
        ax.set_ylabel(yLbl)
        ax.set_xlim(np.min(pltDatArr[:,0]), np.max(pltDatArr[:,0]))
        ax.set_ylim(np.min(pltDatArr[:,1]), np.max(pltDatArr[:,1]))
        if base2logscale:
            ax.set_xscale('log', basex=2)
            ax.set_yscale('log', basey=2)
            ax.xaxis.set_major_locator(LogLocator(base=2.0,numticks=xTicks))
            ax.yaxis.set_major_locator(LogLocator(base=2.0,numticks=yTicks))
        else:
            ax.xaxis.set_major_locator(LinearLocator(numticks=xTicks))
            ax.yaxis.set_major_locator(LinearLocator(numticks=yTicks))

        

        ax.xaxis.set_major_formatter(FormatStrFormatter(xStrFmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(yStrFmt))
        
        #new
        
    
    plt.suptitle(title)

    plt.subplots_adjust(wspace=0.2)
    #plt.savefig('test.png')
    if savefig:
        #fig.tight_layout()
        fig.savefig(figfilename,
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    #bbox_inches='tight',
                    pad_inches=0.01)
    if showplt:
        plt.show()



def plotBar(x, y1, y2=None,
            title='', figSize=(8, 4), tightLayout=True, 
            showMajorGridLines=True, showMinorGridLines=False,
            xLbl='xAxis ->', yLbl='yAxis ->', yStrFmt='%0.0f%%',
            xLogBase = None, tight_pad=2.0, savefig=False,
            figfilename='errplot.pdf', yPercentage = False,
            yLim=None, xTicks=None, xStrFmt=None):
    '''
        x => str list/array
        y => float list/array
        y2 => float list/array
    '''

    assert len(x) == len(y1)
    if y2 is not None: assert len(y1) == len(y2)

    fig = plt.figure(figsize=figSize)
    lw = 2  # line width
    
    x_vals = np.arange(len(x))
    
    cmap = cm.get_cmap('viridis')
    #sns.barplot(x=x,y=y1,hue=y2, ax=plt.gca())#palette=plt.cm.viridis

    ### plots
    #plt.errorbar(x=x_vals, y=means, xerr=None, yerr=stds, marker='o', ms=4, color='green', ls = 'dotted', capsize=6) #cmap=plt.cm.viridis
    w = 0.4
    plt.bar(x_vals-(w/2), y1, width=w, align='center', color=cmap(0.4))

    if y2 is not None:
        plt.bar(x_vals+(w/2), y2, width=w, align='center', color=cmap(0.7))


    plt.xticks(x_vals, x, rotation=45)
    plt.title(title, fontsize=16)
    plt.xlabel(xLbl, fontsize=12)
    plt.ylabel(yLbl, fontsize=12)

    ax = plt.gca()
    ax.autoscale(tight=True)

    plt.legend(['Train Accuracy', 'Test Accuracy'], loc=4)

    ymin = y1.min()
    if y2 is not None: ymin = min(ymin, y2.min())
    
    ymax = y1.max()
    if y2 is not None: ymax = max(ymax, y2.max())

    ax.set_ylim((ymin-2,ymax+1))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yStrFmt))

    if showMajorGridLines and showMinorGridLines:
        ax.yaxis.grid(which='both', linestyle='--')
    elif showMajorGridLines:
        ax.yaxis.grid(which='major', linestyle='--')
    elif showMinorGridLines:
        ax.yaxis.grid(which='minor', linestyle='--')

    if tightLayout: plt.tight_layout(pad=tight_pad)
    

    if savefig:
        fig.tight_layout()
        #fname = os.path.normpath(os.path.join(figdir, figfilename))
        fig.savefig(figfilename,
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    
    plt.show()


def plotImgV4(data_dict, show_plt=False, fig_save_loc='debug_fig.pdf'):
    '''
        supply a dict get a plot

        raw_data
        timestep
        sample
        pred_keypt_data
    '''
    plt.rcParams["font.size"] = "12"
    #print("KEYS: ", list(data_dict.keys()))
    dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, crop_transf_matx, act_orig = data_dict['raw_data']
    keypt_pred_px_1, keypt_pred_px_2, keypt_pred_px_3 = data_dict['pred_keypt_data']
    act_pred_1, act_pred_2, act_pred_3 = data_dict['pred_action_data']
    err3d_1, err3d_2, err3d_3 = data_dict['3d_err_seq']
    timeframe = data_dict['timeframe']

    com_px_crop = affineTransform2D(crop_transf_matx, com_px_orig)
    keypt_px_crop = affineTransform2D(crop_transf_matx, keypt_px_orig)

    w = 0.5
    line_gt = Line2D([0], [0], color='b', linestyle='-')
    line_pred = Line2D([0], [0], color='b', linestyle='--')
    labels = ['Ground Truth', 'Pose Estimation']

    def_cmap = cmap.viridis #bone #viridis # cmap.bone #
    fig = plt.figure(dpi=150, figsize=(10,6)) # dpi=200


    for i, (keypt_pred_px, title_keypt, act_pred) in enumerate(zip(
        [keypt_pred_px_1, keypt_pred_px_2, keypt_pred_px_3],
        ['Baseline (%0.1fmm)' % err3d_1, 'Method \\#2 (%0.1fmm)' % err3d_2, 'Method \\#3 (%0.1fmm)' % err3d_3],
        [act_pred_1, act_pred_2, act_pred_3],
    )):
        keypt_pred_px_crop = affineTransform2D(crop_transf_matx, keypt_pred_px)
        ax = fig.add_subplot(231 + i) #if show_aug_plots else fig.add_subplot(221) if keypt_pred_px is not None else fig.add_subplot(121)
        ax.imshow(dpt_crop, cmap=def_cmap)
        ax.plot(com_px_crop[0], com_px_crop[1], 'rx')
        ax.plot(keypt_px_crop[:,0], keypt_px_crop[:,1], 'g.')
        ax.plot(keypt_pred_px_crop[:,0], keypt_pred_px_crop[:,1], 'b.')

        visualize_joints_2d(ax, keypt_px_crop, joint_idxs=False)
        visualize_joints_2d(ax, keypt_pred_px_crop, joint_idxs=False, linestyle='--')

        ax.legend([line_gt, line_pred], labels, loc=3, prop={'size': 10})
        ax.set_title(title_keypt)
        ax.set_axis_off()
        #print("SUM: ", np.sum(act_pred))
        ax2 = fig.add_subplot(231 + (i+3))
        x = np.arange(act_orig.shape[0])
        ax2.bar(x, act_pred, width=w, align='center', color=cmap.get_cmap('viridis')(0.7), label='Action Predictions') # viridis
        ax2.bar(x, act_orig, width=w+0.1, align='center', color=cmap.get_cmap('viridis')(0.9), alpha=0.4, label='Ground Truth') #cmap.get_cmap('jet')(0.9) #red
        ax2.set_title("Action Distribution (Pred: %d)" % np.argmax(act_pred))
        ax2.legend(loc=3, prop={'size': 10})
        ax2.set_xlabel('Action Class Index')
        ax2.set_ylabel('Probability')



    
    
    #else:

    plt.suptitle("Time Frame: %d, Action Class: %d" % (timeframe, np.argmax(act_orig)))

    plt.subplots_adjust(wspace=0.2)

    
    plt.tight_layout(rect=[0.03, 0.01, 1.01, 0.94])

    if show_plt:
        plt.show()
    #plt.show()
    plt.gcf().savefig(fig_save_loc + "_%d.pdf" % (timeframe),
                format='pdf',
                dpi=300,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0.01)
    #plt.show()
    #return plt.gcf()


def simple2DLinePlot(data_plot,
                     title='', figSize=(8, 6), tightLayout=True, 
                     showMajorGridLines=True, showMinorGridLines=False,
                     xLbl='Time (framestep)', yLbl='Average 3D Error (mm)', scatter=False,
                     xLogBase = None, tight_pad=2.0, savefig=True,
                     legend1 = 'Train', legend2 = 'Test',
                     fig_save_loc='valplot.pdf', yPercentage = False,
                     yLim=None, xTicks=None, xStrFmt="%d",
                     show_plt=True):
    '''
        only for one param -- 2d line plot...
        allows for fill_between for val curve
    '''

    plt.rcParams["font.size"] = "12"

    fig = plt.figure(figsize=figSize)
    lw = 2  # line width

    x = np.arange(data_plot['timeframe'])
    y = data_plot['3d_err_seq_trajec'][0]
    y2 = data_plot['3d_err_seq_trajec'][1]
    z = data_plot['3d_err_seq_trajec'][2]
    peq = data_plot['peq']
    
    # if scatter:
    #     plt.scatter(x,y)
    plt.plot(x, y)
        # if y_stdev is not None: 
        #     plt.fill_between(x, y - y_stdev, y + y_stdev, alpha=0.2, color="navy", lw=lw)
    plt.plot(x,y2)
        
        # if y_stdev2 is not None:
        #     plt.fill_between(x, y2 - y_stdev2, y2 + y_stdev2, alpha=0.2, color="darkorange", lw=lw)

    plt.plot(x,z)

    plt.legend(['Method \\#1 (Baseline)', 'Method \\#2', 'Method \\#3'])

    if xLogBase is not None:
        plt.xscale('log', basex=xLogBase)
        plt.gca().xaxis.set_major_locator(LogLocator(base=xLogBase, numticks=xTicks))
    else:
        plt.gca().xaxis.set_major_locator(LinearLocator(numticks=xTicks))

    if yPercentage == True:
        vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    
    # if yLim is not None and isinstance(yLim, tuple):
    #     plt.gca().set_ylim(yLim[0], yLim[1])

    plt.gca().set_xlim(0, data_plot['timeframe'])
    
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter(xStrFmt))



    
    plt.title("Pose Estimation Error Trajectory for Action Sequence ($p_{eq} = %0.1f$)" % peq)
    plt.xlabel(xLbl)
    plt.ylabel(yLbl)

    if tightLayout: 
        plt.tight_layout()
    
    if savefig:
        fig.tight_layout()
        fig.savefig('%s_peq_%0.1f.pdf' % (fig_save_loc,peq),
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    
    if show_plt:
        plt.show()


if __name__ == "__main__":
    '''
        python results\plot_misc.py -if results\combined_boosting_eta_zeta\summary.csv -t eta_zeta_v11d_contour
        python results\plot_misc.py -if results\visual_pose_depth_action\sample_4_timestep_20_val.npz -t visual_compare_methods
        
    '''
    VAL_CHOICES = ['eta_zeta_v11d_contour', 'visual_compare_methods', 'plot_traj']

    parser = argparse.ArgumentParser(description='Perform plotting given plot_tag.',
                                 formatter_class=RawTextHelpFormatter)
    parser.add_argument('-if', '--input_file', help="Input filename", type=str, default=True)
    parser.add_argument('-np', '--no-popup', action='store_true', help='do not show plot only save')
    parser.add_argument('-t', '--tag', metavar='TAG', type=str, choices=VAL_CHOICES,
                    help='plot choice', required=True)
    parser.add_argument('-od', '--output_dir', type=str,
                        help='output dir, filename is predetermined', default='results/plot_pdfs')

    args = parser.parse_args()
    output_dir = path.normpath(args.output_dir)
    tag = args.tag
    infile = args.input_file
    showplt = not args.no_popup

    assert path.exists(infile), "Input file %s not found!" % infile


    if args.tag == 'eta_zeta_v11d_contour':
        '''
            Input CSV Format:
            #ETA, ZETA, TOPVALERR, TOPVALACC
            0.1, 0.1, *, *
            0.1, ...
            ...
            0.9, 0.9, *, *
        '''
        plot_data = np.loadtxt(infile, delimiter=',')
        save_file = path.join(output_dir, 'combined_v11d_eta_zeta_search_contour.pdf')

        ## put all config options here directly
        surfacePlotGridSearch(plot_data[:,:3],'$\\eta$', '$\\zeta$', \
        xStrFmt='%0.1f', yStrFmt='%0.1f', pltdatArr2=plot_data[:,[0,1,3]],
        zLbl=None, title='Combined Ensemble Fusion Parameter Experiments',
        savefig=True, figfilename=save_file, base2logscale=False, xTicks=3, yTicks=3, pad=0.02,
        plotContour=True, tight_rect=[0.00, 0.01, 1.00, 0.94],
        subplt1Title='Val 3D Error (mm)', subplt2Title='Val Accuracy (\\%)',
        showplt=showplt)
    


    if args.tag == 'visual_compare_methods':
        '''
            Visually compare single frame results and action accuracy
        '''
        dict_obj = np.load(infile)['arr_0'].item() # this is an np array otype object need to extract the divt

        plotImgV4(dict_obj, show_plt=showplt, fig_save_loc='results/plot_pdfs/visual_compare_methods')


    if args.tag == 'plot_traj':
        '''
            Input CSV Format:
            #ETA, ZETA, TOPVALERR, TOPVALACC
            0.1, 0.1, *, *
            0.1, ...
            ...
            0.9, 0.9, *, *
        '''

        dict_obj = np.load(infile)['arr_0'].item() # this is an np array otype object need to extract the divt

        simple2DLinePlot(dict_obj, show_plt=showplt, fig_save_loc='results/plot_pdfs/traj_lot')