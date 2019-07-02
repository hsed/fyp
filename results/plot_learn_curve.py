import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, LogLocator, AutoLocator
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix

import seaborn as sns

import argparse
from argparse import RawTextHelpFormatter
from os import path
from os import listdir

import yaml

## for use with latex
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "14"
plt.rcParams["text.usetex"] = True



def plotDualModelLearningCurves(xyLst, xzLst, xyLst2=None, xzLst2=None,
                             xTicks=None,
                             xLim=(None,None), yLim=(None,None), zLim=(None,None),
                             title='', figSize=(10, 4), xStrFmt='%0.1f',
                             tightLayout=True, showMajorGridLines=True, 
                             showMinorGridLines=False, xLbl='xAxis ->', 
                             yLbl='yAxis ->', zLbl='zAxis ->', 
                             yLbl2='', zLbl2='', legendLoc=2,
                             xLogBase = None, yLogBase=None, zLogBase=None,
                             tight_pad=2.0, tight_rect=[0,0,1,1], savefig=False,
                             legendsYLst = ['TrainY1', 'TrainY2'],
                             legendsZLst = ['TrainZ1', 'TrainZ2'],
                             yStrFmt='%0.1f', zStrFmt='%0.1f',
                             figfilepath='plots/lcplot.pdf',
                             show_plot=True):
    legLoc1, legLoc2 = legendLoc if isinstance(legendLoc, list) else (legendLoc, legendLoc)

    if ((xyLst2 is None) or (xzLst2 is None)):
        fig, axLst = plt.subplots(1,2, figsize=figSize)
        axLst = list(axLst.flatten())
        datLst = [xyLst, xzLst]
        legLst = [legendsYLst, legendsZLst]
        lblLst = [yLbl, zLbl]
        strfmtLst = [yStrFmt, zStrFmt]
        yzLimLst = [yLim, zLim]
        legLocLst = [legLoc1, legLoc1]
    else:
        fig, axLst = plt.subplots(2,2, figsize=figSize)
        axLst = list(axLst.flatten())
        datLst = [xyLst, xzLst, xyLst2, xzLst2]
        legLst = [legendsYLst, legendsZLst, legendsYLst, legendsZLst]
        lblLst = [yLbl, zLbl, yLbl2, zLbl2]
        strfmtLst = [yStrFmt, zStrFmt, yStrFmt, zStrFmt]
        yzLimLst = [yLim, zLim, yLim, zLim]
        legLocLst = [legLoc1, legLoc1, legLoc2, legLoc2]
        

    lw = 2  # line width

    ## datLst is all the y-plots i.e.
    ## 2D list 

    # plot both y and z
    for (ax, pltDatLst, legendLst, lbl, strFmt, lim, legendLoc) in \
        zip(axLst, datLst, 
            legLst,
            lblLst,
            strfmtLst,
            yzLimLst,
            legLocLst):
        
        for (pltDat, legend) in zip(pltDatLst, legendLst):
            ax.plot(pltDat[:,0], pltDat[:,1], label=legend)
        


        ax.legend(loc=legendLoc, prop={'size': 12})        
        ax.set_ylabel(lbl)
        ax.set_xlabel(xLbl)

        if xLogBase is not None:
            ax.set_xscale('log', basex=xLogBase)
            ax.xaxis.set_major_locator(LogLocator(base=xLogBase, numticks=xTicks))
        else:
            # mutuall exclusive
            if xTicks is not None:
                # must supply xTicks to do this
                ax.xaxis.set_major_locator(LinearLocator(numticks=xTicks))
            pass

        ax.xaxis.set_major_formatter(FormatStrFormatter(xStrFmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(strFmt))

        ax.set_xlim(xLim)
        ax.set_ylim(lim)

        if showMajorGridLines and showMinorGridLines:
            ax.grid(which='both')
        elif showMajorGridLines:
            ax.grid(which='major')
        elif showMinorGridLines:
            ax.grid(which='minor')
        



    plt.suptitle(title, fontsize=18)

    if tightLayout: 
        #(left, bottom, right, top),
        #(0,0,1,1)
        plt.tight_layout(rect=tight_rect, pad=tight_pad)
    
    if savefig:
        #fig.tight_layout()
        fig.savefig('%s' % figfilepath,
                    format='pdf',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.01)
    
    if show_plot:
        plt.show()








def extract_data_and_plot(config: dict, show_plot: bool = True):

    keys = set([
        'model_name',
        'runs',
        'run_tags',
        'base_path',
        'scalar_tag',
        'save_path',
        'axis_title',
        'title',
        'max_epochs',
        'legend_loc',
        'x_ticks',
    ])
    keys_present = set(config.keys())
    assert keys.issubset(keys_present), "Keys Expected:\n%a\nKeys Found:\n%a" % (keys, keys_present)
    
    assert isinstance(config['runs'], list)
    assert isinstance(config['run_tags'], list)
    assert len(config['runs']) == len(config['run_tags'])

    model_name     = config['model_name']
    runs           = config['runs']
    base_path      = config['base_path']
    scalar_tag     = config['scalar_tag']
    run_tags       = config['run_tags']
    save_path      = config['save_path']
    axis_title     = config['axis_title']
    title          = config['title']
    max_epochs     = config['max_epochs']
    legend_loc     = config['legend_loc']
    x_ticks        = config['x_ticks']
    scalar_tag_2   = config['scalar_tag_2'] if ('scalar_tag_2' in config) else None
    axis_title_2   = config['axis_title_2'] if ('axis_title_2' in config and 'scalar_tag_2' in config) else ''
    train_data_list = []
    valid_data_list = []
    fig_size = tuple(config['fig_size']) if 'fig_size' in config else (10, 4)

    train_data_list_2 = []
    valid_data_list_2 = []
    
    for run in runs:
        assert path.isdir(path.normpath(path.join(base_path, run))), "Path %s either doesn't exist or is not a directory!" \
            % path.normpath(path.join(base_path, run))
        
        df = pd.read_csv(path.normpath(path.join(base_path, run, 'scalars.csv')), sep=',')

        train_data_list.append(df[['step', 'train/%s' % scalar_tag]].values)
        valid_data_list.append(df[['step', 'valid/%s' % scalar_tag]].values)

        if scalar_tag_2 is not None:
            train_data_list_2.append(df[['step', 'train/%s' % scalar_tag_2]].values)
            valid_data_list_2.append(df[['step', 'valid/%s' % scalar_tag_2]].values)
        





    assert (len(train_data_list) == len(valid_data_list))
    assert (len(train_data_list) == len(run_tags))  
    assert (len(train_data_list_2) == len(valid_data_list_2))


    if scalar_tag_2 is None:
        train_data_list_2 = None
        valid_data_list_2 = None
    #title = 'Loss Curves for Conditional GAN Experiments (Max Epochs: 15) ' + model_name
    
    plotDualModelLearningCurves(xyLst=train_data_list, xzLst=valid_data_list,
                                xyLst2=train_data_list_2, xzLst2=valid_data_list_2, 
                                xLogBase=None, yLogBase=None, zLogBase=None,
                                xStrFmt='%d', yStrFmt='%0.2f', zStrFmt='%0.2f',
                                xLim=(1,max_epochs), xTicks=x_ticks,
                                xLbl='Epochs', yLbl='Train ' + axis_title,
                                zLbl='Validation ' + axis_title,
                                yLbl2='Train ' + axis_title_2, zLbl2='Validation ' + axis_title_2,
                                title=title, figSize=fig_size,
                                legendsYLst=run_tags, legendsZLst=run_tags,
                                tight_pad=0.5, tight_rect=[0.0, 0.0, 1.0, 0.95], tightLayout=True,
                                savefig=True, figfilepath=save_path, legendLoc=legend_loc, show_plot=show_plot)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform plotting given plot_tag.',
                                 formatter_class=RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, required=True, help='config file (can pass multiple)', nargs='+')
    parser.add_argument('-np', '--no-popup', action='store_true', help='do not show plot only save')
    # parser.add_argument('-dnf', '--denoiser_files', help="Denoiser filenames", type=str,
    #                     metavar='FNAME', nargs='+', default=True)
    
    args = parser.parse_args()

    for i, config_file in enumerate(args.config):
        # allow for multiple config passing
        extract_data_and_plot(yaml.load(open(config_file), Loader=yaml.SafeLoader), show_plot=(not args.no_popup))
        print('[MAIN] Processed %d out of %d plots...' % (i+1, len(args.config)))
    

    '''
        python plots_cv.py -dnt aaa -dst bbb -dnf loss_cdcgan_baseline_20190313_1934.csv -dsf loss_cdcgan_baseline_20190313_1934.csv
    '''
    


'''
python plots_cv.py -dnt CGAN_Baseline CDCGAN_Baseline -dnf loss_cgan_orig_untouched_20190314_1846.csv loss_cdcgan_baseline_20190312_1335.csv -sf cgan_vs_cdcgan_loss.pdf

python plots_cv.py -dnt CDCGAN +C1-C0 +C2-C1 +LR+C0-C2 +LS +LF1 +LF2-LF1 +EM +SN-BN-EM -dnf loss_cdcgan_baseline_20190312_1335.csv loss_cdcgan_baseline_20190311_2054.csv loss_cdcgan_baseline_20190311_2114.csv loss_cdcgan_baseline_20190312_0004.csv loss_cdcgan_baseline_20190312_0017.csv loss_cdcgan_baseline_20190313_1641.csv loss_cdcgan_baseline_20190313_1709.csv loss_cdcgan_baseline_20190313_1843.csv loss_cdcgan_baseline_20190314_0243.csv -sf cdcgan_baseline_vs_improve.pdf

python plots_cv.py -dnt CDCGAN +HL+BN-SN +DP +SN-BN -dnf loss_cdcgan_baseline_20190312_1335.csv loss_cdcgan_baseline_20190314_1239.csv loss_cdcgan_baseline_20190314_1433.csv loss_cdcgan_baseline_20190314_1551.csv -sf cdcgan_baseline_vs_hinge_loss_types.pdf

python plots_cv.py -dnt CDCGAN +3G_1D +VB1-3G_1D +VB2-VB1 -dnf loss_cdcgan_baseline_20190312_1335.csv loss_cdcgan_baseline_20190311_1837_3g_1d.csv loss_cdcgan_baseline_20190314_0042_vbn.csv loss_cdcgan_baseline_20190314_0112_vbn.csv -sf cdcgan_baseline_vs_bad_methods.pdf


ls results/*/*.yaml
python results\plot_learn_curve.py -np -c results/combined_hyp_fix_exp/bn_dropout_fix_plots.yaml results/har_atten/har_atten_plots_v1_v2.yaml results/combined_hyp_fix_exp/loss_alpha_fix_plots.yaml  results/har_atten/har_atten_plots_v3_v4_v5.yaml results/combined_hyp_fix_exp/lr_fix_plots.yaml results/har_atten/har_atten_plots_v6_v7_v8.yaml results/combined_hyp_fix_exp/wd_fix_plots.yaml results/hpe_cond/hpe_cond_plots_1.yaml results/combined_new_exp/v11d_two_model_exp.yaml results/hpe_cond/hpe_cond_plots_2.yaml results/combined_new_exp/v2a_backprop_methods.yaml results/hpe_equiprob_cond_exp/hpe_equiprobcond.yaml results/combined_new_exp/v4d_act_feedback_exp.yaml results/hpe_equiprob_prob_exp/hpe_equiprob_prob.yaml results/combined_new_exp/v4d_action_equiprob_exp.yaml

'''