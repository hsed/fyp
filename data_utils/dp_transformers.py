import os
import sys
import multiprocessing
import ctypes

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

from dp_augment import *

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

def standardiseKeyPoints(keypoints_mm, crop_dpt_mm, copy_arr=False):
    '''
        crop_dpt_mm => the z-axis crop length in mm
        returns val in range [-1, 1]
    '''
    ## only one standarisation method, I reckon this is -1 -> 1 standardisation
    ## as by default max values will be -crop3D_sz_mm/2, +crop3D_sz_mm/2
    ## keypoints are the one relative to CoM
    if copy_arr:
        keypoints_mm = np.asarray(keypoints_mm.copy())
    return keypoints_mm / (crop_dpt_mm / 2.)

def unStandardiseKeyPoints(keypoints_std, crop_dpt_mm, copy_arr=False):
    '''
        `keypoints_std` => keypoints in range [-1, 1]
        `crop_dpt_mm` => the z-axis crop length in mm
        returns val in range [-crop3D_sz_mm/2, +crop3D_sz_mm/2]
    '''
    if copy_arr:
        keypoints_std = np.asarray(keypoints_std.copy())
    return keypoints_std * (crop_dpt_mm / 2.)

def plotImg(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, 
            dpt_orig_aug=None, dpt_crop_aug=None, crop_transf_matx=None, 
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
    ax.plot(keypt_px_orig[:,0], keypt_px_orig[:, 1], marker='.', linewidth=0) ## remove line...
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

def saveKeypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')

# from v2v_posenet
def computeDistAcc(pred, gt, dist):
        '''
        pred: (N, K, 3)
        gt: (N, K, 3)
        dist: (M, )
        return acc: (K, M)
        '''
        assert(pred.shape == gt.shape)
        assert(len(pred.shape) == 3)

        N, K = pred.shape[0], pred.shape[1]
        err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2))  # (N, K)

        acc = np.zeros((K, dist.shape[0]))

        for i, d in enumerate(dist):
            acc_d = (err_dist < d).sum(axis=0) / N
            acc[:,i] = acc_d

        return acc


class DeepPriorYTestInverseTransform(object):
    '''
        Quick transformer for y-vals only (keypoint)
        Centers (w.r.t CoM in dB) and standardises (-1,1) y
        
        ## now this is an inverse of above
        Note: This doesn't invert any augmentations so only use for test data
        and obviously do not augment test data!

    '''
    def __init__(self, crop_len_mm=200):
        self.crop_len_mm = crop_len_mm
    
    def __call__(self, sample):
        pred_std_cen_batch, com_batch = sample
        # use ref point to transform back to REAL MM values i.e
        # mm distances of keypoints is w.r.t focal point of img
        ## need to first transform pred_std_centered -> pred_mm_centered using the crop value
        ## then transform pred_mm_centered -> pred_mm_not_centered by appending CoM value
        ## now store this final value in 'keypoints'
        ## also store gt_mm_not_centered in keypoints_gt for future error calc
        
        ## perform the operation in batch .. shuld automatically work with numpy
        #print("\npred_std_cen_batch Shape: ", pred_std_cen_batch.shape)
        #print("\ncom_batch Shape: ", com_batch.shape)
        if len(pred_std_cen_batch.shape) == 3:
            # broadcasting won't work automatically need to adjust array to handle that
            # repetitions will happen along dim=1 so (N, 3) -> (N, 1, 3)
            # Now we can do (N, 21, 3) + (N, 1, 3) as it allows automatic broadcasting along dim=1
            # for (21, 3) + (3,) case this is handled automatically
            com_batch = com_batch[:, None, :]

        return \
            (unStandardiseKeyPoints(pred_std_cen_batch, self.crop_len_mm) + com_batch)



## rename this to deep prior
class DeepPriorXYTransform(object):
    '''
            Deep Prior Transformation class to transform single samples
            by cropping, centering inputs and outputs w.r.t CoM and scaling.

            CoM must be supplied (no calc done) and no augmentations are implemented

            0 <= `abs_rot_lim_deg` <= 180

            `aug_mode_lst` => A list of possible aug to randomly choose from
            `depthmap_px` => Ouput px size of one-sde (we ouput 1:1 crop)

            sc ~ N(1, 0.02**2); tr ~ N(0, 5); rot ~ U[-180, 180]
            from Deep-Prior++ paper
    '''
    def __init__(self, depthmap_px=128, crop_len_mm=200, 
                 fx = 241.42, fy = 241.42, ux = 160.0, uy = 120.0,
                 scale_std=0.02, trans_std=5, abs_rot_lim_deg=180,
                 aug_mode_lst = [AugType.AUG_NONE], debug_mode=False):

        self.fx = fx #241.42
        self.fy = fy #241.42
        self.ux = ux #160.0
        self.uy = uy #120.0
        
        # output sz in px
        self.out_sz_px = (depthmap_px, depthmap_px)
        
        # 3D cube crop sz around com in mm
        self.crop_vol_mm = (crop_len_mm, crop_len_mm, crop_len_mm)

        self.rot_lim = abs_rot_lim_deg if abs_rot_lim_deg <=180 else 180
        self.sc_std = scale_std if scale_std <= 0.02 else 0.02
        self.tr_std = trans_std if trans_std <= 5 else 5
        self.aug_mode_lst = aug_mode_lst

        self.debug_mode = debug_mode    # plot if in debug mode

    def __call__(self, sample):
        ### this function is called before reutrning a sample to transform the sample and corresponding
        ### output using a) data augmentation and b) voxelisation
        ### __init__ is called when object is first defined
        ### __call__ is called for all subsequent calls to object
        ## as x,y,z mm where center is center of image

        ### orig data loading ###
        dpt_orig = sample['depthmap']
        keypt_mm_orig = sample['joints']
        com_mm_orig = sample['refpoint']

        ## convert joints & CoM to img coords
        ## note this is right for original image but wrong for cropped pic
        keypt_px_orig = self.mm2pxMulti(keypt_mm_orig)
        com_px_orig = self.mm2px(com_mm_orig)

        ### cropping + centering ###
        ## convert input image to cropped, centered and resized 2d img
        ## required CoM in px value
        ## convert to 128x128 for network
        ## px_transform_matx for 2D transform of keypoints in px values
        dpt_crop, crop_transf_matx = cropDepth2D(
                                                dpt_orig, com_px_orig,
                                                fx=self.fx, fy=self.fy,
                                                crop3D_mm=self.crop_vol_mm,
                                                out2D_px=self.out_sz_px
                                                )
       
        aug_mode, aug_param = getAugModeParam(self.aug_mode_lst, self.rot_lim, 
                                                self.sc_std, self.tr_std)
        
        # notice we supply {dpt_crop, keypt_px_orig}, we need dpt_crop_aug
        # and keypt_mm_crop_aug
        # dpt_crop_aug is done using the transf func
        # keypt_px_crop_aug is of no use to us (except plotting)
        # keypt_mm_crop_aug can only be found using keypt_px_orig_aug which
        # is what we get using transf func
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

        keypt_mm_crop_aug = self.px2mmMulti(keypt_px_orig_aug) - self.px2mm(com_px_orig_aug)
        
        ### debugging using plots ###
        ## final input (before std) is dpt_crop_aug
        plotImg(dpt_orig, dpt_crop, keypt_px_orig, com_px_orig, 
                crop_transf_matx=crop_transf_matx, aug_transf_matx=aug_transf_matx,
                aug_mode=aug_mode, aug_val=aug_param, dpt_crop_aug=dpt_crop_aug) \
                    if self.debug_mode else None

        ### Standardisation ###
        ## This must be the last step always!
        dpt_final = \
            standardiseImg(dpt_crop_aug, com_mm_orig[2], self.crop_vol_mm[2])[np.newaxis, ...]
        keypt_final = standardiseKeyPoints(keypt_mm_crop_aug, self.crop_vol_mm[2]).flatten()

        
        # return final x, y pair -- HOWEVER FOR TRAINING YOU NEED PCA VERSION OF OUTPUTS!!
        # no need to convert to torch as this is fine as numpy here
        # the dataloader class automatically gets a torch version.

        return (dpt_final, keypt_final)

    
    ## from deep-prior
    def px2mmMulti(self, sample):
        """
            Converts a collection of pixel indices (w.r.t an image
            with original dimensions (un-cropped)) to a collection of mm values
            (w.r.t the original focal point i.e 0mm at focal point).
            :param sample: row vectors in (x,y,z) with x,y in original image 
            coordinates and z in mm
            :return: {(x1,y1,z1), ..., (xN, yN, zN)} in mm 
        """
        ret = np.zeros((sample.shape[0], 3), dtype=np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.px2mm(sample[i])
        return ret

    def px2mm(self, sample):
        """
            Converts pixel indices (w.r.t an image
            with original dimensions (un-cropped)) to mm values
            (w.r.t the original focal point i.e 0mm at focal point).
            :param sample: vector in (x,y,z) with x,y in original image 
            coordinates and z in mm
            :return: (x,y,z) in mm 
        """
        ret = np.zeros((3,), dtype=np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret
    

    def mm2pxMulti(self, sample):
        """
            Converts a collection of mm values (w.r.t the original focal point 
            i.e 0mm at focal point) to a collection of indices with px values
            (w.r.t an image of original dimensions (un-cropped)).
            :param sample: row vectors in (x,y,z) with x,y in original image 
            coordinates and z in mm
            :return: {(x1,y1,z1), ..., (xN, yN, zN)} in mm 
        """
        ret = np.zeros((sample.shape[0], 3), dtype=np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.mm2px(sample[i])
        return ret

    def mm2px(self, sample):
        """
        Denormalize each joint/keypoint from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joint in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), dtype=np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret



class DeepPriorXYTestTransform(DeepPriorXYTransform):
    '''
        This transformer is designed for use during testing
        You are returned a triplet with X, Y, CoM

        Your model returns Y_bar_centered
        You use original CoM to make this Y_bar

        Note: Y_bar_centered -> Y_bar comes from object localisation
        and in hand pose est. this task is not at all handled.

        I.e. for us we assume somehow to get 'CoM' from hand_detector
        which did localisation and stored all such values for later
        use in a file.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class

    def __call__(self, sample):
        final_depth_img, _ = super().__call__(sample) # transform x, y as usual
        keypoints_gt_mm = sample['joints']  # get the actual y (untransformed)
        com_mm = sample['refpoint'] # neede to transform back output from model

        # basically for test we don't need to transform y_coords
        # but we need to supply com_mm so that the ouput from the
        # nn can be corresctly recovered to keypoints_gt_mm range
        # hence this derived class is used.
        return (final_depth_img, keypoints_gt_mm, com_mm)


class DeepPriorYTransform(DeepPriorXYTransform):
    '''
        Quick transformer for y-vals only (keypoint)
        Centers (w.r.t CoM in dB) and standardises (-1,1) y
        y-val -> center

        Overload of XY transform so we can also do y augmentation
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class

    def __call__(self, sample):
        keypt_mm_orig = sample['joints']
        com_mm_orig = sample['refpoint']

        keypt_px_orig = self.mm2pxMulti(keypt_mm_orig)
        com_px_orig = self.mm2px(com_mm_orig)

        # wherever dpt is needed, supply none to by-pass dpt transform

        _, crop_transf_matx = cropDepth2D(
                                            None, com_px_orig,
                                            fx=self.fx, fy=self.fy,
                                            crop3D_mm=self.crop_vol_mm,
                                            out2D_px=self.out_sz_px
                                        )
        
        aug_mode, aug_param = getAugModeParam(self.aug_mode_lst, self.rot_lim, 
                                                self.sc_std, self.tr_std)

        (_, keypt_px_orig_aug, com_px_orig_aug, _) = \
            rotateHand2D(None, keypt_px_orig, com_px_orig, aug_param) \
            if aug_mode == AugType.AUG_ROT \
            else translateHand2D(None, keypt_px_orig, com_px_orig, com_mm_orig, aug_param, 
                                    self.fx, self.fy, crop_transf_matx, self.mm2px, self.crop_vol_mm) \
            if aug_mode == AugType.AUG_TRANS \
            else scaleHand2D(None, keypt_px_orig, com_px_orig, com_mm_orig, aug_param, 
                                    self.fx, self.fy, crop_transf_matx,
                                    self.mm2px, crop3D_mm=self.crop_vol_mm) \
            if aug_mode == AugType.AUG_SC \
            else (None, keypt_px_orig, com_px_orig, np.eye(3, dtype=np.float32))

        keypt_mm_crop_aug = self.px2mmMulti(keypt_px_orig_aug) - self.px2mm(com_px_orig_aug)


        return standardiseKeyPoints(keypt_mm_crop_aug, self.crop_vol_mm[2]).flatten()



class PCATransform():
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
                 cache_dir='checkpoint'):
        
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
            os.mkdir(self.cache_dir)
        
        cache_file = os.path.join(self.cache_dir, 'pca_'+str(self.out_dim)+'_cache.npz')
        np.savez(cache_file, transform_matrix_np=self.transform_matrix_np, mean_vect_np=self.mean_vect_np)


    
    
    ## create inverse transform function?
    
    def fit(self, X, return_X_no_mean=False):
        ## assume input is of torch type
        ## can put if condition here
        if X.dtype != self.dtype or X.device != self.device:
            # make sure to switch to cpu
            X.to(device=self.device, dtype=self.dtype)
        
        if self.transform_matrix_np is not None:
            Warning("PCA transform matx already exists, refitting...")

        # mean normalisation
        #X_mean = torch.mean(X,0)
        #X = X - X_mean.expand_as(X)

        sklearn_pca = PCA(n_components=self.out_dim)
        sklearn_pca.fit(X.cpu().detach().numpy())


        # svd, need to transpose x so each data is in one col now
        #U,S,_ = torch.svd(torch.t(X))

        #print("X.shape:", X.shape, "U.shape:", U.shape)

        ## store U.T as this is the correct matx for single samples i.e. vectors, for multiple i.e. matrix ensure to transpose back!
        #self.transform_matrix_torch = torch.t(U[:,:self.out_dim])
        #self.mean_vect_torch = X_mean

        # if in cpu just return view tensor as ndarray else copy array to cpu and return as ndarray
        self.transform_matrix_np = sklearn_pca.components_.copy()#self.transform_matrix_torch.cpu().clone().numpy()
        self.mean_vect_np = sklearn_pca.mean_.copy() #mean_vect_torch.cpu().clone().numpy().flatten() # ensure 1D

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
            return X - torch.from_numpy(self.mean_vect_np).expand_as(X)
        else:
            return None
    
    
    def fit_transform(self, X):
        ## return transformed features for now
        ## by multiplying appropriate matrix with mean removed X
        # note fit if given return_X_no_mean returns X_no_mean_Torch
        return np.matmul(self.fit(X, return_X_no_mean=True).numpy(), self.transform_matrix_np.T)



class DeepPriorBatchResultCollector():
    def __init__(self, data_loader, transform_output, num_samples):
        self.data_loader = data_loader
        self.transform_output = transform_output
        self.num_samples = num_samples  # need to be exact total calculated using len(test_set)
        
        self.keypoints = None # pred_mm_not_centered
        self.keypoints_gt = None # gt_mm_not_centered
        self.idx = 0
    
    def __call__(self, data_batch):
        ## this function is called when we need to calculate final output error
        
        ### load data straight from disk, y & CoM is loaded straight from the test_loader
        ## x, pred_std_centered, gt_mm_not_centered, CoM = data_batch

        # the first component is input_batch don't need this for now
        # cen => centered i.e. w.r.t CoM
        # nc => not centered i.e. w.r.t focal point of image i.e. similar to gt
        _, pred_std_cen_batch, gt_mm_nc_batch, com_batch = data_batch
        
        pred_std_cen_batch = pred_std_cen_batch.cpu().numpy()
        gt_mm_nc_batch = gt_mm_nc_batch.cpu().numpy()
        com_batch = com_batch.cpu().numpy()

        # an important transformer
        pred_mm_nc_batch = self.transform_output((pred_std_cen_batch, com_batch))

        #print("pred_mm_nc (min, max): (%0f, %0f)\t gt_mm_nc (min, max): (%0f, %0f)" % \
        #        (pred_mm_nc_batch.min(), pred_mm_nc_batch.max(), gt_mm_nc_batch.min(), gt_mm_nc_batch.max()))
        
        ## Note we will have a problem if we have >1 num_batches
        ## and last batch is incomplete, ideally in that case

        if self.keypoints is None:
            # Initialize keypoints until dimensions available now
            self.keypoints = np.zeros((self.num_samples, *pred_mm_nc_batch.shape[1:]))
        if self.keypoints_gt is None:
            # Initialize keypoints until dimensions available now
            self.keypoints_gt = np.zeros((self.num_samples, *gt_mm_nc_batch.shape[1:]))

        batch_size = pred_mm_nc_batch.shape[0] 
        self.keypoints[self.idx:self.idx+batch_size] = pred_mm_nc_batch
        self.keypoints_gt[self.idx:self.idx+batch_size] = gt_mm_nc_batch
        self.idx += batch_size

    def get_result(self):
        ## this will just return predicted keypoints
        return self.keypoints
    
    def get_ahpe_result(self, ahpe_fname, test_model_id, dataset_dir):
        '''
            `ahpe_fname` => msra_test_list.txt
            test_model_id => current unseen test model.

            Pseudo Code:
            1) load val_sample_name list from ahpe_fname
            2) load test_sample_name from self.dataloader.dataset.names
                Note this is test data loader and we need to access
                Underlying test data-set
            3) Filter val_sample_name to only those starting as
                'P%s*' % test_model_id to get only valid names for that
                test model
            4) Now find all indices of test_sample_name that are also in
                val_sample_name
                Now use these indices to select from self.keypoints only
                valid keypoints and save results as
                eval_id_valid.txt
            5) later on u can merge all these files make sure last merge
                is same num as hpce
        '''

        assert (test_model_id >= 0 and test_model_id <= 8), "Error: Test_ID must be >=0 and <=8"
        
        prefix = 'P%d' % test_model_id
        with open(ahpe_fname) as f:
            val_name_arr = \
                np.array([os.path.join(dataset_dir, line.rstrip('\n')) \
                                    for line in f if line.startswith(prefix)])
        dataset_name_arr = np.array(self.data_loader.dataset.names)
        valid_mask_arr = np.isin(dataset_name_arr, val_name_arr, assume_unique=True)
        #print("val_name_arr:\n", dataset_name_arr)
        #print("\nDataset_Name_Arr:\n", dataset_name_arr)
        
        # note our keypt arr is (num_samples, num_joints, num_dim)
        # we just need to filter out num_samples, nothing else!
        val_keyp_arr = self.keypoints[ valid_mask_arr ]
        print("Filter Results: (Mask, Valid Names, Valid Keypoints) -- ",\
                    valid_mask_arr.shape, val_name_arr.shape, val_keyp_arr.shape)

        assert val_name_arr.shape[0] == val_keyp_arr.shape[0], \
                        "Error: Something went wrong with np.in1d..."
        assert self.keypoints.shape[1] == self.keypoints.shape[1]
        assert self.keypoints.shape[2] == self.keypoints.shape[2]

        return val_keyp_arr

    
    def calc_avg_3D_error(self, ret_avg_err_per_joint=False):
        ## use self.keypoints for model's results
        ## use self.keypoints_gt for gt results

        ## R^{500, 21, 3} - R^{500, 21, 3} => R^{500, 21, 3} err
        ## R^{500, 21, 3} == l2_err_dist_per_joint ==> R^{500, 21} <-- find euclidian dist btw gt and pred
        err_per_joint = np.linalg.norm(self.keypoints - self.keypoints_gt, ord=2, axis=2)

        ## R^{500, 21} == avg_err_across_dataset ==> R^{21}
        ## do avg for each joint over errors of all samples
        avg_err_per_joint = err_per_joint.mean(axis=0)

        ## R^{21} == avg_err_across_joints ==> R
        avg_3D_err = avg_err_per_joint.mean()

        if ret_avg_err_per_joint:
            return avg_3D_err, avg_err_per_joint
        else:
            return avg_3D_err

        ## for each test frame
        ## calc euclidian dist for each joint's 3D vector between pred and gt to get error matrix
        ## which is R^{21x3}
        ## each row is error of one joint
        ## each col is x,y & z error respectively.
        ## now reduce x,y,z error to single val using euc. dist aka norm
        ## so error is R^{21}
        ## now reduce this dataset matrix of
        ## R^{Nx21} where N is test size to
        ## R^{21} to get avg error of each joint
        ## FINALLY if needed avg R^{21} -> R^{1}
        ## to get avg 3D error
    
    # from v2v_posenet
    def compute_dist_acc_wrapper(self, max_dist=10, num=100):
        '''
        pred: (N, K, 3)
        gt: (N, K, 3)
        return dist: (K, )
        return acc: (K, num)
        '''
        assert(self.keypoints.shape == self.keypoints_gt.shape)
        assert(len(self.keypoints.shape) == 3)

        dist = np.linspace(0, max_dist, num)
        return dist, computeDistAcc(self.keypoints, self.keypoints_gt, dist)

    # from v2v_posenet, internal func