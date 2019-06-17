import time

from torchvision import transforms

from data_utils import JointsActionDataLoader, CollateJointsSeqBatch, DepthJointsDataLoader,\
                       PersistentDataLoader, AugType

from models import BaselineHARModel, DeepPriorPPModel
from trainer import init_metrics

import torch

from torch.nn.utils.rnn import pack_sequence

import numpy as np

from contextlib import contextmanager
from timeit import default_timer

from tqdm import tqdm
from copy import deepcopy

import torchvision

def normalise_hand_pose(input_pose):
    """ Make wrist origin Normalise bone lengths of input pose to unit"""

    # wrist as origin
    tmp = deepcopy(input_pose.reshape(21,3))
    #init_pos = copy.deepcopy(tmp[0])

    #for i in range(len(tmp)):  # wrist norm
    #    tmp[i] -= init_pos

    output_pose = np.zeros((21, 3))

    links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
                (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]

    #links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    '''
        [
            [0,1], [1,6] , [6,7]  , [7,8]  ,
            [0,2], [2,9] , [9,10] , [10,11],
            [0,3], [3,12], [12,13], [13,14],
            [0,4], [4,15], [15,16], [16,17],
            [0,5], [5,18], [18,19], [19,20],
        ]
    
    a[[1,2,3,4,5]] - a[[0,0,0,0,0]]
    
    joints
    a = [[0,0,0,0,0], [1,2,3,4,5], [6,9,12,15,18], [7,10,13,16,19], [8,11,14,17,20]]

    for i in range(0, len(a) - 1):
        diff_vec = joints[a[i+1]] - joints[a[i]]
        diff_vec = diff_vec / np.linalg.norm(diff_vec)
        joints[a[i+1]] = joints[a[i]] + diff_vec
    
    we need to make sure that whether to copy or not also when we do this what happens to next set of vect?
    i i think this way is fine
    '''
    # note we shuld only do this for action recognition but not for hpe because that will be hard to recover!
    # if we do it for hpe then it'll be like losing information or making things simpler
    # so basically we need to create a transformer to do this in the middle of hpe/har and basically
    # require a transformer to unstandardise and then make it unit bone length

    ## need to set output_pose to be non_origin so thats done here if wrist com is used then this is not required

    output_pose[0] = tmp[0]

    ## for each finger..
    for finger_idx, finger_links in enumerate(links):
        ## lets say u are at first finger...
        ## now for each CONSEQUTIVE pair say (0,1), (1,2), (2,3) etc..
        for idx in range(len(finger_links) - 1):
            # you find vector between joints in this pair lets say vect_1 - vect_0
            # you normalise this vector so vector has unit length
            # now you simply add this vector to the origin join or first coord
            # so that for 2 terms (vA, vB) you do
            # vB = vA + ((vB-vA)/||(vB-vA)||)
            # so the direction will always be the same! Only it'll be the unit norm dist
            old_vec = tmp[finger_links[idx+1]]-tmp[finger_links[idx]]
            old_vec = old_vec/np.linalg.norm(old_vec)
            output_pose[finger_links[idx+1]] = output_pose[finger_links[idx]] + old_vec

    return output_pose


def normalise_hand_pose_v2(input_pose):
    links = [[0,0,0,0,0], [1,2,3,4,5], [6,9,12,15,18], [7,10,13,16,19], [8,11,14,17,20]]

    tmp = deepcopy(input_pose.reshape(21,3))
    output_pose = np.zeros((21, 3))

    output_pose[0] = tmp[0]

    for i in range(0, len(links) - 1):
        diff_vec = tmp[links[i+1]] - tmp[links[i]]
        diff_vec = (diff_vec / np.linalg.norm(diff_vec, axis=1).reshape(-1,1)) # take diff along 3D vals only
        output_pose[links[i+1]] = output_pose[links[i]] + diff_vec
    
    return output_pose

## display elapsed time
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: "%0.4fs" % (default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start



#@profile
def debug():
    ### fix for linux filesystem
    torch.multiprocessing.set_sharing_strategy('file_system')

    with elapsed_timer() as elapsed:

        hpe_train_loader = DepthJointsDataLoader(
                                                data_dir='datasets/hand_pose_action',
                                                dataset_type='train',
                                                batch_size=4,
                                                shuffle=False,
                                                validation_split=0.2,#-1.0,
                                                num_workers=0,# debugging
                                                debug=False,
                                                reduce=True,
                                                use_pca_cache=True,
                                                pca_overwrite_cache=False,#True,#False,
                                                use_msra=False,
                                                output_type='depth_joints', #'depth_action_joints',
                                                data_aug=[AugType.AUG_ROT],
                                                eval_pca_space=False,
                                            )
        hpe_test_loader = hpe_train_loader.split_validation()
        # hpe_test_loader = DepthJointsDataLoader(
        #                                         data_dir='datasets/hand_pose_action',
        #                                         dataset_type='test',
        #                                         batch_size=4,
        #                                         shuffle=False,
        #                                         validation_split=0.0,
        #                                         num_workers=0,# debugging
        #                                         debug=False,
        #                                         reduce=True,
        #                                         use_pca_cache=True,
        #                                         pca_overwrite_cache=True,#False,
        #                                     )
        
        
        #print("\n[%s] Model Summary: " % elapsed())

        print("\n=> [%s] Debugging FWD+BKD Pass" % elapsed())
        

        norm_dist = torch.distributions.normal.Normal(0, 1)

        ## for depth + action set input channels to 2 .. temp for now

        hpe_baseline = DeepPriorPPModel(input_channels=1, predict_action=False, action_cond_ver=0, eval_pca_space=False) #6 , 5 ; 3
        # inputs = norm_dist.sample((10, 2,128,128)) # 10 hand samples
        # targets = norm_dist.sample((10,30))

        # outputs = hpe_baseline(inputs)
        #print("Output: ", outputs.shape, "Target: ", targets.shape)

        from metrics import mse_and_nll_loss, Avg3DError


        optimizer = torch.optim.Adam(hpe_baseline.parameters())
        criterion = torch.nn.MSELoss()#mse_and_nll_loss #torch.nn.MSELoss()

        #persistent_data_loader = PersistentDataLoader(hpe_train_loader)

        metrics = [Avg3DError]
        init_metrics(metrics, hpe_baseline, hpe_train_loader, torch.device('cpu'), torch.float)

        
        print("\n=> [%s] Debugging Data Loader(s)" % elapsed())
        
        tmp_item = None
        max_num_batches = 2#99999

        tst = hpe_train_loader.dataset[0]
        
        with tqdm(total=len(hpe_train_loader), desc="Loading max %d batches for HPE" % max_num_batches) \
            as tqdm_pbar:
            t = time.time()
            for i, item in enumerate(hpe_train_loader):
                if i > max_num_batches:
                   break
                #print("Got ", i, " Shape: ", item[0].shape)
                tmp_item = item # store running last item as the tmp_item
                tqdm_pbar.update(1)
        print("HPE Data Loading Took: %0.2fs\n" % (time.time() - t) )



        print("\n=> [%s] Debugging single batch training for HPE" % elapsed())
        print("Overfitting HPE on 1 batch for 10 epochs...")
        # print("Info: Detected type is %s" % ('TUPLE' if isinstance(item[0], tuple) else \
        #     'TORCH.TENSOR' if isinstance(item[0], torch.Tensor) else 'UNKNOWN'))
        losses = []
        (data, target) = tmp_item[0], tmp_item[1]

        # test bone_length
        a = normalise_hand_pose(target[0].cpu().numpy())

        b = normalise_hand_pose_v2(target[0].cpu().numpy())

        assert np.array_equal(a,b)

        if isinstance(data, tuple):
            data = tuple(sub_data.to(torch.float32) for sub_data in data)
        else:
            data = data.to(torch.float32)

        if isinstance(target, tuple):
            target = tuple(sub_data.to(torch.float32) for sub_data in target)
        else:
            target = target.to(torch.float32)

        # print("DATA_MIN:", data.min(), "DATA_MAX:", data.max(),
        #       "\tTARGET_MIN:", target.min(), "TARGET_MAX:", target.max())

        
        #print("Input Shape:", data.shape, " Output Shape:", target[0].shape, "Action shape", target[1].shape)
        for _ in range(10): #10
            output = hpe_baseline(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward() # calc grads w.r.t weight/bias nodes
            optimizer.step() # update weight/bias params
            losses.append(loss.item())
        print("10 Losses:\n", losses)

        print("\n\n=> [%s] All debugging complete!\n" % elapsed())
        

#### for debugging
if __name__ == "__main__":
    debug()