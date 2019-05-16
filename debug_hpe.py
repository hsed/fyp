import time

from torchvision import transforms

from data_utils import JointsActionDataLoader, CollateJointsSeqBatch, DepthJointsDataLoader,\
                       PersistentDataLoader, AugType

from models import BaselineHARModel, DeepPriorPPModel

import torch

from torch.nn.utils.rnn import pack_sequence

import numpy as np

from contextlib import contextmanager
from timeit import default_timer

from tqdm import tqdm

import torchvision

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
                                                output_type='depth_action_joints',
                                                data_aug=[AugType.AUG_ROT]
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

        hpe_baseline = DeepPriorPPModel(input_channels=1, predict_action=False, action_cond_ver=6) # 5 ; 3
        # inputs = norm_dist.sample((10, 2,128,128)) # 10 hand samples
        # targets = norm_dist.sample((10,30))

        # outputs = hpe_baseline(inputs)
        #print("Output: ", outputs.shape, "Target: ", targets.shape)

        from metrics import mse_and_nll_loss

        optimizer = torch.optim.Adam(hpe_baseline.parameters())
        criterion = torch.nn.MSELoss()#mse_and_nll_loss #torch.nn.MSELoss()

        #persistent_data_loader = PersistentDataLoader(hpe_train_loader)

        
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

        
        from metrics import Avg3DError
        from models import PCADecoderBlock
        avg_3d_err_metric = Avg3DError(cube_side_mm=hpe_train_loader.params['cube_side_mm'],
                                                    ret_avg_err_per_joint=False)
        
        avg_3d_err_metric.pca_decoder = \
            PCADecoderBlock(num_joints=hpe_train_loader.params['num_joints'],
                            num_dims=hpe_train_loader.params['world_dim'],
                            pca_components=hpe_train_loader.params['pca_components'])
        
        ## weights are init as transposed of given
        avg_3d_err_metric.pca_decoder.initialize_weights(weight_np=hpe_train_loader.pca_weights_np,
                                                            bias_np=hpe_train_loader.pca_bias_np)
        #avg_3d_err_metric.pca_decoder= avg_3d_err_metric.pca_decoder.to(device, dtype)
        
        # with tqdm(total=len(hpe_test_loader), desc="Loading max %d batches for HPE" % max_num_batches) \
        #     as tqdm_pbar:
        #     for i, item in enumerate(hpe_test_loader):
        #         inputs , targets = item
        #         avg_3d_err_metric(hpe_baseline(inputs), targets)
        #         if i > max_num_batches:
        #            break
        #         tqdm_pbar.update(1)


        print("\n=> [%s] Debugging single batch training for HPE" % elapsed())
        print("Overfitting HPE on 1 batch for 10 epochs...")
        # print("Info: Detected type is %s" % ('TUPLE' if isinstance(item[0], tuple) else \
        #     'TORCH.TENSOR' if isinstance(item[0], torch.Tensor) else 'UNKNOWN'))
        losses = []
        (data, target) = tmp_item[0], tmp_item[1]

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