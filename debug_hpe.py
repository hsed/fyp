import time

from torchvision import transforms

from data_utils import JointsActionDataLoader, CollateJointsSeqBatch, DepthJointsDataLoader,\
                       PersistentDataLoader

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
                                                validation_split=0.0,
                                                num_workers=0,# debugging
                                                debug=False,
                                                reduce=True,
                                                use_pca_cache=True,
                                                pca_overwrite_cache=True,#False,
                                            )
        
        
        
        #print("\n[%s] Model Summary: " % elapsed())

        print("\n=> [%s] Debugging FWD+BKD Pass" % elapsed())
        

        norm_dist = torch.distributions.normal.Normal(0, 1)
    
        hpe_baseline = DeepPriorPPModel()
        inputs = norm_dist.sample((10, 1,128,128)) # 10 hand samples
        targets = norm_dist.sample((10,30))

        outputs = hpe_baseline(inputs)
        print("Output: ", outputs.shape, "Target: ", targets.shape)

        optimizer = torch.optim.Adam(hpe_baseline.parameters())
        criterion = torch.nn.MSELoss()

        #persistent_data_loader = PersistentDataLoader(hpe_train_loader)

        
        print("\n=> [%s] Debugging Data Loader(s)" % elapsed())
        
        tmp_item = None
        max_num_batches = 2#99999
        
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

        
        # with tqdm(total=len(persistent_data_loader), desc="Reloading max %d batches for HPE" % max_num_batches) \
        #     as tqdm_pbar:
        #     for i, item in enumerate(persistent_data_loader):
        #         if i > max_num_batches:
        #            break
        #         tqdm_pbar.update(1)


        print("\n=> [%s] Debugging single batch training for HPE" % elapsed())
        print("Overfitting HAR on 1 batch for 10 epochs...")
        print("Info: Detected type is %s" % ('TUPLE' if isinstance(item[0], tuple) else \
            'TORCH.TENSOR' if isinstance(item[0], torch.Tensor) else 'UNKNOWN'))
        losses = []
        (data, target) = tmp_item[0].to(torch.float32), tmp_item[1].to(torch.float32)

        print("DATA_MIN:", data.min(), "DATA_MAX:", data.max(),
              "\tTARGET_MIN:", target.min(), "TARGET_MAX:", target.max())

        
        print("Input Shape:", data.shape, " Output Shape:", target.shape)
        for _ in range(10):
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