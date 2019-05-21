import time

from torchvision import transforms

from data_utils import JointsActionDataLoader, CollateJointsSeqBatch, DepthJointsDataLoader, \
                       PersistentDataLoader

from models import BaselineHARModel

import torch

from torch.nn.utils.rnn import pack_sequence

import numpy as np

from contextlib import contextmanager
from timeit import default_timer

from tqdm import tqdm

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
        use_pca = False
        load_depth = False
        pad_seq = True
        ### note this stuff is meant to be wrapped by data loader class!!
        train_loader = JointsActionDataLoader(
                                                data_dir='datasets/hand_pose_action',
                                                dataset_type='test',
                                                batch_size=4,
                                                shuffle=False,
                                                validation_split=0.0,
                                                num_workers=0,
                                                debug=False,
                                                reduce=True,
                                                pad_sequence=pad_seq,# True
                                                max_pad_length=-1, # 100,
                                                load_depth=load_depth, #False,
                                                randomise_params=False,
                                                use_pca=use_pca,
                                            )
        
        lstm_baseline = BaselineHARModel(
                                            in_frame_dim=63 if not use_pca else 30,#63 use 30 if using pca
                                            out_dim=45,
                                            num_lstm_units_per_layer=100,
                                            num_hidden_layers=1,
                                            lstm_dropout_prob=0.2,
                                            use_unrolled_lstm=False#True#False#True
                                        )
        
        
        print("\n[%s] Model Summary: " % elapsed())
        print(lstm_baseline, "\n")

        print("\n=> [%s] Debugging FWD+BKD Pass" % elapsed())
        in_dim = 63 if not use_pca else 30
        sample_input_seq = [
            (np.random.randn(i, in_dim), j) for i,j in zip([3, 5, 10, 12, 7], [0,1,2,3,4])
        ]
        # sample_input_seq = [
        #     (np.random.randn(i, in_dim), j) for i,j in zip([3, 3, 5, 5, 7], [0,1,2,3,4])
        # ]

        coll = CollateJointsSeqBatch(pad_sequence=pad_seq)
        out, target = coll(sample_input_seq)

        ### unneeded now
        # # this is for sorting in ascending order
        # sample_input_seq = [torch.from_numpy(item[0]) for item in sample_input_seq]
        # sample_input_seq.sort(key=lambda a: a.shape[0], reverse=True)
        
        # sample_packed_seq, seq_idx_arr = out
        # assert torch.allclose(sample_packed_seq[0][seq_idx_arr[0]], sample_input_seq[-1][-1])
        # assert torch.allclose(sample_packed_seq[0][seq_idx_arr[1]], sample_input_seq[-2][-1])
        # assert torch.allclose(sample_packed_seq[0][seq_idx_arr[len(seq_idx_arr)-1]], sample_input_seq[0][-1])

        out = out.to(torch.device('cpu'), torch.float) if isinstance(out, torch.nn.utils.rnn.PackedSequence) else tuple(item.to(torch.device('cpu'), torch.float) for item in out)

        outputs = lstm_baseline(out)
        print("Output: ", outputs.shape)
        ## no need one-hot encoding, done automatically by torch!
        ## only supply action_class_idx!
        targets = torch.tensor([1,0,44,15,13])#torch.randn(5, 45)

        optimizer = torch.optim.Adam(lstm_baseline.parameters())
        criterion = torch.nn.NLLLoss()

        
        print("\n=> [%s] Debugging data loader(s) for first 10 batches" % elapsed())
        tmp_item = None
        max_num_batches = 99999

        t = time.time()
        # with tqdm(total=len(train_loader), desc="Loading max %d batches for HAR" % max_num_batches) as tqdm_pbar:
        #     t = time.time()
        #     for i, item in enumerate(train_loader):
        #         if i > max_num_batches:
        #            break
        #         # print("Got ", i, " Shape: ", item[0][0].data.shape)
        #         tmp_item = item # store running last item as the tmp_item
        #         tqdm_pbar.update(1)
        print("HAR Data Loading Took: %0.2fs\n" % (time.time() - t) )
        
        t = time.time()
        with tqdm(total=len(train_loader), desc="Loading max %d batches for HAR" % max_num_batches) as tqdm_pbar:
            for i, item in enumerate(train_loader):
                if i > max_num_batches:
                   break
                #print("Got ", i, " Shape: ", item[0][0].data.shape)
                tmp_item = item # store running last item as the tmp_item
                tqdm_pbar.update(1)
        print("HAR Data Loading Took: %0.2fs\n" % (time.time() - t) )
        
        

        #from copy import deepcopy
        print("\n=> [%s] Debugging single batch training for HAR" % elapsed())
        print("Overfitting HAR on 1 batch for 10 epochs...")
        print("Info: Detected type is %s" % ('TUPLE' if isinstance(item[0], tuple) else \
            'TORCH.TENSOR' if isinstance(item[0], torch.Tensor) else 'UNKNOWN'))
        losses = []
        (data, target) = tmp_item #deepcopy(tmp_item) # we need 'fresh new data' everytime! otherwise some errors in backprop
        for _ in range(10):
            #(data, target) = deepcopy(tmp_item) # tmp_item.copy()
            if isinstance(data, torch.Tensor):
                output = lstm_baseline(data) # direct invocation calls .forward() automatically
            elif isinstance(data, tuple):
                # if its not a tensor its probably a tuple
                # we expect model to handle tuple
                # we send it in similar fashion to *args
                output = lstm_baseline(data)
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