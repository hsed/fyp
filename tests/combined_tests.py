from argparse import Namespace

import numpy as np
import torch
from torch.nn.utils.rnn import *

from datasets import *
from data_utils import *
from data_utils.data_loaders import _check_pca
from models import *
from metrics import *

from contextlib import contextmanager
from timeit import default_timer

## display elapsed time
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: "%0.4fs" % (default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def test_data_collate_fn():
    in_dim = 30
    out_dim = 45

    sample_input_seq_1 = [
        (np.random.randn(i, in_dim), j) for i,j in zip([3, 5, 10, 12, 7], [0,1,2,3,4])
    ]
    expected_targets_1 = torch.tensor([3, 2, 4, 1, 0])
    sample_input_seq_2 = [
        (np.random.randn(i, in_dim), j) for i,j in zip([4, 3, 5, 5, 2], [0,1,2,3,4])
    ]
    expected_targets_2 = torch.tensor([2, 3, 0, 1, 4])

    coll = CollateCustomSeqBatch(pad_sequence=False, max_pad_length=-1, inputs=1, outputs=1, action_in=False, action_out=True)
    coll2 = CollateCustomSeqBatch(pad_sequence=True, max_pad_length=-1, inputs=1, outputs=1, action_in=False, action_out=True)
    in1, target1 = coll(sample_input_seq_1)
    in2, target2 = coll2(sample_input_seq_2)


    sample_input_seq_3 = [
        (np.random.randn(i, in_dim), j, np.random.randn(i, out_dim)) for i,j in zip([4, 3, 5, 5, 2], [0,1,2,3,4])
    ]
    expected_in = torch.tensor([2, 3, 0, 1, 4])
    coll3 = CollateCustomSeqBatch(pad_sequence=False, inputs=2, outputs=1, action_in=True, action_out=False)
    in3, target3 = coll3(sample_input_seq_3)

    print(target1)
    print(target2)
    print(in2)
    print(in3)
    print(target3)

    assert (isinstance(in1, PackedSequence) and isinstance(target1, torch.Tensor))
    assert (isinstance(in2, tuple) and isinstance(in2[0], torch.Tensor) and isinstance(in2[1], torch.Tensor))
    assert (isinstance(in3, tuple) and isinstance(in3[0], PackedSequence) and isinstance(in3[1], torch.Tensor))

    assert torch.equal(target1, expected_targets_1)
    assert torch.equal(target2, expected_targets_2)
    assert torch.equal(in3[1], expected_in)


    sample_input_seq_4 = [
        (np.random.randn(i, in_dim), np.random.randn(i, out_dim)) for i,j in zip([4, 3, 5, 5, 2], [0,1,2,3,4])
    ]
    coll4 = CollateCustomSeqBatch(pad_sequence=False, inputs=1, outputs=1, action_in=False, action_out=False)
    in4, target4 = coll4(sample_input_seq_4)


def test_combined_data_loader_and_model():
    with elapsed_timer() as elapsed:
        pad_seq = False #True Set this to false as most losses are easier to compute using a packed seq rather than pad seq
        train_loader = CombinedDataLoader(
                                                    data_dir='datasets/hand_pose_action',
                                                    dataset_type='train',
                                                    batch_size=2, #4,
                                                    shuffle=False,
                                                    validation_split=0.0,
                                                    num_workers=0,
                                                    debug=False,
                                                    reduce=True,
                                                    pad_sequence=pad_seq,# True
                                                    # # make this a small value for faster training during debugging,
                                                    # this can still limt frames without padding turned on
                                                    # its actually max seq length
                                                    max_pad_length=15, #20, #-1, # 100
                                                    randomise_params=False,
                                                    use_pca=True,# everything is on pca'ed stuff
                                                    forward_type=0, # 3
                                                )
            
        combined_model = CombinedModel(
                                        hpe_checkpoint=None,
                                        har_checkpoint=None,
                                        pca_checkpoint=None,
                                        hpe_args=dict(
                                            input_channels=1,
                                            action_cond_ver=0, # use 0 or 6 => no action cond, 6 => best action cond using film 
                                            dynamic_cond=False, # true -> turn off after X epochs
                                            pca_components=30,
                                            dropout_prob=0.3,
                                            train_mode=True,
                                            init_w=True,
                                            predict_action=False, #true ## new
                                            res_blocks_per_group=5, # 5 -- orig
                                        ),
                                        har_args=dict(
                                            in_frame_dim=30,
                                            num_hidden_layers=1,
                                            use_unrolled_lstm=True, # basically the combined model must use an unrolled version this will be made permanent later
                                        ),
                                        forward_type= 0, #3,
                                        combined_version='0',
                                        force_trainable_params=True, # make all params trainable
        )

        print("\n[%s] Model Loaded, Trainable Params: " % elapsed(), combined_model.param_count())
        #print(combined_model, "\n")

        ## need to finx these
        optimizer = torch.optim.Adam(combined_model.parameters())
        criterion = lambda outputs, targets: mse_seq_and_nll_loss(outputs, targets) #torch.nn.NLLLoss()

        top1_acc_fn = lambda outputs, targets: top1_acc(outputs, targets)

        ### basic ver of avg 3d metric -- a bit messy but no time to clean! TODO: but this code in init area
        ### maybe just directly pass it the train loader params object or something....
        avg_3d_err_fn = Avg3DError(cube_side_mm=train_loader.params['cube_side_mm'],
                                            ret_avg_err_per_joint=False)
        avg_3d_err_fn.pca_decoder = \
            PCADecoderBlock(num_joints=train_loader.params['num_joints'],
                            num_dims=train_loader.params['world_dim'],
                            pca_components=train_loader.params['pca_components'])
        avg_3d_err_fn.pca_decoder.initialize_weights(weight_np=train_loader.pca_weights_np,
                                                            bias_np=train_loader.pca_bias_np)
        

        
        tmp_item = None
        max_num_batches = 2 #-1 # -1 -> Load all batches from reduced set ; N -> load first N batches where N > 0

        t = time.time()
        
        print("\n=> [%s] Debugging data loader(s) for first %d batches" % (elapsed(), max_num_batches))
        with tqdm(total=len(train_loader), desc="Loading batches") as tqdm_pbar:
            for i, item in enumerate(train_loader):
                if i >= max_num_batches and max_num_batches > 0:
                   break
                #print("Got ", i, " Shape: ", item[0][0].data.shape)
                tmp_item = item # store running last item as the tmp_item
                ##print('\n',item[0].data[:4,:5])
                tqdm_pbar.update(1)
        print("Data Loading Took: %0.2fs\n" % (time.time() - t) )
        
        # for param in lstm_baseline.parameters():
        #     param.requires_grad = False
        #print(list(lstm_baseline.parameters()))

        #from copy import deepcopy
        print("\n=> [%s] Debugging single batch training for CombinedModel" % elapsed())
        # print("Info: Detected type is %s" % ('TUPLE' if isinstance(item[0], tuple) else \
        #     'TORCH.TENSOR' if isinstance(item[0], torch.Tensor) else 'UNKNOWN'))
        losses = []
        top1_accs = []
        avg_3d_errs = []
        (data, target) = tmp_item #deepcopy(tmp_item) # we need 'fresh new data' everytime! otherwise some errors in backprop
        epochs = 10
        with tqdm(total=epochs, desc="Overfitting CombinedModel on 1 batch for %d epochs" % epochs) as tqdm_pbar:
            for _ in range(10):
                #(data, target) = deepcopy(tmp_item) # tmp_item.copy()
                output = combined_model(data) # direct invocation calls .forward() automatically
                loss = criterion(output, target)
                loss.backward() # calc grads w.r.t weight/bias nodes
                optimizer.step() # update weight/bias params
                losses.append(loss.item())
                top1_accs.append(top1_acc_fn(output, target))
                avg_3d_errs.append(avg_3d_err_fn(output, target))
                tqdm_pbar.update(1)
        print("10 Losses:\n", losses)
        print("10 Top-1 Acc:\n", top1_accs)
        print("10 Avg3D Errors:\n", avg_3d_errs)

        print("\n\n=> [%s] All debugging complete!\n" % elapsed())


'''
def test_combined_data_loader_and_model_har_vs_combined():
    # a simple tst fn to test output of conmbined mode type 3 and har
    # do something similar to har
'''

if __name__ == "__main__":
    # pytest --rootdir=tests tests\combined_tests.py -v
    # python -m tests.combined_tests 
    import argparse
    #test_data_collate_fn()
    test_combined_data_loader_and_model()