from argparse import Namespace

import numpy as np
import torch

from datasets import *
from data_utils import *
from data_utils.data_loaders import _check_pca
from models import *


def test_addition():
    assert 3 == (1+2)


def test_simple_dataloading():
    '''
        test first sample returns value as expected under default conditions
    '''
    train_loader = JointsActionDataLoader(
                                                data_dir='datasets/hand_pose_action',
                                                dataset_type='test',
                                                batch_size=4,
                                                shuffle=False,
                                                validation_split=0.0,
                                                num_workers=0,
                                                debug=False,
                                                reduce=True,
                                                pad_sequence=False,# True
                                                max_pad_length=-1, # 100,
                                                load_depth=False,
                                                randomise_params=False
                                            )
    sample = next(iter(train_loader))
    sample_tensor = torch.tensor([-0.0258,  0.6907, -0.3394, -0.1375,  0.6528, -0.4441, -0.1389, -0.0786,
                                  -0.1514,  0.0000,  0.0000,  0.0000,  0.1112,  0.1207,  0.0957,  0.2106,
                                  0.2024,  0.1929, -0.3777,  0.2748, -0.0755, -0.5379,  0.0228,  0.1701,
                                  -0.6809, -0.1944,  0.2739, -0.2819, -0.2077,  0.2472, -0.3830, -0.2989,
                                  0.5290, -0.4950, -0.2599,  0.6900, -0.0845, -0.2350,  0.4216, -0.2603,
                                  -0.1190,  0.6353, -0.4000,  0.0624,  0.7071,  0.0724, -0.1018,  0.4733,
                                  -0.0908, -0.0463,  0.6940, -0.2605,  0.1167,  0.7785,  0.1046,  0.1313,
                                  0.4988,  0.0378,  0.0865,  0.6915, -0.1317,  0.1928,  0.7830], dtype=torch.float)
    #print(sample_tensor)
    #print(sample[0][0].data[0] - sample_tensor)
    assert len(sample) == 2 and len(sample[0]) == 2 and len(sample[1]) == 4
    assert torch.allclose(sample[0][0].data[0], sample_tensor, atol=8e-05, rtol=1e-06)
    assert torch.equal(torch.tensor([303, 318, 400, 403]), sample[0][1])
    #assert sample[0]
    #assert 1 ==1
    # sample -> tuple; sample[0] -> tuple; sample[1] -> tensor
    #print(sample[0][0], '\n\n', sample[0][1], '\n\n', sample[1])

def test_simple_training():
    '''
        test first 5 train losses are always same under default conditions and train params
    '''
    train_loader = JointsActionDataLoader(
                                                data_dir='datasets/hand_pose_action',
                                                dataset_type='test',
                                                batch_size=4,
                                                shuffle=False,
                                                validation_split=0.0,
                                                num_workers=0,
                                                debug=False,
                                                reduce=True,
                                                pad_sequence=False,# True
                                                max_pad_length=-1, # 100,
                                                load_depth=False,
                                                randomise_params=False
                    )
    lstm_baseline = BaselineHARModel(
                                            in_frame_dim=63,
                                            out_dim=45,
                                            num_lstm_units_per_layer=100,
                                            num_hidden_layers=1,
                                            lstm_dropout_prob=0.2
                                        )
    optimizer = torch.optim.Adam(lstm_baseline.parameters())
    criterion = torch.nn.NLLLoss()
    losses = []
    (data, target) = next(iter(train_loader))
    for _ in range(5):
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
    
    print("5 Losses:\n", losses)
    assert np.allclose(np.array(losses),
                       np.array([3.7771735191345215, 3.7091832160949707,
                                 3.640038013458252, 3.5674924850463867, 3.489128351211548]))


def test_keypt_crop_vs_no_crop():
    '''
        crop operation on keypt in 3D space should be a no-op so both
        cases of transformations must result in the sample value....
    '''
    train_loader1 = JointsActionDataLoader(
                                                data_dir='datasets/hand_pose_action',
                                                dataset_type='test',
                                                batch_size=4,
                                                shuffle=False,
                                                validation_split=0.0,
                                                num_workers=0,
                                                debug=False,
                                                reduce=True,
                                                pad_sequence=False,# True
                                                max_pad_length=-1, # 100,
                                                load_depth=False,
                                                randomise_params=False
                    )
    train_loader2 = JointsActionDataLoader(
                                            data_dir='datasets/hand_pose_action',
                                            dataset_type='test',
                                            batch_size=4,
                                            shuffle=False,
                                            validation_split=0.0,
                                            num_workers=0,
                                            debug=False,
                                            reduce=True,
                                            pad_sequence=False,# True
                                            max_pad_length=-1, # 100,
                                            load_depth=True,
                                            randomise_params=False
                )
    sample1 = next(iter(train_loader1))
    sample2 = next(iter(train_loader2))

if __name__ == "__main__":
    # pytest --rootdir=tests tests\har_tests.py -v
    # python -m tests.har_tests 
    import argparse
    test_simple_dataloading()
    test_simple_training()
    test_keypt_crop_vs_no_crop()