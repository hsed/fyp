import time

from torchvision import transforms

from datasets.hand_pose_action import HandPoseActionDataset, DatasetMode, TaskMode

from data_utils import JointsActionDataLoader, CollateJointsSeqBatch

from models import BaselineHARModel

import torch

from torch.nn.utils.rnn import pack_sequence

import numpy as np

#@profile
def debug():
    ### fix for linux filesystem
    torch.multiprocessing.set_sharing_strategy('file_system')

    ### note this stuff is meant to be wrapped by data loader class!!
    train_loader = JointsActionDataLoader(
                                            data_dir='datasets/hand_pose_action',
                                            dataset_type='train',
                                            batch_size=4,
                                            shuffle=False,
                                            validation_split=0.0,
                                            num_workers=0,
                                            debug=False
                                         )
    
    lstm_baseline = BaselineHARModel(
                                        in_frame_dim=63,
                                        out_dim=45,
                                        num_lstm_units_per_layer=100,
                                        num_hidden_layers=1,
                                        lstm_dropout_prob=0.2
                                    )

    #train_loader.__iter__().__next__()
    # for i, (data, targets) in enumerate(train_loader):
    #     print("VALUE:\n", data.shape)
    #     quit()
    
    print("\nModel Summary: ")
    lstm_baseline.summary()
    print("\n")

    sample_input_seq = [
        (np.random.randn(i, 63), j) for i,j in zip([3, 5, 10, 12, 7], [0,1,2,3,4])
    ]

    coll = CollateJointsSeqBatch()
    out, target = coll(sample_input_seq)

    # get sort idx for sorting output
    # this is for sorting in ascending order
    ##sort_idx_asc = sorted(range(len(sample_input_seq)), key=lambda k: sample_input_seq[k].shape[0])
    sample_input_seq = [torch.from_numpy(item[0]) for item in sample_input_seq]
    sample_input_seq.sort(key=lambda a: a.shape[0], reverse=True)
    ##sample_packed_seq = pack_sequence(sample_input_seq)
    
    ## new sequence lengths in ASCENDING ORDER
    ## must ensure output is also in ASCENDING order
    ##seq_lengths = np.flipud([s.shape[0] for s in sample_input_seq])

    #seq_lengths[0] -= 1
    #diff_arr = np.append(seq_lengths[0], np.diff(seq_lengths))
    #mult_arr = np.arange(seq_lengths.shape[0], 0, -1)
    #seq_idx_arr = np.cumsum(diff_arr*mult_arr) - 1

    ##seq_idx_arr = np.cumsum(np.append(seq_lengths[0],np.diff(seq_lengths))*np.arange(seq_lengths.shape[0], 0, -1))-1
    
    # need to select indices
    # apparently it has to be a long tensor :(
    ##seq_idx_arr = torch.from_numpy(seq_idx_arr.astype(np.int64))
    # this will return relevant indices for each sample
    # it is basically the point at which that particular sample reaches its end of sequence
    # these indices can be used to idx relevant options from packed seq
    # to test we can check by indexing packes sequence and seeing that each ascending sorted
    # sample by sequence length has its last sequence as the same 


    ##print('Input Seq of %d items -- Shape:\n%a' % (len(sample_input_seq), [s.shape for s in sample_input_seq]))
    ##print('Lengths: %a' % [s for s in seq_lengths])
    
    ##print('Indices: %a' % seq_idx_arr)
    ##print("Packed Seq:\n", sample_packed_seq)

    # assertions -- assuming sample_input_seq is desc sorted
    # sample_packed_seq[0][seq_idx_arr[0]] <==> sample_input_seq[-1][-1]
    # sample_packed_seq[0][seq_idx_arr[1]] <==> sample_input_seq[-2][-1]
    # ...
    # sample_packed_seq[0][seq_idx_arr[len(seq_idx_arr)-1]] <==> sample_input_seq[0][-1]
    sample_packed_seq, seq_idx_arr = out
    assert torch.allclose(sample_packed_seq[0][seq_idx_arr[0]], sample_input_seq[-1][-1])
    assert torch.allclose(sample_packed_seq[0][seq_idx_arr[1]], sample_input_seq[-2][-1])
    assert torch.allclose(sample_packed_seq[0][seq_idx_arr[len(seq_idx_arr)-1]], sample_input_seq[0][-1])

    out = tuple(item.to(torch.device('cpu'), torch.float) for item in out)

    outputs = lstm_baseline(*out)
    print("Output: ", outputs.shape)
    ## no need one-hot encoding, done automatically by torch!
    ## only supply action_class_idx!
    targets = torch.tensor([1,0,44,15,13])#torch.randn(5, 45)

    optimizer = torch.optim.Adam(lstm_baseline.parameters())
    criterion = torch.nn.NLLLoss()

    #optimizer.zero_grad()
    #loss = criterion(outputs,targets)
    #loss.backward()
    #optimizer.step()
    from tqdm import tqdm
    with tqdm(total=len(train_loader)) as tqdm_pbar:
        t = time.time()
        for i, item in enumerate(train_loader):
            #print("Got ", i)
            tqdm_pbar.update(1)
            if i > 9:
                break
        print("Data Loading Took: %0.2fs" % (time.time() - t) )
    losses = []
    
    print("Overfitting on 1 batch for 10 epochs...")
    for _ in range(10):
        optimizer.zero_grad()
        outputs = lstm_baseline(*out)   # direct invocation calls .forward() automatically
        loss = criterion(outputs, targets)
        loss.backward() # calc grads w.r.t weight/bias nodes
        optimizer.step() # update weight/bias params
        losses.append(loss.item())

    print("10 Losses:\n", losses)




#### for debugging
if __name__ == "__main__":
    debug()