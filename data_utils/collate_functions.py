import numpy as np
import torch

from torch.nn.utils.rnn import pack_sequence

class CollateJointsSeqBatch(object):
    def __init__(self):
        pass

    
    def __call__(self, batch):
        '''
            Expect a list of tensors with variable last dim based on F (num frames)
            Returns a tuple of (packed_seq, out_idx_arr)
        '''

        #print(batch)
        #sort_idx_asc = sorted(range(len(batch)), key=lambda k: batch[k].shape[0])
        '''
            Batch Format:
            - The format is a list of exactly what is returned by the last transformer of
              the dataset
            - In our case that's a tuple with first elem as variable sized joints_seq
              and a single idx for action
            - joints_seq is already in numpy.ndarray format
        '''

        # print("Type_of_batch: ", type(batch))
        # print("Type_of_batch[0]: ", type(batch[0]))
        # print("Type_of_batch[0][0]: ", type(batch[0][0]))

        # print("Len_of_batch: ", len(batch))
        # print("Len_of_batch[0]: ", len(batch[0]))
        # print("Len_of_batch[0][0]: ", len(batch[0][0]))

        # batch[i] --> ith (== item_idx) (input,output) tuple
        # batch[i][0] --> ith input
        # batch[i][1] --> ith output


        sort_idx_asc = sorted(range(len(batch)), key=lambda item_idx: batch[item_idx][0].shape[0])
        
        sort_idx_asc = np.array(sort_idx_asc)

        sort_idx_desc = np.flipud(sort_idx_asc)

        #print("Sorts::ASC::DESC\n", sort_idx_asc, sort_idx_desc)

        targets_sorted_asc = np.array([item[1] for item in batch])[ sort_idx_asc ]

        # NOTE: this is a non_numeric array of arrays!
        # Need numpy type due to sorting

        # temporary conversion to np.array of tensors for sorting
        inputs_sorted_desc = list(np.array([torch.from_numpy(item[0]) for item in batch], dtype='O')[ sort_idx_desc ])

        seq_lengths_asc = np.flipud([s.shape[0] for s in inputs_sorted_desc])

        seq_idx_arr = np.cumsum(np.append(seq_lengths_asc[0],np.diff(seq_lengths_asc))*np.arange(seq_lengths_asc.shape[0], 0, -1))-1
        
        # interesting point is that packed_seq support to so we can later convert to correct dtypes
        # if required e.g. double -> float
        packed_inputs = pack_sequence(list(inputs_sorted_desc))
        
        # required to be long
        seq_idx_arr = torch.from_numpy(seq_idx_arr.astype(np.int64))

        targets_sorted_asc = torch.from_numpy(targets_sorted_asc)

        return ((packed_inputs, seq_idx_arr), targets_sorted_asc)