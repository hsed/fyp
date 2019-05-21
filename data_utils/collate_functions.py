import numpy as np
import torch
import h5py

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch._six import int_classes

class CollateJointsSeqBatch(object):
    def __init__(self, pad_sequence=False, max_pad_length=-1):
        self.pad_sequence = pad_sequence
        self.max_pad_length = max_pad_length # -1 or 100

    
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
        ## this function is WRONG!! doesnt work with same sizes e.g. [3,3,5,5,7]

        sort_idx_asc = sorted(range(len(batch)), key=lambda item_idx: batch[item_idx][0].shape[0])
        
        sort_idx_asc = np.array(sort_idx_asc)

        sort_idx_desc = np.flipud(sort_idx_asc)

        #print("Sorts::ASC::DESC\n", sort_idx_asc, sort_idx_desc)

        #targets_sorted_asc = np.array([item[1] for item in batch])[ sort_idx_desc ]
        targets_sorted_asc = np.array([item[1] for item in batch])[ sort_idx_asc ]

        # NOTE: this is a non_numeric array of arrays!
        # Need numpy type due to sorting

        # temporary conversion to np.array of tensors for sorting
        inputs_sorted_desc = list(np.array([torch.from_numpy(item[0]) for item in batch], dtype='O')[ sort_idx_desc ])

        #seq_lengths_asc = np.flipud([s.shape[0] for s in inputs_sorted_desc])

        #seq_idx_arr = np.cumsum(np.append(seq_lengths_asc[0],np.diff(seq_lengths_asc))*np.arange(seq_lengths_asc.shape[0], 0, -1))-1
        
        # interesting point is that packed_seq support to so we can later convert to correct dtypes
        # if required e.g. double -> float
        packed_inputs = pack_sequence(list(inputs_sorted_desc))

        targets_sorted_asc = torch.from_numpy(targets_sorted_asc.astype(np.int64))#required to be long only in windows?

        if self.pad_sequence:
          'also pad the data'
          #print("padding now")
          total_length = None if (self.max_pad_length == -1 or inputs_sorted_desc[0].shape[0] < self.max_pad_length) \
                         else self.max_pad_length
          # sequence idx array is now provided by this but need to do -1
          padded_inputs, batch_sizes = pad_packed_sequence(packed_inputs, batch_first=True, total_length=total_length)
          #print("padded_shape: ", padded_inputs.shape)
          return ((padded_inputs, batch_sizes), targets_sorted_asc)

        else:
          return (packed_inputs, targets_sorted_asc)



class CollateDepthJointsBatch(object):
    def __init__(self, inputs = 1, outputs = 1):
        self.inputs = inputs
        self.outputs = outputs

        #print("[COLLATE_FN] INPUTS, OUTPUTS: ", self.input_indices, self.output_indices)
        pass

    
    def __call__(self, batch):
        '''
            Batch Format:
            - The format is a list of exactly what is returned by the last transformer of
              the dataset
            - In our case that's a tuple with first elem as variable sized joints_seq
              and a single idx for action
            - joints_seq is already in numpy.ndarray format
        '''

        #print("Type_of_batch: ", type(batch))
        #print("Type_of_batch[0]: ", type(batch[0]))
        #print("Type_of_batch[0][0]: ", type(batch[0][0]))

        # print("Len_of_batch: ", len(batch))
        # print("Len_of_batch[0]: ", len(batch[0]))
        # print("Len_of_batch[0][0]: ", len(batch[0][0]))

        # batch[i] --> ith (== item_idx) (input,output) tuple
        # batch[i][0] --> ith input
        # batch[i][1] --> ith output
        def collate_h5py(samples):
          if isinstance(samples[0], h5py._hl.dataset.Dataset):
            return torch.stack([torch.from_numpy(s.value) for s in samples], 0)
          elif isinstance(samples[0], np.ndarray):
            return torch.stack([torch.from_numpy(s) for s in samples], 0)
          elif isinstance(samples[0], int):
            return torch.tensor(samples)
          else:
            raise NotImplementedError("Type %s to torch conversion is unimplemented." % type(samples[0]))
        
        
        # required to be long
        transposed = zip(*batch)
        data = tuple(collate_h5py(samples) for samples in transposed)

        # new for action loading
        
        return (data[0] if self.inputs is 1 else data[:self.inputs]), (data[-1] if self.outputs is 1 else data[self.inputs:self.inputs+self.outputs])