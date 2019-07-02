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

        # sort_idx_asc = sorted(range(len(batch)), key=lambda item_idx: batch[item_idx][0].shape[0])        
        # sort_idx_asc = np.array(sort_idx_asc)
        # sorting in reverse is the correct way rather than sort in ascending and then flipping! 
        # sometimes when items have same value flipping vs reverse sorting differs!
        # we will now ensure all targets are descending sorted too and no flipping is performed during forward call
        sort_idx_desc = sorted(range(len(batch)), key=lambda item_idx: batch[item_idx][0].shape[0], reverse=True) #np.flipud(sort_idx_asc)

        #print("Sorts::ASC::DESC\n", sort_idx_asc, sort_idx_desc)

        #targets_sorted_asc = np.array([item[1] for item in batch])[ sort_idx_desc ]
        targets_sorted_asc = np.array([item[1] for item in batch])[ sort_idx_desc ] #sort_idx_asc

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



class CollateCustomSeqBatch(object):
    def __init__(self, pad_sequence=False, max_pad_length=-1, inputs=1, outputs=1,
                 action_in=False, action_out=True):
        '''
            all samples must be tuple type!!
            (in1, in2, action, out1, out2, action)
            (in1, out1, action)
            (in1, action, out1)
            (action, action)
            (in1, out1)
        '''
        self.pad_sequence = pad_sequence
        self.max_pad_length = max_pad_length # -1 or 100
        self.action_in = action_in
        self.action_out = action_out

        self.inputs_seq = inputs - self.action_in
        self.outputs_seq = outputs - self.action_out

        self.is_seq = tuple(1 for _ in range(self.inputs_seq))
        if self.action_in: self.is_seq += (0,)
        
        self.is_seq += tuple(1 for _ in range(self.outputs_seq))
        if self.action_out: self.is_seq += (0,)
        ## what to do with action?
        # used for padding only, need to also do this right in dataloader by using seqlimiter transformer appropriately
        self.total_pad_length = None if self.max_pad_length < 0 else self.max_pad_length
        
        print("[COLLATEFN] EXPECTED SEQ:", self.is_seq)

    
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

        ## assume length is given by 0th dim of first input

        # tuple (input(s)_tuple, output(s)_tuple)
        # we must always assume its like this
        # input(s)_tuple: (in1, in2, ..., action) action can be present or absent
        # output(s)_tuple: (out1, out2, ..., action) action can be present or absent
        # if inputs_tuple is 1 or 

        #first_item = batch[0]
        batches_tuple = zip(*batch) # unwrap the batch

        
        #assert len(batches_tuple) == len(self.is_seq)
        #if inputs_batch

        # usually first item is of sequence type!
        first_seq_batch_idx = int(np.where(np.array(self.is_seq)!=0)[0][0])
        sort_idx_desc = np.array(sorted(range(len(batch)), key=lambda item_idx: batch[item_idx][first_seq_batch_idx].shape[0], reverse=True)) #np.flipud(sort_idx_asc)

        # NOTE: this is a non_numeric array of arrays!
        # Need numpy type due to sorting

        # temporary conversion to np.array of tensors for sorting
        # here we have a batch of n-dim tuple for every dim it iterates thoughout batch and creates a list
        # its like unzipping and then
        # action_idx is appended in ASCENDING ORDER 
        # ALL OTHER ITEMS ARE SORTED BY DESCENDING ORDER
        sorted_arr = []
        for is_this_seq, batch in zip(self.is_seq, batches_tuple):
          if is_this_seq == 0:
            ## this is not a variable sequence type
            sorted_arr.append(torch.from_numpy(np.array(batch)[ sort_idx_desc ].astype(np.int64)))
          else:
            seq = list(np.array([torch.from_numpy(item) for item in batch], dtype='O')[ sort_idx_desc ])
            
            if self.pad_sequence:
              sorted_arr.append(pad_packed_sequence(pack_sequence(seq), batch_first=True, total_length=self.total_pad_length))
            else:
              sorted_arr.append(pack_sequence(seq))


        # if input/output singleton remove tuple, otherwise present as tuple
        inputs = tuple(sorted_arr[:(self.inputs_seq+self.action_in)]) if (self.inputs_seq+self.action_in) > 1 else sorted_arr[0]
        outputs = tuple(sorted_arr[(self.inputs_seq+self.action_in):]) if (self.outputs_seq+self.action_out) > 1 else sorted_arr[-1]

        #print("COLLATE FN:")
        #print("ins/ outs:", inputs[0].data.shape, outputs[0].data.shape)
        # possibilities: (tuple, tuple) ; (tensor, tuple) ; (tuple, tensor) ; (tensor, tensor)
        return (inputs, outputs)