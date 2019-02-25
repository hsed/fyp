import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict

from models import BaseModel, LinearBlock

from torch.nn.utils.rnn import pack_sequence

class BaselineHARModel(BaseModel):
    '''
        A sequence is passed which consists of F frames.\n
        `in_frame_dim` => Size of a single frame's dim\n
        `out_dim` => Number of output (softmaxed) classes
    '''
    def __init__(self, in_frame_dim=63, out_dim=45, num_lstm_units_per_layer=100, 
                       num_hidden_layers=1, lstm_dropout_prob=0.3):
        
        ## init basemodel
        super(BaselineHARModel, self).__init__()


        self.main_layers = nn.Sequential(OrderedDict([
            ('lstm', nn.LSTM(input_size=in_frame_dim, hidden_size=100,
                    num_layers=num_hidden_layers, dropout=lstm_dropout_prob,
                    batch_first=True))
        ]))

        self.output_layers = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(in_features=num_lstm_units_per_layer, out_features=out_dim)),
            ('softmax', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x, seq_idx_arr):
        #x = self.main_layers(x)
        outputs, (hn, cn) = self.main_layers(x)
        #### NEED TO RESHAPE HERE!
        # outputs, (hn, cn) = lstm(inputs, h0)
        # print(hn[-1])
        # (seq_lengths-1).view(1, -1, 1) => gather indices of outputs that we desire
        # these shuld be the last of each sequence i.e. the point beyond which the batch size
        # drops because one or more sequences have ended 
        # final_outputs[0] -> outputs at each timestep
        # final_outputs[1] -> batch_size at each timestep

        # collect (max_batch_size) output vectors each with dim 100
        # thus shape is max_batch_size x 100
        # Note: Samples are sorted w.r.t ascending order of sequence len
        # This is reverse of how its done by pack_seq
        # need .long() as all params are float by default
        final_output_batch = outputs[0].index_select(0, seq_idx_arr.long())

        #print("Final_Output.shape: ", final_output_batch.shape)
        # https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/4
        x = self.output_layers(final_output_batch)
        return x
