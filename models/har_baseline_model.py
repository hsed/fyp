import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict

from models import BaseModel

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class LinearBlock(nn.Module):
    '''
        Linear Block Consisting of:
        1) nn.Linear
        2) nn.ReLU (optional)
        3) nn.Dropout (optional)
    '''
    ## BN + ReLU # alpha = momentum?? TODO: check
    ## this one causes problem in back prop
    def __init__(self, in_features, out_features, apply_relu=True, dropout_prob=0, use_bias=True):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias=use_bias),
        )
        if apply_relu:
            self.block.add_module("relu_1", nn.ReLU(True))
        if (dropout_prob > 0.0 and dropout_prob < 1.0):
            self.block.add_module("dropout_1", nn.Dropout(dropout_prob))
            ## Do not set inplace in nn.Dropout to true! it must happen as a 'layer' otherwise
            ## backprop is not possible!
            ## took so many hours debuggung this
    
    def forward(self, x):
        return self.block(x)




class BaselineHARModel(BaseModel):
    '''
        A sequence is passed which consists of F frames.\n
        `in_frame_dim` => Size of a single frame's dim\n
        `out_dim` => Number of output (softmaxed) classes
    '''
    def __init__(self, in_frame_dim=63, out_dim=45, num_lstm_units_per_layer=100, 
                       num_hidden_layers=1, lstm_dropout_prob=0.3,
                       use_unrolled_lstm=False):
        
        ## init basemodel
        super(BaselineHARModel, self).__init__()
        self.num_lstm_layers = num_hidden_layers
        self.lstm_layer_dim = num_lstm_units_per_layer

        self.main_layers = nn.LSTM(input_size=in_frame_dim, hidden_size=100,
                                   num_layers=num_hidden_layers, dropout=lstm_dropout_prob,
                                   batch_first=True)

        self.output_layers = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(in_features=num_lstm_units_per_layer, out_features=out_dim)),
            ('softmax', nn.LogSoftmax(dim=1))
        ]))

        self.use_unrolled_lstm = use_unrolled_lstm

    def forward(self, x):
        # x must be either of type PackedSequence or of type tuple
        # note we can't test tuple directly because PackedSequence extends tuple!
        x, vlens = (x, None) if isinstance(x, torch.nn.utils.rnn.PackedSequence) else (x[0], x[1].long())

        if not self.use_unrolled_lstm:
            # outputs will be padded if x will be padded --- in this case vlens is not none
            outputs, (hn, cn) = self.main_layers(x)
            
            final_output_batch = hn.squeeze(0).flip(dims=(0,)) if vlens is None else \
                                 outputs.gather(1, (vlens-1).to(outputs.device)\
                                                            .unsqueeze(1).unsqueeze(2)\
                                                            .expand(-1,-1,outputs.shape[2]))\
                                                            .squeeze(1).flip(dims=(0,))
            # aaa = self.forward_unrolled((x,seq_idx_arr))
            # padded_outputs, vlens = pad_packed_sequence(outputs, batch_first=True)
            # #vlens.device = padded_outputs.device
            # #print("padded_out_device:", padded_outputs.device, vlens.device)
            # #https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/4
            # #https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
            # final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
            #                                                        .unsqueeze(1).unsqueeze(2)\
            #                                                        .expand(-1,-1,padded_outputs.shape[2]))\
            #                                                        .squeeze(1).flip(dims=(0,))

            # if not torch.allclose(aaa, final_output_batch, atol=1e-3):
            #     print("seq_idx_arr", seq_idx_arr)
            #     print("batch_sizes:", x.batch_sizes)
            #     print('outs\n', final_output_batch[:, :10])
            #     print('h_ns\n', aaa[:, :10])
            #     print('max_diff\n', torch.max(final_output_batch - aaa))
            #     assert torch.allclose(aaa, final_output_batch)==True
        else:
            final_output_batch = self.forward_unrolled(x, vlens)


        x = self.output_layers(final_output_batch)
        return x

    def forward_unrolled(self, x, vlens=None):
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        padded_x, vlens = (x, vlens) if vlens is not None else pad_packed_sequence(x, batch_first=True) 
        outputs = []


        h_n = torch.zeros(self.num_lstm_layers*1, padded_x.shape[0], self.lstm_layer_dim,
                          device=padded_x.device, dtype=padded_x.dtype)
        c_n = h_n.clone()

        for i in range(padded_x.shape[1]):
            y, (h_n, c_n) = self.main_layers(padded_x[:,i,:].unsqueeze(1), (h_n, c_n))
            outputs.append(y.squeeze(1))

        padded_outputs = torch.stack(outputs, dim=1)
        
        final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
                                                               .unsqueeze(1).unsqueeze(2)\
                                                               .expand(-1,-1,padded_outputs.shape[2]))\
                                                               .squeeze(1).flip(dims=(0,))
        return final_output_batch