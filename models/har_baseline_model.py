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


        # self.main_layers = nn.Sequential(OrderedDict([
        #     ('lstm', nn.LSTM(input_size=in_frame_dim, hidden_size=100,
        #             num_layers=num_hidden_layers, dropout=lstm_dropout_prob,
        #             batch_first=True))
        # ]))
        self.main_layers = nn.LSTM(input_size=in_frame_dim, hidden_size=100,
                                   num_layers=num_hidden_layers, dropout=lstm_dropout_prob,
                                   batch_first=True)

        self.output_layers = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(in_features=num_lstm_units_per_layer, out_features=out_dim)),
            ('softmax', nn.LogSoftmax(dim=1))
        ]))

        self.use_unrolled_lstm = use_unrolled_lstm

    def forward(self, x):
        # now only tuples are supplied....
        x, seq_idx_arr = x
        #x = self.main_layers(x)
        

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

        # for padded sequence we use gather
        # it basically selects for each elem in seq array
        # that idx for the dim=1 i.e for each matrix select that row
        # unsqueeze is used to match dim and then expand is used to match size
        # result of gather is (B,1,100) with each of B samples consisting of 1x100 row
        # as final output, this is squeezed to (B,100)

        if not self.use_unrolled_lstm:
            #https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/4
            #https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
            outputs, (hn, cn) = self.main_layers(x)
            #_, (hn, cn) = self.main_layers(x)
            final_output_batch = hn.squeeze(0).flip(dims=(0,))
            # aaa = self.forward_unrolled((x,seq_idx_arr))
            # padded_outputs, vlens = pad_packed_sequence(outputs, batch_first=True)
            # #vlens.device = padded_outputs.device
            # #print("padded_out_device:", padded_outputs.device, vlens.device)
            # final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
            #                                                        .unsqueeze(1).unsqueeze(2)\
            #                                                        .expand(-1,-1,padded_outputs.shape[2]))\
            #                                                        .squeeze(1).flip(dims=(0,))

            # final_output_batch = outputs[0].index_select(0, seq_idx_arr.long()) if isinstance(outputs, tuple) \
            #                         else outputs.gather(1, seq_idx_arr.long()\
            #                                     .unsqueeze(1).unsqueeze(2)\
            #                                     .expand(-1,-1,outputs.shape[2])).squeeze(1)
                                    # else outputs[:,-1,:]
            # if not torch.allclose(aaa, final_output_batch, atol=1e-3):
            #     print("seq_idx_arr", seq_idx_arr)
            #     print("batch_sizes:", x.batch_sizes)
            #     print('outs\n', final_output_batch[:, :10])
            #     print('h_ns\n', aaa[:, :10])
            #     print('max_diff\n', torch.max(final_output_batch - aaa))
            #     assert torch.allclose(aaa, final_output_batch)==True
        else:
            final_output_batch = self.forward_unrolled((x,seq_idx_arr))
            
            # _, (hn, cn) = self.main_layers(x)
            # hn_ = hn.squeeze(0).flip(dims=(0,))

            # if not torch.allclose(hn_, final_output_batch, atol=1e-3): # 1e-10 (till 10 epochs) # 1e-4 (only till 30 epochs) # 1e-3
            #     print("seq_idx_arr", seq_idx_arr)
            #     print("batch_sizes:", x.batch_sizes)
            #     print('outs\n', final_output_batch[:, :10])
            #     print('h_ns\n', hn_[:, :10])
            #     print('max_diff\n', torch.max(final_output_batch - hn_))
            #     assert torch.allclose(hn_, final_output_batch)==True

        #print("Final_Output.shape: ", final_output_batch.shape)
        # https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/4

        #out2 = self.forward_unrolled((x,seq_idx_arr))
        x = self.output_layers(final_output_batch)
        return x

    def forward_unrolled(self, x):
        x, seq_idx_arr = x
        #output_deque = deque((), maxlen=x.batch_sizes[0].item()) # create empty deque object
        # need to do this if we are using unroll method
        # basically it signals lstm cell to only run for a single timestep
        # but can't do this here need to do it seperately on case by case basis
        # x.data = x.data.unsqueeze(1)
        
        ## assume x is packed sequence, only packed sequence is supported.
        #assert isinstance(x, torch.nn.utils.rnn.PackedSequence)
        
        padded_x, vlens = pad_packed_sequence(x, batch_first=True)
        outputs = []
        # #vlens.device = padded_outputs.device
        # #print("padded_out_device:", padded_outputs.device, vlens.device)
        # final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
        #                                                        .unsqueeze(1).unsqueeze(2)\
        #                                                        .expand(-1,-1,padded_outputs.shape[2]))\
        #                                                        .squeeze(1).flip(dims=(0,))

        # final_output_batch = outputs[0].index_select(0, seq_idx_arr.long()) if isinstance(outputs, tuple) \
        #                         else outputs.gather(1, seq_idx_arr.long()\
        #                                     .unsqueeze(1).unsqueeze(2)\
        #                                     .expand(-1,-1,outputs.shape[2])).squeeze(1)

        #lstm = self.main_layers[0] # need to access lstm object directly, cant use nn.seq


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
        '''
        ## perform initial pass, for t=0, to get a h_n, c_n vector
        # this automatically creates h_0 = 0, c_0 = 0, this way we don't have to present h_0, c_0 manually
        # (num_layers * num_directions, batch, hidden_size)
        #h_0 = torch.zeros(self.num_lstm_layers*1, x.batch_sizes[0], self.lstm_layer_dim, device=x.data.device, dtype=x.data.dtype)
        #c_0 = h_0.clone()
        _, (h_n,c_n) = lstm(x.data[0:x.batch_sizes[0]].unsqueeze(1))
        samples_done = x.batch_sizes[0].item() # necessary otherwise batch size 0 is affected!
        last_batch_size = x.batch_sizes[0].item()

        # additional timesteps
        for batch_size in x.batch_sizes[1:]:
            ## do the loop here...
            # need to take care when list is getting smaller i.e. we are running out of samples at this point we should do a push
            # note: 1st dim of h_n must always be =1 in order for this to work!
            if batch_size < last_batch_size:
                #print("Last_batch_size: ", last_batch_size, "New Batch Size: ", batch_size)
                #print("OutListLen: ", len(output_deque))
                [output_deque.append(h) for h in h_n[-1, batch_size:].flip(dims=(0,))]
                #print("NewOutListLen: ", len(output_deque))

            _, (h_n,c_n) = lstm(x.data[samples_done:samples_done+batch_size].unsqueeze(1),(h_n[-1:,:batch_size], c_n[-1:, :batch_size]))
            samples_done += batch_size
            last_batch_size = batch_size
        
        # special case only one timestep used or all timesteps were equal
        if len(output_deque) == 0:
            output = h_n.squeeze(0).flip(dims=(0,))
        else:
            # note last item(s) must be added manually! Add all last items as no more propagation will be done!
            [output_deque.append(h) for h in h_n[0, :].flip(dims=(0,))]
            output = torch.stack(list(output_deque)) # device=output_deque[0].device, dtype=output_deque[0].dtype)
        ## return final h_n as the desired output vector, both h_n and y_n (ignored here)
        ## have exact same content in the end but different view so
        ## also different order!, for y original we have smallest to largest order but for
        ## y: batch_size x timesteps x hidden dim
        ## h: timesteps x batch_size x hidden dim
        return output
        '''
