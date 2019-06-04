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
                       use_unrolled_lstm=False,
                       attention_type='disabled'):
        
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

        # api for compatibility
        self.recurrent_layers = self.main_layers
        self.num_recurrent_layers = self.num_lstm_layers
        self.recurrent_layers_dim = self.lstm_layer_dim
        self.action_layers = self.output_layers

        self.use_unrolled_lstm = use_unrolled_lstm

        if attention_type == 'disabled':
            pass # don't add any new module, backwards compatibility
        elif attention_type == 'cnn_v1':
            print("[HAR] Using CNN Attention with k=1 along temporal axis")
            self.attention_layer = nn.Conv2d(1,1,kernel_size=(1,in_frame_dim+100))
            self.attention_forward = self.forward_unrolled_attention_cnn
        elif attention_type[:8] == 'cnn_v1_k' or attention_type[:8] == 'cnn_v2_k':
            '''
                add padding to support new kernal size
                so k_size = 3 -> p = 1

                ENSURE BOTH VALUES ARE ODD NUMBERS
                att2 = nn.Conv2d(1,1,kernel_size=(3,163), padding=(1,0))

                Formula: p = round_down(k_size/2)

                Note: Only add padding along height, no need for width padding!

                Its divided by two because p=1 implies padding on either side so 2 rows
                of padding
            '''
            kernel_sz = int(attention_type[8:10])
            assert (kernel_sz % 2 == 1), "KernelSz must be an odd number!"
            print("[HAR] Using CNN Attention with k=%s along temporal axis" % kernel_sz)
            padding = int(kernel_sz/2) # same effect as floor
            self.attention_layer = nn.Conv2d(1,1,kernel_size=(kernel_sz,in_frame_dim+100),
                                            padding=(padding, 0))
            if int(attention_type[5]) == 1:
                self.attention_forward = self.forward_unrolled_attention_cnn
            elif int(attention_type[5]) == 2:
                self.attention_forward = self.forward_unrolled_attention_cnn_v2
        elif attention_type[:8] == 'v3':
            self.attention_layer = nn.Sequential(
                nn.RNNCell(120*63,100),
                nn.Linear(100,120)
            )
            self.attention_forward = self.forward_unrolled_attention_v3
        elif attention_type[:8] == 'v4':
            self.attention_layer = nn.Linear(120*63,120)
            self.attention_forward = self.forward_unrolled_attention_v4
        elif attention_type[:8] == 'v5':
            #nn.Linear(120*63,120)
            self.attention_layer = nn.Sequential(
                nn.Linear(in_frame_dim+num_lstm_units_per_layer, 1, bias=True), # Bx163 -> 1 (for every timestep)
                nn.ReLU(), # try sigmoid as well...?
            )
            self.attention_forward = self.forward_unrolled_action_attention
        elif attention_type[:8] == 'v6':
            #nn.Linear(120*63,120)
            self.attention_layer = nn.Sequential(
                nn.Linear(in_frame_dim+num_lstm_units_per_layer, 1, bias=True), # Bx163 -> 1 (for every timestep)
                nn.Sigmoid(), # try sigmoid as well...?
            )
            self.attention_forward = self.forward_unrolled_action_attention

    def forward(self, x, return_temporal_probs=False):
        # x must be either of type PackedSequence or of type tuple
        # note we can't test tuple directly because PackedSequence extends tuple!
        x, vlens = (x, None) if isinstance(x, torch.nn.utils.rnn.PackedSequence) else (x[0], x[1].long())
        state_hist = None

        if not getattr(self, 'attention_layer', False):
            if not self.use_unrolled_lstm:
                # outputs will be padded if x will be padded --- in this case vlens is not none
                #print(type(x), x.shape)
                #print("oshape", x.shape)
                outputs, (hn, cn) = self.main_layers(x)
                
                # https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/4
                # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
                # .flip(dims=(0,))
                final_output_batch = hn.squeeze(0) if vlens is None else \
                                    outputs.gather(1, (vlens-1).to(outputs.device)\
                                                                .unsqueeze(1).unsqueeze(2)\
                                                                .expand(-1,-1,outputs.shape[2]))\
                                                                .squeeze(1)#.flip(dims=(0,))
                
                # for debugging -- comment when not needed
                # aaa = self.forward_unrolled(x,vlens)
                # if not torch.allclose(aaa, final_output_batch, atol=1e-3): # 1e-4 bad, 1e-5 bad
                #     print("batch_sizes:", x.batch_sizes)
                #     print('outs\n', final_output_batch[:, :10])
                #     print('h_ns\n', aaa[:, :10])
                #     print('max_diff\n', torch.max(final_output_batch - aaa))
                #     assert torch.allclose(aaa, final_output_batch)==True
                if return_temporal_probs:
                    # only works if batch_size=1 AND packedSeq is used AND no unroll lstm
                    # if batch_size is more fix it for urself!
                    state_hist = outputs.data
            else:
                final_output_batch = self.forward_unrolled(x, vlens)
        else:
            assert self.use_unrolled_lstm, "Must use unrolled LSTM for attention!"
            final_output_batch = self.attention_forward(x, vlens)

        # if last dim == 45 then action is probably calc so use that directly don't calc action again!
        x = self.output_layers(final_output_batch) if not final_output_batch.shape[-1] == 45 else final_output_batch
        x = x if state_hist is None else (x, self.output_layers(state_hist))
        return x

    def forward_unrolled_action_attention(self, y, vlens=None):
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        padded_y, vlens = (y, vlens) if vlens is not None else pad_packed_sequence(y, batch_first=True) 
        h_list = [] # list of hidden vectors
        betas_list = []
        action_lin_list = []

        h_n = torch.zeros(self.num_lstm_layers*1, padded_y.shape[0], self.lstm_layer_dim,
                          device=padded_y.device, dtype=padded_y.dtype)
        c_n = h_n.clone()

        for i in range(padded_y.shape[1]):
            h_i, (h_n, c_n) = self.main_layers(padded_y[:,i,:].unsqueeze(1), (h_n, c_n))
            h_list.append(h_i.squeeze(1))
            beta_i = self.attention_layer(torch.cat([padded_y[:,i,:], h_i.squeeze(1)], dim=1))
            betas_list.append(beta_i)
            
            action_lin_i = self.output_layers[0](h_i.squeeze(1)) # get linear output without softmax operation
            action_lin_list.append(action_lin_i)

        padded_h = torch.stack(h_list, dim=1) # B x T x 100
        padded_betas = torch.stack(betas_list, dim=2) # B x 1 x T
        padded_action_lin = torch.stack(action_lin_list, dim=1) # B x T x 45

        # B x 1 x T -> B x 1 x T (softmaxed)
        #padded_alphas = torch.log(padded_betas, dim=2)
        padded_alphas = F.softmax(padded_betas, dim=2)

        # (B x 1 x T) * (B x T x 45) -> (B x 1 x 45) -> (B x 45)
        action_lin_focused = torch.bmm(padded_alphas, padded_action_lin).squeeze(1)

        final_action = self.output_layers[1](action_lin_focused)
        
        return final_action

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
                                                               .squeeze(1)#.flip(dims=(0,))
        return final_output_batch
    

    def forward_unrolled_attention_cnn(self, x, vlens=None):
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        # x -> keypts
        # y -> hidden state
        # z -> action
        padded_x, vlens = (x, vlens) if vlens is not None else pad_packed_sequence(x, batch_first=True) 
        outputs = []
        x_cat_y = []
        x_list = []


        h_n = torch.zeros(self.num_lstm_layers*1, padded_x.shape[0], self.lstm_layer_dim,
                          device=padded_x.device, dtype=padded_x.dtype)
        c_n = h_n.clone()

        ## note h_n == y but just a different shape

        for i in range(padded_x.shape[1]):
            x_curr = padded_x[:,i,:] # NOTE THESE ARE KEYPOINTS
            #y, (h_n, c_n) = self.main_layers(padded_x[:,i,:].unsqueeze(1), (h_n, c_n))
            #outputs.append(y.squeeze(1))
            x_list.append(x_curr)

            # cat [B x 63] with [B x 100] (need to squeeze as h is originally [1 x B x 100])
            x_cat_y.append(torch.cat([x_curr, h_n.squeeze(0)], dim=1))

            # B x 1 x T x 1 -> B x 1 x T
            betas_reduced = self.attention_layer(torch.stack(x_cat_y, dim=1).unsqueeze(1)).squeeze(3)

            # B x 1 x T -> B x 1 x T
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            # bmm with current history of y's
            # [B x 1 x T] DOT [B x T x 63] -> [B x 1 x 63]
            x_focused = torch.bmm(alphas_reduced, torch.stack(x_list, dim=1))

            # NOTE y is hidden state not keypoints!
            y, (h_n, c_n) = self.main_layers(x_focused, (h_n, c_n))
            outputs.append(y.squeeze(1))

        padded_outputs = torch.stack(outputs, dim=1)
        
        final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
                                                               .unsqueeze(1).unsqueeze(2)\
                                                               .expand(-1,-1,padded_outputs.shape[2]))\
                                                               .squeeze(1)#.flip(dims=(0,))
        return final_output_batch


    def forward_unrolled_attention_cnn_v2(self, x, vlens=None):
        # v2 model
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        # x -> keypts
        # y -> hidden state
        # z -> action
        padded_x, vlens = (x, vlens) if vlens is not None else pad_packed_sequence(x, batch_first=True) 
        outputs = []
        x_cat_y = []
        x_list = []


        h_n = torch.zeros(self.num_lstm_layers*1, padded_x.shape[0], self.lstm_layer_dim,
                          device=padded_x.device, dtype=padded_x.dtype)
        c_n = h_n.clone()

        ## note h_n == y but just a different shape

        for i in range(padded_x.shape[1]):
            x_curr = padded_x[:,i,:] # NOTE THESE ARE KEYPOINTS
            #y, (h_n, c_n) = self.main_layers(padded_x[:,i,:].unsqueeze(1), (h_n, c_n))
            #outputs.append(y.squeeze(1))
            x_list.append(x_curr)

            # cat [B x T x 63] with [B x 1 x 100] using expand along dim 1 (need to transpose as h is originally [1 x B x 100])
            # keypts -> [B x T x 63]; h -> [1 x B x 100]; h_transposed -> [B x 1 x 100]; h_expanded -> [B x T x 100]
            # x_cat_y -> [B x T x 163]
            # along T axis, keypts change but h is constant
            x_interim = torch.stack(x_list, dim=1)
            x_cat_y = torch.cat([x_interim, h_n.transpose(0,1).expand(-1, x_interim.shape[1], -1)],dim=2)

            # B x 1 x T x 1 -> B x 1 x T
            betas_reduced = self.attention_layer(x_cat_y.unsqueeze(1)).squeeze(3)

            # B x 1 x T -> B x 1 x T
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            # bmm with current history of y's
            # [B x 1 x T] DOT [B x T x 63] -> [B x 1 x 63]
            x_focused = torch.bmm(alphas_reduced, torch.stack(x_list, dim=1))

            # NOTE y is hidden state not keypoints!
            y, (h_n, c_n) = self.main_layers(x_focused, (h_n, c_n))
            outputs.append(y.squeeze(1))

        padded_outputs = torch.stack(outputs, dim=1)
        
        final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
                                                               .unsqueeze(1).unsqueeze(2)\
                                                               .expand(-1,-1,padded_outputs.shape[2]))\
                                                               .squeeze(1)#.flip(dims=(0,))
        return final_output_batch



    def forward_unrolled_attention_v3(self, x, vlens=None):
        # v2 model
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        # x -> keypts
        # y -> hidden state
        # z -> action
        padded_x, vlens = (x, vlens) if vlens is not None else pad_packed_sequence(x, batch_first=True) 
        outputs = []
        #x_cat_y = []
        x_list = []
        past_timesteps = 0
        x_buffer_list = [torch.zeros(padded_x.shape[0], 63,
                            device=padded_x.device, dtype=padded_x.dtype) for _ in range(120)] # need to make this a param


        h_n = torch.zeros(self.num_lstm_layers*1, padded_x.shape[0], self.lstm_layer_dim,
                          device=padded_x.device, dtype=padded_x.dtype)
        c_n = h_n.clone()

        ## note h_n == y but just a different shape

        for i in range(padded_x.shape[1]):
            x_curr = padded_x[:,i,:] # NOTE THESE ARE KEYPOINTS
            #y, (h_n, c_n) = self.main_layers(padded_x[:,i,:].unsqueeze(1), (h_n, c_n))
            #outputs.append(y.squeeze(1))
            x_list.append(x_curr)

            past_timesteps += 1 # is there a bug when past_timesteps == 120?

            idx_sel = torch.arange(past_timesteps).to(padded_x.device, torch.long) # possible dtype issue with gpu
            x_buffer_list[i] = x_curr
            x_buffer = torch.stack(x_buffer_list, dim=1)

            # B x 120*63 -> B x 120 -> B x 1 x 120
            betas = self.attention_layer[0](x_buffer.view(x_buffer.shape[0], x_buffer.shape[1]*x_buffer.shape[2]), h_n.squeeze(0))
            betas = self.attention_layer[1](betas).unsqueeze(1)

            # B x 1 x 120 -> B x 1 x T
            betas_reduced = torch.gather(betas, 2, idx_sel.view(1,1,-1).expand(padded_x.shape[0], 1, -1))

            # B x 1 x T -> B x 1 x T (softmaxed)
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            # bmm with current history of y's
            # [B x 1 x T] DOT [B x T x 63] -> [B x 1 x 63]
            x_focused = torch.bmm(alphas_reduced, torch.stack(x_list, dim=1))

            # NOTE y is hidden state not keypoints!
            y, (h_n, c_n) = self.main_layers(x_focused, (h_n, c_n))
            outputs.append(y.squeeze(1))

        padded_outputs = torch.stack(outputs, dim=1)
        
        final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
                                                               .unsqueeze(1).unsqueeze(2)\
                                                               .expand(-1,-1,padded_outputs.shape[2]))\
                                                               .squeeze(1)#.flip(dims=(0,))
        return final_output_batch

    def forward_unrolled_attention_v4(self, x, vlens=None):
        # v2 model
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        # x -> keypts
        # y -> hidden state
        # z -> action
        padded_x, vlens = (x, vlens) if vlens is not None else pad_packed_sequence(x, batch_first=True) 
        outputs = []
        #x_cat_y = []
        x_list = []
        past_timesteps = 0
        x_buffer_list = [torch.zeros(padded_x.shape[0], 63,
                            device=padded_x.device, dtype=padded_x.dtype) for _ in range(120)] # need to make this a param


        h_n = torch.zeros(self.num_lstm_layers*1, padded_x.shape[0], self.lstm_layer_dim,
                          device=padded_x.device, dtype=padded_x.dtype)
        c_n = h_n.clone()

        ## note h_n == y but just a different shape

        for i in range(padded_x.shape[1]):
            x_curr = padded_x[:,i,:] # NOTE THESE ARE KEYPOINTS
            #y, (h_n, c_n) = self.main_layers(padded_x[:,i,:].unsqueeze(1), (h_n, c_n))
            #outputs.append(y.squeeze(1))
            x_list.append(x_curr)

            past_timesteps += 1 # is there a bug when past_timesteps == 120?

            idx_sel = torch.arange(past_timesteps).to(padded_x.device, torch.long) # possible dtype issue with gpu
            x_buffer_list[i] = x_curr
            x_buffer = torch.stack(x_buffer_list, dim=1)

            # B x 120*63 -> B x 120 -> B x 1 x 120
            betas = self.attention_layer(x_buffer.view(x_buffer.shape[0], x_buffer.shape[1]*x_buffer.shape[2])).unsqueeze(1)

            # B x 1 x 120 -> B x 1 x T
            betas_reduced = torch.gather(betas, 2, idx_sel.view(1,1,-1).expand(padded_x.shape[0], 1, -1))

            # B x 1 x T -> B x 1 x T (softmaxed)
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            # bmm with current history of y's
            # [B x 1 x T] DOT [B x T x 63] -> [B x 1 x 63]
            x_focused = torch.bmm(alphas_reduced, torch.stack(x_list, dim=1))

            # NOTE y is hidden state not keypoints!
            y, (h_n, c_n) = self.main_layers(x_focused, (h_n, c_n))
            outputs.append(y.squeeze(1))

        padded_outputs = torch.stack(outputs, dim=1)
        
        final_output_batch = padded_outputs.gather(1, (vlens-1).to(padded_outputs.device)\
                                                               .unsqueeze(1).unsqueeze(2)\
                                                               .expand(-1,-1,padded_outputs.shape[2]))\
                                                               .squeeze(1)#.flip(dims=(0,))
        return final_output_batch