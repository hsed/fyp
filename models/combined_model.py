import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict, deque

from models import BaseModel, DeepPriorPPModel, BaselineHARModel

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence

class CombinedModel(BaseModel):
    def __init__(self, hpe_checkpoint=None, har_checkpoint=None, pca_checkpoint=None,
                 hpe_args = None, har_args = None, forward_type=0, combined_version='0',
                 force_trainable_params=False):
        
        #???self.train_mode = 'u_v_w'????
        super(CombinedModel, self).__init__()
        
        # Note: Args are overridden if loading from cache
        hpe_args = {} if hpe_args is None else hpe_args
        har_args = {} if har_args is None else har_args

        if har_checkpoint is None:
            self.har = BaselineHARModel(**har_args)
        else:
            print("LOADING HAR CHECKPOINT")
            checkpoint = torch.load(har_checkpoint)
            self.har = BaselineHARModel(**checkpoint['config']['arch']['args'])
            self.har.load_state_dict(checkpoint['state_dict'])
        

        ### for some stupid reason pytorch bug the model with no gradients should be initialised in the end
        if hpe_checkpoint is None:
            #pass
            self.hpe = DeepPriorPPModel(**hpe_args)
            # for param in self.hpe.parameters(): # this is now done later...
            #     param.requires_grad = False
                # for module, param in self.hpe.parameters():
                #     self.register_buffer(param.name, param.data)
        else: 
            # note optimizer states are lost ... we can't really do much there
            print("LOADING HPE CHECKPOINT")
            checkpoint = torch.load(hpe_checkpoint)
            self.hpe = DeepPriorPPModel(**checkpoint['config']['arch']['args'])
            self.hpe.load_state_dict(checkpoint['state_dict'])
        

        
        #print(list(self.modules()))
        #quit()

        ## disable hpe params or har params
        # for param in self.hpe.parameters():
        #     param.requires_grad = False

        

        self.forward_type = int(forward_type)
        self.combined_version = str(combined_version)
        self._assert_attribs()
        if not force_trainable_params: self._set_trainable_params() # turn params on or off based on forward type and versions

        self.combined_forward_fn_dict  = {
            '0': self.forward_v0,
        }
    
        
    
    def forward(self, x):
        if self.forward_type == 0:
            return self.combined_forward_fn_dict[self.combined_version](x)
        
        elif self.forward_type == 1:
            raise NotImplementedError
            return 0
        elif self.forward_type == 2:
            raise NotImplementedError
            return 0
        elif self.forward_type == 3:
            return self.forward_keypts_action(x, self.har.recurrent_layers)
    
    def _assert_attribs(self):
        ## some assertions -- these attributes must be present in respective blocks
        assert getattr(self.har, 'recurrent_layers', False)
        assert getattr(self.har, 'action_layers', False)
        assert getattr(self.har, 'num_recurrent_layers', False)
        assert getattr(self.har, 'recurrent_layers_dim', False)
    
    def _set_trainable_params(self):
        if self.forward_type == 0:
            if self.combined_version == '0':
                print("[COMBINED_MODEL] Setting all params to non-trainable")
                for param in self.har.parameters():
                    param.requires_grad = False
                for param in self.hpe.parameters():
                    param.requires_grad = False
        
        elif self.forward_type == 1:
            raise NotImplementedError
        elif self.forward_type == 2:
            # Depth -> Action training HAR+HPE
            pass
        elif self.forward_type == 3:
            # KeyPts -> Joints training HAR
            print("[COMBINED_MODEL] Setting hpe params to non-trainable")
            for param in self.hpe.parameters():
                param.requires_grad = False

    
    def pad_packed_inputs(self, x):
        '''
            perform padding on both x and y as required
        '''
        pass
    

    def extract_outputs(self, outputs_padded: torch.tensor, batch_sizes: torch.tensor):
        '''
            return the correct relevant outputs
        '''
        
        return outputs_padded.gather(1, (batch_sizes-1).to(outputs_padded.device)\
                                                 .unsqueeze(1).unsqueeze(2)\
                                                 .expand(-1,-1,outputs_padded.shape[2]))\
                                                 .squeeze(1)
    

    def get_h0_c0(self, x_padded):
        h0 = torch.zeros(self.har.num_recurrent_layers*1, x_padded.shape[0],
                          self.har.recurrent_layers_dim,
                          device=x_padded.device, dtype=x_padded.dtype)
        c0 = h0.clone()

        return (h0, c0)
    


    def forward_cnn(self, x_padded, extras=None):
        '''
            use of padded x for forward pass through cnn
        '''
        pass

    def forward_rnn_unrolled(self, x_padded, lstm_block):
        '''
            assume rnn/lstm is uni-directional
        '''
        pass


    ### misc forwards only for testing
    def forward_keypts_action(self, x, lstm_block):
        '''
            x -> keypts (N,63) or pca of keypts (N, 30)
            lstm_block -> must be lstm not gru or anything else!
        '''
        # return self.har(x) #for debugging only
        # print("x_type: ", type(x))
        # quit()
        x_padded, batch_sizes = pad_packed_sequence(x, batch_first=True) if isinstance(x, PackedSequence) else (x[0], x[1].long())
        (h_n, c_n) = self.get_h0_c0(x_padded)
        outputs = []

        for i in range(x_padded.shape[1]):
            y, (h_n, c_n) = lstm_block(x_padded[:,i,:].unsqueeze(1), (h_n, c_n))
            outputs.append(y.squeeze(1))
        
        outputs_padded = torch.stack(outputs, dim=1)

        x_interim = self.extract_outputs(outputs_padded, batch_sizes)

        return self.har.action_layers(x_interim)
    
    def forward_depths_action(self, x):
        pass


    def forward_v0(self, x):
        '''
            only to be used during eval mode

            this is the trivial baseline in notes
        '''
        #print(type(x), len(x), type(x[0]), type(x[1]))
        #print(type(x), len(x))
        #print(type(x[0]), type(x[1]))
        # assert eval mode?
        ## we detach gradients here
        #with torch.no_grad():
        # perform computations here
        #on cpu need to perform some kind of batch splitting..?
        #print(x.data.shape)
        y = self.hpe(x.data.unsqueeze(1))
        #print("y_Shape", y.shape)

        y = PackedSequence(data=y, batch_sizes=x.batch_sizes)
        #print(y)
        # return both predictions of keypts and action class prediction
        return (y, self.har(y))
    

    def forward_v1(self, x):
        '''
            add a single step previous frame as action condition to hpe or something?
        '''