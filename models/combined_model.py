import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict, deque
from copy import deepcopy
import yaml

from models import BaseModel, DeepPriorPPModel, BaselineHARModel

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence, pack_padded_sequence

class CombinedModel(BaseModel):
    def __init__(self, hpe_checkpoint=None, har_checkpoint=None, pca_checkpoint=None,
                 hpe_act_checkpoint = None,
                 hpe_args = None, har_args = None, forward_type=0, combined_version='0',
                 force_trainable_params=False, action_classes=45, act_0c_alpha=0.01,
                 joints_dim=63, max_seq_length=120,
                 ensure_batchnorm_fixed_eval=False,
                 ensure_dropout_fixed_eval=False,
                 ensure_batchnorm_dropout_fixed_eval=False):
        
        #???self.train_mode = 'u_v_w'????
        super(CombinedModel, self).__init__()
        
        # Note: Args are overridden if loading from cache
        hpe_args = {} if hpe_args is None else hpe_args
        har_args = {} if har_args is None else har_args

        if har_checkpoint is None:
            self.har = BaselineHARModel(**har_args)
        else:
            print("[COMBINED_MODEL] LOADING HAR CHECKPOINT: %s" % har_checkpoint)
            checkpoint = torch.load(har_checkpoint)
            #print("CONFIG\n", yaml.dump(checkpoint['config']['arch']['args']))
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
            print("[COMBINED_MODEL] LOADING HPE CHECKPOINT: %s" % hpe_checkpoint)
            checkpoint = torch.load(hpe_checkpoint)
            self.hpe = DeepPriorPPModel(**checkpoint['config']['arch']['args'])
            self.hpe.load_state_dict(checkpoint['state_dict'], strict=False)
        

        if hpe_act_checkpoint is not None:
            print("[COMBINED_MODEL] LOADING HPE ACT CHECKPOINT: %s" % hpe_act_checkpoint)
            checkpoint_ = torch.load(hpe_act_checkpoint)
            self.hpe_act = DeepPriorPPModel(**checkpoint_['config']['arch']['args'])
            self.hpe_act.load_state_dict(checkpoint_['state_dict'])
            # for param in self.hpe_act.parameters():
            #     param.requires_grad = False
        # note must ensure dtypes / device is correct at runtime
        # NEED LOG VERSION AS WE USE LOG SOFTMAX AND NLL LOSS!!
        self.act_zeros = torch.zeros(action_classes, device=next(self.parameters()).device)
        self.act_ones = torch.ones(action_classes, device=next(self.parameters()).device)
        self.equiprob_act = (1/action_classes) * self.act_ones.clone()
        
        self.equiprob_act_log = torch.log(self.equiprob_act.clone())
        #print(list(self.modules()))
        #quit()
        #print("[COMBINED_MODEL] Act 0c ALPHA:", act_0c_alpha)
        self.act_0c_alpha=act_0c_alpha
        ## disable hpe params or har params
        # for param in self.hpe.parameters():
        #     param.requires_grad = False
        self.ensure_pca_fixed_weights = True # true by default

        self.ensure_batchnorm_fixed_eval = ensure_batchnorm_fixed_eval
        self.ensure_dropout_fixed_eval = ensure_dropout_fixed_eval
        
        # legacy cmdarg
        if ensure_batchnorm_dropout_fixed_eval:
            print("[COMBINED_MODEL] WARN: ENSURE_BATCHNORM_DROPOUT_FIXED_EVAL IS LEGACY ARG!")
            self.ensure_batchnorm_fixed_eval = True
            self.ensure_dropout_fixed_eval = True

        

        self.forward_type = int(forward_type)
        self.combined_version = str(combined_version)

        #### temporary code for debugging only ####
        if self.combined_version == '0c' and not getattr(self, "hpe_act", False):
            print('[COMBINED_MODEL] SPECIAL DEBUG CODE: CREATING TEMP HPE ACT')
            hpe_act_args = deepcopy(hpe_args)
            hpe_act_args['action_cond_ver'] = 6
            self.hpe_act = DeepPriorPPModel(**hpe_act_args)
        
        #### new attention mechanism ####
        if self.combined_version[0] == '5' or self.combined_version[0] == '6':
            print('[COMBINED_MODEL] V5X/6X: Adding extra module for attention mechanism')
            self.attention_layer = nn.Linear(joints_dim*max_seq_length, max_seq_length)
            self.max_seq_length = max_seq_length
            self.joints_dim = joints_dim
        
        if self.combined_version[0] == '7':
            print('[COMBINED_MODEL] V7X: Adding extra cnn module for attention mechanism with hidden concat')
            self.attention_layer = nn.Conv2d(1,1,kernel_size=(1,joints_dim+self.har.recurrent_layers_dim))
            self.max_seq_length = max_seq_length
            self.joints_dim = joints_dim

        self._assert_attribs()
        if not force_trainable_params: self._set_trainable_params() # turn params on or off based on forward type and versions

        self.combined_forward_fn_dict  = {
            '0': self.forward_v0,
            '0a': self.forward_v0_action,
            '0b': self.forward_v0_action_unrolled_legacy, # create action from feedback, use 2 models, 2 passes per timestep
            '0c': self.forward_v0_action_feedback, # create action from feedback, use 2 models, 2 batched passes
            '1a': self.forward_v1_hpe,
            '2a': self.forward_v0,
            '2b': self.forward_v0,
            '2b2': self.forward_v0,
            '2c': self.forward_v0,
            '2d': self.forward_v0,
            '3a': self.forward_v0_unrolled,
            '3d': self.forward_v0_unrolled,
            '4d': self.forward_v0_unrolled_action_feedback,
            '5a': self.forward_v0_unrolled_attention,
            '5d': self.forward_v0_unrolled_attention,
            '5f': self.forward_v0_unrolled_attention,
            '6d': self.forward_v0_unrolled_action_feedback_attention,
            '7d': self.forward_v0_unrolled_attention_cnn,
        }

        # set here to ensure init_metric fn can access these values
        self.train_pca_space = self.hpe.train_pca_space
        self.eval_pca_space = self.hpe.eval_pca_space
    
        
    
    def forward(self, x):
        #x, batch_sizes
        #x = PackedSequence(data=x, batch_sizes=batch_sizes)
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
            # use index to get first char
            if self.combined_version[0] == '0':
                print("[COMBINED_MODEL] V0: Setting all params to non-trainable")
                for param in self.har.parameters():
                    param.requires_grad = False
                for param in self.hpe.parameters():
                    param.requires_grad = False
                if getattr(self, 'hpe_act', False):
                    for param in self.hpe_act.parameters():
                        param.requires_grad = False
            
            elif self.combined_version == '1a':
                # disable har, only train hpe
                print("[COMBINED_MODEL] V1A: Setting only har params to non-trainable")
                for param in self.har.parameters():
                    param.requires_grad = False
            
            elif self.combined_version == '2a' or self.combined_version == '3a'  or self.combined_version == '5a':
                print("[COMBINED_MODEL] V2A/V3A/V5A: Setting HPE-RES, HPE-PCA, HAR-LSTM & HAR-LIN params to non-trainable")
                for param in self.har.parameters():
                    param.requires_grad = False
                for param in self.hpe.main_layers.parameters():
                    param.requires_grad = False
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = False
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
            
            elif self.combined_version == '2b':
                print("[COMBINED_MODEL] V2B: Setting HPE-PCA, HAR-LSTM & HAR-LIN params to non-trainable")
                for param in self.har.parameters():
                    param.requires_grad = False
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = False
                    #self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
            
            elif self.combined_version == '2b2':
                print("[COMBINED_MODEL] V2B2: Setting all HPE layers incl. PCA to be trainable. Rest non-trainable")
                for param in self.har.parameters():
                    param.requires_grad = False
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = True
                self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
            
            elif self.combined_version == '2c':
                print("[COMBINED_MODEL] V2C: Setting all lin and pca layers trainable")
                for param in self.hpe.main_layers.parameters():
                    param.requires_grad = False
                for param in self.har.recurrent_layers.parameters():
                    param.requires_grad = False
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = True
                self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")

            elif self.combined_version == '2d' or self.combined_version == '3d' \
                 or self.combined_version == '5d' or self.combined_version == '7d':
                print("[COMBINED_MODEL] V2D/V3D/V5D/V7D: Setting all layers layers (incl. PCA) trainable.")
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = True
                self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
            

            elif self.combined_version == '4d' or self.combined_version == '6d':
                print("[COMBINED_MODEL] V4D/V6D: Setting all layers layers (incl. PCA) trainable, ensuring HPE is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = True
                self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False

                assert self.hpe.action_cond_ver != 0, "Please use ActCond arch for HPE!"

            

            elif self.combined_version == '2e':
                print("[COMBINED_MODEL] V2E: Setting all linear layers trainable")
                for param in self.hpe.final_layer.parameters():
                    param.requires_grad = False
                self.hpe.train_pca_space = False
                for param in self.hpe.linear_layers.parameters():
                    param.requires_grad = False
                for param in self.har.action_layers.parameters():
                    param.requires_grad = False
            
            elif self.combined_version == '5f':
                print("[COMBINED_MODEL] V5F: Setting all HPE layers non-trainable.")
                for param in self.hpe.parameters():
                    param.requires_grad = False
                self.ensure_pca_fixed_weights = True
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
            
            else:
                raise NotImplementedError("[COMBINED_MODEL] No init scheme found for %s" % self.combined_version)

        
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

    
    def initialize_pca_layer(self, w,b):
        # handled by hpe
        self.hpe.initialize_pca_layer(w,b, ensure_fixed_weights=self.ensure_pca_fixed_weights)
        if getattr(self, 'hpe_act', False):
            self.hpe_act.initialize_pca_layer(w,b, ensure_fixed_weights=self.ensure_pca_fixed_weights)
    

    def extract_final_timestep(self, outputs_padded: torch.tensor, lengths: torch.tensor):
        '''
            return the outputs at the last time-step for each relevant sample in a batch

            REQUIRED DIM:
                B x T x H
                
                B => Batch_SZ
                T => Time_steps
                H => Hidden Size

                e.g. (16, 120, 100)
                Anything else won't work!
            e.g 
            input:\n
            [[1,2,3,4],
             [5,6,7,0],
             [8,0,0,0],

            batch_size: [4,3,1]

            output: [1,2,3,4,5,6,7,8]


        '''
        
        return outputs_padded.gather(1, (lengths-1).to(outputs_padded.device)\
                                                 .unsqueeze(1).unsqueeze(2)\
                                                 .expand(-1,-1,outputs_padded.shape[2]))\
                                                 .squeeze(1)
    

    def get_h0_c0(self, x_padded):
        h0 = torch.zeros(self.har.num_recurrent_layers*1, x_padded.shape[0],
                          self.har.recurrent_layers_dim,
                          device=x_padded.device, dtype=x_padded.dtype)
        c0 = h0.clone()

        return (h0, c0)
    

    def get_a0(self, x_padded):
        '''
            return 1/45 * torch.ones() expanded to fit batch size
        '''
        return self.equiprob_act.unsqueeze(0).expand(x_padded.shape[0], -1).to(x_padded.device, x_padded.dtype)

    def get_log_a0(self, x_padded):
        return self.equiprob_act_log.unsqueeze(0).expand(x_padded.shape[0], -1).to(x_padded.device, x_padded.dtype)
    
    def get_ones_a0(self, x_padded):
        return self.act_ones.unsqueeze(0).expand(x_padded.shape[0], -1).to(x_padded.device, x_padded.dtype)
    
    def get_zeros_a0(self, x_padded):
        return self.act_ones.unsqueeze(0).expand(x_padded.shape[0], -1).to(x_padded.device, x_padded.dtype)


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

        x_interim = self.extract_final_timestep(outputs_padded, batch_sizes)

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
        y = PackedSequence(data=y, batch_sizes=x.batch_sizes)
        z = self.har(y)

        return (y, z)
    
    def forward_v0_unrolled(self,x):
        '''
            x => PackedSeq of depths only (no act information)

            Non-action conditioned back_prop

            x => Depth
            y => Keypts
            h => Hidden State from HAR.LSTM
            z => Action
        '''
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)

        y = []
        h = [] # store hidden states from HAR.LSTM block (before HAR.LIN block)

        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM

        ## need to somehow set batch norm layers to a fixed layer so that this doesn't happen
        ## maybe we can make it fixed output or make it trainable=False and test?
        ## basically a different batch size can cause a lot of problems with output

        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            # simple non-action pass-through
            # unsqueeze for channel dim
            y_i = self.hpe(x_padded[:,i,...].unsqueeze(1))
            y.append(y_i)

            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_i.unsqueeze(1), (h_n, c_n))
            h.append(h_i.squeeze(1))

            # actions.append(torch.exp(self.har.action_layers(y.squeeze(1)))) # for future use
        
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        y_padded = torch.stack(y, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        # extract final state making it equal to compact/rolled version during forward pass
        h_padded = torch.stack(h, dim=1)
        h_final_timestep = self.extract_final_timestep(h_padded, lengths)

        z = self.har.action_layers(h_final_timestep)

        return (y, z)
    
    def forward_v0_unrolled_action_feedback(self,x):
        '''
            Action conditioned back_prop
            Assumes Self.HPE can accept action

            x => Depth_Packed (GT Actions NOT REQUIRED)
            y => Keypts
            h => Hidden State from HAR.LSTM
            z => Action
        '''
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)

        y = []
        h = [] # store hidden states from HAR.LSTM block (before HAR.LIN block)

        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM
        z_n = self.get_a0(x_padded) # get_equiprob z vector

        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            # simple non-action pass-through
            # unsqueeze for channel dim
            y_i = self.hpe((x_padded[:,i,...].unsqueeze(1), z_n))
            y.append(y_i)

            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_i.unsqueeze(1), (h_n, c_n))
            h.append(h_i.squeeze(1))

            z_n = torch.exp(h_i.squeeze(1)) # these are log_probs, need to make them probs!
            # actions.append(torch.exp(self.har.action_layers(y.squeeze(1)))) # for future use
        
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        y_padded = torch.stack(y, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        # extract final state making it equal to compact/rolled version during forward pass
        h_padded = torch.stack(h, dim=1)
        h_final_timestep = self.extract_final_timestep(h_padded, lengths)

        z = self.har.action_layers(h_final_timestep)

        return (y, z)



    def forward_v0_unrolled_attention(self, x):
        '''
            U/C
        '''
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)

        y = []
        h = [] # store hidden states from HAR.LSTM block (before HAR.LIN block)

        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM

        batch_size = x_padded.shape[0]
        past_timesteps = 0
        y_buffer_list = [torch.zeros(batch_size, self.joints_dim,
                               device=x_padded.device, dtype=x_padded.dtype) for _ in range(self.max_seq_length)]


        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            # simple non-action pass-through
            # unsqueeze for channel dim
            y_i = self.hpe(x_padded[:,i,...].unsqueeze(1))
            y.append(y_i)
            
            # cloning messes up prev timestep gradients for hpe!
            #y_buffer = y_buffer.clone() # need this for in-place operation // .detach() not needed gets implied (i think) in clone

            ###### attention #####
            past_timesteps += 1 # is there a bug when past_timesteps == 120?

            idx_sel = torch.arange(past_timesteps).to(x_padded.device, torch.long) # possible dtype issue with gpu
            #y_buffer[:,i,:] = y_i # just add here directly
            y_buffer_list[i] = y_i
            y_buffer = torch.stack(y_buffer_list, dim=1)
            ## possible dtype for arange error here...
            
            y_reduced = torch.gather(y_buffer, 1, idx_sel.view(1,-1,1).expand(batch_size, -1, self.joints_dim))
            #print("Y_RED_SHAPE", y_reduced.shape) # 5x10x63

            # possible device issue with arange
            betas = self.attention_layer(y_buffer.view(y_buffer.shape[0], y_buffer.shape[1]*y_buffer.shape[2])).unsqueeze(1)
            betas_reduced = torch.gather(betas, 2, idx_sel.view(1,1,-1).expand(batch_size, 1, -1))
            #print("BETAS_RED_SHAPE", betas_reduced.shape) # 5x1x10

            # calc softmax along last dim
            # note we don't use nn.Module here as it is calc amongst reduced tensors!
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            y_focused_reduced = torch.bmm(alphas_reduced, y_reduced)
            #print("Y_AFTER_ATTENTION:", y_focused_reduced.shape) # 5x1x63


            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_focused_reduced, (h_n, c_n))
            h.append(h_i.squeeze(1))


        ### Now pass y to HAR!
        ##
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        
        y_padded = torch.stack(y, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        # extract final state making it equal to compact/rolled version during forward pass
        h_padded = torch.stack(h, dim=1)
        h_final_timestep = self.extract_final_timestep(h_padded, lengths)

        z = self.har.action_layers(h_final_timestep)

        return (y, z)
    

    def forward_v0_unrolled_attention_cnn(self, x):
        '''
            U/C
        '''
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)

        y = []
        h = [] # store hidden states from HAR.LSTM block (before HAR.LIN block)
        y_cat_h = []
        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM

        batch_size = x_padded.shape[0]
        past_timesteps = 0
        # y_buffer_list = [torch.zeros(batch_size, self.joints_dim,
        #                        device=x_padded.device, dtype=x_padded.dtype) for _ in range(self.max_seq_length)]


        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            # simple non-action pass-through
            # unsqueeze for channel dim
            y_i = self.hpe(x_padded[:,i,...].unsqueeze(1))
            y.append(y_i)
            
            # cat [B x 63] with [B x 100] (need to squeeze as h is originally [1 x B x 100])
            y_cat_h.append(torch.cat([y_i, h_n.squeeze(0)], dim=1))

            # B x 1 x T x 1 -> B x 1 x T
            betas_reduced = self.attention_layer(torch.stack(y_cat_h, dim=1).unsqueeze(1)).squeeze(3)

            # B x 1 x T -> B x 1 x T
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            # bmm with current history of y's
            # [B x 1 x T] DOT [B x T x 63] -> [B x 1 x 63]
            y_focused = torch.bmm(alphas_reduced, torch.stack(y, dim=1))
            #print("Y_AFTER_ATTENTION:", y_focused.shape) # 5x1x63


            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_focused, (h_n, c_n))
            h.append(h_i.squeeze(1))


        ### Now pass y to HAR!
        ##
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        
        y_padded = torch.stack(y, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        # extract final state making it equal to compact/rolled version during forward pass
        h_padded = torch.stack(h, dim=1)
        h_final_timestep = self.extract_final_timestep(h_padded, lengths)

        z = self.har.action_layers(h_final_timestep)

        return (y, z)
    
    def forward_v0_unrolled_action_feedback_attention(self, x):
        '''
            U/C
        '''
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)

        y = []
        h = [] # store hidden states from HAR.LSTM block (before HAR.LIN block)

        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM
        z_n = self.get_a0(x_padded) # get_equiprob z vector

        batch_size = x_padded.shape[0]
        past_timesteps = 0
        y_buffer_list = [torch.zeros(batch_size, self.joints_dim,
                               device=x_padded.device, dtype=x_padded.dtype) for _ in range(self.max_seq_length)]


        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            # simple non-action pass-through
            # unsqueeze for channel dim
            y_i = self.hpe((x_padded[:,i,...].unsqueeze(1), z_n))
            y.append(y_i)
            
            # cloning messes up prev timestep gradients for hpe!
            #y_buffer = y_buffer.clone() # need this for in-place operation // .detach() not needed gets implied (i think) in clone

            ###### attention #####
            past_timesteps += 1 # is there a bug when past_timesteps == 120?

            idx_sel = torch.arange(past_timesteps).to(x_padded.device, torch.long) # possible dtype issue with gpu
            y_buffer_list[i] = y_i
            y_buffer = torch.stack(y_buffer_list, dim=1)
            ## possible dtype for arange error here...
            
            y_reduced = torch.gather(y_buffer, 1, idx_sel.view(1,-1,1).expand(batch_size, -1, self.joints_dim))
            #print("Y_RED_SHAPE", y_reduced.shape) # 5x10x63

            # possible device issue with arange
            betas = self.attention_layer(y_buffer.view(y_buffer.shape[0], y_buffer.shape[1]*y_buffer.shape[2])).unsqueeze(1)
            betas_reduced = torch.gather(betas, 2, idx_sel.view(1,1,-1).expand(batch_size, 1, -1))
            #print("BETAS_RED_SHAPE", betas_reduced.shape) # 5x1x10

            # calc softmax along last dim
            # note we don't use nn.Module here as it is calc amongst reduced tensors!
            alphas_reduced = F.softmax(betas_reduced, dim=2)

            # note we dont squeee here as we have to unsqueeze later for timestep anyways...
            y_focused_reduced = torch.bmm(alphas_reduced, y_reduced)
            #print("Y_AFTER_ATTENTION:", y_focused_reduced.shape) # 5x1x63


            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_focused_reduced, (h_n, c_n))
            h.append(h_i.squeeze(1))
            z_n = torch.exp(h_i.squeeze(1)) # these are log_probs, need to make them probs!


        ### Now pass y to HAR!
        ##
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        
        y_padded = torch.stack(y, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        # extract final state making it equal to compact/rolled version during forward pass
        h_padded = torch.stack(h, dim=1)
        h_final_timestep = self.extract_final_timestep(h_padded, lengths)

        z = self.har.action_layers(h_final_timestep)

        return (y, z)

    def forward_v0_action(self, x):
        ### thi is not a valid model only used for testing
        ### x is tuple of two packed sequences
        ### batch_sizes should be same for both
        depth, batch_sizes, action = x[0].data, x[0].batch_sizes, x[1].data
        ## pass with conditioning
        y = self.hpe((depth.unsqueeze(1), action)) #self.hpe_act((depth.unsqueeze(1), action))
        #print("y_Shape", y.shape)

        y = PackedSequence(data=y, batch_sizes=batch_sizes)

        # tmpz = self.har(y)
        # print(tmpz.shape)
        # print(action.shape)
        # quit()
        #print(y)
        # return both predictions of keypts and action class prediction
        return (y, self.har(y))

    def forward_v0_action_feedback(self, x):
        depth, batch_sizes, action = x[0].data, x[0].batch_sizes, x[1].data
        #y_gt = x[2].data
        #a0 = self.get_a0(depth)
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1)) #self.hpe((depth.unsqueeze(1), action)) #self.hpe((depth.unsqueeze(1), action)) #self.hpe(depth.unsqueeze(1)) #action #self.hpe((depth.unsqueeze(1), a0))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        
        #y = self.hpe_act((depth.unsqueeze(1), action)) #self.hpe_act((depth.unsqueeze(1), action))

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        #return (y_, z_log_)

        _, lengths = pad_packed_sequence(x[0], batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        
        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        alpha = self.act_0c_alpha #0.1 #0.0001
        z_log_mean = (alpha * z_log) + ((1-alpha) * z_log_)

        y_mean = (alpha * y.data) + ((1-alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        # print("\nGT ACTION:           ", torch.argmax(action, dim=1)[:batch_sizes[0]])
        # #print("PRED ACTION_PASS_1_ORIG_SHAPE:   ", torch.argmax(z_, dim=1))
        # print("PRED ACTION_PASS_1:    ", torch.argmax(z_seq_, dim=1)[:batch_sizes[0]]) #_AFTER_PACKING
        # print("PRED ACTION_PASS_2:    ", torch.argmax(z_log, dim=1))
        # print("PRED ACTION_MEAN_PASS: ", torch.argmax(z_log_mean, dim=1))
        
        return (y_mean, z_log_mean) #(y, z_log_mean)
        #torch.cat([a[i].repeat(b[i],1) for i in range(b.shape[0])])
    
    def forward_v0_action_unrolled_legacy(self, x):
        ### thi is not a valid model only used for testing
        ### x is tuple of two packed sequences
        ### batch_sizes should be same for both
        depth, batch_sizes, action = x[0].data, x[0].batch_sizes, x[1].data

        #print("BS SHAPE", batch_sizes.shape)
        #self.equiprob_act.unsqueeze(0).expand(x_padded.shape[0], -1).to(x_padded.device, x_padded.dtype)
        ## pass with conditioning
        x_padded, lengths = pad_packed_sequence(x[0], batch_first=True)
        a_padded, _ = pad_packed_sequence(x[1], batch_first=True)

        #print("BS2 SHAPE", batch_sizes.shape)

        outputs = []
        actions = []

        a_equiprob = self.get_zeros_a0(x_padded) # a_n #get_a0

        for i in range(x_padded.shape[1]):
            y = self.hpe(x_padded[:,i,...].unsqueeze(1)) #self.hpe_act((x_padded[:,i,...].unsqueeze(1), a_padded[:,i,...])) #a_equiprob a_padded[:,i,...]
            #a_0 = a_n.detach()
            outputs.append(y)
           
            #print("Y_SHAPE: ", y.shape)
            #vlens = torch.ones(y.shape[0], device=y.device, dtype=y.dtype)
            #y_interim = pack_padded_sequence(torch.stack(outputs, dim=1).unsqueeze(1), batch_sizes[:i+1], batch_first=True)
            #z_ = self.har(y, )
            #print(z_.shape)
            #quit()
            # #actions.append(a_n)
        
        # ## extract action at final timestep -- this will be our prediction for the entire sequence
        ## we can also perform other operations like mean, max etc
        #actions = torch.stack(actions, dim=1)
        outputs_padded = torch.stack(outputs, dim=1)

        (h_n, c_n) = self.get_h0_c0(outputs_padded)
        actions = []

        for i in range(outputs_padded.shape[1]):
            y, (h_n, c_n) = self.har.recurrent_layers(outputs_padded[:,i,:].unsqueeze(1), (h_n, c_n))
            actions.append(torch.exp(self.har.action_layers(y.squeeze(1))))

        actions_padded = torch.stack(actions, dim=1)
        #print("a_orig_padded", a_padded.shape, "a_new_padded", actions_padded.shape)
        #quit()
        #actions = self.extract_final_timestep(actions, batch_sizes)
        #outputs = pack_padded_sequence(outputs, lengths, batch_first=True)
        ### need to pad_packed_sequence
        outputs = []
        for i in range(x_padded.shape[1]):
            y = self.hpe_act((x_padded[:,i,...].unsqueeze(1), actions_padded[:,i,...])) #a_equiprob a_padded[:,i,...]
            #a_0 = a_n.detach()
            outputs.append(y)
        #z = self.har(outputs)
        outputs = torch.stack(outputs, dim=1)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)
        #print("Z_SHAPE:", z.shape)
        #print("a_equiprob_shape:,", a_equiprob.shape)
        #return (outputs, actions)

        #  for i in range(x_padded.shape[1]):
        #     y = self.hpe_act((x_padded[:,i,...].unsqueeze(1), a_padded[:,i,...])) #a_equiprob a_padded[:,i,...]
        #     #a_0 = a_n.detach()
        #     outputs.append(y)
        
        #self.hpe.action_cond_ver = 0
        #self.hpe.use_resnet_conditioning = False
        #y = self.hpe((depth.unsqueeze(1), None)) #action
        #print("y_Shape", y.shape)

        #y = PackedSequence(data=y, batch_sizes=batch_sizes)
        #print(y)
        # return both predictions of keypts and action class prediction
        return (outputs, self.har(outputs))
    

    def forward_v1_hpe(self, x):
        '''
            add a single step previous frame as action condition to hpe or something?

            only train hpe in this case do not train har maybe set har params to non trainable

            single step means detaching every loop
            you can add more steps maybe detaching every other 3 or something
        '''

        ## x is packed sequence
        # inputs: x
        # out: keypts, action value
        # targets: keypts, action_idx
        
        # we feed in previous output

        ## note hpe should have conditioning information so act cond 6 and output action information
        assert isinstance(x, PackedSequence)
        x_padded, batch_sizes = pad_packed_sequence(x, batch_first=True)
        
        ### note try a0 as log likehilood vector or just normal equiprob vect
        a_n = self.get_a0(x_padded) # a_n
        a_0 = a_n.clone()
        

        outputs = []
        actions = []

        for i in range(x_padded.shape[1]):
            y, a_n = self.hpe((x_padded[:,i,...].unsqueeze(1), a_0))
            a_0 = a_n.detach()
            outputs.append(y)
            actions.append(a_n)
        
        # ## extract action at final timestep -- this will be our prediction for the entire sequence
        ## we can also perform other operations like mean, max etc
        actions = torch.stack(actions, dim=1)
        outputs = torch.stack(outputs, dim=1)

        actions = self.extract_final_timestep(actions, batch_sizes)
        outputs = pack_padded_sequence(outputs, batch_sizes, batch_first=True)
        ### need to pad_packed_sequence

        return (outputs, actions)
        #outputs = torch.stack(outputs, dim=1)

        ## make packed
        #outputs = self.extract_final_outputs(outputs, batch_sizes)
        # for i in range(x_padded.shape[1]):
        #     y, a_n = self.hpe(x_padded[:,i,...].unsqueeze(1), a_n)
        #     outputs.append(y)



    def forward_v2(self, x):
        '''
            add a single step previous frame as action condition to hpe or something?

            only train hpe in this case do not train har maybe set har params to non trainable
        '''

        ### feed hidden state directly back to 

        ## x is packed sequence
        # inputs: x
        # out: keypts, action value
        # targets: keypts, action_idx
        
        # we feed in previous output

        ## note hpe should have conditioning information so act cond 6 and output action information
        assert isinstance(x, PackedSequence)
        x_padded, batch_sizes = pad_packed_sequence(x, batch_first=True)
        
        a_n = self.get_a0(x_padded)

        outputs = []
        actions = []

        for i in range(x_padded.shape[1]):
            y, a_n = self.har(x_padded[:,i,...].unsqueeze(1), a_n)
            outputs.append(y)
    

    def train(self, mode=True):
        super(CombinedModel, self).train(mode=mode)
        #print("train called")
        
        print('')
        ## new code to ensure same results in rolled and unrolled combined version
        for m in self.modules():
            if (self.ensure_batchnorm_fixed_eval and isinstance(m, nn.BatchNorm2d)) or \
                (self.ensure_dropout_fixed_eval and isinstance(m, nn.Dropout)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False # also make non-trainable
        
        bnorm_training = None
        dropout_training = None
        for m in self.modules():
            # first first instance and get whether module is in train or eval mode
            if isinstance(m, nn.BatchNorm2d):
                bnorm_training = m.training
            elif isinstance(m, nn.Dropout):
                dropout_training = m.training
            if bnorm_training is not None and dropout_training is not None:
                break
        print("[COMBINED_MODEL] SPECIAL_NOTICE   B_NORM_TRAIN_MODE: %s DROPOUT_TRAIN_MODE: %s" % (bnorm_training, dropout_training))
        return self