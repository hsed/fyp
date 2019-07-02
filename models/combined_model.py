import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict, deque
from copy import deepcopy
from functools import partial
import yaml

from models import BaseModel, DeepPriorPPModel, BaselineHARModel

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence, pack_padded_sequence

class CombinedModel(BaseModel):
    def __init__(self, hpe_checkpoint=None, har_checkpoint=None, pca_checkpoint=None,
                 hpe_act_checkpoint = None, har_act_checkpoint=None,
                 hpe_args = None, har_args = None, forward_type=0, combined_version='0',
                 force_trainable_params=False, action_classes=45, act_0c_alpha=0.01,
                 joints_dim=63, max_seq_length=120,
                 ensure_batchnorm_fixed_eval=False,
                 ensure_dropout_fixed_eval=False,
                 ensure_batchnorm_dropout_fixed_eval=False,
                 temporal_smoothing=-1, trainable_smoothing=False,
                 ensemble_eta=0.5, ensemble_zeta=0.5,
                 trainable_enemble_hyp=False):
        
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
            print("[COMBINED_MODEL] Loading HPE Checkpoint: %s" % hpe_checkpoint)
            print("[COMBINED_MODEL] OVERWRITING ACTION_COND with HPE_ARGS config: %s" % hpe_args['action_cond_ver'])
            checkpoint = torch.load(hpe_checkpoint)
            checkpoint['config']['arch']['args']['action_cond_ver'] = hpe_args['action_cond_ver']
            if 'action_equiprob_chance' in hpe_args:
                print("[COMBINED_MODEL] OVERWRITING ACTION_EQUIPROB_CHANCE with HPE_ARGS config: %s" % hpe_args['action_equiprob_chance'])
                checkpoint['config']['arch']['args']['action_equiprob_chance'] = hpe_args['action_equiprob_chance']
            self.hpe = DeepPriorPPModel(**checkpoint['config']['arch']['args'])
            self.hpe.load_state_dict(checkpoint['state_dict'], strict=False)
        

        if hpe_act_checkpoint is not None:
            print("[COMBINED_MODEL] Loading HPE ACT Checkpoint: %s" % hpe_act_checkpoint)
            print("[COMBINED_MODEL] OVERWRITING ACTION_COND with HPE_ACT_ACTION_COND: %s" % hpe_args['hpe_act_action_cond_ver'])
            checkpoint_ = torch.load(hpe_act_checkpoint)
            checkpoint_['config']['arch']['args']['action_cond_ver'] = hpe_args['hpe_act_action_cond_ver']
            if 'action_equiprob_chance' in hpe_args:
                print("[COMBINED_MODEL] OVERWRITING ACTION_EQUIPROB_CHANCE with HPE_ARGS config: %s" % hpe_args['action_equiprob_chance'])
                checkpoint_['config']['arch']['args']['action_equiprob_chance'] = hpe_args['action_equiprob_chance']
            self.hpe_act = DeepPriorPPModel(**checkpoint_['config']['arch']['args'])
            self.hpe_act.load_state_dict(checkpoint_['state_dict'])
            # for param in self.hpe_act.parameters():
            #     param.requires_grad = False
        
        if har_act_checkpoint is not None:
            print("[COMBINED_MODEL] NEW: LOADING HAR_ACT CHECKPOINT: %s" % har_act_checkpoint)
            checkpoint__ = torch.load(har_act_checkpoint)
            # print("******* NEW CONFIG\n", yaml.dump(checkpoint__['config']['arch']['args']))
            self.har_act = BaselineHARModel(**checkpoint__['config']['arch']['args'])
            self.har_act.load_state_dict(checkpoint__['state_dict'])


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

        print("[COMBINED_MODEL] Using Combined Version: %s" % self.combined_version)

        #### temporary code for debugging only ####
        if self.combined_version == '0c' and not getattr(self, "hpe_act", False):
            print('[COMBINED_MODEL] SPECIAL DEBUG CODE: CREATING TEMP HPE ACT')
            hpe_act_args = deepcopy(hpe_args)
            hpe_act_args['action_cond_ver'] = 6
            self.hpe_act = DeepPriorPPModel(**hpe_act_args)
        
        #### temporary code for debugging only ####
        if self.combined_version[0:2] == '12' and not getattr(self, "har_act", False):
            print('[COMBINED_MODEL] SPECIAL DEBUG CODE: CREATING TEMP HAR ACT USING HAR ARGS')
            self.har_act = BaselineHARModel(**har_args)
        
        #### temporary code for debugging only ####
        if self.combined_version[0:2] == '15' and not getattr(self, "har_act", False):
            print('[COMBINED_MODEL] SPECIAL DEBUG CODE: CREATING TEMP HAR ACT USING HAR ARGS')
            self.har_act = BaselineHARModel(**har_args)
        
        if self.combined_version[0:2] == '16':
            if not getattr(self, "har_act", False):
                print('[COMBINED_MODEL] SPECIAL DEBUG CODE: CREATING TEMP HAR ACT USING HAR ARGS')
                har_act_args = deepcopy(har_args)
                har_act_args['attention_type'] = 'v5'
                self.har_act = BaselineHARModel(**har_act_args)
            if not getattr(self, "hpe_act", False):
                print('[COMBINED_MODEL] SPECIAL DEBUG CODE: CREATING TEMP HPE ACT')
                hpe_act_args = deepcopy(hpe_args)
                hpe_act_args['action_cond_ver'] = 6
                self.hpe_act = DeepPriorPPModel(**hpe_act_args)
        
        #### new attention mechanism ####
        if self.combined_version[0] == '5':
            print('[COMBINED_MODEL] V5X/6X: Adding extra module for attention mechanism')
            self.attention_layer = nn.Linear(joints_dim*max_seq_length, max_seq_length)
            self.max_seq_length = max_seq_length
            self.joints_dim = joints_dim
        
        if self.combined_version[0] == '7':
            print('[COMBINED_MODEL] V7X: Adding extra cnn module for attention mechanism with hidden concat')
            self.attention_layer = nn.Conv2d(1,1,kernel_size=(1,joints_dim+self.har.recurrent_layers_dim))
            self.max_seq_length = max_seq_length
            self.joints_dim = joints_dim
        
        #### check
        if self.combined_version[0] == '8' or self.combined_version[0:2] == '10':
            if temporal_smoothing < 0 or temporal_smoothing > 1:
                print('[COMBINED_MODEL] Gamma=%0.2f is invalid, using default gamma=0.4 for combined ver 8x' % temporal_smoothing)
                temporal_smoothing = 0.4
            if trainable_smoothing:
                print('[COMBINED_MODEL] Gamma=%0.2f will be made trainable!' % temporal_smoothing)
                self.temporal_smoothing_param = torch.nn.Parameter(torch.tensor(temporal_smoothing))
                temporal_smoothing = self.temporal_smoothing_param
        
        if self.combined_version[0:2] == '12' or self.combined_version[0:2] == '13' \
            or self.combined_version == '11d3':
            print("[COMBINED_MODEL] act_0c_alpha=%0.4f aka student_teacher_fusion will be made TRAINABLE!" % act_0c_alpha)
            self.act_0c_alpha = torch.nn.Parameter(torch.tensor(float(act_0c_alpha)))
        
        if self.combined_version == '11d5':
            print('[COMBINED_MODEL] Eta=%0.4f, Zeta=%0.4f' % (ensemble_eta, ensemble_zeta))
            if trainable_enemble_hyp:
                print("[COMBINED_MODEL] ensemble eta and zeta trainable!")
                self.ensemble_eta = torch.nn.Parameter(torch.tensor(float(ensemble_eta)))
                self.ensemble_zeta = torch.nn.Parameter(torch.tensor(float(ensemble_zeta)))
            else:
                self.ensemble_eta = ensemble_eta
                self.ensemble_zeta = ensemble_zeta
            ensemble_eta = self.ensemble_eta # useful if made a param
            ensemble_zeta = self.ensemble_zeta # useful if made a param

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
            '2d': self.forward_v0, # 2nd best ... baseline
            '2f': self.forward_v0,
            '3a': self.forward_v0_unrolled,
            '3d': self.forward_v0_unrolled, # 2nd best
            '4d': self.forward_v0_unrolled_action_feedback, # best
            '5a': self.forward_v0_unrolled_attention,
            '5d': self.forward_v0_unrolled_attention,
            '5f': self.forward_v0_unrolled_attention,
            '6d': self.forward_v0_unrolled_action_feedback_attention, # new, rewritten
            '7d': self.forward_v0_unrolled_attention_cnn,
            '8d': self.forward_v0_unrolled_action_feedback_smoothing,
            '8d1': partial(self.forward_v0_unrolled_action_feedback_smoothing, gamma=0.4, har_smoothing=False),
            '8d2': partial(self.forward_v0_unrolled_action_feedback_smoothing, gamma=0.4, har_smoothing=True),
            '8d3': partial(self.forward_v0_unrolled_action_feedback_smoothing, gamma=temporal_smoothing, har_smoothing=False),
            '8d4': partial(self.forward_v0_unrolled_action_feedback_smoothing, gamma=temporal_smoothing, har_smoothing=True),
            '10d': partial(self.forward_v0_unrolled_action_feedback_attention_smoothing, gamma=temporal_smoothing, har_smoothing=True),
            '10d2': partial(self.forward_v0_unrolled_action_feedback_attention_smoothing, gamma=temporal_smoothing, har_smoothing=False),
            '11d': self.forward_v0_action_feedback_cleaned,
            '11d2': self.forward_v0_action_feedback_cleaned,
            '11d3': self.forward_v0_action_feedback_cleaned,
            '11d4': partial(self.forward_v0_action_feedback_temporal_smoothing, gamma=temporal_smoothing),
            '11d5': partial(self.forward_v0_action_feedback_eta_zeta, eta=ensemble_eta, zeta=ensemble_zeta),
            '12d': self.forward_v0_action_feedback_dual_hpe_har,
            '12d2': self.forward_v0_action_feedback_dual_hpe_har,
            '13d': self.forward_v0_action_feedback_refined,
            '14d': self.forward_v0_action_feedback_cleaned,
            '15d': partial(self.forward_v0_action_feedback_student_teacher, return_y_teacher=False),
            '15d2': partial(self.forward_v0_action_feedback_student_teacher, return_y_teacher=True),
            '15d3': partial(self.forward_v0_action_feedback_student_teacher, return_y_teacher=True, return_type=2),
            '16d': self.forward_v0_action_feedback_dual_temporal_atten,
            '17d': self.forward_v0_action_feedback_logits,
            '18d': self.forward_v0_action_feedback_z_arith_mean,
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
                    print("[COMBINED_MODEL] v0c uses Act0cAlpha: %0.2f" % self.act_0c_alpha)
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
                for param in self.hpe.parameters():
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
            

            elif self.combined_version == '4d' or self.combined_version == '6d' or self.combined_version[:2] == '8d' or \
                 self.combined_version == '10d' or self.combined_version == '10d2':
                print("[COMBINED_MODEL] V4D/V6D/V8D/V10D: Setting all layers layers (incl. PCA) trainable, ensuring HPE is wActCond.")
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
            
            elif self.combined_version == '2f':
                print("[COMBINED_MODEL] V2F: Setting only HAR trainable")
                for param in self.hpe.parameters():
                    param.requires_grad = False
                self.hpe.train_pca_space = False
                for param in self.har.parameters():
                    param.requires_grad = True
            
            elif self.combined_version == '5f':
                print("[COMBINED_MODEL] V5F: Setting all HPE layers non-trainable.")
                for param in self.hpe.parameters():
                    param.requires_grad = False
                self.ensure_pca_fixed_weights = True
                self.hpe.train_pca_space = False
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
            
            elif self.combined_version == '11d' or self.combined_version == '13d' \
                 or self.combined_version == '14d' or self.combined_version == '11d3' \
                 or self.combined_version == '11d4' or self.combined_version == '11d5' \
                 or self.combined_version == '17d' or self.combined_version == '18d':
                print("[COMBINED_MODEL] V11D/V13D/14D: Setting all layers layers (except hpe1//incl. PCA2) trainable, ensuring HPEAct is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                print("[COMBINED_MODEL] V11/V13/V14 uses Act0cAlpha: %0.2f" % self.act_0c_alpha)
                for param in self.hpe.parameters():
                    param.requires_grad = False
                self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False
                self.hpe_act.train_pca_space = False

                for param in self.har.parameters():
                    param.requires_grad = True
                for param in self.hpe.parameters():
                    param.requires_grad = False
                for param in self.hpe_act.parameters():
                    param.requires_grad = True

                assert self.hpe_act.action_cond_ver != 0, "Please use ActCond arch for HPEACT!"
            
            elif self.combined_version == '11d2':
                print("[COMBINED_MODEL] V11D2: Setting all layers layers (incl. PCA2) trainable, ensuring HPEAct is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                print("[COMBINED_MODEL] V11 uses Act0cAlpha: %0.2f" % self.act_0c_alpha)
                ### note this model requires batch size 2 for training as size 4 runs out of mem!
                self.ensure_pca_fixed_weights = False
                self.hpe.train_pca_space = False
                self.hpe_act.train_pca_space = False

                for param in self.har.parameters():
                    param.requires_grad = True
                for param in self.hpe.parameters():
                    param.requires_grad = True
                for param in self.hpe_act.parameters():
                    param.requires_grad = True

                assert self.hpe_act.action_cond_ver != 0, "Please use ActCond arch for HPEACT!"
            

            elif self.combined_version == '12d':
                print("[COMBINED_MODEL] V12D: Setting all layers layers (except hpe1//har1) trainable, ensuring HPEAct is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                print("[COMBINED_MODEL] V12 uses Act0cAlpha (StudentTeacherFusion): %0.2f" % self.act_0c_alpha)
                self.ensure_pca_fixed_weights = False
                
                for param in self.hpe.parameters():
                    param.requires_grad = False
                for param in self.har.parameters():
                    param.requires_grad = False
                
                self.hpe.train_pca_space = False
                self.hpe_act.train_pca_space = False

                for param in self.har_act.parameters():
                    param.requires_grad = True
                for param in self.hpe_act.parameters():
                    param.requires_grad = True

                assert self.hpe_act.action_cond_ver != 0, "Please use ActCond arch for HPEACT!"
            
            elif self.combined_version == '12d2':
                print("[COMBINED_MODEL] V12D2: Setting hpe2+har1+har2 (except hpe1) trainable, ensuring HPEAct is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                print("[COMBINED_MODEL] V12 uses Act0cAlpha (StudentTeacherFusion): %0.2f" % self.act_0c_alpha)
                self.ensure_pca_fixed_weights = False
                
                for param in self.hpe.parameters():
                    param.requires_grad = False
                for param in self.har.parameters():
                    param.requires_grad = True
                
                self.hpe.train_pca_space = False
                self.hpe_act.train_pca_space = False

                for param in self.har_act.parameters():
                    param.requires_grad = True
                for param in self.hpe_act.parameters():
                    param.requires_grad = True

                assert self.hpe_act.action_cond_ver != 0, "Please use ActCond arch for HPEACT!"
            
            elif self.combined_version[:3] == '15d':
                print("[COMBINED_MODEL] V15D: Setting all layers layers (except hpe_act//har_act) trainable, HPEAct is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                self.ensure_pca_fixed_weights = False

                for param in self.hpe_act.parameters():
                    param.requires_grad = False
                for param in self.har_act.parameters():
                    param.requires_grad = False
                
                for param in self.hpe.parameters():
                    param.requires_grad = True
                for param in self.har.parameters():
                    param.requires_grad = True
                
                self.hpe.train_pca_space = False
                self.hpe_act.train_pca_space = False

                assert self.hpe_act.action_cond_ver != 0, "Please use ActCond arch for HPEACT!"
            
            elif self.combined_version[:3] == '16d':
                print("[COMBINED_MODEL] V16D: Setting HPE_ACT, HAR, HAR_ACT_ATTEN trainable, HPEAct is wActCond.")
                print("[COMBINED_MODEL] Enforcing train_pca_space = False for HPE")
                self.ensure_pca_fixed_weights = False

                for param in self.hpe_act.parameters():
                    param.requires_grad = True
                for param in self.har_act.parameters():
                    param.requires_grad = False
                for param in self.har_act.attention_layer.parameters():
                    # make only attention trainable
                    param.requires_grad = True
                
                for param in self.hpe.parameters():
                    param.requires_grad = False
                for param in self.har.parameters():
                    param.requires_grad = True
                
                self.hpe.train_pca_space = False
                self.hpe_act.train_pca_space = False

                assert self.hpe_act.action_cond_ver != 0, "Please use ActCond arch for HPEACT!"


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

            z_n = torch.exp(self.har.action_layers(h_i.squeeze(1))) #torch.exp(h_i.squeeze(1)) # these are log_probs, need to make them probs!
            # actions.append(torch.exp(self.har.action_layers(y.squeeze(1)))) # for future use
        
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        y_padded = torch.stack(y, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        # extract final state making it equal to compact/rolled version during forward pass
        h_padded = torch.stack(h, dim=1)
        h_final_timestep = self.extract_final_timestep(h_padded, lengths)

        z = self.har.action_layers(h_final_timestep)

        return (y, z)
    

    def forward_v0_unrolled_action_feedback_smoothing(self,x, gamma=0.4, har_smoothing=True): #False
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

        y_prev = self.hpe((x_padded[:,0,...].unsqueeze(1), z_n))

        # if isinstance(gamma, torch.nn.Parameter):
        #     gamma = gamma.to(x_padded.device,x_padded.dtype) # ensure correct device maybe useless

        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            # simple non-action pass-through
            # unsqueeze for channel dim
            y_i = self.hpe((x_padded[:,i,...].unsqueeze(1), z_n))

            # moving average
            y_mavg = gamma*y_prev + (1-gamma)*y_i
            y_prev = y_mavg
            y.append(y_mavg)

            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm
            if har_smoothing:
                y_i = y_mavg # use smoothed y_i as input to har -- probably gonna have bad consequences
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_i.unsqueeze(1), (h_n, c_n))
            h.append(h_i.squeeze(1))

            z_n = torch.exp(self.har.action_layers(h_i.squeeze(1))) # these are log_probs, need to make them probs!
            # actions.append(torch.exp(self.har.action_layers(y.squeeze(1)))) # for future use
        
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
            NEW: V6D HAS CHANGED!!
        '''
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)
        y_list = []
        h_list = [] # list of hidden vectors
        betas_list = []
        action_lin_list = []

        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM
        z_n = self.get_a0(x_padded) # get_equiprob z vector

        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            y_i = self.hpe((x_padded[:,i,...].unsqueeze(1), z_n))
            y_list.append(y_i)

            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_i.unsqueeze(1), (h_n, c_n))
            h_list.append(h_i.squeeze(1))

            z_n = torch.exp(self.har.action_layers(h_i.squeeze(1))) # these are log_probs, need to make them probs!

            beta_i = self.har.attention_layer(torch.cat([y_i, h_i.squeeze(1)], dim=1))
            betas_list.append(beta_i)
            
            action_lin_i = self.har.action_layers[0](h_i.squeeze(1)) # get linear output without softmax operation
            action_lin_list.append(action_lin_i)
        
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        y_padded = torch.stack(y_list, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        h_padded = torch.stack(h_list, dim=1) # B x T x 100
        betas_padded = torch.stack(betas_list, dim=2) # B x 1 x T
        action_lin_padded = torch.stack(action_lin_list, dim=1) # B x T x 45

        # B x 1 x T -> B x 1 x T (softmaxed)
        #padded_alphas = torch.log(betas_padded, dim=2)
        padded_alphas = F.softmax(betas_padded, dim=2)

        # (B x 1 x T) * (B x T x 45) -> (B x 1 x 45) -> (B x 45)
        action_lin_focused = torch.bmm(padded_alphas, action_lin_padded).squeeze(1)

        z_final = self.har.action_layers[1](action_lin_focused)
        
        return (y, z_final)
    

    def forward_v0_unrolled_action_feedback_attention_smoothing(self, x, gamma=0.4, har_smoothing=True):
        '''
            NEW: V10D
        '''
        # now tuple implies the first item is padded sequence and second item is a batch_sizes (vlens) tensor
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)
        y_list = []
        h_list = [] # list of hidden vectors
        betas_list = []
        action_lin_list = []

        (h_n, c_n) = self.get_h0_c0(x_padded) # get init state_vector for LSTM
        z_n = self.get_a0(x_padded) # get_equiprob z vector
        y_prev = self.hpe((x_padded[:,0,...].unsqueeze(1), z_n))

        # timestep is dim[1]
        for i in range(x_padded.shape[1]):
            y_i = self.hpe((x_padded[:,i,...].unsqueeze(1), z_n))

            y_mavg = gamma*y_prev + (1-gamma)*y_i
            y_prev = y_mavg
            y_list.append(y_mavg)
            #y_list.append(y_i)

            if har_smoothing:
                y_i = y_mavg

            # unsqueeze for time_len=1
            # resqueeze (at another dim) cause hidden output has another dim for uni or bidirect lstm 
            h_i, (h_n, c_n) = self.har.recurrent_layers(y_i.unsqueeze(1), (h_n, c_n))
            h_list.append(h_i.squeeze(1))

            z_n = torch.exp(self.har.action_layers(h_i.squeeze(1))) # these are log_probs, need to make them probs!

            beta_i = self.har.attention_layer(torch.cat([y_i, h_i.squeeze(1)], dim=1))
            betas_list.append(beta_i)
            
            action_lin_i = self.har.action_layers[0](h_i.squeeze(1)) # get linear output without softmax operation
            action_lin_list.append(action_lin_i)
        
        # convert back to packedseq making it equal to compact/rolled version during forward pass
        y_padded = torch.stack(y_list, dim=1)
        y = pack_padded_sequence(y_padded, lengths, batch_first=True)

        h_padded = torch.stack(h_list, dim=1) # B x T x 100
        betas_padded = torch.stack(betas_list, dim=2) # B x 1 x T
        action_lin_padded = torch.stack(action_lin_list, dim=1) # B x T x 45

        # B x 1 x T -> B x 1 x T (softmaxed)
        #padded_alphas = torch.log(betas_padded, dim=2)
        padded_alphas = F.softmax(betas_padded, dim=2)

        # (B x 1 x T) * (B x T x 45) -> (B x 1 x 45) -> (B x 45)
        action_lin_focused = torch.bmm(padded_alphas, action_lin_padded).squeeze(1)

        z_final = self.har.action_layers[1](action_lin_focused)
        
        return (y, z_final)


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
    

    def forward_v0_action_feedback_cleaned(self, x):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        alpha = self.act_0c_alpha #0.1 #0.0001 # TODO: ADD OPTION TO MAKE THIS A TORCH PARAM!!
        z_log_mean = (alpha * z_log) + ((1-alpha) * z_log_)

        y_mean = (alpha * y.data) + ((1-alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        return (y_mean, z_log_mean)
    
    def forward_v0_action_feedback_dual_hpe_har(self, x):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
            v12d
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        # new use different lstm model in second stage!
        z_log = self.har_act(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        #alpha =  #0.1 #0.0001 #
        z_log_mean = (self.act_0c_alpha * z_log) + ((1-self.act_0c_alpha) * z_log_)

        y_mean = (self.act_0c_alpha * y.data) + ((1-self.act_0c_alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        return (y_mean, z_log_mean)
    

    def forward_v0_action_feedback_refined(self, x):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
            v13d
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        #z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        #0.1 #0.0001 # TODO: ADD OPTION TO MAKE THIS A TORCH PARAM!!
        #z_log_mean = (alpha * z_log) + ((1-alpha) * z_log_)

        y_mean = (self.act_0c_alpha * y.data) + ((1-self.act_0c_alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)
        z_log_mean = self.har(y_mean)

        return (y_mean, z_log_mean)
    


    def forward_v0_action_feedback_student_teacher(self, x, return_y_teacher=False, return_type=1):
        '''
            in: x -> (PackedSeq, PackedSeq) -> (Depth, Action)
            out: y,z, z_teacher -> (PackedSeq, Tensor, Tensor) -> (Keypt, TeacherAct, StudentAct)
        '''
        ### x is tuple of two packed sequences
        ### batch_sizes should be same for both
        depth, batch_sizes, action = x[0].data, x[0].batch_sizes, x[1].data
        
        ## pass with conditioning through hpe_act -- teacher should be untrainable!
        ## need to handle case when in valid mode
        y_teacher = self.hpe_act((depth.unsqueeze(1), action)) #self.hpe_act((depth.unsqueeze(1), action))
        y_teacher = PackedSequence(data=y_teacher, batch_sizes=batch_sizes)
        
        # depth, batch_sizes = x.data, x.batch_sizes
        
        ## -- should be ordinary no cond model! -- should be trainable!
        y_student = self.hpe(depth.unsqueeze(1))
        y_student = PackedSequence(data=y_student, batch_sizes=batch_sizes)
        
        if return_type == 1:
            z_teacher = self.har_act(y_teacher)
            z_student = self.har(y_student)
        elif return_type == 2:
            # skip softmax layer, this is done in loss func
            _, (hn_teacher, _) = self.har_act.recurrent_layers(y_teacher)
            z_teacher = self.har_act.action_layers[0](hn_teacher.squeeze(0))

            _, (hn_student, _) = self.har.recurrent_layers(y_student) 
            z_student = self.har.action_layers[0](hn_student.squeeze(0))
        else:
            raise NotImplementedError

        # due to top1 acc using [-1] for calc we supply teacher in middle for now!
        
        return (y_student, z_teacher, z_student) if not return_y_teacher else \
               (y_student, y_teacher, z_teacher, z_student)
    

    def forward_v0_action_feedback_temporal_smoothing(self, x, gamma=0.4):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        alpha = self.act_0c_alpha #0.1 #0.0001 # TODO: ADD OPTION TO MAKE THIS A TORCH PARAM!!
        z_log_mean = (alpha * z_log) + ((1-alpha) * z_log_)

        y_mean = (alpha * y.data) + ((1-alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        ## smooth y_mean
        y_padded, lengths = pad_packed_sequence(y_mean, batch_first=True)
        y_prev = y_padded[:, 0, ...]
        y_list = []
        y_list.append(y_prev)
        for i in range(1, y_padded.shape[1]):
            y_i = y_padded[:, i, ...]
            y_mavg = gamma*y_prev + (1-gamma)*y_i
            y_prev = y_mavg
            y_list.append(y_mavg)
        y_mean_smooth = pack_padded_sequence(torch.stack(y_list, dim=1), lengths, batch_first=True)

        return (y_mean_smooth, z_log_mean)
    




    def forward_v0_action_feedback_eta_zeta(self, x, eta=0.5, zeta=0.5):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        z_log_mean = (zeta * z_log) + ((1-zeta) * z_log_)

        y_mean = (eta * y.data) + ((1-eta) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        return (y_mean, z_log_mean)


    def forward_v0_action_feedback_dual_temporal_atten(self, x):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        # h_ is a packed seq of hidden states
        h_, (_, _) = self.har.recurrent_layers(y_)

        y_h_ = PackedSequence(data=torch.cat([y_.data, h_.data], dim=1), batch_sizes=batch_sizes)
        betas_packed = PackedSequence(data=self.har_act.attention_layer(y_h_.data), batch_sizes=batch_sizes) # ? x 1

        # get logits -- before softmax!
        z_logits_packed = PackedSequence(data=self.har.action_layers[0](h_.data), batch_sizes=batch_sizes)
        
        betas_padded = pad_packed_sequence(betas_packed, batch_first=True)[0].transpose(1,2) # B x 1 x T
        alphas_padded = F.softmax(betas_padded, dim=2) # B x 1 x T
        z_logits_padded = pad_packed_sequence(z_logits_packed, batch_first=True)[0] # B x T x 45

        # (B x 1 x T) * (B x T x 45) -> (B x 1 x 45) -> (B x 45)
        z_logits_focused = torch.bmm(alphas_padded, z_logits_padded).squeeze(1)
        
        z_log_ = self.har.action_layers[1](z_logits_focused) # B x 45 --> focused
        #z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data
        
        
        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        alpha = self.act_0c_alpha #0.1 #0.0001 # TODO: ADD OPTION TO MAKE THIS A TORCH PARAM!!
        z_log_mean = (alpha * z_log) + ((1-alpha) * z_log_)

        y_mean = (alpha * y.data) + ((1-alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        return (y_mean, z_log_mean)



    def forward_v0_action_feedback_logits(self, x):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like...
        _, (h_, _) = self.har.recurrent_layers(y_) 
        z_logit_ = self.har.action_layers[0](h_.squeeze(0))
        z_log_ = self.har.action_layers[1](z_logit_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        
        _, (h, _) = self.har.recurrent_layers(y)
        z_logit = self.har.action_layers[0](h.squeeze(0)) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        
        alpha = self.act_0c_alpha #0.1 #0.0001 # TODO: ADD OPTION TO MAKE THIS A TORCH PARAM!!
        z_logit_mean = (alpha * z_logit) + ((1-alpha) * z_logit_)
        z_log_mean = self.har.action_layers[1](z_logit_mean)

        y_mean = (alpha * y.data) + ((1-alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        return (y_mean, z_log_mean)
    
    def forward_v0_action_feedback_z_arith_mean(self, x):
        '''
            in: x -> PackedSeq
            out: y,z -> (PackedSeq, Tensor)
        '''
        depth, batch_sizes = x.data, x.batch_sizes
        
        ## pass with conditioning
        y_ = self.hpe(depth.unsqueeze(1))
        y_ = PackedSequence(data=y_, batch_sizes=batch_sizes)
        

        # interim action -- output is log softmax need to perform exp to get it one_hot like... 
        z_log_ = self.har(y_)
        z_ = torch.exp(z_log_)

        _, lengths = pad_packed_sequence(x, batch_first=True)
        z_seq_ = pack_sequence([z_[i].repeat(lengths[i],1) for i in range(lengths.shape[0])]).data

        y = self.hpe_act((depth.unsqueeze(1), z_seq_))
        y = PackedSequence(data=y, batch_sizes=batch_sizes) # must wrap as packed_seq before sending to har
        z_log = self.har(y) # note these are logsoftmax numbers, but for argmax operation it doesn't matter
        z     = torch.exp(z_log)


        alpha = self.act_0c_alpha #0.1 #0.0001 # TODO: ADD OPTION TO MAKE THIS A TORCH PARAM!!
        #z_log_mean = (alpha * z_log) + ((1-alpha) * z_log_)

        z_mean = (alpha*z) + ((1-alpha) * z_)
        z_log_mean = torch.log(z_mean)

        y_mean = (alpha * y.data) + ((1-alpha) * y_.data)
        y_mean = PackedSequence(data=y_mean, batch_sizes=batch_sizes)

        return (y_mean, z_log_mean)


    
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
    

    # def on_epoch_train(self, epochs_trained):
        # if getattr(self, 'hpe_act', False):
        #     if self.hpe_act.action_cond_ver == 7.182:
        #         self.act_0c_alpha += 0.2
        #         print("new act 0c alpha", self.act_0c_alpha)

        #         prob = self.hpe_act.equiprob_chance.probs.cpu().item()
        #         if prob < 1.0:
        #             prob += 0.01
        #             self.hpe_act.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([prob]))
        #             print("NEW PROB: ", )
        #if epochs_trained == 15 and getattr(self, 'hpe_act', False):
            #if self.hpe_act.action_cond_ver == 7.182:
                #self.hpe_act.on_epoch_train(epochs_trained)
                
                
                
        # if self.combined_version == '14d':
        #     print("[COMBINED_MODEL] HPE_OLD_GRAD: %s, HPE_ACT_OLD_GRAD: %s" % \
        #           (next(self.hpe.parameters()).requires_grad, next(self.hpe_act.parameters()).requires_grad))
        #     print("[COMBINED_MODEL] Trainable Param Old:", self.param_count())
        #     grad_required = next(self.hpe.parameters()).requires_grad
        #     grad_required = not grad_required # invert
        #     for param in self.hpe.parameters():
        #         # invert hpe params
        #         param.requires_grad = grad_required
        #     grad_required = not grad_required
        #     for param in self.hpe_act.parameters():
        #         # the opposite of what happened to HPE
        #         param.requires_grad = grad_required
        #     print("[COMBINED_MODEL] HPE_NEW_GRAD: %s, HPE_ACT_NEW_GRAD: %s" % \
        #           (next(self.hpe.parameters()).requires_grad, next(self.hpe_act.parameters()).requires_grad))
        #     print("[COMBINED_MODEL] Trainable Param New:", self.param_count())
            # switch hpe vs hpe act trainable...
            #print(self.trainable_)