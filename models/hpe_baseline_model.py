import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict

## remove later
import torch

from models import BaseModel


### Notes planes are like channels!
### so 1 plane = 1 channel e.g. 'B' in RGB

### kernel_size === kernel_size this is defined as a single int usually but can be tuple.

def get_padding_size(kernel_size, pad_type="valid"):
    if pad_type == "valid":
        return 0
    elif pad_type == "same":
        ## this eqn is valid if stride is always assumed to be 1
        ## if stide is not 1 then this gives output size as if stride was not applied
        ## sometimes its desired to use stride to do downsampling
        return ((kernel_size-1)//2)
    else:
        raise NotImplementedError("Padding must be either \"valid\" or \"same\"")


class ExtendedSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = super(ExtendedSequential, self).__dict__ #get super's dict with init stuff and functions
        #print(super(ExtendedSequential, self).__dict__)
        #print(self.__dict__)
    # def __getattr__(self, name):
    #     print("Attrib..: ", name)
    #     if name == '_modules':

    #     val = super(ExtendedSequential, self).__getattr__(name)
    #     print("val: ", val)
    #     return val

    def forward(self, input):
        for module in self._modules.values():
            input = module(input) if not isinstance(input, tuple) \
                    else (module(input[0]), input[1]) if not getattr(module, 'accept_tuple', False) \
                    else (module(input), input[1]) # consume + bypass
            
            # special case for nests extended seq
            input = input[0] if (isinstance(input[0], tuple) and getattr(module, 'accept_tuple', False)) else input
        return input


class LinearEmbeddingBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, use_bias=True,
                 pre_onehot=True, use_embed=False):
        '''
            A simple embedding block with support for long/one_hot and nn.Embedding/nn.Linear

            M -> input_features
            O -> output features

            Supported inputs:
            NxM encodings of M-feature inputs (pre_onehot must be true)
            NxM one-hot encodings of M-unique_categories
            N indices of M-unique_categories
        '''
        super(LinearEmbeddingBlock, self).__init__()
        in_features = input_features
        out_features = output_features

        self.pre_onehot = pre_onehot
        self.use_embed = use_embed
        self.eye = torch.eye(in_features) # for one-hot conversion
        self.main_layer = nn.Linear(in_features, out_features, bias=use_bias) if not use_embed \
                          else nn.Embedding(in_features, out_features)
    

    def forward(self, context):
        if not self.pre_onehot and not self.use_embed:
            # N (float) -> N (long) -> Nx45 (45 dim one-hot)
            context = self.eye[context.long()].to(context.device, context.dtype)
        elif self.pre_onehot and self.use_embed:
            context = torch.argmax(context, dim=1)
        return self.main_layer(context)

class FiLMBlock(nn.Module):
    def __init__(self, num_unique_contexts: int, target_channels: int, use_embed=True, use_bias=True, context_dim=1,
                 use_attention=False, attention_dim=(None,None), use_onehot=False, pre_onehot=True,
                 accept_tuple_input=False):
        '''
            Arguments:
                num_unique_contexts: for embedd dict size, how many possible context values
                target_channels: what you aim to get out of embed for each gammas and betas
                context_dim: for fc layer
            Return:
                output : feature maps modulated with betas and gammas (FiLM parameters)

            Inspired from: https://github.com/ap229997/Neural-Toolbox-PyTorch/blob/master/film_layer.py
            
            However we don't use dynamic fc
            We also do (gamma)* not (1+gamma)* ?? maybe try both

            Note attention cannot be implemented as no way to keep track of spatial dim
        '''
        super(FiLMBlock, self).__init__()

        self.target_channels = target_channels
        self.use_embed = use_embed
        self.use_attention = use_attention
        self.use_onehot = use_onehot
        self.pre_onehot = pre_onehot
        #print("FILM CALLED WITH TARGET CHANNELS: ", self.target_channels)

        self.eye = torch.eye(num_unique_contexts) # for use later with onehot encodin

        if self.use_onehot: self.use_embed = False # mutually exclusive

        self.main_layer = nn.Embedding(num_unique_contexts, target_channels*2) \
                            if self.use_embed \
                            else nn.Linear((context_dim if not self.use_onehot else num_unique_contexts),
                                           target_channels*2, bias=use_bias)
        

        if self.use_attention and attention_dim != (None, None):
            self.attention_layer = nn.Sequential(OrderedDict([
                nn.Linear(context_dim, attention_dim[0]*attention_dim[1], bias=use_bias),
                nn.Softmax(dim=1),
            ]))

        self.accept_tuple = accept_tuple_input


    def forward(self, feature_maps, context=None):
        ## TO EDIT!!
        #self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape
        # FiLM parameters needed for each channel in the feature map
        # hence, feature_size defined to be same as no. of channels
        #self.feature_size = feature_maps.data.shape[1]

        # if context is None:
        #     ## note this is used for dynamic condioning
        #     return feature_maps
        if self.accept_tuple and isinstance(feature_maps, tuple):
            #print("A TUPLE!!")
            feature_maps, context = feature_maps
            #print("FEATURE_SIZE:", feature_maps.shape)
            #print("CONTEXT_SIZE:", context.shape)

        if self.use_embed:
            # N.float -> N.long
            if self.pre_onehot:
                context = torch.argmax(context, dim=1)
            context = context.long()
        if self.use_onehot and not self.pre_onehot:
            context = self.eye[context.long()].to(context.device, context.dtype)
        elif not self.use_embed and not self.use_onehot:
            if len(context.shape) == 1:
                # N -> N x 1
                context = context.unsqueeze(1)
        
        if len(feature_maps.shape) != 4:
            # this is probably a linear_layer, make it look like a conv layer with H=W=1
            #print("INPUT IS 1D")
            input_1d = True
            feature_maps = feature_maps.unsqueeze(2).unsqueeze(3)
        else:
            input_1d = False
        #print("***FILM WAS CALLED!! X_CHANNELS, TARGET_CHANNELS: %d, %d***" % (feature_maps.shape[1], self.target_channels))
        # linear transformation of context to FiLM parameters
        # ? = 1 or > 1
        # Nx? -> Nx2C -> Nx2Cx1x1 -> Nx2CxWxH
        film_params = self.main_layer(context).unsqueeze(2).unsqueeze(3).expand(-1,-1,feature_maps.shape[2], feature_maps.shape[3])

        # stack the FiLM parameters across the spatial dimension
        # film_params = torch.stack([film_params]*self.height, dim=2)
        # film_params = torch.stack([film_params]*self.width, dim=3)

        # slice the film_params to get betas and gammas
        # Nx2CxWxH -> NxCxWxH & NxCxWxH
        gammas = film_params[:, :self.target_channels, :, :]
        betas = film_params[:, self.target_channels:, :, :]

        if feature_maps.shape[1] != self.target_channels:
            print("***SHAPE MISMATCH***")

        if feature_maps.shape != gammas.shape or feature_maps.shape != betas.shape:
            print("SHAPES (FMAP; GAMMAS; BETAS):", feature_maps.shape, gammas.shape, betas.shape)

        # modulate the feature map with FiLM parameters
        output = (1 + gammas) * feature_maps + betas

        if self.use_attention:
            attention = self.attention_layer(output)
            output = output * attention.reshape(-1,1,feature_maps.shape[2], feature_maps.shape[3])

        if input_1d:
            ## if it was a 1d input make it 1d again
            #print("OUTPUT SHAPE OLD", output.shape)
            output = output.reshape(-1, self.target_channels) # for some reason squeeze is not working here...
            #print("OUTPUT SHAPE NEW", output.shape)
        return output


class LinearDropoutBlock(nn.Module):
    ## BN + ReLU # alpha = momentum?? TODO: check
    ## this one causes problem in back prop
    def __init__(self, in_features, out_features, apply_relu=True, dropout_prob=0, use_bias=True):
        super(LinearDropoutBlock, self).__init__()
        self.block = ExtendedSequential(
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


class BNReLUBlock(nn.Module):
    ## BN + ReLU # alpha = momentum?? TODO: check
    def __init__(self, in_channels, eps=0.0001, momentum=0.1, context_layer=None):
        super(BNReLUBlock, self).__init__()

        self.block = ExtendedSequential(
            nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum), 
            nn.ReLU(True),
        )

        self.context_layer = context_layer
        self.accept_tuple = True

        if context_layer is not None:
            #print("SET_BN_WEIGHT_FIXED")
            self.block[0].weight.requires_grad = False
            self.block[0].bias.requires_grad = False
            #print("[BR_BLOCK] In_CHANNELS VS CONTEXT:", in_channels, context_layer.target_channels)
        
    
    def forward(self, x):
        ## add support for gammas betas here!
        # if isinstance(x, tuple):
        #     print("BN_RELU_BLOCK_GOT_TUPLE")
        # if isinstance(x, tuple) and self.context_layer is not None:
        #     print("have context")
        # if self.context_layer is not None:
        #     print("[BR_BLOCK] TRAIN_CHANNELS VS CONTEXT VS IS TUPLE:", x.shape[1], 
        #           self.context_layer.target_channels, isinstance(x, tuple))
        return self.block(x) if not isinstance(x, tuple) \
                             else self.block[1](self.context_layer(self.block[0](x[0]), x[1])) if self.context_layer is not None \
                             else self.block(x[0])


class ConvPoolBlock(nn.Module):
    ## TODO: add docs
    def __init__(self, in_channels, out_channels, kernel_size, pool_size,
                        stride=1,
                        doBatchNorm=False, doReLU=False, pad="valid", use_cond=False):
        super(ConvPoolBlock, self).__init__()
        self.padding = get_padding_size(kernel_size, pad)
        self.use_cond = False

        self.block = ExtendedSequential()
        self.block.add_module("conv_1",
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding)
        )
        ## double check to make sure this is as expected
        ## currently w ~ N(0, sqrt(2/var_weight_{fwd_pass}))
        I.kaiming_normal_(self.block.conv_1.weight, a=0, nonlinearity='relu')

        self.block.add_module("pool_1", 
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )


        if doReLU:
            self.block.add_module("relu_1", nn.ReLU(True))
            ## maybe need to add normailsation

        if doBatchNorm:
            self.block.add_module("batchnorm_1", nn.BatchNorm2d(out_channels))
        

        self.film = None # placeholder
    
    def forward(self, x):
        return self.block(x)
            # if not isinstance(x, tuple) \
            # else (self.block(x[0]), x[1]) if not self.use_cond \
            #     else self.film(x)


class BNReLUConvBlock(nn.Module):
    ## BN + ReLU + Conv, no pooling, use 2x2 stride aka stride=2 to downsample layer, also padding is fixed to same
    def __init__(self, in_channels, out_channels, kernel_size, stride, context_layer = None):
        super(BNReLUConvBlock, self).__init__()
        # all resnet blocks have got "same" paddining
        # however if stride == 2 then the output will be input/2 but this is ok
        self.padding = get_padding_size(kernel_size, "same")

        self.block = ExtendedSequential(
            BNReLUBlock(in_channels, context_layer=context_layer),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding)
        )

        #self.film_layer = context_layer # could be none type
        ## double check to make sure this is as expected
        ## do all weight setting in end!! i.e. based on layer types
        self.accept_tuple = True
    
    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, middle_stride, context_layers_dict=None):
        super(ResBlock, self).__init__()


        if (out_channels != in_channels*2 and out_channels != in_channels):
            raise NotImplementedError("Only factor of {1, 2} upsampling of channels is implemented.")
        
        if (middle_stride != 1 and middle_stride != 2):
            raise NotImplementedError("Only middle_stride={1, 2} is supported.")
        
        #if context_layers_dict != None:
        #film_layer = context_layers_dict#['emb_cin'], context_layers_dict['emb_cout']

        '''
            In Resnet we typically perform H(x) = R(x) + x

            Here H(x) is the actual output required or true output distribution.
            R(x) is what the model actually learns, this is called the 'residue' or 'difference'
            between input 'x' and output 'H(x)' or 'res branch' i.e. R(x) = H(x) - x

            By learning the difference, just like coding schemes and compression techniques in DIP its lower
            magnitude of information and makes things easier to learn and 'graspable' by the network.

            Note: x is usally denoted skip-connection as it simply skips input to output stage.
        '''

        ## bottleneck type
        ## all 3 layers pad s.t. if stride=1 then input==output
        ## i.e. same padding without taking stride into account
        self.res_branch = ExtendedSequential(
            ## context object is only used as a placeholder to disable batchnorm training so bn happens but without trainable weights!
            ## first downsample channels e.g. 32x32x32 -> 32x32x16
            BNReLUConvBlock(in_channels, out_channels//4, kernel_size=1, stride=1, context_layer=context_layers_dict[0]),

            ## then downsample input e.g. 32x32x16 -> 16x16x16
            ## the bottleneck
            ## this part is different from deep-prior but is used everywhere else
            BNReLUConvBlock(out_channels//4, out_channels//4, kernel_size=3, stride=middle_stride, context_layer=context_layers_dict[1]),

            ## now upsample channels e.g. 16x16x16 -> 16x16x64
            # so width&hright reduced, depth increased
            # note: by default last layer must upsample bottleneck by 4
            # as convention so this method works if input_c==output_c
            # or even if input_c*2==output_c
            BNReLUConvBlock(out_channels//4, out_channels, kernel_size=1, stride=1, context_layer=context_layers_dict[1]), #context_layers_dict[1]
        )

        if in_channels == out_channels:
            ## this is used when in_channels == out_channel
            self.skip_con = ExtendedSequential()
        else:
            self.skip_con = ExtendedSequential(
                # context_layer=context_layers_dict[0] -- note this is only to disable bn TODO: fix this and explicitly disable batchnorm
                BNReLUConvBlock(in_channels, out_channels, kernel_size=1, stride=middle_stride)
            )
        
        self.accept_tuple = True
        self.out_film_layer = context_layers_dict[0] if context_layers_dict is not None else None # context_layers_dict[2] # or 0
    
    def forward(self, x):
        # make sure to pass along the context
        # note bug here!!
        # x can be of type tuple, which can cause a serious error where the '+' operation is
        # applied on tuple and not tensor so the tuple just gets extended to like size 4 and the addition operation
        # is actually never performed, to fix this we must ensure that no tuple passes through the inidividual branches
        # this will ensure that the output of each branch is a tensor and thus the addition is a tensor addition
        x = self.out_film_layer(x[0], x[1]) if self.out_film_layer is not None \
            else x[0] if isinstance(x, tuple) else x  #(self.out_film_layer(x[0], x[1]), x[1])
        return self.res_branch(x) + self.skip_con(x)

        #x,a=x[0],x[1]
        #x = self.res_branch(x) + self.skip_con(x)
        #return self.out_film_layer(x, a) if self.out_film_layer is not None else x

        


class ResGroup(nn.Module):
    def __init__(self, in_channels, out_channels, first_middle_stride=1, blocks=5, use_context=False,
                 context_params={'num_unique_contexts': 45, 'use_embed': True, 'use_onehot': True}):
        super(ResGroup, self).__init__()
        ## makes a residual layer of n_blocks
        ## expansion is always 4x of bottleneck
        if use_context:
            context_layers_dict = OrderedDict([
                ('emb_cin', FiLMBlock(target_channels=in_channels, **context_params)),
                ('emb_cout', FiLMBlock(target_channels=out_channels, **context_params)),
                ('emb_cout_quarter', FiLMBlock(target_channels=out_channels//4, **context_params))
            ])
        else:
            context_layers_dict = {'emb_cin': None, 'emb_cout': None, 'emb_cout_quarter': None}

        # first_middle_stride: the stride of bottleneck layer of first resBlock
        layers = []
        layers.append(ResBlock(in_channels, out_channels, middle_stride=first_middle_stride, 
                               context_layers_dict=(context_layers_dict['emb_cin'], context_layers_dict['emb_cout_quarter'], 
                                                    context_layers_dict['emb_cout'])))
        for _ in range(1, blocks):
            ## all of these will have same dim
            layers.append(ResBlock(out_channels, out_channels, middle_stride=1, 
                                   context_layers_dict=(context_layers_dict['emb_cout'], context_layers_dict['emb_cout_quarter'],
                                                        context_layers_dict['emb_cout'])))
        
        self.resblocks = ExtendedSequential(*layers)
        self.accept_tuple = True
    
    def forward(self, x):
        return self.resblocks(x) # bypass is handled by nnseq u dont do it here

class PCADecoderBlock(nn.Module):
    '''
        Useful as standalone for metric calc
    '''
    def __init__(self, num_joints=21, num_dims=3, pca_components=30,
                 no_grad=True):
        self.num_joints, self.num_dims = num_joints, num_dims
        super(PCADecoderBlock, self).__init__()

        ## back projection layer
        self.main_layer = nn.Linear(pca_components, self.num_joints*self.num_dims)

        if no_grad:
            self.main_layer.weight.requires_grad = False
            self.main_layer.bias.requires_grad = False
    
    def forward(self, x, reshape=True):
        return self.main_layer(x).view(-1, self.num_joints, self.num_dims) \
            if reshape else self.main_layer(x)
    

    def initialize_weights(self, weight_np, bias_np):
        '''
            Sets weights as transposed of what is supplied and bias is set as is
        '''
        self.main_layer.weight = \
            torch.nn.Parameter(torch.tensor(weight_np.T, dtype=self.main_layer.weight.dtype, device=self.main_layer.weight.device))
        self.main_layer.bias = \
            torch.nn.Parameter(torch.tensor(bias_np, dtype=self.main_layer.bias.dtype, device=self.main_layer.weight.device))
        
        self.main_layer.weight.requires_grad = False
        self.main_layer.bias.requires_grad = False





class DeepPriorPPModel(BaseModel): #nn.Module
    def __init__(self, input_channels=1, num_joints=21, num_dims=3, pca_components=30, dropout_prob=0.3,
                 train_mode=True, weight_matx_np=None, bias_matx_np=None, init_w=True, predict_action=False,
                 action_classes=45, action_cond_ver=0, dynamic_cond=False, res_blocks_per_group=5,
                 eval_pca_space=False, train_pca_space=False, fixed_pca=True, preload_pca=True,
                 action_equiprob_chance=1.0):
        self.num_joints, self.num_dims = num_joints, num_dims
        self.predict_action = predict_action
        self.action_cond_ver = action_cond_ver
        self.dynamic_cond = dynamic_cond
        self.eval_pca_space = eval_pca_space # when TRUE output is PCA (default) at test, else output at test is num_dims*num_joints
        self.train_pca_space = train_pca_space # this is generally always true but we may need to keep it to false for combined model
        self.preload_pca = preload_pca # a trainer class 'may' preload pca components
        self.fixed_pca = fixed_pca # whether to make pca layer trainable this needs to be set during init otherwise opt wouldn't register

        super(DeepPriorPPModel, self).__init__()
        channelStages = [32, 64, 128, 256, 256]
        strideStages = [2, 2, 2, 1]

        print("[HPE_MODEL] Action Cond Ver: %0.2f" % self.action_cond_ver)
        context_params = {}
        
        # new action conditioning info
        use_resnet_conditioning = False
        use_lin_conditioning = False
        if self.action_cond_ver == 1:
            input_channels += 1
        elif self.action_cond_ver == 2:
            ## in order to make this work must remove transform actiononehot encoder!
            num_channels = 45
            input_channels += num_channels # so 15+1 = 16 for first input
            self.embed_layer = LinearEmbeddingBlock(action_classes, num_channels, pre_onehot=True, use_embed=True)
        elif self.action_cond_ver == 3:
            # Note: Action cond ver 3 re-written
            num_channels = 45
            input_channels += num_channels # so 15+1 = 16 for first input
            self.embed_layer = LinearEmbeddingBlock(action_classes, action_classes, pre_onehot=True)
            # a test for gammas, betas
            #num_embed = 2
            #self.embed_layer = nn.Embedding(action_classes, 2)
        elif self.action_cond_ver == 4:
            self.film_layer = FiLMBlock(action_classes, 1)
        elif self.action_cond_ver == 4.1:
            self.film_layer = FiLMBlock(action_classes, 1, use_embed=False, use_onehot=True)
        elif self.action_cond_ver == 5:
            use_resnet_conditioning = True
            context_params = {'num_unique_contexts': 45, 'use_embed': True, 'use_onehot': False}
        elif self.action_cond_ver == 6:
            use_resnet_conditioning = True
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True}
        elif self.action_cond_ver == 7:
            ### use of equiprob conditioning ###
            use_resnet_conditioning = True
            #use_lin_conditioning = True
            self.use_equiprob_cond = True
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True}
        elif self.action_cond_ver == 7.1:
            ### use of true conditioning on linear // NO CONV###
            #use_resnet_conditioning = True
            use_lin_conditioning = True
            self.use_equiprob_cond = False #True
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        elif self.action_cond_ver == 7.11:
            ### use of true conditioning on linear + conv ###
            use_resnet_conditioning = True
            use_lin_conditioning = True
            self.use_equiprob_cond = False #True
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        elif self.action_cond_ver == 7.12:
            ### use of true conditioning on linear ++  // NO CONV ###
            use_resnet_conditioning = False
            use_lin_conditioning = 2 #True :: 2 makes it also use the lowest layer...
            self.use_equiprob_cond = False #True
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        elif self.action_cond_ver == 7.121:
            ### use of ALL conditioning on linear++ + conv ###
            use_resnet_conditioning = True
            use_lin_conditioning = 2
            self.use_equiprob_cond = False #True
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        elif self.action_cond_ver == 7.13:
            ### use of true conditioning on linear ++  // NO CONV ###
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            use_resnet_conditioning = False
            use_lin_conditioning = True # do not add condition in last layer
            self.use_equiprob_cond = False #True
            LABEL_FLIP_PROB = action_equiprob_chance #0.95 # 0.4 # 0.8
            self.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([LABEL_FLIP_PROB]))
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        elif self.action_cond_ver == 7.14:
            ### use of true conditioning on linear ++  // NO CONV ###
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            use_resnet_conditioning = False
            use_lin_conditioning = 2 # do not add condition in last layer
            self.use_equiprob_cond = False #True
            LABEL_FLIP_PROB = action_equiprob_chance #0.4
            print("[HPE] ACT_COND_ 7.14 LABEL_FLIP_PROB:", LABEL_FLIP_PROB)
            self.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([LABEL_FLIP_PROB]))
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        elif self.action_cond_ver == 7.15:
            ### use of true conditioning on linear ++  // NO CONV ###
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            use_resnet_conditioning = True
            use_lin_conditioning = False # do not add condition in last layer
            self.use_equiprob_cond = False #True
            LABEL_FLIP_PROB = action_equiprob_chance #0.99 #0.95 #0.999 #0.90 # 0.4 # 0.8
            print('[HPE] USING LABEL_FLIP_PROB:', LABEL_FLIP_PROB)
            self.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([LABEL_FLIP_PROB]))
            #self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
        #### new ####
        elif self.action_cond_ver == 7.16:
            
            ## this is like v7 but on linear cond
            ### use of equiprob conditioning ###
            #use_resnet_conditioning = True
            use_lin_conditioning = True
            self.use_equiprob_cond = True
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True, 'accept_tuple_input': True}
        elif self.action_cond_ver == 7.17:
            ### use of equiprob conditioning ###
            ## this is like v7 but on linear cond++
            #use_resnet_conditioning = True
            use_lin_conditioning = 2
            self.use_equiprob_cond = True
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True, 'accept_tuple_input': True}
        elif self.action_cond_ver == 7.18:
            ### use of equiprob conditioning ###
            ## this is like v7 but on linear+res cond
            use_resnet_conditioning = True
            use_lin_conditioning = True
            self.use_equiprob_cond = True
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True, 'accept_tuple_input': True}
        elif self.action_cond_ver == 7.181 or self.action_cond_ver == 7.182:
            ### use of equiprob conditioning with variable chance ###
            ## this is like v7.15 but on linear+res cond
            use_resnet_conditioning = True
            use_lin_conditioning = True
            self.use_equiprob_cond = False #True
            LABEL_FLIP_PROB = action_equiprob_chance #0.99 #0.95 #0.999 #0.90 # 0.4 # 0.8
            print('[HPE] USING LABEL_FLIP_PROB:', LABEL_FLIP_PROB)
            self.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([LABEL_FLIP_PROB]))
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True, 'accept_tuple_input': True}
        elif self.action_cond_ver == 7.19:
            ### use of equiprob conditioning ###
            ## this is like v7 but on linear++ + res cond
            use_resnet_conditioning = True
            use_lin_conditioning = 2
            self.use_equiprob_cond = True
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True, 'accept_tuple_input': True}
        ### end new ###
        elif self.action_cond_ver == 7.2:
            # this is like act_cond_6 but with also lin cond
            use_resnet_conditioning = True
            use_lin_conditioning = True
            self.use_equiprob_cond = False
            self.equiprob_act = (1/action_classes) * torch.ones(action_classes)
            context_params = {'num_unique_contexts': 45, 'use_embed': False, 'use_onehot': True,
                              'accept_tuple_input': True}
            


        self.main_layers = ExtendedSequential(OrderedDict([
            ('conv_pool_1', ConvPoolBlock(input_channels, channelStages[0],
                kernel_size=5, pool_size=2, pad="same", stride=1)),
            ('res_group_1', ResGroup(
                channelStages[0], channelStages[1], first_middle_stride=strideStages[0],
                use_context=use_resnet_conditioning, context_params=context_params,
                blocks=res_blocks_per_group)),
            ('res_group_2', ResGroup(
               channelStages[1], channelStages[2], first_middle_stride=strideStages[1],
               use_context=use_resnet_conditioning, context_params=context_params,
               blocks=res_blocks_per_group)),
            ('res_group_3', ResGroup(
               channelStages[2], channelStages[3], first_middle_stride=strideStages[2],
               use_context=use_resnet_conditioning, context_params=context_params,
               blocks=res_blocks_per_group)),
            ('res_group_4', ResGroup(
               channelStages[3], channelStages[4], first_middle_stride=strideStages[3],
               use_context=use_resnet_conditioning, context_params=context_params,
               blocks=res_blocks_per_group)),
            ('bn_relu_1', BNReLUBlock(channelStages[4])),
            
        ]))

        if not use_lin_conditioning:
            self.linear_layers = ExtendedSequential(OrderedDict([
                #sadly the input here must be computed manually
                ('lin_relu_1', LinearDropoutBlock(8*8*256, 1024, 
                    apply_relu=True, dropout_prob=dropout_prob)),
                ('lin_relu_2', LinearDropoutBlock(1024, 1024, 
                    apply_relu=True, dropout_prob=dropout_prob)),
                ('lin_relu_3_pca', LinearDropoutBlock(1024, pca_components, 
                    apply_relu=False, dropout_prob=0.)),  # 0 dropout => no dropout layer to add
            ]))
        else:
            self.linear_layers = ExtendedSequential(OrderedDict([
                ('lin_relu_1', LinearDropoutBlock(8*8*256, 1024, 
                    apply_relu=True, dropout_prob=dropout_prob)),
                ('film_lin_1', FiLMBlock(target_channels=1024, **context_params)),
                ('lin_relu_2', LinearDropoutBlock(1024, 1024, 
                    apply_relu=True, dropout_prob=dropout_prob)),
                ('film_lin_2', FiLMBlock(target_channels=1024, **context_params)),
                ('lin_relu_3_pca', LinearDropoutBlock(1024, pca_components, 
                    apply_relu=False, dropout_prob=0.)),  # 0 dropout => no dropout layer to add
                
            ]))
            if use_lin_conditioning == 2:
                print("[HPE] ALSO ADDED EXTRA LAYER OF LIN")
                self.linear_layers.add_module('film_lin_3', FiLMBlock(target_channels=pca_components, **context_params))

        ### add action layer but dont append to linear layers
        ### predict_action
        if self.predict_action:
            # NEW
            self.action_layer = ExtendedSequential(OrderedDict([
                ('lin', nn.Linear(in_features=pca_components, out_features=action_classes)),
                ('softmax', nn.LogSoftmax(dim=1))
            ]))

        ## back projection layer
        self.final_layer = PCADecoderBlock(num_joints=num_joints,
                                           num_dims=num_dims)
        
        self.register_buffer('is_pca_loaded', torch.zeros(1, device=next(self.parameters()).device))

        # currently done according to resnet implementation (He init conv layers)
        # also init last layer with pca__inverse_transform vals and make it fixed
        # option to delay init for a future time
        if init_w:
            self.initialize_weights(w_final=weight_matx_np, b_final=bias_matx_np)


    def forward(self, x):
        #x = x.float()   # cast to float,temp fix..., later whole shuld be float
        #print('TEST DATA NEW: ', len(x), x[0].shape, x[1].shape)
        if not isinstance(x, tuple):
            # by default ensure x is a tuple
            x, a = (x, None)
        else:
            # if isinstance(x, tuple):
            ## new depth+action
            ## both these elements are tensors but of different types
            x, a = x # (dpt, action) in tuple
            # quit()
        # overwrite a object with relevant data based on action type

        if self.action_cond_ver == 0:
            a = None
        elif self.action_cond_ver == 1:
            ## a simple embedding methoddoesn't require long
            ## if long is needed it should be back converted here,
            ## long -> float -> long is safe for ints no data lost just a bit ineff
            # [BATCH_SIZE] -> [BATCH_SIZE, 1, 1, 1] -> [BATCH_SIZE, 1, 128, 128]
            a_idx = torch.argmax(a, dim=1).float() #encodings are one hot by default convert io indices
            a = a_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = torch.cat((x, a), 1)  # concat on channel axis -> [BATCH_SIZE, 2, 128, 128]
        elif self.action_cond_ver == 2:
            # [BATCH_SIZE] -> [BATCH_SIZE, 45] -> [BATCH_SIZE, 45, 1, 1] -> [BATCH_SIZE, 45, 128, 128]
            a = self.embed_layer(a.long()).unsqueeze(2).unsqueeze(3).expand(-1,-1,x.shape[2], x.shape[3])
            x = torch.cat((x, a), 1)  # concat on channel axis -> [BATCH_SIZE, 46, 128, 128]
            #print("A_SHAPE", a.shape)
        elif self.action_cond_ver == 3:
            # film in the beginning only.... NOTE THIS IS REPURPOSED NOW NO LONGER FILM!!
            # THIS NOW ONE HOT COMPATIBLE EMBEDDING
            # a = self.embed_layer(a.long()).unsqueeze(2).unsqueeze(3).expand(-1,-1,x.shape[2], x.shape[3])
            # gammas = a[:, 0, :, :].unsqueeze(1)
            # betas = a[:, 1, :, :].unsqueeze(1)
            # x = (1+gammas)*x + betas
            # [BATCH_SIZE] -> [BATCH_SIZE, 45] -> [BATCH_SIZE, 45, 1, 1] -> [BATCH_SIZE, 45, 128, 128]
            a = self.embed_layer(a).unsqueeze(2).unsqueeze(3).expand(-1,-1,x.shape[2], x.shape[3])
            x = torch.cat((x, a), 1)  # concat on channel axis -> [BATCH_SIZE, 46, 128, 128]
        elif self.action_cond_ver == 4 or self.action_cond_ver == 4.1:
            # film in the beginning only.... now done properly....
            x = self.film_layer(x, a)
        elif self.action_cond_ver == 5 or self.action_cond_ver == 6:
            pass # all config set during init
        elif self.action_cond_ver == 7.182:
            act_samples = []
            for i in range(x.shape[0]):
                if bool(self.equiprob_chance.sample().item()):
                    act_samples.append(self.equiprob_act.to(x.device, x.dtype))
                else:
                    act_samples.append(a[i])
            a = torch.stack(act_samples, dim=0)
        elif self.action_cond_ver >= 7 and self.action_cond_ver < 7.2:
            if self.action_cond_ver == 7.13 or self.action_cond_ver == 7.14 or self.action_cond_ver == 7.15 \
                or self.action_cond_ver == 7.181:
                ## do some randoming...
                
                # probabilistic
                self.use_equiprob_cond = bool(self.equiprob_chance.sample().item())
                    

            
            # [7.0, 7.2)
            if self.use_equiprob_cond:
                a = self.equiprob_act.unsqueeze(0).expand(x.shape[0], -1).to(x.device, x.dtype)
            else:
                pass # use actual action
        elif self.action_cond_ver == 7.2:
            # dynamic equiprob ..?
            # try torch bernoulli random....
            pass
        else:
            print("Input Len:", len(x), "Input Type:", type(x),
                    "Depth Shape:", x[0].shape, "Action Shape:", x[1].shape)
            raise NotImplementedError("Unknown ActionCond Version")

        # from henceforth x is always of type tuple
        #if a is None:
        #    x = self.linear_layers(self.main_layers(x).view(-1, 8*8*256))
        #else:   
        x = self.main_layers((x, a))
        
        ## must convert x into (n_samples, 8*8*256 vector)
        x = (x[0].view(-1, 8*8*256), x[1])
        
        x = self.linear_layers(x)[0]
        # discard action info here
        # if self.action_cond_ver >= 7:
        #     for layer in self.linear_layers:
        #         if isinstance(layer, FiLMBlock):
        #             x = (layer(x[0], x[1]), x[1])
        #             print("shapes:", x[0].shape, x[1].shape)
        #         else:
        #             x = (layer(x[0]), x[1])
        #     x = x[0]  # discard action info here

        y = self.action_layer(x) if self.predict_action else None
        # usually people use self.training which is built in but in our case
        # that would lead to val returining y in keypoint space
        # rather than oca space
        # reshape (63,) -> (21, 3) done automatically

        if ((self.training and not self.train_pca_space) or (not self.training and not self.eval_pca_space)):
            x = self.final_layer(x, reshape=False)
        
        if self.predict_action:
            x = (x,y)
        
        return x

        # return \
        #     (x if self.training and  else self.final_layer(x)) if not self.predict_action \
        #         else ((x, y) if self.train_mode else (self.final_layer(x), y))
    
    def forward_eval(self, x):
        ## use this function at evaluation i.e. testing
        ## in this case output is self.num_joints*self.num_dims
        ## this is done by passing through an additional layer
        ## this last layer has PRE-TRAINED WEIGHTS FROM PCA
        ## the weight matrix is U
        ## the bias is the y_gt3D_train_MEAN vector
        ## U is (sorted evect matx as col) from covar matx of y_gt3D_train

        ## only pass through these layers when not training
        x = self.forward(x)
        x = self.final_layer(x)
        return x.view(-1, self.num_joints, self.num_dims)    #optional

    # TODO: change this to as done in deep-prior?
    # currently as porvided by torchvision resnet implementation
    def initialize_weights(self, w_final=None, b_final=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Embedding): # untested // can also do on nn.Linear // note this is intrusive to change
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        
        # init final layer weights and make them fixed -- used only for test error
        if (w_final is not None and b_final is not None and self.preload_pca):
            # we use torch.tensor() method to create a copy of the array provided.
            # as this method always copies data
            print('[HPE] Init PCA Weights and Bias during global init...')
            self.final_layer.initialize_weights(w_final, b_final)
        if self.fixed_pca:
            print('[HPE] Setting PCA layer as fixed')
            for param in self.final_layer.parameters():
                param.requires_grad = False # make it fixed

    def initialize_pca_layer(self, w,b, ensure_fixed_weights=True):
        if self.preload_pca:
            if self.is_pca_loaded == torch.zeros(1, device=next(self.parameters()).device):
                print('[HPE] Init PCA Weights and Bias only...')
                self.final_layer.initialize_weights(w, b)
                if ensure_fixed_weights:
                    print("Making PCA Weights Fixed in Final Layer")
                    for param in self.final_layer.parameters():
                        param.requires_grad = False # make it fixed
                else:
                    print('[HPE] Warning PCA weights may or may not be fixed...')
                self.is_pca_loaded = torch.ones(1, device=next(self.parameters()).device)
            else:
                print("[HPE] PCA Trainable?", next(self.final_layer.parameters()).requires_grad)
                print("[HPE] is_pca_loaded: %a => PCA already loaded from global save file so it won't be reloaded." % self.is_pca_loaded)
        else:
            print('[HPE] Warning: Model has a PCA layer but preloading PCA is turned off.')

    def on_epoch_train(self, epochs_trained):
        if epochs_trained >= 10 and self.dynamic_cond and self.action_cond_ver == 5:
            print("[HPE_MODEL] Turning off action cond (5 -> 0)!!")
            self.action_cond_ver = 0
        
        # if epochs_trained == 15 and self.action_cond_ver == 7.182:
        #     # note here you should ckech first whats current prob and if it is not 0.01 thenn make it that
        #     # or slowly decay...
        #     print("[HPE_MODEL] Making equiprob act prob lower 0.80!!")
        #     self.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([0.80]))

        # if epochs_trained % 5 and self.action_cond_ver == 7.15:
        #     # note here you should ckech first whats current prob and if it is not 0.01 thenn make it that
        #     # or slowly decay...
        #     print("[HPE_MODEL] Making equiprob act prob high!")
        #     print("[HPE_MODEL] Old Prob: %f" % self.equiprob_chance.probs.item(), end='')
        #     new_prob = self.equiprob_chance.probs.item() - 0.1
        #     print(" New Prob: %f" % new_prob)
        #     self.equiprob_chance = torch.distributions.Bernoulli(torch.tensor([new_prob]))

    # ## makes a residual layer of n_blocks
    # ## expansion is always 4x of bottleneck
    # @staticmethod
    # def _make_res_layer(in_channels, out_channels, first_middle_stride=1, blocks=5):
    #     # first_middle_stride: the stride of bottleneck layer of first resBlock
    #     layers = []
    #     layers.append(ResBlock(in_channels, out_channels, middle_stride=first_middle_stride))
    #     for _ in range(1, blocks):
    #         ## all of these will have same dim
    #         layers.append(ResBlock(out_channels, out_channels, middle_stride=1))

    #     return nn.Sequential(*layers)



if __name__ == "__main__":
    from torch.distributions import normal
    
    ### for model testing
    norm_dist = normal.Normal(0, 1)
    
    DP = DeepPriorPPModel()
    inputs = norm_dist.sample((10, 1,128,128)) # 10 hand samples
    targets = norm_dist.sample((10,30))
    outputs = DP(inputs)

    print("Inputs Shape: ", inputs.shape,
            "\tOutputs Shape: ", outputs.shape,
            "\tTargets Shape: ", targets.shape)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(DP.parameters())
    
    losses = []
    
    print("Overfitting on 1 batch for 10 epochs...")
    for i in range(10):
        optimizer.zero_grad()
        outputs = DP(inputs)   # direct invocation calls .forward() automatically
        loss = criterion(outputs, targets)
        loss.backward() # calc grads w.r.t weight/bias nodes
        optimizer.step() # update weight/bias params
        losses.append(loss.item())

    print("10 Losses:\n", losses)


