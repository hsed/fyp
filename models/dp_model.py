import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch import tensor as T
from collections import OrderedDict

## remove later
import torch

### TODO: MUST EDIT ALL!!!

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


class LinearDropoutBlock(nn.Module):
    ## BN + ReLU # alpha = momentum?? TODO: check
    ## this one causes problem in back prop
    def __init__(self, in_features, out_features, apply_relu=True, dropout_prob=0, use_bias=True):
        super(LinearDropoutBlock, self).__init__()
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


class BNReLUBlock(nn.Module):
    ## BN + ReLU # alpha = momentum?? TODO: check
    def __init__(self, in_channels, eps=0.0001, momentum=0.1):
        super(BNReLUBlock, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum), 
            nn.ReLU(True),
        )
    
    def forward(self, x):
        return self.block(x)


class ConvPoolBlock(nn.Module):
    ## TODO: add docs
    def __init__(self, in_channels, out_channels, kernel_size, pool_size,
                        stride=1,
                        doBatchNorm=False, doReLU=False, pad="valid"):
        super(ConvPoolBlock, self).__init__()
        self.padding = get_padding_size(kernel_size, pad)

        self.block = nn.Sequential()
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
    
    def forward(self, x):
        return self.block(x)


class BNReLUConvBlock(nn.Module):
    ## BN + ReLU + Conv, no pooling, use 2x2 stride aka stride=2 to downsample layer, also padding is fixed to same
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BNReLUConvBlock, self).__init__()
        # all resnet blocks have got "same" paddining
        # however if stride == 2 then the output will be input/2 but this is ok
        self.padding = get_padding_size(kernel_size, "same")

        self.block = nn.Sequential(
            BNReLUBlock(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding)
        )
        ## double check to make sure this is as expected
        ## do all weight setting in end!! i.e. based on layer types
    
    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, middle_stride):
        super(ResBlock, self).__init__()


        if (out_channels != in_channels*2 and out_channels != in_channels):
            raise NotImplementedError("Only factor of {1, 2} upsampling of channels is implemented.")
        
        if (middle_stride != 1 and middle_stride != 2):
            raise NotImplementedError("Only middle_stride={1, 2} is supported.")

        ## bottleneck type
        ## all 3 layers pad s.t. if stride=1 then input==output
        ## i.e. same padding without taking stride into account
        self.res_branch = nn.Sequential(
            ## first downsample channels e.g. 32x32x32 -> 32x32x16
            BNReLUConvBlock(in_channels, out_channels//4, kernel_size=1, stride=1),

            ## then downsample input e.g. 32x32x16 -> 16x16x16
            ## the bottleneck
            ## this part is different from deep-prior but is used everywhere else
            BNReLUConvBlock(out_channels//4, out_channels//4, kernel_size=3, stride=middle_stride),

            ## now upsample channels e.g. 16x16x16 -> 16x16x64
            # so width&hright reduced, depth increased
            # note: by default last layer must upsample bottleneck by 4
            # as convention so this method works if input_c==output_c
            # or even if input_c*2==output_c
            BNReLUConvBlock(out_channels//4, out_channels, kernel_size=1, stride=1),
        )

        if in_channels == out_channels:
            ## this is used when in_channels == out_channel
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                BNReLUConvBlock(in_channels, out_channels, kernel_size=1, stride=middle_stride)
            )
    
    def forward(self, x):
        return self.res_branch(x) + self.skip_con(x)




class DeepPriorPPModel(nn.Module):
    def __init__(self, input_channels=1, num_joints=21, num_dims=3, pca_components=30, dropout_prob=0.3,
                 train_mode=True, weight_matx_np=None, bias_matx_np=None):
        self.num_joints, self.num_dims = num_joints, num_dims
        self.train_mode = train_mode # when True output is PCA, else output is num_dims*num_joints

        super(DeepPriorPPModel, self).__init__()
        channelStages = [32, 64, 128, 256, 256]
        strideStages = [2, 2, 2, 1]

        self.main_layers = nn.Sequential(OrderedDict([
            ('conv_pool_1', ConvPoolBlock(input_channels, channelStages[0],
                kernel_size=5, pool_size=2, pad="same", stride=1)),
            ('res_group_1', DeepPriorPPModel._make_res_layer(
                channelStages[0], channelStages[1], first_middle_stride=strideStages[0])),
            ('res_group_2', DeepPriorPPModel._make_res_layer(
               channelStages[1], channelStages[2], first_middle_stride=strideStages[1])),
            ('res_group_3', DeepPriorPPModel._make_res_layer(
               channelStages[2], channelStages[3], first_middle_stride=strideStages[2])),
            ('res_group_4', DeepPriorPPModel._make_res_layer(
               channelStages[3], channelStages[4], first_middle_stride=strideStages[3])),
            ('bn_relu_1', BNReLUBlock(channelStages[4])),
            
        ]))

        self.linear_layers = nn.Sequential(OrderedDict([
            #sadly the input here must be computed manually
            ('lin_relu_1', LinearDropoutBlock(8*8*256, 1024, 
               apply_relu=True, dropout_prob=dropout_prob)),
            ('lin_relu_2', LinearDropoutBlock(1024, 1024, 
              apply_relu=True, dropout_prob=dropout_prob)),
            ('lin_relu_3_pca', LinearDropoutBlock(1024, pca_components, 
              apply_relu=False, dropout_prob=0.)),  # 0 dropout => no dropout layer to add
        ]))

        ## back projection layer
        self.final_layer = nn.Linear(pca_components, self.num_joints*self.num_dims)

        # currently done according to resnet implementation (He init conv layers)
        # also init last layer with pca__inverse_transform vals and make it fixed
        self._initialize_weights(w_final=weight_matx_np, b_final=bias_matx_np)


    def forward(self, x):
        #x = x.float()   # cast to float,temp fix..., later whole shuld be float
        x = self.main_layers(x)
        
        ## must convert x into (n_samples, 8*8*256 vector)
        x = x.view(-1, 8*8*256)
        x = self.linear_layers(x)

        return x
    
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
    def _initialize_weights(self, w_final=None, b_final=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        #print("PARAMS: \n", list(self.children()))
        #print("WEIGHTS: \n", self.final_layer.weight, "\nShape: ", self.final_layer.weight.shape)
        #print("\nBIAS: \n", self.final_layer.bias, "\nShape: ", self.final_layer.bias.shape)
        
        # init final layer weights and make them fixed -- used only for test error
        if (w_final is not None and b_final is not None):
            # we use torch.tensor() method to create a copy of the array provided.
            # as this method always copies data
            self.final_layer.weight = \
                torch.nn.Parameter(torch.tensor(w_final, dtype=self.final_layer.weight.dtype))
            self.final_layer.bias = \
                torch.nn.Parameter(torch.tensor(b_final, dtype=self.final_layer.bias.dtype))
            
            self.final_layer.weight.requires_grad = False
            self.final_layer.bias.requires_grad = False
            #print("\n\nNEW WEIGHTS: \n", self.final_layer.weight, "\nShape: ", self.final_layer.weight.shape)
            #print("\nNEW BIAS: \n", self.final_layer.bias, "\nShape: ", self.final_layer.bias.shape, "\n")
    

    ## makes a residual layer of n_blocks
    ## expansion is always 4x of bottleneck
    @staticmethod
    def _make_res_layer(in_channels, out_channels, first_middle_stride=1, blocks=5):
        # first_middle_stride: the stride of bottleneck layer of first resBlock
        layers = []
        layers.append(ResBlock(in_channels, out_channels, middle_stride=first_middle_stride))
        for _ in range(1, blocks):
            ## all of these will have same dim
            layers.append(ResBlock(out_channels, out_channels, middle_stride=1))

        return nn.Sequential(*layers)



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


