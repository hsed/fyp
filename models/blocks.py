import torch.nn as nn

'''
    Define here all re-usable mini-blocks and macro-blocks
'''

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