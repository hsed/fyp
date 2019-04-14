import logging
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        self.logger.info('Trainable parameters: {}'.format(self.param_count()))
        self.logger.info(self)
    
    def param_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(self.param_count())
        # print(super(BaseModel, self))
