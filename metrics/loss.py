import torch.nn.functional as F


def nll_loss(output, target):
    # to ensure that we use long for target as
    # nll_loss requires that for indexing, we will simply copy it over
    return F.nll_loss(output, target)