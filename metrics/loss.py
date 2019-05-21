import torch.nn.functional as F


def nll_loss(output, target):
    # to ensure that we use long for target as
    # nll_loss requires that for indexing, we will simply copy it over
    # all datatypes are set to float during training, we need long here
    # if one hot encodings are supplied for action, convert it to long
    target = target.long() if target.dim() == 1 else torch.argmax(target, dim=1).long()
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)




def mse_and_nll_loss(outputs: tuple, targets: tuple):
    '''
        outputs and targets: (HandSkeleton3D, ActionClassIdx)  
    '''

    output_hpe, output_probvect = outputs
    target_hpe, target_cidx = targets
    target_cidx = target_cidx.long() if target_cidx.dim() == 1 else torch.argmax(target_cidx, dim=1).long()

    # combine the two losses with eual weights, TODO: adjust loss weights in future...
    return F.mse_loss(output_hpe, target_hpe) + F.nll_loss(output_probvect, target_cidx.long())