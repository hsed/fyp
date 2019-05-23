import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple
from torch import Tensor

def nll_loss(output, target):
    # to ensure that we use long for target as
    # nll_loss requires that for indexing, we will simply copy it over
    # all datatypes are set to float during training, we need long here
    # if one hot encodings are supplied for action, convert it to long
    target = target.long() if target.dim() == 1 else torch.argmax(target, dim=1).long()
    return F.nll_loss(output, target)


def mse_seq_and_nll_loss(output: Tuple[PackedSequence, Tensor], target: Tuple[PackedSequence, Tensor],
                         alpha: float = 0.02) -> Tensor:
    # note: currently only packed sequences are supported for mse, nll must be of tensor type
    # we need an alpha param for fusing of two losses
    # note nll_loss at beginning is ~3 and mse_loss is ~0.06 so
    # 3*0.02 ~= 0.06; 0.06*0.98 ~= 0.06 so we have the same range
    #print("Types:", type(target[0]), type(target[1]))
    output_seqs, output_preds = output
    target_seqs, target_preds = target

    # if one hot encodings are supplied for action, convert it to long, else convert float to long
    target_preds = target_preds.long() if target_preds.dim() == 1 else torch.argmax(target_preds, dim=1).long()
    #print("types: ", output_seqs.data.dtype, target_seqs.data.dtype)
    return alpha*F.nll_loss(output_preds, target_preds) + (1.0-alpha)*F.mse_loss(output_seqs.data, target_seqs.data)



def mse_seq_loss(output: PackedSequence, target: PackedSequence):
    return F.mse_loss(output.data, target.data)


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