import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple
from torch import Tensor
import torch
from functools import partial

def nll_loss(output, target):
    # to ensure that we use long for target as
    # nll_loss requires that for indexing, we will simply copy it over
    # all datatypes are set to float during training, we need long here
    # if one hot encodings are supplied for action, convert it to long
    target = target.long() if target.dim() == 1 else torch.argmax(target, dim=1).long()
    return F.nll_loss(output, target)




def mse_seq_and_nll_loss(output: Tuple[PackedSequence, Tensor], target: Tuple[PackedSequence, Tensor],
                         alpha: float = 0.000002) -> Tensor: #0.02
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



def seq_mse_and_nll_and_kl_loss(output: Tuple[PackedSequence, Tensor, Tensor], target: Tuple[PackedSequence, Tensor],
                         alpha: float = 0.000002, beta: float = 0.5) -> Tensor: #0.02
    '''
        first tensor is expected to be of teacher, second of student
        both should be log probs
    '''
    # note: currently only packed sequences are supported for mse, nll must be of tensor type
    # we need an alpha param for fusing of two losses
    # note nll_loss at beginning is ~3 and mse_loss is ~0.06 so
    # 3*0.02 ~= 0.06; 0.06*0.98 ~= 0.06 so we have the same range
    #print("Types:", type(target[0]), type(target[1]))
    ##  kl loss formula student probs need to be log probs, teacher probs must be probs
    output_seqs, output_teacher_log_preds, output_student_log_preds = output
    target_seqs, target_preds = target
    output_teacher_preds = torch.exp(output_teacher_log_preds)

    # if one hot encodings are supplied for action, convert it to long, else convert float to long
    target_preds = target_preds.long() if target_preds.dim() == 1 else torch.argmax(target_preds, dim=1).long()
    #print("types: ", output_seqs.data.dtype, target_seqs.data.dtype)
    return (beta*F.kl_div(output_student_log_preds, output_teacher_preds, reduction='batchmean')) + (1.0-beta)*(alpha*F.nll_loss(output_student_log_preds, target_preds) + (1.0-alpha)*F.mse_loss(output_seqs.data, target_seqs.data))


def seq_mse_and_nll_and_kl_and_mse_loss(output: Tuple[PackedSequence, PackedSequence, Tensor, Tensor], 
                                        target: Tuple[PackedSequence, Tensor],
                                        alpha: float = 0.000002, beta: float = 0.002,
                                        gamma: float = 0.0002) -> Tensor: #0.02
    '''
        first tensor is expected to be of teacher, second of student
        both should be log probs
    '''
    # note: currently only packed sequences are supported for mse, nll must be of tensor type
    # we need an alpha param for fusing of two losses
    # note nll_loss at beginning is ~3 and mse_loss is ~0.06 so
    # 3*0.02 ~= 0.06; 0.06*0.98 ~= 0.06 so we have the same range
    #print("Types:", type(target[0]), type(target[1]))
    ##  kl loss formula student probs need to be log probs, teacher probs must be probs
    output_seqs, teacher_seq, output_teacher_log_preds, output_student_log_preds = output
    target_seqs, target_preds = target
    output_teacher_preds = torch.exp(output_teacher_log_preds)

    # if one hot encodings are supplied for action, convert it to long, else convert float to long
    target_preds = target_preds.long() if target_preds.dim() == 1 else torch.argmax(target_preds, dim=1).long()
    #print("types: ", output_seqs.data.dtype, target_seqs.data.dtype)
    return (beta*(gamma*F.kl_div(output_student_log_preds, output_teacher_preds, reduction='batchmean') + \
                    (1.0-gamma)*F.mse_loss(output_seqs.data, teacher_seq.data))) \
                    + (1.0-beta)*(alpha*F.nll_loss(output_student_log_preds, target_preds) \
                        + (1.0-alpha)*F.mse_loss(output_seqs.data, target_seqs.data))

def seq_mse_and_nll_and_kl_and_mse_loss_v2(output: Tuple[PackedSequence, PackedSequence, Tensor, Tensor], 
                                        target: Tuple[PackedSequence, Tensor],
                                        alpha: float = 0.000002, beta: float = 0.002,
                                        gamma: float = 0.0002, T: float = 1.2) -> Tensor: #0.02
    '''
        first tensor is expected to be of teacher, second of student
        both should be log probs
    '''
    # note: currently only packed sequences are supported for mse, nll must be of tensor type
    # we need an alpha param for fusing of two losses
    # note nll_loss at beginning is ~3 and mse_loss is ~0.06 so
    # 3*0.02 ~= 0.06; 0.06*0.98 ~= 0.06 so we have the same range
    #print("Types:", type(target[0]), type(target[1]))
    ##  kl loss formula student probs need to be log probs, teacher probs must be probs
    output_seqs, teacher_seq, output_teacher_preds, output_student_preds = output
    target_seqs, target_preds = target
    output_student_log_preds = F.log_softmax(output_student_preds, dim=1)
    
    stud_soft_preds = F.log_softmax(output_student_preds/T, dim=1)
    teach_soft_preds = F.softmax(output_teacher_preds/T, dim=1)

    # if one hot encodings are supplied for action, convert it to long, else convert float to long
    target_preds = target_preds.long() if target_preds.dim() == 1 else torch.argmax(target_preds, dim=1).long()
    #print("types: ", output_seqs.data.dtype, target_seqs.data.dtype)
    return (beta*(gamma*F.kl_div(stud_soft_preds, teach_soft_preds, reduction='batchmean')*(T**2) + \
                    (1.0-gamma)*F.mse_loss(output_seqs.data, teacher_seq.data))) \
                    + (1.0-beta)*(alpha*F.nll_loss(output_student_log_preds, target_preds) \
                        + (1.0-alpha)*F.mse_loss(output_seqs.data, target_seqs.data))



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



class CombinedSeqLoss(object):
    def __init__(self, alpha=0.0002, beta=-1, gamma=-1, sigma=-1, loss_type='mse_and_nll'):
        loss_fn_dict = {
            'mse_and_nll': mse_seq_and_nll_loss,
            'mse_and_nll_and_kl': partial(seq_mse_and_nll_and_kl_loss, beta=beta), # alpha will be supplied as below
            'mse_and_nll_and_kl_and_mse': partial(seq_mse_and_nll_and_kl_and_mse_loss, beta=beta, gamma=gamma),
            'mse_and_nll_and_kl_and_mse_v2': partial(seq_mse_and_nll_and_kl_and_mse_loss_v2, beta=beta, gamma=gamma)
        }

        assert (loss_type in loss_fn_dict.keys()), "Loss %s is not implemented, available options: %a" % \
               (loss_type, list(loss_fn_dict.keys()))

        self.alpha = alpha
        self.beta = beta    
        self.gamma = gamma  # unused
        self.sigma = sigma  # unused

        self.loss_fn = loss_fn_dict[loss_type]

    
    def __call__(self, output: Tuple[PackedSequence, Tensor], target: Tuple[PackedSequence, Tensor]) -> Tensor:
        return self.loss_fn(output, target, self.alpha)