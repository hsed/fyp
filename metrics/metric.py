import torch
from data_utils import unStandardiseKeyPointsCube
from datasets import BaseDataType
from models import PCADecoderBlock

from torch.nn.utils.rnn import PackedSequence

import numpy as np

import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple
from torch import Tensor
import torch

def kl_only_seq(output: Tuple[PackedSequence, Tensor, Tensor], target: Tuple[PackedSequence, Tensor]) -> Tensor:
    # get the last two items from this tuple
    # tuple can be len 4 or 3 either way its last two objects
    teacher_preds, student_preds = output[-2], output[-1]
    teacher_preds = torch.exp(teacher_preds)
    return F.kl_div(student_preds, teacher_preds, reduction='batchmean')

def mse_only_seq_2(output: Tuple[PackedSequence, PackedSequence, Tensor, Tensor], target: Tuple[PackedSequence, Tensor]) -> Tensor:
    student_keypts, teacher_keypts, _, _ = output # only valid if output is 4 len seq
    return F.mse_loss(student_keypts, teacher_keypts)

def mse_only_seq(output: Tuple[PackedSequence, Tensor], target: Tuple[PackedSequence, Tensor]) -> Tensor:
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
    return F.mse_loss(output_seqs.data, target_seqs.data)

def nll_only_seq(output: Tuple[PackedSequence, Tensor], target: Tuple[PackedSequence, Tensor]) -> Tensor:
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
    return F.nll_loss(output_preds, target_preds)


def combined_only_seq(output: Tuple[PackedSequence, Tensor], target: Tuple[PackedSequence, Tensor]) -> Tensor:
    output_seqs, output_preds = output
    target_seqs, target_preds = target

    # if one hot encodings are supplied for action, convert it to long, else convert float to long
    target_preds = target_preds.long() if target_preds.dim() == 1 else torch.argmax(target_preds, dim=1).long()

    return (100*F.mse_loss(output_seqs.data, target_seqs.data) + F.nll_loss(output_preds, target_preds))/2


# accuracy
def top1_acc(output, target):
    # always assumed that predicts or action classes are the last element of either tuple
    output = output[-1] if isinstance(output, tuple) else output
    target = target[-1] if isinstance(target, tuple) else target
    with torch.no_grad():
        target = target.long() if target.dim() == 1 else torch.argmax(target, dim=1).long()
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

# top-k accuracy
def top3_acc(output, target, k=3):
    output = output[-1] if isinstance(output, tuple) else output
    target = target[-1] if isinstance(target, tuple) else target
    with torch.no_grad():
        target = target.long() if target.dim() == 1 else torch.argmax(target, dim=1).long()
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


class Avg3DError(object):
    '''
        calc avg 3D error by first unstandardising network output
        and then comparing with targets

        Note: targets are already assumed to be pre-centered w.r.t CoM so
        decentering is not performed on output i.e. we assume both outputs and
        targets have the same origin (CoM) in euclidian space

        Note 2: the `cube_side_mm` param is used to decide on the scaling factor to
        unstandardise.
    '''
    def __init__(self, cube_side_mm = 200, ret_avg_err_per_joint=False, eval_pca_space=True,
                 train_pca_space=True, num_joints=21, num_dims=3):
        ## all other values are assumed default e.g. num joints 
        ## and camera intrinsics, see BaseTransformer defn. for def. params
        #self.unstandardiser = JointUnstandardiser(cube_side_mm = cube_side_mm)
        # eval_pca_space=True -> if true need to use pca decoder to project to keypoint space, else projection already
        # assumed done by model and output has shape (?, num_joints*num_dims)
        self.cube_side_mm = cube_side_mm
        self.ret_avg_err_per_joint = ret_avg_err_per_joint
        self.eval_pca_space = eval_pca_space
        self.train_pca_space = train_pca_space
        #self.train_pca_space = True # Always true, no need to set as argument
        self.num_dims = num_dims
        self.num_joints = num_joints

        #print("AVG£D ERROR EVALPCA SPACE :", self.eval_pca_space)

        self.pca_decoder = None
        self.__name__ = 'avg_3d_err_mm'
        ##self.pca
    
    def init_pca(self, pca_init_kwargs: dict, weight_np, bias_np, device, dtype):
        '''
            Add a PCA decoder class to self
        '''
        self.pca_decoder = PCADecoderBlock(**pca_init_kwargs)
        self.pca_decoder.initialize_weights(weight_np=weight_np, bias_np=bias_np)
        self.pca_decoder = self.pca_decoder.to(device, dtype)

    def __call__(self, output, target, return_mm_data=False):
        '''
            Note: input is torch tensors
        '''

        if isinstance(output, tuple) and isinstance(target, tuple):
            #do some plots here....
            # np.savez('tests/temp.npz', outs=torch.nn.utils.rnn.pad_packed_sequence(output[0], batch_first=True, padding_value=-999.0),
            #          targets=torch.nn.utils.rnn.pad_packed_sequence(target[0], batch_first=True, padding_value=-999.0))
            # quit()
            # support for action tuple types
            # basically only get the rgression based data_types not class info
            # we now also support output of being PackedSequence type in which case simply extract data element
            output = output[0] if not isinstance(output[0], PackedSequence) else output[0].data
            target = target[0] if not isinstance(target[0], PackedSequence) else target[0].data
        
        # skip decoding if:
        # output is in eval_mode (validation) and eval_pca_space is False
        # output is in train_mode (training) and train_pca_space is False
        decode_pca = False if ((not output.requires_grad and not self.eval_pca_space) or \
                               (output.requires_grad and not self.train_pca_space)) else True

        with torch.no_grad():
            ## pca -> keypoint space
            ## this needs to be a torch decoder
            ## (?x30) -> (?x21x3)
            #print("SHAPES: ", output.shape, target.shape)
            output = self.pca_decoder(output, reshape=True) if decode_pca \
                     else output.reshape(-1, self.num_joints, self.num_dims)
            target = self.pca_decoder(target, reshape=True) if decode_pca \
                     else target.reshape(-1, self.num_joints, self.num_dims) # assume target is already shaped to (21,3)
            #print("OUT_SHAPE: ", output.shape, "TARGET_SHAPE: ", target.shape)
            #quit()#target = self.pca_decoder(target, reshape=True)

            ## -1,1 -> -depth_len/2, +depth_len/2
            output = unStandardiseKeyPointsCube(output, self.cube_side_mm)
            target = unStandardiseKeyPointsCube(target, self.cube_side_mm)
            #output = self.unstandardiser(output)[BaseDataType.JOINTS]

            ## R^{500, 21, 3} == avg_err_per_joint ==> R^{500, 21}
            err_per_joint = torch.norm(output - target, p=2, dim=2)
            
            ## R^{500, 21} == avg_err_across_dataset ==> R^{21}
            ## do avg for each joint over errors of all samples
            avg_err_per_joint = err_per_joint.mean(dim=0)


            ## R^{21} == avg_err_across_joints ==> R
            avg_3D_err = avg_err_per_joint.mean()
            
            if return_mm_data:
                return avg_3D_err, output, target
            elif self.ret_avg_err_per_joint:
                return avg_3D_err, avg_err_per_joint
            else:
                return avg_3D_err



    

   

    
    
