import torch
from data_utils import unStandardiseKeyPointsCube
from datasets import BaseDataType


# accuracy
def top1_acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

# top-k accuracy
def top3_acc(output, target, k=3):
    with torch.no_grad():
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
    def __init__(self, cube_side_mm = 200, ret_avg_err_per_joint=False):
        ## all other values are assumed default e.g. num joints 
        ## and camera intrinsics, see BaseTransformer defn. for def. params
        #self.unstandardiser = JointUnstandardiser(cube_side_mm = cube_side_mm)
        self.cube_side_mm = cube_side_mm
        self.ret_avg_err_per_joint = ret_avg_err_per_joint

        self.pca_decoder = None
        self.__name__ = 'avg_3d_err_mm'
        ##self.pca

    def __call__(self, output, target, return_mm_data=False):
        '''
            Note: input is torch tensors
        '''

        if isinstance(output, tuple) and isinstance(target, tuple):
            # support for action tuple types
            # basically only get the rgression based data_types not class info
            output = output[0]
            target = target[0]
        
        ## pca -> keypoint space
        ## this needs to be a torch decoder
        ## (?x30) -> (?x21x3)
        output = self.pca_decoder(output, reshape=True)
        target = self.pca_decoder(target, reshape=True)
        #print("OUT_SHAPE: ", output.shape, "TARGET_SHAPE: ", target.shape)
        #target = self.pca_decoder(target, reshape=True)

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



    

   

    
    
