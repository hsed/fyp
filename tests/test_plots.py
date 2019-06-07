##### function to send sample and create gif
import os
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX
import logging
import yaml
from data_utils import CombinedDataLoader, CollateCustomSeqBatch
from data_utils.debug_plot import plotImg, plotImgV2
from models import CombinedModel
import torch
import numpy as np
from metrics import Avg3DError
from trainer import init_metrics
from test import _tensor_to
from tensorboardX.utils import figure_to_image
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]='3'



from tensorboardX.utils import _prepare_video
from torch.nn.utils.rnn import PackedSequence
from moviepy import editor as mpy
import tempfile
from IPython.display import Image
from tensorboardX.summary import _calc_scale_factor

def make_gif(image_seq, fps=4, print_filepath=False, filename=None, filedir=None):
    #print(debug_sample_tuple[0].shape)
    idx_dict = {
        'orig_depth': 0
    }
    
#     if len(debug_sample_tuple[idx_dict['orig_depth']].shape) == 3:
#         print("CHANGING TxWxH => 1xTx1XWxH")
    # https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/summary.py
    
    #tensor = imag
    tensor = image_seq
    if len(tensor.shape) == 3:
        C = 1
        T, H, W = tensor.shape
    elif len(tensor.shape) == 4:
        T, C, H, W =  tensor.shape
    tensor = tensor.reshape(1, T, C, H, W) # prepare video requires this format
    tensor = _prepare_video(tensor)
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    
    #print(tensor)
    #t, h, w, c = tensor.shape
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
    if filename is None:
        filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False, dir=filedir).name
    clip.write_gif(filename, verbose=False, progress_bar=False)
    if print_filepath:
        print("GIF Path:", filename)
    
    return filename
    

def show_gif_inline(filename):
    with open(filename,'rb') as file:
        display(Image(data=file.read(), format='gif'))
        ## add option here to delete later

#depth_seqs = [make_gif(item) for item in val_data]

#print(max(a)) # 1901
#print(len(val_data)) #575

def plot_img_seq(data_tuple, keypt_pred_px, pred_err, action_pred_probs,
                 keypt_pred_px_2=None, pred_err_2=-1, action_pred_probs_2=None,
                 all_action_pred_probs_1=None, all_action_pred_probs_2=None):
    # note the are all seq
    dpt_orig, dpt_crop, dpt_joints_orig_px, dpt_com_orig_px, dpt_crop_transf_matx = data_tuple[:5]
    gt_action = data_tuple[-1][-1] # last elem of whole tuple is output, last elem of output is action
    
    ## make it compatible with list operation
    keypt_pred_px = [None for _ in range(dpt_orig.shape[0])] if keypt_pred_px is None \
                    else keypt_pred_px
    keypt_pred_px_2 = [None for _ in range(dpt_orig.shape[0])] if keypt_pred_px_2 is None \
                    else keypt_pred_px_2
    
    all_action_pred_probs_1 = [action_pred_probs for _ in range(dpt_orig.shape[0])] if all_action_pred_probs_1 is None \
                    else all_action_pred_probs_1
    all_action_pred_probs_2 = [action_pred_probs_2 for _ in range(dpt_orig.shape[0])] if all_action_pred_probs_2 is None \
                    else all_action_pred_probs_2

    pred_err = [pred_err for _ in range(dpt_orig.shape[0])]
    pred_err_2 = [pred_err_2 for _ in range(dpt_orig.shape[0])]
    
    def plt_fig(t):
        return plotImgV2(dpt_orig[t], dpt_crop[t], dpt_joints_orig_px[t], dpt_com_orig_px[t],
                         dpt_crop_transf_matx[t], action_gt=gt_action, final_action_pred_probs=action_pred_probs,
                         final_action_pred_probs_2=action_pred_probs_2,
                         keypt_pred_px_1=keypt_pred_px[t], pred_err_1=pred_err[t],
                         action_pred_probs_1=all_action_pred_probs_1[t], # need to change this to per timestep
                         keypt_pred_px_2=keypt_pred_px_2[t], pred_err_2=pred_err_2[t],
                         action_pred_probs_2=all_action_pred_probs_2[t],# need to change this to per timestep
                         show_aug_plots=False, return_fig=True, 
                         frame_curr=t, frame_max=dpt_orig.shape[0])#,
                       #keypt_pred_px=y_px[t], pred_err=err_3d)
    img_arr = np.stack([figure_to_image(plt_fig(t)) for t in range(dpt_orig.shape[0])])

    return img_arr



def plot_and_show_inline(data_tuple, keypt_pred_px, pred_err, action_pred_probs,
                         keypt_pred_px_2=None, pred_err_2=-1, action_pred_probs_2=None,
                         all_action_pred_probs_1=None, all_action_pred_probs_2=None,
                         filename=None, filedir='results/plot_data'):
    os.makedirs(filedir, exist_ok=True)
    show_gif_inline(make_gif(plot_img_seq(data_tuple, keypt_pred_px, pred_err, action_pred_probs,
                                          keypt_pred_px_2, pred_err_2, action_pred_probs_2,
                                          all_action_pred_probs_1, all_action_pred_probs_2), fps=2, print_filepath=True,
                             filedir=filedir, filename=filename))

    
    


class DebugData(object):
    def __init__(self, device=torch.device('cuda'), dtype=torch.float):
        self.device = device
        self.dtype = dtype
        #logger = logging.getLogger(__name__)
        #writer = WriterTensorboardX("logs/temp_log", logger, True)
        self.config = yaml.load(open("configs/combined/combined_act_in_act_out.yaml"), Loader=yaml.SafeLoader)
        self._fix_config()
        
        self.data_loader = CombinedDataLoader(**self.config['data_loader']['args'])
        self.val_data_loader = self.data_loader.split_validation()
        self.model = CombinedModel(**self.config['arch']['args']).to(self.device,self.dtype)
        self.model.eval() # set to eval mode
        
        self._fix_val_data_loader()
        #self._preload_val_data()
        
        self._setup_collator()
        self._setup_metrics()
    
    def _fix_config(self, print_final=True):
        self.config['data_loader']['args']['batch_size'] = 8 # 576
        self.config['data_loader']['args']['shuffle'] = False
        self.config['data_loader']['args']['num_workers'] = 0 #8 #0
        self.config['data_loader']['args']['forward_type'] = -1

        if print_final:
            print(self.config['data_loader']['args'])
    
    def _fix_val_data_loader(self):
        self.val_data_loader.dataset.transform = self.data_loader.debug_transforms
        
        # sort samples
        self.val_data_loader.batch_sampler.sampler.indices = np.sort(self.val_data_loader.batch_sampler.sampler.indices)
        
        # take a list of samples and return as is, dont do as is
        self.val_data_loader.collate_fn = (lambda k: k)
        
    def _preload_val_data(self, break_val=np.iinfo(np.uint32).max):
        self.val_data = [] #val_data # temporary #[]
        for i, item in tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader), desc='Pre-loading validation samples'):
            self.val_data += item
            if i >= break_val:
                break
        return self
    
    def _setup_collator(self):
        ## must match data_loader...
        coll = self.data_loader.collate_fn
        self.coll_single = (lambda sample: coll([sample])) # collate a single sample only
    
    def _setup_metrics(self):
        ### add som more metrics like acc!
        
        errMetric = [Avg3DError]
        init_metrics(errMetric, model=self.model, data_loader=self.data_loader, device=self.device, dtype=self.dtype)
        errMetric = errMetric[0]
        
        self.metrics = [ errMetric, ]
        
        
    
    
    def __getitem__(self, index):
        # should be a tuple but not multiple data with/without tuple
        # ins => (depth, action_seq, joints)
        # targets => (joints, action)
        com_orig = self.val_data[index][-2]
        ins, targets = self.coll_single(self.val_data[index][-1])
        print("depth   -> inputs[0].shape:", ins[0].data.shape)
        print("act_seq -> inputs[1].shape:", ins[1].data.shape)
        print("keypts  -> inputs[2].shape:", ins[2].data.shape)
        print("keypts  -> targets[0].shape:", targets[0].data.shape)
        print("act     -> targets[1].shape:", targets[1].data.shape)
        
        ### atm we only get hpe results
        ##
        gt_keypts = _tensor_to(targets[0], device=self.device, dtype=self.dtype)
        
        #depths, batch_sizes = ins[0].data.unsqueeze(1), ins[0].batch_sizes
        depths_packed_seq = _tensor_to(ins[0], device=self.device, dtype=self.dtype)
        actions_seq = _tensor_to(ins[1].data, device=self.device, dtype=self.dtype)
        
        
        # 1 => x -> hpe -> y ; 2 => x+z -> hpe_wAct -> y
        pred_keypts_1, pred_keypts_2, pred_act_probs_1, pred_act_probs_2, pred_all_act_probs_1, pred_all_act_probs_2 = \
                                self.forward_pass((depths_packed_seq, actions_seq), type='hpe_hpe_act_har')
        #pred_keypts = self.forward_pass((depths, actions_seq), type='hpe_act')
        pred_keypts_1 = PackedSequence(data=pred_keypts_1,batch_sizes=depths_packed_seq.batch_sizes)
        pred_keypts_2 = PackedSequence(data=pred_keypts_2,batch_sizes=depths_packed_seq.batch_sizes)
        
        depths = depths_packed_seq.data
        print("\ndepths.shape", depths.shape, depths.dtype, depths.device)
        print("gt_keypts.shape", gt_keypts.data.shape, gt_keypts.data.dtype, gt_keypts.data.device)
        print("pred_keypts.shape", pred_keypts_1.data.shape, pred_keypts_1.data.dtype, pred_keypts_1.data.device)
        
        err_3d_1, y_mm_, y_mm = self.metrics[0](pred_keypts_1, gt_keypts, return_mm_data=True)
        err_3d_1 = err_3d_1.item()
        y_mm_uncentered_ = y_mm_.cpu().numpy() + com_orig[:, np.newaxis, ...]
        y_px_pred_1 = np.stack([self.data_loader.mm2px_multi(sample) for sample in y_mm_uncentered_])
        print("3d_error: %0.3fmm" % err_3d_1)
        
        err_3d_2, y_mm_, y_mm = self.metrics[0](pred_keypts_2, gt_keypts, return_mm_data=True)
        err_3d_2 = err_3d_2.item()
        y_mm_uncentered_ = y_mm_.cpu().numpy() + com_orig[:, np.newaxis, ...]
        y_px_pred_2 = np.stack([self.data_loader.mm2px_multi(sample) for sample in y_mm_uncentered_])
        print("3d_error_2: %0.3fmm" % err_3d_2)

        
        pred_act_probs_1 = torch.exp(pred_act_probs_1).detach().cpu().numpy().flatten()
        pred_act_probs_2 = torch.exp(pred_act_probs_2).detach().cpu().numpy().flatten()
        pred_all_act_probs_1 = torch.exp(pred_all_act_probs_1).detach().cpu().numpy()
        pred_all_act_probs_2 = torch.exp(pred_all_act_probs_2).detach().cpu().numpy()
        # print(pred_act_probs_1.shape)
        # print(self.model.har.use_unrolled_lstm)
        # quit()
        
        plot_and_show_inline(self.val_data[index], keypt_pred_px=y_px_pred_1, pred_err=err_3d_1,
                            action_pred_probs=pred_act_probs_1,
                            keypt_pred_px_2=y_px_pred_2, pred_err_2=err_3d_2,
                            action_pred_probs_2=pred_act_probs_2,
                            all_action_pred_probs_1=pred_all_act_probs_1,
                            all_action_pred_probs_2=pred_all_act_probs_2)
    
    
    def forward_pass(self, x, type='hpe'):
        if type == 'hpe':
            return self.model.hpe(x)
        if type == 'hpe_act':
            return self.model.hpe_act(x)
        if type == 'har':
            return self.model.har(x)
        if type == 'hpe_act_har':
            pass #y_ = self.hpe(depth.unsqueeze(1))
        if type == 'hpe_hpe_act_har':
            x_packed, z = x # split to depth, action
            x, batch_sizes = x_packed.data.unsqueeze(1), x_packed.batch_sizes 
            
            # two differnt models
            y_hpe = self.model.hpe(x)
            y_hpe_act = self.model.hpe_act((x,z))
            
            
            # two different inputs to same model
            z_hpe, z_temporal_hpe = self.model.har(PackedSequence(data=y_hpe, batch_sizes=batch_sizes), return_temporal_probs=True)
            z_hpe_act, z_temporal_hpe_act = self.model.har(PackedSequence(data=y_hpe_act, batch_sizes=batch_sizes), return_temporal_probs=True)
            
            #print("SHape", z_temporal_hpe.shape)
            return (y_hpe, y_hpe_act, z_hpe, z_hpe_act, z_temporal_hpe, z_temporal_hpe_act)
        else:
            raise NotImplementedError
    
    
    def __len__(self):
        return len(self.val_data)
    

if __name__ == "__main__":
    debug_data_obj = DebugData(device=torch.device('cuda'))
    val_data = debug_data_obj._preload_val_data(break_val=9).val_data

    #val_data_sample = next(iter(debug_data_obj.val_data_loader))
    #debug_data_obj.val_data = val_data
    #debug_data_obj[0]
    #val_data
    debug_data_obj[8] #debug_data_obj[-37]