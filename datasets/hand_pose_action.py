import os
import sys
import struct
import time
from enum import IntEnum
from typing import List, Any, Callable

#import cv2 # no longer using this
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


from .base_data_types import ExtendedDataType as DT, \
                             BaseDatasetType as DatasetMode, \
                             BaseTaskType as TaskMode


## starting from 0, each component of y_gt_mm_keypoints is
# --0(wrist)
# --1(thumb_mcp), 2(index_mcp), 3(middle_mcp), 4(ring_mcp), 5(pinky_mcp)
# --6(thumb_pip), 7(thumb_dip), 8(thumb_tip),
# --9(index_pip), 10(index_dip), 11(index_tip),
# --12(middle_pip), 13(middle_dip), 14(middle_tip),
# --15(ring_pip), 16(ring_dip), 17(ring_tip),
# --18(pinky_pip), 19(pinky_dip), 20(pinky_tip),
# total 21 joints, each having 3D co-ords

# COM, currently MCP --> 3*world_dim: 3*world_dim+3



class Num2Str(object):
    '''
        For consistent keynames for use with h5 files
    '''
    def __init__(self, pad=8):
        self.pad_str = "%" + "0" + str(pad) + "d"
    
    def __call__(self, i):
        return (self.pad_str % i)


class HandPoseActionDataset(Dataset):
    def __init__(self, root, data_mode, task_mode, transform=None, reduce=False,
                retrieve_depth=True, preload_depth=False):
        '''
            `data_mode` => 'train' // 'test'
            `reduce` => Train only on 1 gesture and 2 subjects, CoM won't work correctly
            `retrieve_depth` => Whether or not to retrieve depth files from disk when __call__
                                method is invoked.
            `preload_depth` => Directly load all depth maps from cache file to RAM for faster training
                               requires sufficient RAM, no effect when `retrieve_depth` is False
            
            Currently this class is used to load data to train a HAR
            Another class
        '''
        self.dpt_img_width = 640 #320
        self.dpt_img_height = 480 #240

        # TO CONFIRM TODO: !!!!
        self.min_depth = 100
        self.max_depth = 700
        
        self.joint_num = 21
        self.world_dim = 3
        
        # self.reorder_idx = np.array([
        #     0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12,
        #     13, 14, 4, 15, 16, 17, 5, 18, 19, 20
        # ])
        
        self.test_pos = -1
        

        
        ### setup directories
        self.root = root
        self.skeleton_dir = os.path.join(self.root, 'Hand_pose_annotation_v1')
        self.video_dir = os.path.join(self.root, 'Video_files') 
        #self.info_dir = os.path.join(self.root, 'Subjects_info')
        self.subj_dirnames = ['Subject_1'] if reduce else \
            ['Subject_%d' % i for i in range(1, 7)]
        #self.center_dir = center_dir # not in use
        self.reduce = reduce

        # setup modes
        if data_mode not in DatasetMode._value2member_map_:
            raise RuntimeError("Invalid dataset type, choose from: ",
                               DatasetMode._value2member_map_.keys())
        if task_mode not in TaskMode._value2member_map_:
            raise RuntimeError("Invalid task type, choose from: ",
                               TaskMode._value2member_map_.keys())

        self.data_mode = DatasetMode._value2member_map_[data_mode]
        self.task_mode = TaskMode._value2member_map_[task_mode]

        self.transform = transform
        
        # currently a very weak check, only checks for skeletons
        if not self._check_exists(): raise RuntimeError('Invalid Hand Pose Action Dataset')

        # 1 => '00000001'; For cache naming
        self.num2str = Num2Str(pad=8)

        ### initialise empty lists
        self.names = []
        self.joints_world = []
        self.coms_world = []
        self.actions = []
        self.depthmaps = []
        self.aug_modes = None
        self.aug_params = None

        self.RAND_SEED = 0 

        self.ignore_cache_for_hpe = False # 
        self._load()

        self.retrieve_depth = retrieve_depth
        self.preload_depth = preload_depth
        
        if retrieve_depth:
            self._verify_depth_cache(data_mode, task_mode)
            if preload_depth: self._preload_depth() # load depth to RAM

        ## open depth_map cache in multi-read mode for dataloaders
        ## bugs with this... 
        ## see https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.depthmap_cachefile = None

        #print("Names:")
        #[print(name) for name in self.names[108:115]]
    

    def __getitem__(self, index):
        ## the x-values are loaded 'on-the-fly' as and when required.
        ## this function is called internally by pytorch whenever a new sample needs to
        ## be loaded.
        #depthmap = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)
        # any depth to allow 16-it images as the depths are 16-bit here, use option to load from file here 

        ## need to do this here cause of bug
        if self.depthmap_cachefile is None and self.retrieve_depth is True \
           and self.preload_depth is False and self.ignore_cache_for_hpe is False:
            self.depthmap_cachefile = h5py.File(self.depthmap_cachepath, 'r', libver='latest', swmr=True)

        if self.task_mode == TaskMode.HAR:
            # commented code is depriciated, only kept for ref; ~10min (old) vs ~1s new!
            # depthmaps = np.stack(
            #     [np.asarray(Image.open(img_path), dtype=np.uint16) for \
            #         img_path in self.names[index]]
            # )
            sample = {
                DT.NAME_SEQ: self.names[index], # sample names => R^{NUM_FRAMES x 1}
                DT.JOINTS_SEQ: self.joints_world[index], # 3d joints => R^{NUM_FRAMES x 63}
                DT.COM_SEQ: self.coms_world[index], # => R^{NUM_FRAMES x 3}
                DT.DEPTH_SEQ: None if self.retrieve_depth is False \
                              else self.depthmap_cachefile[self.num2str(index)].value \
                              if self.preload_depth is False \
                              else self.depthmaps[index], #depthmaps => R^{NUM_FRAMES x 480 x 640}
                DT.ACTION: self.actions[index], # action => R^{1}
            }

        elif self.task_mode == TaskMode.HPE:
            #depthmap = cv2.imread(self.names[index], cv2.IMREAD_ANYDEPTH)
            # once depthmap is indexed from h5py file, its a numpy array
            sample = {
                DT.NAME: self.names[index], # sample name => R^{1}
                DT.JOINTS: self.joints_world[index], # 3d joints of the sample => R^{63}
                DT.COM: self.coms_world[index], # => R^{3}
                DT.DEPTH: None if self.retrieve_depth is False \
                          else np.asarray(Image.open(self.names[index]), dtype=np.uint16) \
                          if self.ignore_cache_for_hpe \
                          else self.depthmap_cachefile[self.num2str(0)][index] \
                          if self.preload_depth is False \
                          else self.depthmaps[index], #depthmap => R^{480 x 640}
                DT.ACTION: self.actions[index], # action => R^{1}
                DT.AUG_MODE: None if self.aug_modes is None else self.aug_modes[index],
                DT.AUG_PARAMS: None if self.aug_params is None else self.aug_params[index]
            }

        #print("SHAPE::: ", sample[DT.DEPTH_SEQ].shape, "DTYPE::: ", sample[DT.DEPTH_SEQ].dtype)

        ## a lot happens here.
        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        
        self._compute_dataset_size()

        self._compute_action_classes()

        self.num_samples = \
            self.train_size if (self.data_mode == DatasetMode.TRAIN) \
                            else self.test_size
        


        action_split_file = os.path.join(self.root, 
                                         'data_split_action_recognition.txt')
        
        with open(action_split_file) as f:
            all_lines = f.read().splitlines()
            
            # reduce dataset for testing only
            if self.reduce:
                all_lines = [line for line in all_lines \
                    if (('Training' in line) or ('Test' in line) or ('Subject_1' in line))]
            
            if (self.data_mode == DatasetMode.TRAIN):
                # get only train samples
                all_lines = all_lines[1:self.test_pos]
            elif (self.data_mode == DatasetMode.TEST):
                # get only test samples
                all_lines = all_lines[self.test_pos+1:]
            
           # print("All_lines", all_lines[:4], all_lines[-4:])

        # at this stage we only have either test or train data
        for line in all_lines:
            line, action_idx_str = line.split(' ')[0], line.split(' ')[1]

            with open(os.path.join(self.skeleton_dir, line, 'skeleton.txt')) as sk_f:
                # each sample is ONE WHOLE skeleton
                sk_lines = sk_f.read().splitlines()

            if self.task_mode == TaskMode.HAR:
                # add in bulks
                # e.g. Subject_1/open_juice_bottle/2 0
                self.names.append(
                    [os.path.join(self.video_dir, line, 'depth', img) for img in \
                        sorted(os.listdir(
                            os.path.join(self.video_dir, line, 'depth')
                        ))
                    ]
                )

                # to get joints as a matrix per sample do:
                # self.joints_world[0].reshape(-1, 21, 3)
                ## append a variable sized matrix
                self.joints_world.append(
                    np.stack(
                        [[float(item) for item in sk_line.split(' ')[1:]] for sk_line in sk_lines]
                    ).astype(np.float32)
                )

                # get gt middle_mcp world co-ords x,y,z of current sample (last item appended)
                self.coms_world.append(
                    self.joints_world[-1][:, 3*self.world_dim : 3*self.world_dim+3]
                )

                self.actions.append(
                    int(action_idx_str)
                )

            elif self.task_mode == TaskMode.HPE:
                # add frame_wise
                # we merge list with list of new items
                # NEW: FIXED DIR LISTING WHICH CAUSED WRONG DEPTHMAPS ASSOCIATED WITH KEYPOINTS FROM TXT. NOW LIST IS SORTED
                self.names += \
                    [os.path.join(self.video_dir, line, 'depth', img) for img in \
                        sorted(os.listdir(
                            os.path.join(self.video_dir, line, 'depth')
                        ))
                    ]

                new_joints_lst = \
                    [np.array([float(item) for item in sk_line.split(' ')[1:]]) for \
                        sk_line in sk_lines]
                
                self.joints_world += new_joints_lst
                
                # sample is 1D np.array, we extract mcp xyz from last added samples
                self.coms_world += \
                    [sample[3*self.world_dim : 3*self.world_dim+3] for \
                         sample in new_joints_lst]

                
                # all samples in seq must have the same action label
                self.actions += \
                    [int(action_idx_str) for _ in range(len(new_joints_lst))]
        
        assert(len(self.names) == self.num_samples)
        assert(len(self.joints_world) == self.num_samples)
        assert(len(self.coms_world) == self.num_samples)
        assert(len(self.actions) == self.num_samples)

    def _verify_depth_cache(self, data_mode_str, task_mode_str):
        '''
            In this function, the depthmaps are first searched via h5py cache file, 
            if found then nothing else is done as data will be loaded on the fly
            if not found then all required depthmaps are loaded and consequently
            stored into a h5py dataset file as a cache in the system, there is no
            option to disable cache as its required to improve data loading during
            training as otherwise there is a huge difference when loading depth maps

            filepath:
            datasets/hand_pose_action/data_{test|train}_{har|hpe}_cache.h5; libver=latest; swmr=True
        '''
        filepath = 'datasets/hand_pose_action/data_%s_%s_%scache.h5' % \
                   (data_mode_str, task_mode_str, "" if not self.reduce else "reduced_")
        filepath = os.path.normpath(filepath)

        
        if os.path.isfile(filepath):
            ## basic assertion
            with h5py.File(filepath, "r", libver='latest') as f:
                all_ok = (
                    len(f.keys()) == len(self.names) \
                        if self.task_mode == TaskMode.HAR
                        else f[self.num2str(0)].shape[0] == len(self.names)
                )
                if not all_ok:
                    from warnings import warn
                    warn("Inconsistent cache file: %s\nRebuilding cache..." % filepath)
                else:
                    self.depthmap_cachepath = filepath
                    return

            # if all ok simply dont do anything

        if not os.path.isdir(os.path.split(filepath)[0]):
            os.mkdir(os.path.split(filepath)[0])
        
        print("Building cache file: %s..." % filepath)
        
        # note np.asarray doesnt perform any copying so file is still kept
        # opened, for the HAR mode this is fine cause for each sequence we create a dataset and
        # thus file is copied and written to disk, atmost num_frames files are opened at once
        # for HPE mode we have to be careful here, we have to use np.array instead to create a copy
        # because we are also stacking here

        with h5py.File(filepath, 'w', libver='latest') as f:
            ## note item is either a string or list type of paths
            ## depending on dataset mode
            if self.task_mode == TaskMode.HAR:
                for i,item in enumerate(tqdm(self.names, desc='Converting depthmaps into h5py format')):
                    ## stack sequences of 1-channel depth imgs
                    ## multiple datasets for HAR type
                    data = np.stack(
                            [np.asarray(Image.open(img_path)) for# dtype=np.uint16
                                img_path in item], axis=0
                    )
                    f.create_dataset(
                        self.num2str(i),
                        data=data,
                        dtype=np.int32,# cause pytorch doesn't support uint16
                        compression="gzip",#lzf ## zlib ## szip 
                        compression_opts=7,
                        #chunks dont really help here because chunk cache per dataset
                        #is set to 1MB, unless we can change it even if we do, it differs
                        #or different datasets also if we directly store data to memory
                        #in first run its kinda helpless to use chunking
                        #chunks=(data.shape[0], data.shape[1], data.shape[2])
                    )

            elif self.task_mode == TaskMode.HPE:
                img_shape = np.array(Image.open(self.names[0])).shape
                num_files = len(self.names)
                f.create_dataset(
                        self.num2str(0), # for HPE dataset is called as if its a single item for HAR
                        dtype=np.int32,
                        shape=(num_files, img_shape[0], img_shape[1]),
                        compression="gzip",
                        compression_opts=7,
                        chunks=(1, img_shape[0], img_shape[1])
                )
                data_handle = f[self.num2str(0)]
                for i, item in tqdm(iterable=enumerate(self.names), desc='Converting depthmaps into h5py format',
                                    total=num_files):
                    data_handle[i] = np.array(Image.open(item))
        self.depthmap_cachepath = filepath


    def _preload_depth(self):
        '''
            Preload all depthmaps to main memory (RAM)
        '''
        with h5py.File(self.depthmap_cachepath, 'r', libver='latest') as f:
            if self.task_mode == TaskMode.HAR:
                for key,dataset in tqdm(f.items(),
                                        desc='Preloading depthmaps to RAM'):
                    self.depthmaps.append(dataset.value)
            elif self.task_mode == TaskMode.HPE:
                # preloading as list is faster than doing no.stack on
                # top of preload as list
                self.depthmaps = [
                                    f[self.num2str(0)][i] for i in tqdm(range(f[self.num2str(0)].shape[0]),
                                                                        desc='Preloading depthmaps to RAM')
                                 ]
                
                #f[self.num2str(0)].value # the first dataset is entire dataset

    def _compute_dataset_size(self):
        '''
            Read `data_split_action_recognition.txt`
            If mode is HAR then just read numbers off directly from file
            Otherwise, if mode is HPE:
                Get keyword '<ACTION_LABEL> <ACTION_INSTANCE>'
                Search for keyword in subject info txt
                Get corresponding number of frames and add to total
            
            Also store position of test samples in split file
        '''
        self.train_size, self.test_size = 0, 0

        action_split_file = os.path.join(self.root, 
                                         'data_split_action_recognition.txt')
        
        with open(action_split_file) as f:
            all_lines = f.read().splitlines()

            if self.reduce:
                from warnings import warn
                warn("Warning: Using reduced dataset (only Subj_1)")
                all_lines = [line for line in all_lines \
                    if (('Training' in line) or ('Test' in line) or ('Subject_1' in line))]

            if (self.task_mode == TaskMode.HAR):
                    # # changes made for support for reduced dataset
                    # # self.train_size = \
                    # #     int(([s.split(' ')[1] for i, s in \
                    # #                 enumerate(all_lines) if 'Training' in s])[0])
                    # # test_pos, test_size = \
                    # #     ([(i, s.split(' ')[1]) for i, s in \
                    # #                 enumerate(all_lines) if 'Test' in s])[0]
                    # # self.test_pos, self.test_size = test_pos, int(test_size)

                    curr_mode = ''
                    for i, line in enumerate(all_lines):
                        if 'Training' in line:
                            curr_mode = 'TRAIN'
                            continue    # go to next line
                        elif 'Test' in line:
                            self.test_pos = i
                            curr_mode = 'TEST'
                            continue
                        else:
                            if curr_mode == 'TRAIN':
                                self.train_size += 1
                            elif curr_mode == 'TEST':
                                self.test_size += 1

                    #####
            elif (self.task_mode == TaskMode.HPE):
                    curr_mode = ''
                    for i, line in enumerate(all_lines):
                        if 'Training' in line:
                            curr_mode = 'TRAIN'
                            continue    # go to next line
                        elif 'Test' in line:
                            self.test_pos = i
                            curr_mode = 'TEST'
                            continue
                        else:
                            # e.g. Subject_1/open_juice_bottle/2 0
                            line = line.split(' ')[0]

                            if curr_mode == 'TRAIN':
                                self.train_size += \
                                    sum(1 for line in \
                                                open(os.path.join(self.skeleton_dir, 
                                                                line, 'skeleton.txt')))
                            elif curr_mode == 'TEST':
                                self.test_size += \
                                    sum(1 for line in \
                                                open(os.path.join(self.skeleton_dir, 
                                                                line, 'skeleton.txt')))


    def _check_exists(self):
        # Check basic data
        # Subj_1, ..., Subj_6   <SUBJECT>
        #   ->  charge_cell_phone, ..., write   <ACTION_LABEL>
        #       -> 1, ..., 4    <INSTANCE_LABEL>
        #           -> skeleton.txt     <GT_Y_PTS>

        for subj in self.subj_dirnames:
            for action_lbl in os.listdir(os.path.join(self.skeleton_dir, subj)):
                for inst_lbl in os.listdir(os.path.join(self.skeleton_dir, subj, action_lbl)):
                    annot_file = os.path.join(self.skeleton_dir, subj, action_lbl, inst_lbl, 'skeleton.txt')    
                    if not os.path.exists(annot_file):
                        print('Error: annotation file {} does not exist'.format(annot_file))
                        return False

        # in future check pre-computed centers

        return True


    def _compute_action_classes(self):
        action_info_file = os.path.join(self.root, 
                                         'action_object_info.txt')
        self.action_class_dict = {}

        with open(action_info_file) as f:
            all_lines = f.read().splitlines()

            # remove first line as its a header
            all_lines = all_lines[1:]
            for i, line in enumerate(all_lines):
                self.action_class_dict[i] = line.split(' ')[1]
               
            self.action_classes = len(self.action_class_dict)
    

    def make_transform_params_static(self, AugType: IntEnum, getAugModeParamFn: Callable[[List[IntEnum]], Any],
                                     custom_aug_modes:List[IntEnum]=None):
        '''
            If no randomisation is requested then try to produce deterministic params for transformation.
            Because this requires knowledge of how many samples exist it cannot be done by transformers
            Instead params will be supplied by dataset __call__ function along with the usual stuff
        '''
        print("[FHAD] Note: Using deterministic params for transformation!")
        
        ## now do something
        if custom_aug_modes is not None:
            print("[FHAD] Using Supplied AugModes: %a" % custom_aug_modes)
            valid_aug_modes = np.array(custom_aug_modes)
        else:
            valid_aug_modes = np.arange(len(AugType))
            print("[FHAD] Using ALL AugModes: %a" % valid_aug_modes)
        
        #print("[MSRA] AugModeLims (RotAbsLim, ScaleStd, TransStd): ", self.aug_lims.abs_rot_lim_deg,
        #        self.aug_lims.scale_std, self.aug_lims.trans_std)
        ## reset seed
        np.random.seed(self.RAND_SEED)
        self.aug_modes = np.random.choice(valid_aug_modes, replace=True, size=len(self))
        self.aug_modes = list(map(lambda i: AugType(i), self.aug_modes)) # convert to enumtype
        
        self.aug_params = [
            getAugModeParamFn([aug_mode]) \
                                for aug_mode in self.aug_modes
        ]



