import os
import sys
import struct
import time

from enum import IntEnum

import cv2
import numpy as np
from torch.utils.data import Dataset

from base import BaseDataType as DT

# renaming to avoid error in this file
from base import BaseDatasetType as DatasetMode
from base import BaseTaskType as TaskMode

#### TO HEAVILY EDIT TO SUPPORT HAND ACTION DATASET
## starting from 0, each component of y_gt_mm_keypoints is
# --0(wrist)
# --1(thumb_mcp), 2(index_mcp), 3(middle_mcp), 4(ring_mcp), 5(pinky_mcp)
# --6(thumb_pip), 7(thumb_dip), 8(thumb_tip),
# --9(index_pip), 10(index_dip), 11(index_tip),
# --12(middle_pip), 13(middle_dip), 14(middle_tip),
# --15(ring_pip), 16(ring_dip), 17(ring_tip),
# --18(pinky_pip), 19(pinky_dip), 20(pinky_tip),
# total 21 joints, each having 3D co-ords




# def load_depthmap(filename, img_width, img_height, max_depth):
#     '''
#         Given a bin file for one sample e.g. 000000_depth.bin
#         Load the depthmap
#         Load the gt bounding box countaining sample hand
#         Clean the image by:
#             - setting depth_values within bounding box as actual
#             - setting all other depth_values to MAX_DEPTH
#         I.e. all other stuff is deleted
        
#     '''
#     with open(filename, mode='rb') as f:
        
        
        
#         ## be careful here thats not the way of deep_prior
#         ### thus we have commented this line!
#         #depth_image[depth_image == 0] = max_depth 
#         ## in deep_prior max_depth is kept at 0 and only changed in the end.

#         ## plot here to see how it looks like

#         return depth_image



class HandPoseActionDataset(Dataset):
    def __init__(self, root, data_mode, task_mode,  test_subject_id, transform=None, reduce=False):
        '''
            `reduce` => Train only on 1 gesture and 2 subjects, CoM won't work correctly
            `use_refined_com` => True: Use GT MCP (ID=5) ref; False: Use refined CoM pretrained.
                                Currently disabled
            
            Currently this class is used to load data to train a HAR
            Another class
        '''
        self.dpt_img_width = 640 #320
        self.dpt_img_height = 480 #240

        # TO CONFIRM TODO: !!!!
        self.min_depth = 100
        self.max_depth = 700
        
        self.fx_d = 475.065948
        self.fy_d = 475.065857

        self.px_d = 315.944855   # aka u0_d
        self.py_d = 245.287079   # aka v0_d

        # depth intrinsic transform matrix
        self.cam_intr_d = np.array([[self.fx_d, 0, self.px_d],
                                    [0, self.fy_d, self.py_d], 
                                    [0, 0, 1]])
        
        self.joint_num = 21
        self.world_dim = 3
        
        self.reorder_idx = np.array([
            0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12,
            13, 14, 4, 15, 16, 17, 5, 18, 19, 20
        ])
        
        self.test_pos = -1
        
        self.subject_num = 3 if reduce else 6 ## number of subjects

        
        ### setup directories
        self.root = root
        self.skeleton_dir = os.path.join(self.root, 'Hand_pose_annotation_v1')
        self.video_dir = os.path.join(self.root, 'Video_files') 
        self.info_dir = os.path.join(self.root, 'Subjects_info')
        self.subj_dirnames = ['Subject_1'] if reduce else \
            ['Subject_%d' % i for i in range(1, 7)]
        #self.center_dir = center_dir # not in use

        # setup modes
        if data_mode not in DatasetMode._value2member_map_:
            raise RuntimeError("Invalid dataset type, choose from: ",
                               DatasetMode._value2member_map_.keys())
        if task_mode not in TaskMode._value2member_map_:
            raise RuntimeError("Invalid task type, choose from: ",
                               TaskMode._value2member_map_.keys())

        self.data_mode = DatasetMode._value2member_map_[data_mode]
        self.task_mode = TaskMode._value2member_map_[task_mode]

        self.test_subject_id = test_subject_id ## do testing using this ID's data
        self.transform = transform
        #self.use_refined_com = use_refined_com not in use

        assert self.test_subject_id >= 0 and self.test_subject_id < len(self.subj_dirnames)

        # currently a very weak check, only checks for skeletons
        if not self._check_exists(): raise RuntimeError('Invalid Hand Pose Action Dataset')
        
        ### load all the y_values and corresponding corrected CoM values using
        ### 'train.txt' and 'center_train_refined'
        self._load()
    
    def __getitem__(self, index):
        ## the x-values are loaded 'on-the-fly' as and when required.
        ## this function is called internally by pytorch whenever a new sample needs to
        ## be loaded.
        #depthmap = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)
        # any depth to allow 16-it images as the depths are 16-bit here

        if self.task_mode == TaskMode.HAR:
            # stack sequences of 1-channel depth imgs
            depthmaps = np.stack(
                [cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) for \
                    img_path in self.names[index]]
            )
            sample = {
                DT.NAME_SEQ: self.names[index], # sample names => R^{NUM_FRAMES x 1}
                DT.JOINTS_SEQ: self.joints_world[index], # 3d joints => R^{NUM_FRAMES x 63}
                DT.COM_SEQ: self.coms_world[index], # => R^{NUM_FRAMES x 3}
                DT.DEPTH_SEQ: depthmaps, # depthmaps => R^{NUM_FRAMES x 480 x 640}
                DT.ACTION: self.actions[index], # action => R^{1}
            }

        elif self.task_mode == TaskMode.HPE:
            depthmap = cv2.imread(self.names[index], cv2.IMREAD_ANYDEPTH)
            sample = {
                DT.NAME: self.names[index], # sample name => R^{1}
                DT.JOINTS: self.joints_world[index], # 3d joints of the sample => R^{63}
                DT.COM: self.coms_world[index], # => R^{3}
                DT.DEPTH: depthmap, # depthmap => R^{480 x 640}
                DT.ACTION: self.actions[index] # action => R^{1}
            }



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
        
        
        
        ### skeleton ground truth ###
        # this will be a list of variable sized 2D matrices (HAR) OR
        # fixed sized vectors (HPE)
        self.names = []
        self.joints_world = []
        self.coms_world = []
        self.actions = []
        


        action_split_file = os.path.join(self.root, 
                                         'data_split_action_recognition.txt')
        
        with open(action_split_file) as f:
            all_lines = f.read().splitlines()
            
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
                        os.listdir(
                            os.path.join(self.video_dir, line, 'depth')
                        )
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

                # get gt middle_mcp world co-ords x,y,z of current sample
                self.coms_world.append(
                    self.joints_world[-1][:, 3*self.world_dim : 3*self.world_dim+3]
                )

                self.actions.append(
                    int(action_idx_str)
                )

            elif self.task_mode == TaskMode.HPE:
                # add frame_wise
                # we merge list with list of new items
                self.names += \
                    [os.path.join(self.video_dir, line, 'depth', img) for img in \
                        os.listdir(
                            os.path.join(self.video_dir, line, 'depth')
                        )
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

            if (self.task_mode == TaskMode.HAR):
                    self.train_size = \
                        int(([s.split(' ')[1] for i, s in \
                                    enumerate(all_lines) if 'Training' in s])[0])
                    test_pos, test_size = \
                        ([(i, s.split(' ')[1]) for i, s in \
                                    enumerate(all_lines) if 'Test' in s])[0]
                    self.test_pos, self.test_size = test_pos, int(test_size)
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

