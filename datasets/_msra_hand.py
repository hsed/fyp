import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset

## starting from 1, each component of y_gt_mm_keypoints is
# --1(wrist) -- index0
# --2(index_mcp), 3(index_pip), 4(index_dip), 5(index_tip)
# --6(middle_mcp), 7(middle_pip), 8(middle_dip), 9(middle_tip)
# --10(ring_mcp), 11(ring_pip), 12(ring_dip), 13(ring_tip)
# --14(little_mcp), 15(little_pip), 16(little_dip), 17(little_tip)
# --18(thumb_mcp), 19(thumb_pip), 20(thumb_dip), 21(thumb_tip) -- index20


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    # from pixel,pixel,mm values to mm,mm,mm values
    # i.e. from a depth_map 2d matrix containing dpeth in mm
    # to actual 3d cords plottable in 3d view
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z ## thi is unchanged
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    return p_x, p_y


def depthmap2points(image, fx, fy):
    ## convert (x,y,z) with x,y in img (pixel) values and z in mm
    ## to (x,y,z) ALL in mm values
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points

## is this the inverse of the above function?? confirm....
## this returns something like np.meshgrid so like x,y values
## it returns x,y and corresponding z is points[i,2] which is unchanged
## so what you can do is:
## let tmp_img = np.zeros((h,w)) so 2d matrix
##  for each pixel i from 1 -> 76800:
##    do: tmp_img[pixes[i,0], pixels[i,1]] = pixels[2]
## check if this works by checking with original depthmap
## if it works with orig res then u can also resize
## also use plots to check if the shape looks somewhat ok.
## see plotting function in old code        
def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels


def pixelsdepth2depthmap(pixels, deptharray, img, img_width, img_height):
    tmp_img = np.zeros((img_height,img_width))
    #tmp_


def load_depthmap(filename, img_width, img_height, max_depth):
    '''
        Given a bin file for one sample e.g. 000000_depth.bin
        Load the depthmap
        Load the gt bounding box countaining sample hand
        Clean the image by:
            - setting depth_values within bounding box as actual
            - setting all other depth_values to MAX_DEPTH
        I.e. all other stuff is deleted
        
    '''
    with open(filename, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I'*6, data[:6*4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f'*num_pixel, data[6*4:])

        cropped_image = np.asarray(cropped_image).reshape(bottom-top, -1)
        depth_image = np.zeros((img_height, img_width), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image
        
        ## be careful here thats not the way of deep_prior
        ### thus we have commented this line!
        #depth_image[depth_image == 0] = max_depth 
        ## in deep_prior max_depth is kept at 0 and only changed in the end.

        ## plot here to see how it looks like

        return depth_image


class MARAHandDataset(Dataset):
    def __init__(self, root, center_dir, mode, test_subject_id, transform=None, reduce=False,
                 use_refined_com=False):
        '''
            `reduce` => Train only on 1 gesture and 2 subjects, CoM won't work correctly
            `use_refined_com` => True: Use GT MCP (ID=5) ref; False: Use refined CoM pretrained.
        '''
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.fx = 241.42
        self.fy = 241.42
        self.joint_num = 21
        self.world_dim = 3
        self.folder_list = ['1'] if reduce else \
            ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
        self.subject_num = 3 if reduce else 9 ## number of subjects

        self.root = root
        self.center_dir = center_dir
        self.mode = mode
        self.test_subject_id = test_subject_id ## do testing using this ID's data
        self.transform = transform
        self.use_refined_com = use_refined_com

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        if not self._check_exists(): raise RuntimeError('Invalid MSRA hand dataset')
        
        ### load all the y_values and corresponding corrected CoM values using
        ### 'train.txt' and 'center_train_refined'
        self._load()
    
    def __getitem__(self, index):
        ## the x-values are loaded 'on-the-fly' as and when required.
        ## this function is called internally by pytorch whenever a new sample needs to
        ## be loaded.
        depthmap = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)
        
        ## Note: For deep-prior we DO NOT want to do this.
        ## are input should remain as 2D img with depth values
        ## points = depthmap2points(depthmap, self.fx, self.fy)
        
        ## originally points are (240,320, 3)
        ## later they are reshaped to (240*320==76800, 3)
        ## points = points.reshape((-1, 3))

        #self.ref_pts[index], # 3d ref point of centre of mass, this is the REFINED CoM points
        refpt = self.joints_world[index][5] if not self.use_refined_com else self.ref_pts[index]

        sample = {
            'name': self.names[index], # sample name
            #'points': points, # 3d points of the sample i.e. 3d point cloud view. unneeded atm
            'joints': self.joints_world[index], # 3d joints of the sample
            'refpoint': refpt,
            'depthmap': depthmap, # also send the depth map as sample
        }

        ## a lot happens here.
        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim), dtype=np.float32)
        self.ref_pts = np.zeros((self.num_samples, self.world_dim), dtype=np.float32)
        self.names = []

        # Collect reference center points strings
        ## ref points are collected based on TEST SSUBJ ID
        ## center_train_TESTSUBJID_refined.txt implies the refined net was trained using 
        ## all subjects except for the test subject and then evaluated for all subjects (incl test subject)
        ## the train file contains results for all data so 9 * 17 * 500
        ## the test file contains results only for one data so 17 * 500
        if self.mode == 'train': ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        else: ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
                ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for mid in range(self.subject_num):
            ## for each model_id
            if self.mode == 'train': model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test': model_chk = (mid == self.test_subject_id)
            else: raise RuntimeError('unsupported mode {}'.format(self.mode))
            
            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')

                    lines = []
                    with open(annot_file) as f:
                        lines = [line.rstrip() for line in f]

                    # skip first line as it contains an int: `num of samples`
                    for i in range(1, len(lines)):
                        # referece point, this contains REFINED CoM 3d co-ord
                        splitted = ref_pt_str[file_id].split()
                        if splitted[0] == 'invalid':
                            print('Warning: found invalid reference frame')
                            file_id += 1
                            continue
                        else:
                            self.ref_pts[frame_id, 0] = float(splitted[0]) #CoM x
                            self.ref_pts[frame_id, 1] = float(splitted[1]) #CoM y
                            self.ref_pts[frame_id, 2] = float(splitted[2]) # CoM z

                        # joint points... the gt (y-val) values in 3D for 21 joints
                        splitted = lines[i].split()
                        for jid in range(self.joint_num):
                            self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])    ## ATTENTION: NOTE THE NEGATION THUS THIS IS SAME AS DEEP-PRIOR
                        
                        filename = os.path.join(self.root, 'P'+str(mid), fd, '{:0>6d}'.format(i-1) + '_depth.bin')
                        self.names.append(filename)

                        frame_id += 1   ## this may differ from i if any CoM is invalid otherwise it won't
                        ## frame_id is used to ensure we only set samples with valid CoM values in our training/testing data matrix
                        ## all other samples are ignored.
                        file_id += 1 ## this is exactly same as i, so frame_id == i , so extra

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id:
                    self.test_size += num
                else: 
                    self.train_size += num

    def _check_exists(self):
        # Check basic data
        for mid in range(self.subject_num):
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                if not os.path.exists(annot_file):
                    print('Error: annotation file {} does not exist'.format(annot_file))
                    return False

        # Check precomputed centers by v2v-hand model's author
        for subject_id in range(self.subject_num):
            center_train = os.path.join(self.center_dir, 'center_train_' + str(subject_id) + '_refined.txt')
            center_test = os.path.join(self.center_dir, 'center_test_' + str(subject_id) + '_refined.txt')
            if not os.path.exists(center_train) or not os.path.exists(center_test):
                print('Error: precomputed center files do not exist')
                return False

        return True
