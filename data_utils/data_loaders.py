from torchvision import datasets, transforms
#from torch.utils.data.dataloader import default_collate

from datasets import HandPoseActionDataset

from .base_data_loader import BaseDataLoader
from .base_data_types import BaseDataType
from .data_transformers import *
from .collate_functions import CollateJointsSeqBatch, CollateDepthJointsBatch

import time

from tqdm import tqdm

from copy import deepcopy

'''
    __init__ will auto import all modules in this folder!
'''

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



'''
    make a data loader class here that acceps whether har or hpe and train or test
    then from then onwards all transformers can be defined WITHIN the scope of loader as shown here
    even the dataset!

    so u can init the correct transformer or even tell it which mode u r in hpe/har

'''


class JointsActionDataLoader(BaseDataLoader):
    '''
        To load joints and actions of same batch_size but variable sequence length for joints
        In future we somehow need to pack/pad sequences using 'pack_sequence'

        Use validation_split to extract some samples for validation
    '''

    def __init__(self, data_dir, dataset_type, batch_size, shuffle, 
                validation_split=0.0, num_workers=1, debug=False, reduce=False):

        t = time.time()
        #not needed atm as NLLloss needs only class idx
        #ActionOneHotEncoder(action_classes=45)
        trnsfrm = transforms.Compose([
                                        JointReshaper(),
                                        JointSeqCentererStandardiser(),
                                        ToTuple(extract_type='joints_action_seq')
                                    ])
        self.dataset = HandPoseActionDataset(data_dir, dataset_type, 'har', transform=trnsfrm, reduce=reduce)


        ## initialise super class appropriately
        super(JointsActionDataLoader, self).\
            __init__(self.dataset, batch_size, shuffle, validation_split, num_workers, 
                     collate_fn=CollateJointsSeqBatch())

        if debug:
            print("Data Loaded! Took: %0.2fs" % (time.time() - t))
            test_sample = self.dataset[0]
            print("Sample Final Shape: ", test_sample[0].shape, test_sample[1].shape)
            print("Sample Final Values:\n", test_sample[0], "\n", test_sample[1])
            print("Sample Joints_Std_MIN_MAX: ", test_sample[0].min(), test_sample[0].max())




class DepthJointsDataLoader(BaseDataLoader):
    '''
        To load joints and corresponding depthmaps for training of HPE/HPG model seperately
        in future, action label per frame of action can also be loaded for a action conditioned
        HPE training

        Use validation_split to extract some samples for validation
    '''

    def __init__(self, data_dir, dataset_type, batch_size, shuffle, 
                validation_split=0.0, num_workers=1, debug=False, reduce=False):

        t = time.time()
        #not needed atm as NLLloss needs only class idx
        #ActionOneHotEncoder(action_classes=45)
        trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': 200,
            'debug_mode': debug
        }
        trnsfrm = transforms.Compose([
                                        JointReshaper(**trnsfrm_base_params),
                                        DepthCropper(**trnsfrm_base_params),
                                        DepthAndJointsAugmenter(
                                            aug_mode_lst=[
                                                AugType.AUG_NONE,
                                                AugType.AUG_ROT,
                                                AugType.AUG_TRANS
                                            ],
                                            **trnsfrm_base_params),
                                        DepthStandardiser(**trnsfrm_base_params),
                                        JointCentererStandardiser(**trnsfrm_base_params),
                                        ToTuple(extract_type='depth_joints')
                                    ])
        self.dataset = HandPoseActionDataset(data_dir, dataset_type, 'hpe',
                                             transform=trnsfrm, reduce=reduce)


        ## initialise super class appropriately
        super(DepthJointsDataLoader, self).\
            __init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                     collate_fn=CollateDepthJointsBatch())# need to init collate fn

        if debug:
            print("Data Loaded! Took: %0.2fs" % (time.time() - t))
            test_sample = self.dataset[0]
            print("Sample Final Shape: ", test_sample[0].shape, test_sample[1].shape)
            print("Sample Final Values:\n", test_sample[0], "\n", test_sample[1])
            print("Sample Joints_Std_MIN_MAX: ", test_sample[0].min(), test_sample[0].max())




class PersistentDataLoader(object):
    '''
        A memory heavy function to store all data in the dataset at once into memory
        Data is extracted using an underlying dataloader y iterating over dataloader once
        and storing results to disk.

        It helps cause for all subsequent epochs, data is readily available
    '''
    
    def __init__(self, dataloader, load_on_init=True):
        '''
            Load on init => Load dataset as soon as class is initialised
        '''
        #self.verbose = verbose
        self.dataloader = dataloader
        self.data = None

        self.batch_size = dataloader.batch_size
        self.n_samples = dataloader.n_samples if hasattr(dataloader, 'n_samples') else \
                          len(dataloader.sampler) if hasattr(dataloader, 'sampler') else \
                              len(dataloader.dataset)
        self.length = len(self.dataloader)

        ### do some checks here to check if dataloader is of the right class

        if load_on_init:
            self.load_data()

    def load_data(self):
        '''
            Load all data to RAM by iterating through a pytorch dataloader
        '''
        self.data = [
            deepcopy(item) for item in tqdm(self.dataloader, total=len(self.dataloader), 
                                                     desc="Loading transformed dataset to memory")
        ]

        del self.dataloader
    

    def __getitem__(self, index):
        # Just to be safe
        if self.data is None:
            self.load_data()
        
        return self.data[index]
    
    def __len__(self):
        return self.length