from torchvision import datasets, transforms
#from torch.utils.data.dataloader import default_collate

from base import BaseDataLoader, BaseDataType

from datasets import HandPoseActionDataset

from .data_transformers import JointSeqCenterer, ActionOneHotEncoder, ToTuple

from .collate_functions import CollateJointsSeqBatch

import time

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
                validation_split=0.0, num_workers=1, debug=False):

        t = time.time()
        #not needed atm as NLLloss needs only class idx
        #ActionOneHotEncoder(action_classes=45)
        trnsfrm = transforms.Compose([
                                        JointSeqCenterer(),
                                        ToTuple(extract_type='joints_action_seq')
                                    ])
        self.dataset = HandPoseActionDataset(data_dir, dataset_type, 'har',
                                            3, transform=trnsfrm, reduce=False)


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