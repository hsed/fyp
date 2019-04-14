from torchvision import datasets, transforms
#from torch.utils.data.dataloader import default_collate

from datasets import BaseDataType, HandPoseActionDataset, \
                     FHADCameraIntrinsics, DepthParameters, \
                     MSRACameraIntrinsics, MSRAHandDataset

from .base_data_loader import BaseDataLoader
from .data_transformers import *
from .collate_functions import CollateJointsSeqBatch, CollateDepthJointsBatch

from ._dp_transformers import DeepPriorXYTransform, DeepPriorYTransform

from .data_augmentors import AugType

import time

from tqdm import tqdm

from copy import deepcopy

from argparse import Namespace

'''
    __init__ will auto import all modules in this folder!
'''

def _check_pca(data_dir, pca_transformer, data_transforms,
               transform_base_params, y_pca_len=int(2e5), use_msra=False,
               randomise_params=True):
    if pca_transformer.transform_matrix_np is None:
    # each sample is 1x21x3 so we use cat to make it 3997x21x3
    # id we use stack it intriduces a new dim so 3997x1x21x3
    # load all y_sample sin tprch array
    # note only train subjects are loaded!
    

        ### dont load depthmaps here!!
        ### pca is always trained on train data, never on test!
        ### validation set is seen by pca but it doesnt really matter we
        ### are not tweaking pca settings (30 is kept as standard)
        if use_msra:
            print("[PCA_CHECKER] Info: Using MSRA for PCA...")
            y_set = MSRAHandDataset(root=data_dir, center_dir='', mode='train', test_subject_id=0,
                                    transform=data_transforms, reduce=False, use_refined_com=False, 
                                    retrieve_depth=False, randomise_params=randomise_params)
            
            #print("FIRST ITEM:\n", y_set[0])
            #quit()

            if not randomise_params:
                print('[PCA_CHECKER] Randomise Params are off, deterministic PCA will be computed')
                rot_lim=transform_base_params['aug_lims'].abs_rot_lim_deg
                sc_std=transform_base_params['aug_lims'].scale_std
                tr_std=transform_base_params['aug_lims'].trans_std

                allowed_aug_modes = np.arange(len(AugType))
                if isinstance(data_transforms, transforms.Compose):
                    for item in data_transforms.transforms:
                        if getattr(item, 'aug_mode_lst', False):
                            #print("FOUND LIST", item.aug_mode_lst)
                            allowed_aug_modes = item.aug_mode_lst
                print('[PCA_CHECKER] Deterministic AugMode List: ', allowed_aug_modes)
                y_set.make_transform_params_static(AugType, \
                    (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
                     custom_aug_modes= allowed_aug_modes)



        else:
            print("Info: Using FHAD for PCA...")
            y_set = HandPoseActionDataset(data_dir, 'train', 'hpe',
                                        transform=data_transforms, reduce=False, retrieve_depth=False)
        
        
        
        y_pca_len = y_pca_len #int(2e5)

        if randomise_params:
            # default option
            y_idx_pca = np.random.choice(len(y_set), y_pca_len, replace=True)
        else:
            # special case
            np.random.seed(getattr(y_set, 'RAND_SEED', 0)) # fix a seed from dataset or just 0
            y_idx_pca = np.random.choice(len(y_set), y_pca_len, replace=True)
        #print(y_idx_pca, y_idx_pca.shape)
        #y_loader = torch.utils.data.DataLoader(y_set, batch_size=1, shuffle=True, num_workers=0)
        #print('==> Collating %d y_samples for PCA ..' % y_pca_len)
        
        fullYList = []
        for item in tqdm(y_idx_pca, 
                                desc='Collating y_samples for PCA ..'):  #y_loader
            fullYList.append(y_set[item])
        
        y_train_samples = torch.from_numpy(np.stack(fullYList)) #tuple(y_loader) #torch.cat()
        #print(fullList)
        print("\nY_GT_STD SHAPE: ", y_train_samples.shape, 
                "[Min, Max]: [%0.4f, %0.4f]" % (y_train_samples.numpy().min(), y_train_samples.numpy().max()), "\n")
        # in future just use fit command, fit_transform is just for testing
        print('==> fitting PCA ..')
        y_low_dim = pca_transformer.fit_transform(y_train_samples)
        print("PCA_Y_SHAPE: ", y_low_dim.shape,
              "[Min, Max]: [%0.4f, %0.4f]" % (y_low_dim.min(), y_low_dim.max()), "\n")
        print('==> PCA fitted ..')
        #quit()
        del y_train_samples
        del fullYList
        #del y_loader
        del y_set
    
    ## save link to pca weight and bias
    ## save these to provide to model later
    return pca_transformer.transform_matrix_np, pca_transformer.mean_vect_np


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
        self.dataset = HandPoseActionDataset(data_dir, dataset_type, 'har', transform=trnsfrm, reduce=reduce,
                                             retrieve_depth=False)


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

        TODO: Augmentations are disabled as not working, will work on them in future
    '''

    def __init__(self, data_dir, dataset_type, batch_size, shuffle, pca_components=30,
                validation_split=0.0, num_workers=1, debug=False, reduce=False,
                test_mm_err=False,
                use_pca_cache=True, pca_overwrite_cache=False, preload_depth=False,
                use_msra=False, data_aug=None, pca_data_aug=None,
                use_orig_transformers=False, use_orig_transformers_pca=False,
                randomise_params=True, crop_depth_ver=0, pca_size=int(2e5),
                crop_pad_3d=[40, 40, 50.], cube_side_mm=190):
        '''
            preload depth is not really needed, pytorch is intelligent and after
            first epoch everything is preloaded i.e. shared memory is used for
            dataset because training after first epoch is much faster
            even without preloading
        '''

        if reduce:
            num_workers = 0
            print('[DATALOADER] In reduce mode, setting num_workers = 0')

        # force no new pca calc in test mode
        if dataset_type == 'test' and pca_overwrite_cache:
            pca_overwrite_cache = False
            print("[DATALOADER] PCA Overwrite Cache and Test Type are mutually exclusive, a cache must have been written during train mode...")
        
        if use_msra:
            print("WARNING: USING MSRA DATASET")
            print("Several settings will be overidden")
            data_dir = '../deep-prior-pp-pytorch/datasets/MSRA15'
            if dataset_type == 'train':
                # force pca override in train_mode
                print("PCA overwritten in MSRA train mode!")
                #pca_overwrite_cache = True#True # ensure correct cache is calculated

        if use_orig_transformers:
            print("WARNING: Using Original Transformers, Only working with MSRA!")
        if use_orig_transformers_pca:
            print("WARNING: Using Original Transformers for PCA, Only working with MSRA!")

        t = time.time()

        ## load pca if overwrite is false and use_cache is true
        ## doesn't distinguish between msra and fhad, just loads
        pca_transformer = PCATransformer(n_components=pca_components,
                                         use_cache=use_pca_cache,
                                         overwrite_cache=pca_overwrite_cache)
        
        #not needed atm as NLLloss needs only class idx
        #ActionOneHotEncoder(action_classes=45)
        trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': cube_side_mm, #210 if not use_msra else 190, #200,#220, #200, #400 for fhad
            'cam_intrinsics': FHADCameraIntrinsics if not use_msra else MSRACameraIntrinsics,
            'dep_params': DepthParameters,
            'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180),
            'crop_depth_ver': crop_depth_ver,
            'crop_pad_3d': tuple(crop_pad_3d), # tuple required for transformers, but yaml loads as list by def
            'debug_mode': debug,
        }

        # only for training
        # aug_mode_lst = [
        #     AugType.AUG_NONE,
        #     AugType.AUG_ROT,
        #     AugType.AUG_TRANS
        # ]
        # pca_aug_mode_lst = [
        #     AugType.AUG_NONE,
        #     AugType.AUG_ROT,
        #     AugType.AUG_TRANS
        # ]

        train_transform_list = [
            JointReshaper(**trnsfrm_base_params), # NOOP for MSRA
            DepthCropper(**trnsfrm_base_params),
        ]

        if data_aug is not None and isinstance(data_aug, list):
            train_aug_list = list(map(lambda i: AugType(i), data_aug))
            train_transform_list.append(
                DepthAndJointsAugmenter(aug_mode_lst=train_aug_list,**trnsfrm_base_params),
            )
            print("Using data augmentation %a for training..." % train_aug_list)

        train_transform_list += [
            DepthStandardiser(**trnsfrm_base_params),
            JointCentererStandardiser(**trnsfrm_base_params),
            pca_transformer,
            ToTuple(extract_type='depth_joints')
        ]

        val_transform_list = train_transform_list.copy()
        if isinstance(val_transform_list[-5], DepthAndJointsAugmenter):
            #print("Deleting depthjointaug for val")
            del val_transform_list[-5]
            #print("New lists:\n", train_transform_list, '\n', val_transform_list)

        ### test or train transforms ###
        if dataset_type == 'train':
            trnsfrm = transforms.Compose(train_transform_list) if not use_orig_transformers else \
                        transforms.Compose([
                            DeepPriorXYTransform(aug_mode_lst=train_aug_list),
                            pca_transformer,
                            ToTuple(extract_type='depth_joints')
                        ])
            
            val_transfrm = transforms.Compose(val_transform_list) if not use_orig_transformers else \
                            transforms.Compose([
                                                DeepPriorXYTransform(aug_mode_lst=[AugType.AUG_NONE]),
                                                pca_transformer,
                                                ToTuple(extract_type='depth_joints')
                                            ])
        elif dataset_type == 'test':
            if test_mm_err:
                # NOTE: No longer in use...
                ## output is 21,3 for later pca untransformed of output testing
                print("Testset target is mm values centered and standardised...")
                #print("Note: [NEW] Transformer doesn't standardise target!")
                trnsfrm = transforms.Compose([
                                JointReshaper(**trnsfrm_base_params),
                                DepthCropper(**trnsfrm_base_params),
                                DepthStandardiser(**trnsfrm_base_params),
                                JointCentererStandardiser(flatten_shape=False, **trnsfrm_base_params), #JointCenterer(**trnsfrm_base_params), #JointCentererStandardiser(flatten_shape=False, **trnsfrm_base_params),
                                ToTuple(extract_type='depth_joints')
                            ])
            else:
                print("Testset target is pca output after centering and standardising...")
                trnsfrm = transforms.Compose([
                                JointReshaper(**trnsfrm_base_params),
                                DepthCropper(**trnsfrm_base_params),
                                DepthStandardiser(**trnsfrm_base_params),
                                JointCentererStandardiser(flatten_shape=True, **trnsfrm_base_params),
                                pca_transformer,
                                ToTuple(extract_type='depth_joints')
                            ])

        
        ### pca transforms ###
        pca_trnsfrm_list = [JointReshaper(**trnsfrm_base_params), DepthCropper(**trnsfrm_base_params)]
        if pca_data_aug is not None and isinstance(pca_data_aug, list):
            pca_aug_list = list(map(lambda i: AugType(i), pca_data_aug))
            pca_trnsfrm_list.append(
                DepthAndJointsAugmenter(aug_mode_lst=pca_aug_list,**trnsfrm_base_params),
            )
            print("Using data augmentation %a for pca (if cache overwrite is true)..." % pca_aug_list)
        pca_trnsfrm_list += [JointCentererStandardiser(**trnsfrm_base_params), ToTuple(extract_type='joints')]
        
        pca_trnsfrms = transforms.Compose(pca_trnsfrm_list) if not use_orig_transformers_pca else \
                       transforms.Compose([
                            DeepPriorYTransform(aug_mode_lst=pca_aug_list), # NOTE: this was set to train_aug_list all along!
                            ToTuple(extract_type='joints')
                       ])


        #### dataset loading ####
        rot_lim=trnsfrm_base_params['aug_lims'].abs_rot_lim_deg
        sc_std=trnsfrm_base_params['aug_lims'].scale_std
        tr_std=trnsfrm_base_params['aug_lims'].trans_std

        ## note randomise params=False not possible in fhad currently

        if use_msra:
            self.dataset = MSRAHandDataset(root=data_dir, center_dir='', mode=dataset_type, 
                                           test_subject_id=0, transform=trnsfrm, reduce=reduce,
                                           use_refined_com=False, randomise_params=randomise_params)
            ### new ###
            if not randomise_params:
                print("[DATA_LOADER] Randomise_Opt = False, Will try to enformce deterministic output")
                ## very important to call function below!
                self.dataset.make_transform_params_static(AugType, \
                    (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
                     custom_aug_modes=train_aug_list)

            if validation_split > 0.0:
                # validation idx are provided by val_sample from within train_set
                # we need to do this due to different transformations
                self.val_dataset = MSRAHandDataset(root=data_dir, center_dir='', mode=dataset_type, 
                                                   test_subject_id=0, transform=val_transfrm, reduce=reduce,
                                                   use_refined_com=False, randomise_params=randomise_params)
                if not randomise_params:
                    self.val_dataset.make_transform_params_static(AugType, \
                        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
                        custom_aug_modes=train_aug_list)
        else:
            self.dataset = HandPoseActionDataset(data_dir, dataset_type, 'hpe',
                                                 transform=trnsfrm, reduce=reduce,
                                                 retrieve_depth=True, preload_depth=preload_depth)
            if validation_split > 0.0:
                self.val_dataset = HandPoseActionDataset(data_dir, dataset_type, 'hpe',
                                                 transform=val_transfrm, reduce=reduce,
                                                 retrieve_depth=True, preload_depth=preload_depth)

        ### before exiting ensure PCA is pre-computed (using cache on disk)
        ### if not we need to compute
        ### for train: used as pre-processing
        ### for test: pca matx and mean is used for weights, bias as last layer
        ### NEW! Also add determinism if possible
        self.pca_weights_np, self.pca_bias_np = _check_pca(data_dir, pca_transformer, pca_trnsfrms,
                                                           trnsfrm_base_params, use_msra=use_msra,
                                                           y_pca_len=pca_size,
                                                           randomise_params=randomise_params)
        
        #print("ITEM_X:\n", list(self.dataset[0][0][0, 64,:]))
        #quit()

        self.params = trnsfrm_base_params
        self.params['pca_components'] = pca_components
        

        ## initialise super class appropriately
        super(DepthJointsDataLoader, self).\
            __init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                     collate_fn=CollateDepthJointsBatch(),
                     randomise_params=randomise_params)# need to init collate fn

        if debug:
            print("Data Loaded! Took: %0.2fs" % (time.time() - t))
            test_sample = self.dataset[0]
            print("Sample Final Shape: ", test_sample[0].shape, test_sample[1].shape)
            print("Sample Final Values:\n", test_sample[0], "\n", test_sample[1])
            print("Sample Joints_Std_MIN_MAX: ", test_sample[0].min(), test_sample[0].max())


        if not self.randomise_params:
            print('[DATALOADER] Setting torch manual seed...')
            torch.manual_seed(getattr(self.dataset, 'RAND_SEED', 0))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            

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



class DeepPriorTestDataLoader(BaseDataLoader):
    def __init__(self, data_dir, dataset_type, batch_size, shuffle, pca_components=30,
                validation_split=0.0, num_workers=1, debug=False, reduce=False,
                test_mm_err=False,
                use_pca_cache=True, pca_overwrite_cache=False, preload_depth=False,
                use_msra=True):

        from data_utils._dp_transformers import DeepPriorXYTestTransform, \
                                                DeepPriorYTestInverseTransform, DeepPriorBatchResultCollector
        trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': 200,#TODO make this redundant and use DepthParams!
            'debug_mode': debug,
            'cam_intrinsics': FHADCameraIntrinsics if not use_msra else MSRACameraIntrinsics,
            'dep_params': DepthParameters
        }

        # transform_test = DeepPriorXYTestTransform(depthmap_px=128, crop_len_mm=200,
        #                                       aug_mode_lst=[AugType.AUG_NONE])
        transform_test = transforms.Compose([
                                            JointReshaper(**trnsfrm_base_params),
                                            DepthCropper(**trnsfrm_base_params),
                                            DepthStandardiser(**trnsfrm_base_params),
                                            ToTuple(extract_type='depth_orig_com_orig_joints')
                                    ])

        if use_msra:
            print("Using MSRA for testing")
            data_dir = '../deep-prior-pp-pytorch/datasets/MSRA15'
            self.dataset = MSRAHandDataset(data_dir, '', 'test', test_subject_id=0,
                                    transform=transform_test, reduce=reduce,
                                    use_refined_com=False)
        else:
            
            self.dataset = HandPoseActionDataset(data_dir, dataset_type, 'hpe',
                                                transform=transform_test, reduce=reduce, 
                                                retrieve_depth=True, preload_depth=preload_depth)


        
        #quit()

        ### we assume here unpca is already performed
        ### transform_output: std_centered -> mm_uncentered
        ## TODO: make compatible with FHAD; parametrize crop len mm
        self.transform_output = DeepPriorYTestInverseTransform(crop_len_mm=200)
        self.test_res_collector = DeepPriorBatchResultCollector(self.dataset, self.transform_output, len(self.dataset))

        pca_transformer = PCATransformer(n_components=pca_components,
                                         use_cache=use_pca_cache,
                                         overwrite_cache=pca_overwrite_cache)

        pca_trnsfrms = transforms.Compose([
                                            JointReshaper(**trnsfrm_base_params),
                                            DepthCropper(**trnsfrm_base_params),
                                            #### disable augmentation
                                            # DepthAndJointsAugmenter( # TODO: implem
                                            #     aug_mode_lst=pca_aug_mode_lst,
                                            #     **transform_base_params),
                                            JointCentererStandardiser(**trnsfrm_base_params),
                                            ToTuple(extract_type='joints') # TODO: see if correct
                                        ])
        

        self.pca_weights_np, self.pca_bias_np = _check_pca(data_dir, pca_transformer, pca_trnsfrms,
                                                           trnsfrm_base_params, use_msra=use_msra)

        super(DeepPriorTestDataLoader, self).\
            __init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
