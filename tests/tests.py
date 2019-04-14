from argparse import Namespace

import numpy as np
import torch

from datasets import *
from data_utils import *
from data_utils.data_loaders import _check_pca

from tqdm import tqdm

def simple_transformer_tests():
    aug_mode_lst=[0,1,2,3] # None, Rot, Scale, Trans
    
    print("[TESTS1] AUG_MODE_LIST: ", list(map(lambda i: AugType(i), aug_mode_lst)))
    print("[TESTS1] NO PCA TESTED")
    ### using orig transformers
    trnsfrm = transforms.Compose([
                            DeepPriorXYTransform(aug_mode_lst=aug_mode_lst, debug_mode=False),
                            ##pca_transformer,
                            ToTuple(extract_type='depth_joints') # 'depth_joints_debug'
                        ])
    
    trnsfrm_base_params = {
        'num_joints': 21,
        'world_dim': 3,
        'cube_side_mm': 200,
        'debug_mode': True,
        'cam_intrinsics': MSRACameraIntrinsics,
        'dep_params': DepthParameters,
        'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180)
        }

    rot_lim=trnsfrm_base_params['aug_lims'].abs_rot_lim_deg
    sc_std=trnsfrm_base_params['aug_lims'].scale_std
    tr_std=trnsfrm_base_params['aug_lims'].trans_std

    dat = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=trnsfrm,
                        test_subject_id=0, randomise_params=False)
    dat.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
         custom_aug_modes=aug_mode_lst)
    
    print("\n------ second dataset --------\n")

    dat2 = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=trnsfrm,
                        test_subject_id=0, randomise_params=False)
    dat2.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
         custom_aug_modes=aug_mode_lst)
    
    print("First item test...")

    print(dat[0])

    print("\n Second item test...")
    print(dat2[0])

    print("[TESTS1] Equal?", "x =>", np.array_equal(dat[0][0], dat2[0][0]), "y =>", np.array_equal(dat[0][1], dat2[0][1]))
    
    assert len(dat) == len(dat2)
    for i in tqdm(range(len(dat)), desc='Testing (x,y) Values'):
        item1, item2 = dat[i], dat2[i]
        assert np.array_equal(item1[0], item2[0]) # test x equal
        assert np.array_equal(item1[1], item2[1]) # test y equal


def simple_transformer_tests_2():
    '''
        test new vs old transforms method!
    '''
    debug = False # set to true to see plots
    trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': 200,
            'debug_mode': debug,
            'cam_intrinsics': MSRACameraIntrinsics,
            'dep_params': DepthParameters,
            'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180)
    }
    aug_mode_lst=[0,1,2,3] # None, Rot, Scale, Trans
    
    print("[TESTS2] AUG_MODE_LIST: ", list(map(lambda i: AugType(i), aug_mode_lst)))
    print("[TESTS2] NO PCA TESTED")
    ### using orig transformers
    trnsfrm_old = transforms.Compose([
                            DeepPriorXYTransform(aug_mode_lst=aug_mode_lst, debug_mode=debug),
                            ##pca_transformer,
                            ToTuple(extract_type='depth_joints')
                        ])
    trnsfrm_new = transforms.Compose([
            JointReshaper(**trnsfrm_base_params), # NOOP for MSRA
            DepthCropper(**trnsfrm_base_params),
            DepthAndJointsAugmenter(aug_mode_lst=aug_mode_lst,**trnsfrm_base_params),
            DepthStandardiser(**trnsfrm_base_params),
            JointCentererStandardiser(**trnsfrm_base_params),
            ToTuple(extract_type='depth_joints') # 'depth_joints_debug'
        ])

    rot_lim=trnsfrm_base_params['aug_lims'].abs_rot_lim_deg
    sc_std=trnsfrm_base_params['aug_lims'].scale_std
    tr_std=trnsfrm_base_params['aug_lims'].trans_std

    dat = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=trnsfrm_old,
                        test_subject_id=0, randomise_params=False)
    dat.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
         custom_aug_modes=aug_mode_lst)
    
    print("\n------ second dataset --------\n")

    dat2 = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=trnsfrm_new,
                        test_subject_id=0, randomise_params=False)
    dat2.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
         custom_aug_modes=aug_mode_lst)
    
    print("First item test...")

    print(dat[0])

    print("\n Second item test...")
    print(dat2[0])

    print("[TESTS2] Equal?", "x =>", np.array_equal(dat[0][0], dat2[0][0]), "y =>", np.array_equal(dat[0][1], dat2[0][1]))
    
    assert len(dat) == len(dat2)
    for i in tqdm(range(len(dat)), desc='[TESTS] Testing (x,y) Values (%% Equal So Far) '):
        item1, item2 = dat[i], dat2[i]
        assert np.array_equal(item1[0], item2[0]) # test x equal
        assert np.array_equal(item1[1], item2[1]) # test y equal
    '''
        try assert
        if failed then from dataset object gather params print.. and break for loop
    '''

def pca_transformer_tests():
    trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': 200,
            'debug_mode': False, # for plotting or some extra info, doesn't really make sense for pca
            'cam_intrinsics': MSRACameraIntrinsics,
            'dep_params': DepthParameters,
            'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180)
    }
    pca_len = int(2e5) # 

    pca_aug_list = [0, 1, 2, 3]
    print("[TESTS3] AUG_MODE_LIST: ", list(map(lambda i: AugType(i), pca_aug_list)))
    pca_transformer1 = PCATransformer(n_components=30,
                                         use_cache=False,
                                         overwrite_cache=False)
    pca_transformer2 = PCATransformer(n_components=30,
                                         use_cache=False,
                                         overwrite_cache=False)

    pca_trnsfrms = transforms.Compose([
                            DeepPriorYTransform(aug_mode_lst=pca_aug_list), # NOTE: this was set to train_aug_list all along!
                            ToTuple(extract_type='joints')
                       ])

    pca1_weights, pca1_bias = _check_pca('../deep-prior-pp-pytorch/datasets/MSRA15', pca_transformer1, pca_trnsfrms,
               trnsfrm_base_params, y_pca_len=pca_len, use_msra=True, randomise_params=False)
    
    pca2_weights, pca2_bias = _check_pca('../deep-prior-pp-pytorch/datasets/MSRA15', pca_transformer2, pca_trnsfrms,
               trnsfrm_base_params, y_pca_len=pca_len, use_msra=True, randomise_params=False)
    
    ### assertions
    print("[TESTS3] PCA EQUAL? WEIGHTS: ", np.array_equal(pca1_weights, pca2_weights),
          "BIAS: ", np.array_equal(pca1_bias, pca2_bias))
    



def pca_transformer_tests2():
    '''
        in this test we compare old and new transformers....

    '''
    trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': 200,
            'debug_mode': False, # for plotting or some extra info, doesn't really make sense for pca
            'cam_intrinsics': MSRACameraIntrinsics,
            'dep_params': DepthParameters,
            'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180)
    }
    pca_len = int(2e5) # 

    pca_aug_list = [0,1,2, 3] #[0, 1, 2, 3]

    print("[TESTS4] AUG_MODE_LIST: ", list(map(lambda i: AugType(i), pca_aug_list)))
    pca_transformer1 = PCATransformer(n_components=30,
                                         use_cache=False,
                                         overwrite_cache=False)
    pca_transformer2 = PCATransformer(n_components=30,
                                         use_cache=False,
                                         overwrite_cache=False)

    pca_trnsfrms_old = transforms.Compose([
                            DeepPriorYTransform(aug_mode_lst=pca_aug_list), # NOTE: this was set to train_aug_list all along!
                            ToTuple(extract_type='joints')
                       ])
    
    ### done exactly as seen in dataloader
    pca_trnsfrm_new_list = [JointReshaper(**trnsfrm_base_params), DepthCropper(**trnsfrm_base_params)]
    pca_trnsfrm_new_list.append(
                DepthAndJointsAugmenter(aug_mode_lst=pca_aug_list,**trnsfrm_base_params),
            )
    pca_trnsfrm_new_list += [JointCentererStandardiser(**trnsfrm_base_params), ToTuple(extract_type='joints')]
    pca_trnsfrms_new = transforms.Compose(pca_trnsfrm_new_list)

    pca1_weights, pca1_bias = _check_pca('../deep-prior-pp-pytorch/datasets/MSRA15', pca_transformer1, pca_trnsfrms_old,
               trnsfrm_base_params, y_pca_len=pca_len, use_msra=True, randomise_params=False)
    
    pca2_weights, pca2_bias = _check_pca('../deep-prior-pp-pytorch/datasets/MSRA15', pca_transformer2, pca_trnsfrms_new,
               trnsfrm_base_params, y_pca_len=pca_len, use_msra=True, randomise_params=False)
    
    ### assertions
    print("[TESTS4] PCA EQUAL? WEIGHTS: ", np.array_equal(pca1_weights, pca2_weights),
          "BIAS: ", np.array_equal(pca1_bias, pca2_bias))


def combined_transformers_test():
    '''
        final test on transformer:
            PCA_ROT+SC+TRANS + TRAIN_ROT+SC+TRANS
        
        do all then compare new vs old
    '''
    debug = False
    trnsfrm_base_params = {
            'num_joints': 21,
            'world_dim': 3,
            'cube_side_mm': 200,
            'debug_mode': debug, # for plotting or some extra info, doesn't really make sense for pca
            'cam_intrinsics': MSRACameraIntrinsics,
            'dep_params': DepthParameters,
            'aug_lims': Namespace(scale_std=0.02, trans_std=5, abs_rot_lim_deg=180)
    }
    pca_len = int(2e5) # 
    aug_mode_lst=[0,1,2,3] # None, Rot, Scale, Trans
    pca_aug_list = [0,1,2, 3] #[0, 1, 2, 3]

    print("[TESTS5] AUG_MODE_LIST: ", list(map(lambda i: AugType(i), aug_mode_lst)))
    print("[TESTS5] PCA_AUG_MODE_LIST: ", list(map(lambda i: AugType(i), pca_aug_list)))
    pca_transformer1 = PCATransformer(n_components=30,
                                         use_cache=False,
                                         overwrite_cache=False)
    pca_transformer2 = PCATransformer(n_components=30,
                                         use_cache=False,
                                         overwrite_cache=False)
    
    trnsfrm_old = transforms.Compose([
                            DeepPriorXYTransform(aug_mode_lst=aug_mode_lst, debug_mode=debug),
                            pca_transformer1,
                            ToTuple(extract_type='depth_joints')
                    ])
    trnsfrm_new = transforms.Compose([
            JointReshaper(**trnsfrm_base_params), # NOOP for MSRA
            DepthCropper(**trnsfrm_base_params),
            DepthAndJointsAugmenter(aug_mode_lst=aug_mode_lst,**trnsfrm_base_params),
            DepthStandardiser(**trnsfrm_base_params),
            JointCentererStandardiser(**trnsfrm_base_params),
            pca_transformer2,
            ToTuple(extract_type='depth_joints') # 'depth_joints_debug'
    ])


    pca_trnsfrms_old = transforms.Compose([
                            DeepPriorYTransform(aug_mode_lst=pca_aug_list), # NOTE: this was set to train_aug_list all along!
                            ToTuple(extract_type='joints')
                       ])
    
    ### done exactly as seen in dataloader
    pca_trnsfrm_new_list = [JointReshaper(**trnsfrm_base_params), DepthCropper(**trnsfrm_base_params)]
    pca_trnsfrm_new_list.append(
                DepthAndJointsAugmenter(aug_mode_lst=pca_aug_list,**trnsfrm_base_params),
            )
    pca_trnsfrm_new_list += [JointCentererStandardiser(**trnsfrm_base_params), ToTuple(extract_type='joints')]
    pca_trnsfrms_new = transforms.Compose(pca_trnsfrm_new_list)

    pca1_weights, pca1_bias = _check_pca('../deep-prior-pp-pytorch/datasets/MSRA15', pca_transformer1, pca_trnsfrms_old,
               trnsfrm_base_params, y_pca_len=pca_len, use_msra=True, randomise_params=False)
    
    print("\n----- second pca -----\n")
    pca2_weights, pca2_bias = _check_pca('../deep-prior-pp-pytorch/datasets/MSRA15', pca_transformer2, pca_trnsfrms_new,
               trnsfrm_base_params, y_pca_len=pca_len, use_msra=True, randomise_params=False)
    


    ### now dataset...
    print("\n----- now dataset -----\n")
    rot_lim=trnsfrm_base_params['aug_lims'].abs_rot_lim_deg
    sc_std=trnsfrm_base_params['aug_lims'].scale_std
    tr_std=trnsfrm_base_params['aug_lims'].trans_std

    dat = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=trnsfrm_old,
                          test_subject_id=0, randomise_params=False)
    dat.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
         custom_aug_modes=aug_mode_lst)
    
    print("\n------ second dataset --------\n")

    dat2 = MSRAHandDataset('../deep-prior-pp-pytorch/datasets/MSRA15', '', 'train', transform=trnsfrm_new,
                        test_subject_id=0, randomise_params=False)
    dat2.make_transform_params_static(AugType, \
        (lambda aug_mode_list: getAugModeParam(aug_mode_list, rot_lim, sc_std, tr_std)[1]),
         custom_aug_modes=aug_mode_lst)
    
    # print("First item test...")

    # print(dat[0])

    # print("\n Second item test...")
    # print(dat2[0])

    
    assert len(dat) == len(dat2)

    ### assertions
    print("\n ------- ASSERTIONS ------\n")
    print("[TESTS5] PCA EQUAL?\t WEIGHTS: ", np.array_equal(pca1_weights, pca2_weights),
          "BIAS: ", np.array_equal(pca1_bias, pca2_bias))
    assert np.array_equal(pca1_weights, pca2_weights)
    assert np.array_equal(pca1_bias, pca2_bias)

    print("[TESTS5] In/Out Equal?", "x =>", np.array_equal(dat[0][0], dat2[0][0]), "y =>", np.array_equal(dat[0][1], dat2[0][1]))
    
    for i in tqdm(range(len(dat)), desc='[TESTS5] Testing (x,y) Values (%% Equal So Far) '):
        item1, item2 = dat[i], dat2[i]
        assert np.array_equal(item1[0], item2[0]) # test x equal
        assert np.array_equal(item1[1], item2[1]) # test y equal

def dataloader_tests():
    print("[TESTS6] Testing dataloaders with old and new methods...")

    PCA_SIZE = int(2e5) # 1000 #
    MAX_BATCHES_TO_TEST = 999999 # 2 # 
    
    base_dataloader_params = {
        'data_dir': "datasets/hand_pose_action",
        'dataset_type': "train",
        'batch_size': 128,
        'shuffle': False,
        'validation_split': 0.2, #0.0, #0.2,  # 0.2 #-1.0 !!!! TO CHANGE
        'pca_components': 30,
        'use_pca_cache': False,
        'num_workers': 0, #4,  # 8
        'debug': False,
        'reduce': False,
        # True now this is not neede, after first epoch preloading is automatically done
        'preload_depth': False,
        'pca_overwrite_cache': False,  # True
        'use_msra': True,
        'data_aug': [0, 1, 2, 3],
        'pca_data_aug': [0, 1, 2, 3],
        'pca_size': PCA_SIZE, # int(2e5)
        'randomise_params': False,
    }
    
    loader1_params = {
        **base_dataloader_params,
        'use_orig_transformers': True,
        'use_orig_transformers_pca': True,
    }

    loader2_params = {
        **base_dataloader_params,
        'use_orig_transformers': False,
        'use_orig_transformers_pca': False,
    }

    dataloader1 = DepthJointsDataLoader(**loader1_params)
    validdataloader1 = dataloader1.split_validation()

    print('\n---- DATALOADER2 ----\n')

    dataloader2 = DepthJointsDataLoader(**loader2_params)
    validdataloader2 = dataloader2.split_validation()

    print("[TESTS6 DATASET EQUAL?]", np.array_equal(dataloader1.dataset[0][0], dataloader2.dataset[0][0]))

    for i, items in tqdm(enumerate(zip(dataloader1, dataloader2)), desc='[TESTS6] Testing TRAINSET M Batches, %% Equal => '):
                if i > MAX_BATCHES_TO_TEST:
                   break
                #print("Got ", i, " Shape: ", item[0].shape)
                inputs1, targets1 = items[0] # store running last item as the tmp_item
                inputs2, targets2 = items[1]
                #print("SHAPE: ", inputs1.shape, inputs2.shape)
                #print("INPUT0: ", inputs1[0], inputs2[0])
                #assert torch.all(torch.lt(torch.abs(torch.add(inputs1, -inputs2)), 1e-12))
                assert torch.equal(inputs1, inputs2)
                assert torch.equal(targets1, targets2)
    

    for i, items in tqdm(enumerate(zip(validdataloader1, validdataloader2)), 
        desc='[TESTS6] Testing VALSET M Batches, %% Equal => '):
                if i > MAX_BATCHES_TO_TEST:
                   break
                #print("Got ", i, " Shape: ", item[0].shape)
                inputs1, targets1 = items[0] # store running last item as the tmp_item
                inputs2, targets2 = items[1]
                assert torch.equal(inputs1, inputs2)
                assert torch.equal(targets1, targets2)

    #print(tmp_item) 

    #torch.tensor_equal

if __name__ == "__main__":
    import argparse
    #simple_transformer_tests()
    #simple_transformer_tests_2()
    #pca_transformer_tests()
    #pca_transformer_tests2()
    #combined_transformers_test()
    dataloader_tests()
