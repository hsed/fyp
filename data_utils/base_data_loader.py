import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from datasets.base_data_types import BaseDatasetType

class SubsetDeterministicSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (idx for idx in self.indices)

    def __len__(self):
        return len(self.indices)

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                val_dataset=None, randomise_params=True):
        self.validation_split = validation_split
        self.val_dataset = val_dataset
        self.shuffle = shuffle
        self.randomise_params = randomise_params
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0 or abs(split) > 1.0:
            return None, None
        if split > 0.0:
            testset_as_val = False
        elif split < 0.0:
            #return None, None
            ## in this special case we require that a seperate validation set is provided
            if self.val_dataset is None:
                print("[SPLIT_SAMPLER] Val Set was NONE and split was <= 0.0 CANNOT DO SPLIT")
                return None, None
            else:
                # we will use val_set as our samples
                # it could be that this is the test set as defined by the dataloader
                # this implies how much out of the test set do u wish to use for validation
                #n_samples = len(self.val_dataset)
                testset_as_val = True
                split = abs(split) # can't be negative!
                
                data_mode = getattr(self.val_dataset,'data_mode', False)
                if data_mode is False:
                    print("[SPLIT_SAMPLER] WARNING: Split <= 0 but unable to detect val_set mode, please ensure it is test_set!")
                elif data_mode != BaseDatasetType.TEST:
                    print("[SPLIT_SAMPLER] WARNING: SPLIT <= 0 BUT VAL_SET != TEST_SET, COLLUSION IN TRAIN TEST SET POSSIBLE!!!")

        
        # this is either the trainset if val_split > 0 else the val set which may or may not be test set
        # data loader must ensure that in this scenario val_set is train set!
        # otherwise you would get data snooping i.e. samples both in train and val
        n_samples = self.n_samples if not testset_as_val else len(self.val_dataset)

        # get all indices of samples from entire train set
        idx_full = np.arange(n_samples)

        # shuffle these indices and pass them through a random sampler object or deterministic sample object
        # the determ sampler always return the same 'pre-shuffled' array of indices
        # the random sampler randomises the indices at every call which happens at end of every epoch
        # note for test set this technology is not implemented and need to be done
        # note this flag is added to child classes
        
        self._reset_rand_seed()
        np.random.shuffle(idx_full)

        len_valid = int(n_samples * split)

        valid_idx = idx_full[0:len_valid]

        if not testset_as_val:
            # proceed as usual
            train_idx = np.delete(idx_full, np.arange(0, len_valid))
        else:
            # now we basically wish touse the entire train set!
            print("[SPLIT_SAMPLER] Split was <= 0.0, Using entire train set for training!")
            train_idx = np.arange(self.n_samples) # now this is train_set samples!
            self._reset_rand_seed() # for determinism
            np.random.shuffle(train_idx) # must shuffle atleast once otherwise you start at saddle point!
        
        # A sampler object must provide __iter__ and __len__, numpy provides this!
        train_sampler = SubsetRandomSampler(train_idx) if self.randomise_params else SubsetDeterministicSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx) if self.randomise_params else SubsetDeterministicSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
        
    def split_validation(self):
        '''
            Use this extract a ref to a data_loader that works only on val sampler

            If you wish to use different transforms for val_set, then pass an additional
            dataset class as val_dataset
        '''
        if self.val_dataset is not None:
            print("[SPLIT_VAL] Using custom val_dataset obj")
            kwargs = self.init_kwargs.copy()
            kwargs.update({'dataset': self.val_dataset})
        else:
            kwargs = self.init_kwargs

        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **kwargs)
    

    def _reset_rand_seed(self):
        if not self.randomise_params:
            np.random.seed(getattr(self.dataset,'RAND_SEED', 0))
        else:
            np.random.seed(0)