import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, Sampler


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
        if split <= 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        # note this flag is added to child classes
        if not self.randomise_params:
            np.random.seed(getattr(self.dataset,'RAND_SEED', 0))
        else:
            np.random.seed(0) 
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
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
            print("Using custom val_dataset obj")
            kwargs = self.init_kwargs.copy()
            kwargs.update({'dataset': self.val_dataset})
        else:
            kwargs = self.init_kwargs

        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **kwargs)
    
