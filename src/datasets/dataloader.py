import torch
import itertools
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .sampler import RepeatBatchSampler

class TrainDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, collate_fn=None):
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        
        batch_sampler = RepeatBatchSampler(
            sampler, batch_size, drop_last, shuffle
        )
        
        super().__init__(dataset, batch_sampler=batch_sampler,
                         num_workers=num_workers, pin_memory=pin_memory,
                         collate_fn=collate_fn)

class EvalDataLoader:
    def __init__(self, dataset, num_repeat, collate_fn):
        self.dataset = dataset
        self.num_repeat = num_repeat
        self.colla_fn = collate_fn
        # We repeat each sample in the dataset for the number of times it is required to be evaluated.
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)  
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = next(self.iterator)  
        
        repeated_data = []
        data["samples"] = data["samples"][0]  # (1, C, H, W) -> (C, H, W)
        data["labels"] = data["labels"]
        data["filenames"] = data["filenames"][0]
        data["clsnames"] = data["clsnames"][0]
        data["anom_type"] = data["anom_type"][0]
        for _ in range(self.num_repeat):
            repeated_data.append(data)
    
        collated_data = self.colla_fn(repeated_data)
        
        return collated_data
    
    def __len__(self):
        return len(self.dataloader)