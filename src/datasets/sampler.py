from torch.utils.data import BatchSampler, Sampler
import random

class RepeatBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, shuffle):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(self.sampler)
        if len(indices) == 0:
            return
        num_samples = len(indices)
        num_repeats = max(1, -(-self.batch_size // num_samples))  # 天井関数で繰り返し回数を計算

        repeated_indices = indices * num_repeats  # インデックスを繰り返す
        if self.shuffle:
            random.shuffle(repeated_indices)
        
        batch = []
        for idx in repeated_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size