import os
import torch
import numpy as np
import torchvision
import math
import time
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Iterator, List, Optional, Union
import bisect

class ImageNetTrain(Dataset):
    def __init__(self, ratio=[0.25,0.5], momentum = 1., batch_size = None, num_epoch=None, delta = None, quantiles=[20,85]):
        self.dataset = datasets.ImageFolder(
                            './datas/ImageNet/train',
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.full(len(self.dataset),1.)
        self.transform = dataset.transform
        self.weights = np.full(len(self.dataset),1.)
        self.save_num = 0
        self.momentum = momentum
        self.batch_size = batch_size
        self.quantiles = quantiles

    def __setscore__(self, indices, values):
        self.scores[indices] = self.momentum * values + (1.-self.momentum)*self.scores[indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        if self.transform:
            data = self.transform(data)
        elif self.dataset.transform:
            data = dataset.transform(data)
        return data, target, index, weight

    def __bucketize__(self,quantile_thresholds,leq=False):
        select_func = bisect.bisect_left if leq else bisect.bisect_right
        well_learned_samples = [[] for i in range(len(quantile_thresholds)+1)]
        for id,value in enumerate(self.scores):
            well_learned_samples[select_func(quantile_thresholds,value)].append(id)
        return well_learned_samples


    def __balance_weight__(self, perm):
        # Put elements with recaled weight to start and end of batch balanced, so that cutmix with batch operation don't
        # need to deal with weight.
        if self.batch_size is None or self.batch_size<=0:
            print('Batchsize not specified! Cannot balance weight for cutmix and mixup')
            return perm

        if torch.distributed.is_available():
            num_replicas = torch.distributed.get_world_size()
        else: num_replicas = 1

        num_samples = math.ceil(len(perm) / num_replicas)

        for cid in range(num_replicas):
            rlimit = (cid+1)*num_samples
            for l in range(cid*num_samples,min(rlimit,len(perm)),self.batch_size):
                local_rebalence = []
                remaining = []
                r = min(l+self.batch_size,rlimit,len(perm))  #left close right open
                for j in range(l,r):
                    if self.weights[perm[j]] > 1:
                        local_rebalence.append(perm[j])
                    else:
                        remaining.append(perm[j])
                perm[l:r] = local_rebalence[:len(local_rebalence)//2] + remaining + local_rebalence[len(local_rebalence)//2:]
        return perm

#     def prune(self, leq = False):  #code for test
#         well_learned_samples = list(range(len(self.dataset)))
#         selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)
#         self.weights[selected]=1./self.ratio
#         np.random.shuffle(well_learned_samples)
#         return self.__balance_weight__(well_learned_samples)

    def prune(self, leq = False):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        quantile_thresholds = [np.percentile(self.scores,q,axis=0) for q in self.quantiles]
        bucktized_samples = self.__bucketize__(quantile_thresholds,leq)
        pruned_samples = []
        pruned_samples.extend(bucktized_samples[-1])
        well_learned_samples = bucktized_samples[:-1]
        selected_q = [np.random.choice(well_learned_samples[i], \
            int(self.ratio[i]*len(well_learned_samples[i])),replace=False) for i in range(len(well_learned_samples))]

        self.reset_weights()
        for i,selected in enumerate(selected_q):
            if len(selected)>0:
                self.weights[selected]=1./self.ratio[i]
                pruned_samples.extend(selected)
        print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        return self.__balance_weight__(pruned_samples)

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune(self.seed>1)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        batch_size: Optional[int] = None
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
#         self.sampler.reset()
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]

        self.total_size = self.num_samples * self.num_replicas

        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        indexes_of_indexes = indices
#         indexes_of_indexes = super().__iter__()  # change this line
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
#         return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output

def is_master():
    if not torch.distributed.is_available():
        return True

    if not torch.distributed.is_initialized():
        return True

    if torch.distributed.get_rank()==0:
        return True

    return False

def split_index(t):
    low_mask = 0b111111111111111
    low = torch.tensor([x&low_mask for x in t])
    high = torch.tensor([(x>>15)&low_mask for x in t])
    return low,high

def recombine_index(low,high):
    original_tensor = torch.tensor([(high[i]<<15)+low[i] for i in range(len(low))])
    return original_tensor


    def copy_properties(self,src):
        for attr in dir(src):
            if attr not in {'__init__','copy_properties'}:
                setattr(self,attr,getattr(src,attr))

