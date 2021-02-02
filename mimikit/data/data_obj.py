import numpy as np
import torch
from copy import copy
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torch.utils.data._utils.collate import default_convert


class DataObj(Dataset):
    def __init__(self, data_object):
        # determine whether an indirect object is needed and
        # return the simplest form for the given object
        self._object = None
        self.style = 'map'
        self.init_data_object(data_object)

    def init_data_object(self, obj):
        if obj is not None:
            if DataObj._items_are_tuples(obj):
                self._object = obj
            elif isinstance(obj, (tuple, list)):
                if any(DataObj._items_are_tuples(x) for x in obj):
                    lst = [SingleDataObj(x) if not DataObj._items_are_tuples(x) else x 
                           for x in obj]
                    self._object = CompositeDataTuple(lst)
                else:
                    self._object = DataTuple(obj)
            else:
                self._object = SingleDataObj(obj)
            self._cached_len = len(self._object)

    # If no data_object is given at instantiation, the object can be used
    # as a wrapper.
    def __call__(self, data_object):
        new_obj = copy(self)
        new_obj.init_data_object(data_object)
        return new_obj

    def __getitem__(self, index):
        return self._object[index]

    def __len__(self):
        return self._cached_len

    def cache_len(self):
        return len(self._object)

    @property
    def n_features(self):
        return len(self[0])

    @property
    def dtypes(self):
        return self.get_dtypes()

    def get_dtypes(self):
        return tuple(x.dtype for x in self[0])

    @property
    def shapes(self):
        return self.get_shapes()

    def get_shapes(self):
        return tuple(x.shape for x in self[0])

    def to_tensor(self):
        new_obj = copy(self)
        data_object = new_obj._object
        if isinstance(data_object, DataObj):
            new_obj._object = data_object.to_tensor()
        return new_obj

    def split(self, splits):
        """
        @param splits: Sequence of floats or ints possibly containing None. The sequence of elements
        corresponds to the proportion (floats), the number of examples (ints) or the absence of
        train-set, validation-set, test-set, other sets... in that order.
        @return: the *-sets
        """
        nones = []
        if any(x is None for x in splits):
            if splits[0] is None:
                raise ValueError("the train-set's split cannot be None")
            nones = [i for i, x in zip(range(len(splits)), splits) if x is None]
            splits = [x for x in splits if x is not None]
        if all(type(x) is float for x in splits):
            splits = [x / sum(splits) for x in splits]
            N = len(self)
            # leave the last one out for now because of rounding
            as_ints = [int(N * x) for x in splits[:-1]]
            # check that the last is not zero
            if N - sum(as_ints) == 0:
                raise ValueError("the last split rounded to zero element. Please provide a greater float or consider "
                                 "passing ints.")
            as_ints += [N - sum(as_ints)]
            splits = as_ints
        sets = list(random_split(self, splits))
        if any(nones):
            sets = [None if i in nones else sets.pop(0) for i in range(len(sets + nones))]
        return tuple(sets)

    @staticmethod
    def _items_are_tuples(x):
        return isinstance(x, DataObj)

    def to(self, device):
        self._object.to(device)


# We need three basic types of basic data object sources:
#
# Tuple of indexables distributing the __getitem__ method over 
# the indexables to build output tuple
#
# Single Object with __getitem__ returning the output element inside a 
# tuple with only one element.  The result is the same as a DataTuple with
# only one element but it doesn't need to map over the tuple when indexing.
#
# A Composite Tuple that can contain derived DataObjects (returning tuples)
# The CompositeDataTuple returns flat tuples (it internally flattens after
# indexing)

class DataTuple(DataObj, tuple):
    def __init__(self, *args):
        self._cached_len = min([len(x) for x in self])
        self.style = 'map'

    def __str__(self):
        return "DistribIndexTuple(" + str(tuple(self)) + ")"

    def __repr__(self):
        return "DistribIndexTuple(" + str(tuple(self)) + ")"
 
    def __getitem__(self, index):
        return tuple(x[index] for x in self)

    def __add__(self, other):
        return DistribIndexTuple(tuple(self) + tuple(other))

    def __len__(self):
        return self._cached_len

    def eltlen(self):
        return super().__len__()

    def indexable(self, index):
        return super().__getitem__(index)

    def get_shapes(self):
        return [x.shape for x in self[0]]

    def to_tensor(self):
        return DataTuple(default_convert(x) for x in tuple(self))

    def to(self, device):
        for x in tuple(self):
            x.to(device)

        
class SingleDataObj(DataObj):
    def __init__(self, obj):
        self._object = obj
        self.style = 'map'

    def __getitem__(self, index):
        return (self._object[index],)
    
    def __len__(self):
        return len(self._object)

    def indexable(self, index):
        return self._object

    def eltlen(self):
        return 1

    def to_tensor(self):
        return SingleDataObj(default_convert(self._object))

    def to(self, device):
        self._object.to(device)


class CompositeDataTuple(DataTuple):
    def __init__(self, *args):
        self._cached_len = min([len(x) for x in self])
        self.style = 'map'

    def __getitem__(self, index):
        return tuple(x for tupl in tuple(self) for x in tupl[index])

    def __add__(self, other):
        return CompositeDataTuple(tuple(self) + tuple(other))

    def eltlen(self):
        return super().__len__()

    def indexable(self, index):
        return super().__getitem__(index)

    def to_tensor(self):
        return CompositeDataTuple(x.to_tensor() for x in tuple(self))


class SequenceSlicer(DataObj):
    def __init__(self, seqs, data_object=None):
        self.seqs = [SequenceSlicer.complete_seq_spec(s) for s in seqs]
        self.init_data_object(data_object)
        self.style = 'map'

    @staticmethod
    def complete_seq_spec(spec):
        if isinstance(spec, int):
            return 0, spec, 1
        if len(spec) == 3:
            return spec
        if len(spec) == 2:
            return spec + (1,)
        if len(spec) == 1:
            return (0,) + spec + (1,)
        else:
            raise ValueError("Expected int or a tuple with 1, 2, or 3 elements for sequence specification.")

    def __len__(self):
        return min([(len(self._object) - length - shift + 1) // stride 
                    for shift, length, stride in self.seqs])

    def __getitem__(self, index):
        output = [self._object[index + shift: index + shift + length: stride]
                  for shift, length, stride in self.seqs]
        return tuple(element for tupl in output for element in tupl)


class DataSlicer(DataObj):
    def __init__(self, sliced_seqs, data_object=None):
        self.slices = [tuple(slice(*x) if isinstance(x, tuple) else slice(None) for x in slices)
                       for slices in sliced_seqs]
        self.init_data_object(data_object)
        self.style = 'map'

    def __getitem__(self, index):
        output = [[obj[s] for obj in self._object[index]] for s in self.slices]
        return tuple(element for tupl in output for element in tupl)


class AddNoise(DataObj):
    def __init__(self, noise_amount, data_object=None):
        self.noise_amount = noise_amount
        self.init_data_object(data_object)

    def __getitem__(self, index):
        return tuple(element + self.noise_amount * np.random.randn(*element.shape)
                     for element in self._object[index])


def rand_in_range(mini, maxi):
    return mini + np.random.rand(1)[0] * (maxi - mini)


def coin(probability):
    return np.random.rand(1)[0] < probability


class RandomMul(DataObj):
    def __init__(self, prob, range=(0.8, 1.2), data_object=None):
        self.prob = prob
        self.range = range
        self.init_data_object(data_object)

    def __getitem__(self, index):
        if coin(self.prob):
            return tuple(element * rand_in_range(*self.range)
                         for element in self._object[index])
        else:
            return self._object[index]


class ExtendingAugmentation(DataObj):
    def init_data_object(self, data_object):
        super().init_data_object(data_object)
        self.original_len = len(data_object)
        self._cached_len = self.original_len + self.n_extensions


# Extending transformations that generate additional data on the fly
# but feed all original data points in each epoch to the model
class ExtendRandomMul(ExtendingAugmentation):
    def __init__(self, n_extensions, range=(0.8, 1.2), data_object=None):
        self.range = range
        self.n_extensions = n_extensions
        self.init_data_object(data_object)

    def __getitem__(self, index):
        if index >= self.original_len:
            return tuple(element * rand_in_range(*self.range)
                         for element in self._object[np.random.randint(1)])
        else:
            return self._object[index]
