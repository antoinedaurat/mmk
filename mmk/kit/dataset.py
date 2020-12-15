import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset, Subset, random_split, DataLoader
from torch.utils.data._utils.collate import default_convert
import pytorch_lightning as pl
from functools import update_wrapper
import warnings


def map_if_multi(attr):
    def decorator(fn):
        def wrapper(self, *arg, **kwargs):
            if isinstance(getattr(self, attr), tuple):
                return tuple(getattr(x, fn.__name__)(*arg, **kwargs)[0] for x in getattr(self, attr))
            else:
                # for consistency, we always return a tuple
                # noinspection PyRedundantParentheses
                return (fn(self, *arg, **kwargs), )

        return update_wrapper(wrapper, fn,
                              # for some reason, __mod__ causes errors when using functools.wraps
                              ("__name__", '__qualname__', '__doc__', '__annotations__'),
                              ("__dict__",))
    return decorator


class Dataset(TorchDataset):

    @property
    def data(self):
        """
        give access to the underlying data to wrappers and co
        """
        return self._object

    @property
    def n_features(self):
        return self._n_features

    @property
    def dims(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def style(self):
        return self._style

    @property
    def attrs(self):
        """
        @return: `attrs` : dict of user-specified infos he wants to keep close (e.g. {"sr": 22050, "time-axis": 0})
        """
        return self._attrs

    def __init__(self,
                 data_object=None,
                 wrappers=None,
                 augmentations=None,
                 attrs=None):
        """
        Here we initialize the instance.
        """
        self._object = data_object
        self._wrapper = wrappers
        self._augmentation = augmentations
        self._attrs = attrs
        self._n_features = 1 if not isinstance(data_object, tuple) else len(data_object)
        self._is_multi = self._n_features > 1

        if self._is_multi:
            # we cast to Dataset recursively
            self._object = tuple(x if isinstance(x, Dataset) else Dataset(x) for x in self._object)

        self._style = self.get_style()
        if any(style is None for style in self._style):
            raise ValueError("Expected data_object to either implement __getitem__ and __len__, or __iter__."
                             " object of class %s implements none of those." % str(type(data_object)))
        if self._is_multi:
            if not all(self._style[0] == style for style in self._style[1:]):
                raise TypeError("Expected all data_objects to be of the same style. Got %s" % str(self._style))
            lengths = tuple(len(obj) for obj in self._object)
            if self._style[0] is "map" and not all(lengths[0] == n for n in lengths[1:]):
                raise ValueError("Expected all 'map-style' data_objects to be of same lengths. Got %s" % \
                                 str(lengths))
        self._dtype = self.get_dtype()
        if any(dtype is None for dtype in self._dtype):
            raise TypeError("Expected elements of `data_object` to be of torch, numpy or built-in numerical dtype."
                            " Got following dtypes %s" % str(self._dtype))
        self._shape = self.get_shape()
        self._device = self.get_device()

    def __call__(self, *args, **kwargs):
        """
        Method used to wrap around an other Dataset. Only used in the wrapper classes.
        """
        pass

    def __len__(self):
        return len(self._object[0]) if self._is_multi else len(self._object)

    @map_if_multi("_object")
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return tuple(self._object[i] for i in item)
        return self._object[item]

    def __iter__(self):
        if isinstance(self._object, tuple):
            return zip(*[iter(obj) for obj in self._object])
        return iter(self._object)

    @staticmethod
    def has_len(data_object):
        return getattr(data_object, "__len__", False)

    @staticmethod
    def _get_style(data_object):
        if getattr(data_object, "__getitem__", False) and getattr(data_object, "__len__", False):
            return "map"
        elif getattr(data_object, "__iter__", False):
            return "iter"
        return None

    @map_if_multi("_object")
    def get_style(self):
        return Dataset._get_style(self._object)

    @staticmethod
    def _get_elem(data_object):
        if Dataset._get_style(data_object) == "map":
            return data_object[0]
        elif Dataset._get_style(data_object) == "iter":
            return next(iter(data_object))
        return None

    @map_if_multi("_object")
    def get_elem(self):
        return Dataset._get_elem(self._object)

    @staticmethod
    def _get_shape(data_object):
        shape = getattr(data_object, "shape", None)
        if shape is None:
            # we try to fet lengths recursively and add `None` when we get elements with no length
            # thus, in the case of a hierarchy of generators, returning at least the number of 'axes'
            N = len(data_object) if Dataset.has_len(data_object) else None
            shape = (N, )
            elem = Dataset._get_elem(data_object)
            while elem is not None and Dataset._get_elem(elem) != elem:
                shape = (*shape, len(elem) if Dataset.has_len(elem) else None)
                elem = Dataset._get_elem(elem)
            # last None in the shape is bottom-level element
            shape = shape[:-1]
        return shape

    @map_if_multi("_object")
    def get_shape(self):
        return Dataset._get_shape(self._object)

    @staticmethod
    def _get_dtype(data_object):
        elem = Dataset._get_elem(data_object)
        dtype = getattr(data_object, "dtype", None)
        if dtype is None:
            dtype = getattr(elem, "dtype", None) or (type(elem) if type(elem) in (int, float, complex) else None)
            while dtype is None and elem is not None and Dataset._get_elem(elem) != elem:
                elem = Dataset._get_elem(elem)
                dtype = getattr(elem, "dtype", None) or (type(elem) if type(elem) in (int, float, complex) else None)
        if isinstance(dtype, torch.dtype):
            pass
        elif dtype in (float, int, complex):
            dtype = getattr(torch, np.dtype(dtype).name, None)
        elif getattr(dtype, "name", False):  # it's a numpy array!
            dtype = getattr(torch, dtype.name, dtype)
        return dtype

    @map_if_multi("_object")
    def get_dtype(self):
        return Dataset._get_dtype(self._object)

    @staticmethod
    def _get_device(data_object):
        elem = Dataset._get_elem(data_object)
        device = getattr(data_object, "device", None)
        if device is None:
            device = getattr(elem, "device", "cpu")
        if isinstance(device, torch.device):
            device = device.type
        return device

    @map_if_multi("_object")
    def get_device(self):
        return Dataset._get_device(self._object)

    @map_if_multi("_object")
    def to_tensor(self):
        if issubclass(type(self._object), torch.Tensor):
            return
        elif "iter" in self.style:
            raise TypeError("Cannot convert 'iter' style objects to tensor.")
        maybe_tensor = default_convert(self._object[:])
        if issubclass(type(maybe_tensor), torch.Tensor):
            self._object = maybe_tensor
            return
        try:
            self._object = torch.tensor(self._object)
        except Exception as e:
            raise e
        return

    @map_if_multi("_object")
    def select(self, indices, inplace=False):
        if not self.has_len(self._object):
            raise TypeError("Cannot select indices of objects that don't implement `__len__`")
        if inplace:
            self._object = self._object[np.sort(indices)]
        else:  # TODO : Should be a proper wrapper:
            self._object = Subset(self._object, indices)

    @map_if_multi("_object")
    def to(self, device):
        """move any underlying tensor to some device"""
        if isinstance(self._object, torch.Tensor):
            self._object.to(device)

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

    def load(self, **kwargs):
        """pack self into a batch producer (Dataloader)"""
        return DataLoader(self, **kwargs)

    def random_examples(self, n):
        """get n random examples"""
        if "map" not in self.style:
            raise NotImplemented("Cannot sample random examples from generator objects")
        idx = np.random.randint(0, len(self), n)
        features = zip(*[self[i] for i in idx])
        features = [torch.stack(feat) if isinstance(feat[0], torch.Tensor) else np.stack(feat)
                    for feat in features]
        return features

    def to_datamodule(self,
                      to_tensor=False,
                      to_gpu=False,
                      splits=None,
                      **loader_kwargs):
        if to_gpu:
            self.to_tensor()
            if not torch.cuda.is_available():
                warnings.warn("You requested to move data to the gpu with `to_gpu=True` "
                              "but, currently, no gpu is available.")
            else:
                self.to("cuda")

        if to_tensor and not to_gpu:
            self.to_tensor()

        if splits is None:
            splits = (1.,)
        sets = self.split(splits)

        class DataModule(pl.LightningDataModule):
            train_ds = None
            val_ds = None
            test_ds = None
            loader_kwargs = {}

            def train_dataloader(self):
                if self.train_ds is not None:
                    return DataLoader(self.train_ds, **self.loader_kwargs)

            def val_dataloader(self):
                if self.val_ds is not None:
                    return DataLoader(self.val_ds, **self.loader_kwargs)

            def test_dataloader(self):
                if self.test_ds is not None:
                    return DataLoader(self.test_ds, **self.loader_kwargs)

        dm = DataModule()
        for ds, attr in zip(sets, ["train_ds", "val_ds", "test_ds"]):
            setattr(dm, attr, ds)
        setattr(dm, "loader_kwargs", loader_kwargs)
        return dm
