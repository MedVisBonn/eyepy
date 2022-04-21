# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Union

import numpy as np

from eyepy.core.eyemeta import EyeMeta

# PR1 and EZ map to 14 and PR2 and IZ map to 15. Hence both names can be used
# to access the same data
SEG_MAPPING = {
    "ILM": 0,
    "BM": 1,
    "RNFL": 2,
    "GCL": 3,
    "IPL": 4,
    "INL": 5,
    "OPL": 6,
    "ONL": 7,
    "ELM": 8,
    "IOS": 9,
    "OPT": 10,
    "CHO": 11,
    "VIT": 12,
    "ANT": 13,
    "PR1": 14,
    "PR2": 15,
    "RPE": 16,
}


class LazyMeta(EyeMeta):
    def __init__(self, *args, **kwargs):
        """The Meta object is a dict with additional functionalities.

        The additional functionallities are:
        1. A string representation suitable for printing the meta information.

        2. Checking if a keys value is a callable before returning it. In case
        it is a callable, it sets the value to the return value of the callable.
        This is used for lazy loading OCT data. The meta information for the OCT
        and all B-Scans is only read from the file when accessed.

        An instance of the meta object can be created as you would create an
        ordinary dictionary.

        For example:

            + Meta({"SizeX": 512})
            + Meta(SizeX=512)
            + Meta([(SizeX, 512), (SizeY, 512)])
        """
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        value = self._store[key]
        if callable(value):
            self[key] = value()
        return self._store[key]


class LazyAnnotation(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self._bscan = None

    def __getitem__(self, key):
        value = self._store[key]
        if callable(value):
            self[key] = value(self.bscan)
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    # TODO: Make the annotation printable to get an overview
    # def __str__(self):
    #     return f"{os.linesep}".join(
    #         [f"{f}: {self[f]}" for f in self if f != "__empty"])

    # def __repr__(self):
    #     return self.__str__()

    @property
    def bscan(self):
        if self._bscan is None:
            raise AttributeError("bscan is not set for this Annotation.")
        return self._bscan

    @bscan.setter
    def bscan(self, value: "Bscan"):
        self._bscan = value


class LazyLayerAnnotation(MutableMapping):
    def __init__(self, data, layername_mapping=None, max_height=2000):
        self._data = data
        self.max_height = max_height
        if layername_mapping is None:
            self.mapping = SEG_MAPPING
        else:
            self.mapping = layername_mapping

    @property
    def data(self):
        if callable(self._data):
            self._data = self._data()
        return self._data

    def __getitem__(self, key):
        data = self.data[self.mapping[key]]
        nans = np.isnan(data)
        empty = np.nonzero(
            np.logical_or(
                np.less(data, 0, where=~nans),
                np.greater(data, self.max_height, where=~nans),
            )
        )
        data = np.copy(data)
        data[empty] = np.nan
        if np.nansum(data) > 0:
            return data
        else:
            raise KeyError(f"There is no data given for the {key} layer")

    def __setitem__(self, key, value):
        self.data[self.mapping[key]] = value

    def __delitem__(self, key):
        self.data[self.mapping[key], :] = np.nan

    def __iter__(self):
        inv_map = {v: k for k, v in self.mapping.items()}
        return iter(inv_map.values())

    def __len__(self):
        return len(self.data.shape[0])

    def layer_indices(self, key):
        layer = self[key]
        nan_indices = np.isnan(layer)
        col_indices = np.arange(len(layer))[~nan_indices]
        row_indices = np.rint(layer).astype(int)[~nan_indices]

        return (row_indices, col_indices)


class LazyEnfaceImage:
    def __init__(self, data, name=None):
        self._data = data
        self._name = name

    @property
    def data(self):
        """Return the enface image as numpy array."""
        if callable(self._data):
            self._data = self._data()
        return self._data

    @property
    def name(self):
        if self._name is None:
            raise ValueError("This EnfaceImage has no respective filename")
        else:
            return self._name


class LazyBscan:
    def __new__(
        cls,
        data,
        annotation=None,
        meta=None,
        data_processing=None,
        oct_obj=None,
        name=None,
        *args,
        **kwargs,
    ):
        # Make all meta fields accessible as attributes of the BScan without
        # reading them. Set a property instead
        def meta_func_builder(x):
            return lambda self: self.meta[x]

        if meta is not None:
            for key in meta:
                setattr(cls, key, property(meta_func_builder(key)))
        return object.__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Union[np.ndarray, Callable],
        annotation: Optional[LazyAnnotation] = None,
        meta: Optional[Union[Dict, LazyMeta]] = None,
        data_processing: Optional[Callable] = None,
        oct_obj: Optional["Oct"] = None,
        name: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        data : A numpy array holding the raw B-Scan data or a callable which
            returns a raw B-Scan. Raw means that it represents the unprocessed
            stored data. The actual dtype and value range depends on the storage
            format.
        annotation: Dict holding B-Scan annotations
        meta : A dictionary holding the B-Scans meta informations or
        oct_obj : Reference to the OCT Volume holding the B-Scan
        name : Filename of the B-Scan if B-Scan is save as individual file
        """
        self._scan_raw = data
        self._scan = None
        self._meta = meta
        self._oct_obj = oct_obj

        self._annotation = annotation

        if data_processing is None:
            self._data_processing = lambda x: x
        else:
            self._data_processing = data_processing

        self._name = name

    @property
    def oct_obj(self):
        if self._oct_obj is None:
            raise AttributeError("oct_obj is not set for this Bscan object")
        return self._oct_obj

    @oct_obj.setter
    def oct_obj(self, value):
        self._oct_obj = value

    @property
    def name(self):
        if self._name is None:
            self._name = str(self.index)
        return self._name

    @property
    def index(self):
        return self.oct_obj.bscans.index(self)

    @property
    def meta(self):
        """A dict holding all Bscan meta data."""
        return self._meta

    @property
    def annotation(self):
        """A dict holding all Bscan annotation data."""
        if self._annotation is None:
            self._annotation = LazyAnnotation({})
        elif callable(self._annotation):
            self._annotation = self._annotation()
        self._annotation.bscan = self
        return self._annotation

    @property
    def scan_raw(self):
        """An array holding a single raw B-Scan.

        The dtype is not changed after the import. If available this is
        the unprocessed output of the OCT device. In any case this is
        the unprocessed data imported by eyepy.
        """
        if callable(self._scan_raw):
            self._scan_raw = self._scan_raw()
        return self._scan_raw

    @property
    def scan(self):
        """An array holding a single B-Scan with the commonly used contrast.

        The array is of dtype <ubyte> and encodes the intensities as
        values between 0 and 255.
        """
        if self._scan is None:
            self._scan = self._data_processing(self.scan_raw)
        return self._scan

    @property
    def shape(self):
        return self.scan.shape

    @property
    def layers(self):
        if "layers" not in self.annotation:
            l_shape = np.zeros((max(SEG_MAPPING.values()) + 1, self.oct_obj.SizeX))
            self.annotation["layers"] = LazyLayerAnnotation(l_shape)
        if callable(self.annotation["layers"]):
            self.annotation["layers"] = self.annotation["layers"]()
        return self.annotation["layers"]


class LazyVolume:
    """.vol header
    -----------
    All fields from the .vol header (the oct meta data) can be accessed as attributes of the
    HeyexOct object.

    SLO
    ---
    The attribute `slo` of the HeyexOct object gives access to the IR SLO image
    and returns it as a numpy.ndarray of dtype `uint8`.

    B-Scans
    -------
    Individual B-Scans can be accessed using `oct_scan[index]`. The returned
    HeyexBscan object exposes all B-Scan header fields as attributes and the
    raw B-Scan image as `numpy.ndarray` of type `float32` under the attribute
    `scan_raw`. A transformed version of the raw B-Scan which is more similar to
    the Heyex experience can be accessed with the attribute `scan` and returns
    the 4th root of the raw B-Scan scaled to [0,255] as `uint8`.

    Segmentations
    -------------
    B-Scan segmentations can be accessed for individual B-Scans like
    `bscan.segmentation`. This return a numpy.ndarray of shape (NumSeg, SizeX)
    The `segmentation` attribute of the HeyexOct object returns a dictionary,
    where the key is a number and the value is a numpy.ndarray of shape
    (NumBScans, SizeX)."""

    def __new__(
        cls,
        bscans: List[LazyBscan],
        localizer=None,
        meta=None,
        data_path=None,
        *args,
        **kwargs,
    ):
        # Set all the meta fields as attributes

        if meta is not None:

            def meta_func_builder(x):
                return lambda self: self.meta[x]

            for key in meta:
                # Every key in meta becomes a property for the new class. The
                # property calls self.meta[key] to retrieve the keys value

                # This lambda func returns a lambda func which accesses the meta
                # object for the key specified in the first lambda. Like this we
                # read the file only on access.
                setattr(cls, key, property(meta_func_builder(key)))

        return object.__new__(cls, *args, **kwargs)

    def __init__(
        self,
        bscans: List[Union[Callable, LazyBscan]],
        localizer: Optional[Union[Callable, LazyEnfaceImage]] = None,
        meta: Optional[LazyMeta] = None,
        data_path: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        bscans :
        meta :
        drusenfinder :
        """
        self.bscans = bscans
        self._localizer = localizer
        self._meta = meta
        self._tform_localizer_to_oct = None

        self._eyepy_id = None
        if data_path is None:
            self.data_path = Path.home() / ".eyepy"
        self.data_path = Path(data_path)

    def __getitem__(self, index) -> Union[LazyBscan, List[LazyBscan]]:
        """The B-Scan at the given index."""
        if type(index) == slice:
            return [self[i] for i in range(*index.indices(len(self)))]
        else:
            bscan = self.bscans[index]
            if callable(bscan):
                self.bscans[index] = bscan()
            self.bscans[index].oct_obj = self
            return self.bscans[index]

    def __len__(self):
        """The number of B-Scans."""
        return len(self.bscans)

    @property
    def shape(self):
        return (self.meta["NumBScans"], self.meta["SizeY"], self.meta["SizeX"])

    @property
    def localizer(self):
        """A numpy array holding the OCTs localizer enface if available."""
        try:

            return self._localizer.data
        except AttributeError:
            raise AttributeError("This OCT object has no localizer image.")

    @property
    def volume_raw(self):
        """An array holding the OCT volume.

        The dtype is not changed after the import. If available this is
        the unprocessed output of the OCT device. In any case this is
        the unprocessed data imported by eyepy.
        """
        return np.stack([x.scan_raw for x in self.bscans], axis=0)

    @property
    def volume(self):
        """An array holding the OCT volume with the commonly used contrast.

        The array is of dtype <ubyte> and encodes the intensities as
        values between 0 and 255.
        """
        return np.stack([x.scan for x in self.bscans], axis=0)

    @property
    def layers_raw(self):
        """Height maps for all layers combined into one volume.

        Layers for all B-Scans are stacked such that we get a volume L x B x W
        where L are different Layers, B are the B-Scans and W is the Width of
        the B-Scans.

        A flip on the B-Scan axis is needed to locate the first B-Scan at the
        bottom of the height map.
        """
        return np.flip(np.stack([x.layers.data for x in self], axis=1), axis=1)

    @property
    def layers(self):
        """Height maps for all layers accessible by the layers name."""
        nans = np.isnan(self.layers_raw)
        empty = np.nonzero(
            np.logical_or(
                np.less(self.layers_raw, 0, where=~nans),
                np.greater(self.layers_raw, self.meta["SizeY"], where=~nans),
            )
        )

        data = self.layers_raw.copy()
        data[empty] = np.nan
        return {
            name: data[i, ...]
            for name, i in self[0].layers.mapping.items()
            if np.nansum(data[i, ...]) != 0
        }

    @property
    def meta(self):
        """A dict holding all OCT meta data.

        The object can be printed to see all available meta data.
        """
        if self._meta is None:
            raise AttributeError("This volume has no meta data")
        return self._meta
