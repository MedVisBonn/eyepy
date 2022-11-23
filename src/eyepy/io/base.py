import abc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from nptyping import NDArray
from nptyping import Shape


class BaseLayerAnnotation(metaclass=abc.ABCMeta):
    def __init__(self, data, meta):
        pass


class BaseAscanAnnotation(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class BaseVolumeAnnotation(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class BaseShapeAnnotation(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class BasePixelAnnotation(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class BaseOCTVolume(metaclass=abc.ABCMeta):
    def __init__(self):
        self._data = None
        self._meta = None
        self._layer_annotations = None
        self._ascan_annotations = None
        self._volume_annotations = None

    @property
    @abc.abstractmethod
    def data(self) -> NDArray[Shape["*, *, *"], Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self) -> Dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def layer_annotations(self) -> List[BaseLayerAnnotation]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ascan_annotations(self) -> List[BaseAscanAnnotation]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def volume_annotations(self) -> List[BaseVolumeAnnotation]:
        raise NotImplementedError


class BaseEnface(metaclass=abc.ABCMeta):
    def __init__(self):
        self._data = None
        self._meta = None
        self._shape_annotations = None
        self._pixel_annotations = None

    @property
    @abc.abstractmethod
    def data(self) -> NDArray[Shape["*, *, *"], Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self) -> Dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape_annotations(self) -> List[BaseShapeAnnotation]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pixel_annotations(self) -> List[BasePixelAnnotation]:
        raise NotImplementedError


class BaseOCTOpener(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def read(
        self, path: Union[str, Path]
    ) -> Tuple[
        Union[BaseOCTVolume, List[BaseOCTVolume]], Union[BaseEnface, List[BaseEnface]]
    ]:
        """Read OCT data from disk

        Args:
            path: Filepath

        Returns:
            tuple holding one or more OCT volumes found in the file and one or more enface images
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write(
        self,
        path: Union[str, Path],
        volumes: Union[BaseOCTVolume, List[BaseOCTVolume]],
        localizers: Union[BaseEnface, List[BaseEnface]],
    ) -> None:
        """Write OCT data to disk

        Args:
            path: Filepath
            volumes: OCT volume or List of OCT volumes if the format support saving multiple volumes
            localizers: Enface image or List of enface images if the format support saving multiple enface images

        Returns:
            None
        """
        raise NotImplementedError
