from __future__ import annotations

import datetime
import json
import os
from typing import Any, Iterable, MutableMapping, Union


class EyeMeta(MutableMapping):
    """"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """

        Args:
            *args:
            **kwargs:
        """
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def as_dict(self) -> dict:
        """Return a copy of the metadata as a regular dictionary.

        Datetime objects are converted to ISO format strings.

        Returns:
            Dictionary containing all metadata key-value pairs
        """
        data = self._store.copy()

        for key in data:
            if isinstance(data[key], datetime.datetime):
                data[key] = data[key].isoformat()
        return data

    def copy(self) -> 'EyeMeta':
        """Create a shallow copy of this metadata object.

        Creates a new instance with the same metadata. Note that this is a
        shallow copy - the dictionary values themselves are not deeply copied.
        This is typically sufficient since metadata values are usually
        primitives (int, float, str) or immutable objects (datetime).

        This is useful when creating modified versions of metadata
        without affecting the original, particularly during image
        transformations.

        Returns:
            New EyeMeta instance with the same data

        Example:
            >>> original_meta = EyeMeta(key1='value1', key2='value2')
            >>> copied_meta = original_meta.copy()
            >>> copied_meta['key1'] = 'modified'
            >>> print(original_meta['key1'])  # Still 'value1'
        """
        return self.__class__(**dict(self._store))

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __str__(self) -> str:
        return f'{os.linesep}'.join(
            [f'{f}: {self[f]}' for f in self if f != '__empty'])

    def __repr__(self) -> str:
        return self.__str__()


class EyeEnfaceMeta(EyeMeta):
    """"""

    def __init__(self, scale_x: float, scale_y: float, scale_unit: str,
                 **kwargs: Any) -> None:
        """A dict with required keys to hold meta data for enface images of the
        eye.

        Args:
            scale_x: Horizontal scale of the enface pixels
            scale_y: Vertical scale of the enface pixels
            scale_unit: Unit of the scale. e.g. µm if scale is given in µm/pixel
            **kwargs:
        """
        super().__init__(scale_x=scale_x,
                         scale_y=scale_y,
                         scale_unit=scale_unit,
                         **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> 'EyeEnfaceMeta':
        """Create an EyeEnfaceMeta instance from a dictionary.

        Args:
            data: Dictionary containing metadata key-value pairs

        Returns:
            New EyeEnfaceMeta instance
        """
        for key in ['visit_date', 'exam_time']:
            if key in data.keys() and data[key] is not None:
                data[key] = datetime.datetime.fromisoformat(data[key])
        return cls(**data)

    def copy(self) -> 'EyeEnfaceMeta':
        """Create a shallow copy of this enface metadata object.

        Creates a new instance with the same metadata. Note that this is a
        shallow copy - the dictionary values themselves are not deeply copied.
        This is typically sufficient since metadata values are usually
        primitives (int, float, str) or immutable objects (datetime).

        Returns:
            New EyeEnfaceMeta instance with the same data

        Example:
            >>> meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')
            >>> copied = meta.copy()
            >>> copied['scale_x'] = 20.0
            >>> print(meta['scale_x'])  # Still 10.0
        """
        return self.__class__(**dict(self._store))


class EyeBscanMeta(EyeMeta):
    """"""

    def __init__(
        self,
        start_pos: tuple[float, float],
        end_pos: tuple[float, float],
        pos_unit: str,
        **kwargs: Any,
    ) -> None:
        """A dict with required keys to hold meta data for OCT B-scans.

        Args:
            start_pos: B-scan start position on the enface (x, y) in physical coordinates, not pixel indices
            end_pos: B-scan end position on the enface (x, y) in physical coordinates, not pixel indices
            pos_unit: Unit of the positions (e.g., 'mm')
            **kwargs: Additional metadata
        """
        start_pos = tuple(start_pos)
        end_pos = tuple(end_pos)
        super().__init__(start_pos=start_pos,
                         end_pos=end_pos,
                         pos_unit=pos_unit,
                         **kwargs)


class EyeVolumeMeta(EyeMeta):
    """"""

    def __init__(
        self,
        scale_z: float,
        scale_x: float,
        scale_y: float,
        scale_unit: str,
        bscan_meta: list[EyeBscanMeta],
        **kwargs: Any,
    ):
        """A dict with required keys to hold meta data for OCT volumes.

        Args:
            scale_z: Distance between neighbouring B-scans
            scale_x: Horizontal scale of the B-scan pixels
            scale_y: Vertical scale of the B-scan pixels
            scale_unit: Unit of the scale. e.g. µm if scale is given in µm/pixel
            bscan_meta: A list holding an EyeBscanMeta object for every B-scan of the volume
            **kwargs:
        """
        super().__init__(
            scale_z=scale_z,
            scale_x=scale_x,
            scale_y=scale_y,
            scale_unit=scale_unit,
            bscan_meta=bscan_meta,
            **kwargs,
        )

    def as_dict(self) -> dict:
        """

        Returns:

        """
        data = super().as_dict()
        data['bscan_meta'] = [bm.as_dict() for bm in data['bscan_meta']]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'EyeVolumeMeta':
        """

        Args:
            data:

        Returns:

        """
        data['bscan_meta'] = [EyeBscanMeta(**d) for d in data['bscan_meta']]
        return cls(**data)
