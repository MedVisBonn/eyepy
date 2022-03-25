import datetime
import json
import os
from typing import List, MutableMapping, Tuple


class EyeMeta(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def as_dict(self):
        data = self._store.copy()
        for key in ["visit_date", "exam_time"]:
            if key in data.keys() and data[key] is not None:
                data[key] = data[key].isoformat()
        return data

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __str__(self):
        return f"{os.linesep}".join([f"{f}: {self[f]}" for f in self if f != "__empty"])

    def __repr__(self):
        return self.__str__()


class EyeEnfaceMeta(EyeMeta):
    def __init__(self, scale_x: float, scale_y: float, scale_unit: str, **kwargs):
        """A dict with required keys to hold meta data for enface images of the eye

        Args:
            scale_x: Horizontal scale of the enface pixels
            scale_y: Vertical scale of the enface pixels
            scale_unit: Unit of the scale. e.g. µm if scale is given in µm/pixel
            **kwargs:
        """
        super().__init__(
            scale_x=scale_x, scale_y=scale_y, scale_unit=scale_unit, **kwargs
        )

    @classmethod
    def from_dict(cls, data: dict):
        for key in ["visit_date", "exam_time"]:
            if key in data.keys() and data[key] is not None:
                data[key] = datetime.datetime.fromisoformat(data[key])
        return cls(**data)


class EyeBscanMeta(EyeMeta):
    def __init__(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        pos_unit: str,
        **kwargs,
    ):
        """A dict with required keys to hold meta data for OCT B-scans

        Args:
            start_pos: B-scan start on the enface
            end_pos: B-scan end on the enface
            pos_unit: Unit of the positions
            **kwargs:
        """
        super().__init__(
            start_pos=start_pos, end_pos=end_pos, pos_unit=pos_unit, **kwargs
        )


class EyeVolumeMeta(EyeMeta):
    def __init__(
        self,
        scale_z: float,
        scale_x: float,
        scale_y: float,
        scale_unit: str,
        bscan_meta: List[EyeBscanMeta],
        **kwargs,
    ):
        """A dict with required keys to hold meta data for OCT volumes

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

    def as_dict(self):
        data = super().as_dict()
        data["bscan_meta"] = [bm.as_dict() for bm in data["bscan_meta"]]
        return data

    @classmethod
    def from_dict(cls, data: dict):
        data["bscan_meta"] = [EyeBscanMeta(**d) for d in data["bscan_meta"]]
        for key in ["visit_date", "exam_time"]:
            if key in data.keys() and data[key] is not None:
                data[key] = datetime.datetime.fromisoformat(data[key])
        return cls(**data)
