import numpy as np
from skimage import img_as_ubyte


class DynamicDefaultDict(dict):
    """A defaultdict for which the factory function has access to the missing key"""

    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


def vol_intensity_transform(data):
    selection_0 = data == np.finfo(np.float32).max
    selection_data = data <= 1

    new = np.log(data[selection_data] + 2.44e-04)
    new = (new + 8.3) / 8.285

    data[selection_data] = new
    data[selection_0] = 0
    data = np.clip(data, 0, 1)
    return img_as_ubyte(data)


def default_intensity_transform(data):
    return data


intensity_transforms = {
    "default": default_intensity_transform,
    "vol": vol_intensity_transform,
}
