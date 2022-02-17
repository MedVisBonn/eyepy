import numpy as np
import matplotlib.pyplot as plt


class EyeEnface:
    def __init__(self, data, meta, annotations=None):
        self.data = data
        self.annotations = annotations
        self.meta = meta

    @property
    def scale_x(self):
        return self.meta["scale_x"]

    @property
    def scale_y(self):
        return self.meta["scale_y"]

    @property
    def size_x(self):
        return self.shape[1]

    @property
    def size_y(self):
        return self.shape[0]

    @property
    def laterality(self):
        return self.meta["laterality"]

    @property
    def shape(self):
        return self.data.shape

    def plot(self, ax=None, region=np.s_[...]):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.data[region], cmap="gray")

    def register(self):
        pass
