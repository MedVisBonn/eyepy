import numpy as np
import matplotlib.pyplot as plt
from eyepy import config
from skimage.transform._geometric import GeometricTransform


class EyeData:
    def __init__(
        self,
        volume: "EyeVolume",
        localizer: "EyeEnface",
        transformation: GeometricTransform,
    ):
        self.volume = volume
        self.localizer = localizer
        self.localizer_transformation = (
            transformation  # Localizer to OCT transformation
        )

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

    @property
    def drusen_projection(self):
        # Sum the all B-Scans along their first axis (B-Scan height)
        # Swap axis such that the volume depth becomes the projections height not width
        # We want the first B-Scan to be located at the bottom hence flip along axis 0
        return np.flip(np.swapaxes(np.sum(self.drusen, axis=0), 0, 1), axis=0)

    @property
    def drusen_enface(self):
        """Drusen projection warped into the localizer space."""
        return transform.warp(
            self.drusen_projection.astype(float),
            self.tform_oct_to_localizer,
            output_shape=self.localizer_shape,
            order=0,
        )

    # Data Access:
    # Bscans r
    # Projections r(w)
    # Shadows r(w)
    # Annotations rw
    # Registrations rw
    # Meta rw

    # Bscan View into the volume

    # Projection views of volume in enface space

    # Projection views of volume annotation in enface space

    # Save and load data

    # Export annotations only
