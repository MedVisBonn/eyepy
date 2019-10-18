import abc


class OctReader(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def read(self, filepath):
        """Returns an OCT object"""
        pass

    @abc.abstractmethod
    def read_meta(self, filepath):
        """Returns only the OCT meta data"""
        pass

    @abc.abstractmethod
    def read_bscans(self, filepath):
        """Returns only the B-scans"""
        pass

    @abc.abstractmethod
    def read_nir(self, filepath):
        """Returns only the near-infrared fundus reflectance (NIR) acquired by an scanning laser ophthalmoscope (SLO)"""
        pass
