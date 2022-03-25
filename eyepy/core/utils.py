class DynamicDefaultDict(dict):
    """A defaultdict for which the factory function has access to the missing key"""

    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]
