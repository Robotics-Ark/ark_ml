class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name):
        if name not in self._registry:
            raise ValueError(f"{name} not found in registry. Available: {list(self._registry.keys())}")
        return self._registry[name]


# Global registries
ALGOS = Registry()
DATASETS = Registry()
MODELS = Registry()
