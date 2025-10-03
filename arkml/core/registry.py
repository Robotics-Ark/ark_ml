"""Lightweight registries that map string identifiers to ArkML components."""


class Registry:
    """Simple decorator-based container for storing classes by name."""

    def __init__(self):
        self._registry = {}

    def register(self, name):
        """
        Register a class under the provided name via decorator usage.
        Args:
            name: Name of the class to register.

        Returns:
            Registered class.
        """

        def decorator(cls):
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name):
        """
        Retrieve a previously registered class by name or raise an error.
        Args:
            name: Name of class to retrieve.

        Returns:
            Class object or raise an error.
        """
        if name not in self._registry:
            raise ValueError(
                f"{name} not found in registry. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]


# Global registries
ALGOS = Registry()
MODELS = Registry()
