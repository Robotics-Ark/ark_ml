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
        Retrieve a registered class by name or raise an error.
        Args:
            name: Name of class to retrieve.

        Returns:
            Class object or raise an error.
        """
        if name == "DiffusionPolicyModel":
            import arkml.algos.diffusion_policy.algorithm
            import arkml.algos.diffusion_policy.models
        elif name == "PiZeroNet":
            import arkml.algos.vla.pizero.algorithm
            import arkml.algos.vla.pizero.models
        elif name == "act":
            import arkml.algos.act.algorithm
            import arkml.algos.act.models
        elif name == "sb3rl":
            import arkml.algos.rl.sb3_algorithm
            import arkml.algos.rl.sb3_models
        elif name == "pi05":
            import arkml.algos.vla.pi05.algorithm
            import arkml.algos.vla.pi05.models
        elif name == "Pi05Policy":
            import arkml.algos.vla.pi05.algorithm
            import arkml.algos.vla.pi05.models
        else:
            raise ValueError(f"Unknown model {name}")

        return self._registry[name]


# Global registries
ALGOS = Registry()
MODELS = Registry()
