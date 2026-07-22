"""Package-specific exceptions."""


class GNSMError(Exception):
    """Base exception for recoverable GNSM failures."""


class ConfigurationError(GNSMError):
    """Raised when a configuration is missing or internally inconsistent."""


class OptionalDependencyError(GNSMError):
    """Raised when an optional model integration is used without its dependency."""
