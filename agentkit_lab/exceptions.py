"""Project-specific exceptions."""


class AgentKitError(Exception):
    """Base exception for the project."""


class MissingDependencyError(AgentKitError):
    """Raised when a required runtime package is missing."""


class ProviderConfigurationError(AgentKitError):
    """Raised when the live LLM provider is not configured."""


class InternalServiceError(AgentKitError):
    """Base class for recoverable service failures."""


class InternalDatabaseUnavailableError(InternalServiceError):
    """Raised when the mock internal database returns a server error."""


class InternalDatabaseBadRequestError(InternalServiceError):
    """Raised when the mock internal database receives invalid input."""


class InternalDatabaseTimeoutError(InternalServiceError):
    """Raised when the mock internal database times out."""


class InternalDatabaseAuthError(InternalServiceError):
    """Raised when the mock internal database rejects authentication."""
