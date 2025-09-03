"""Custom exceptions for the application."""

class DataModelingException(Exception):
    """Base exception for data modeling errors."""
    pass

class LLMServiceException(DataModelingException):
    """Exception raised when LLM service fails."""
    pass

class ValidationException(DataModelingException):
    """Exception raised when data validation fails."""
    pass

class StorageException(DataModelingException):
    """Exception raised when storage operations fail."""
    pass 