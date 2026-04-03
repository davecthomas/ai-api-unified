"""
Typed exceptions for PII redaction middleware failures.

These exceptions provide deterministic failure semantics for dependency-resolution
and runtime redaction execution issues.
"""


class PiiRedactionError(RuntimeError):
    """
    Base exception for all PII redaction middleware failures.
    """


class PiiRedactionDependencyUnavailableError(PiiRedactionError):
    """
    Raised when strict redaction mode requires dependencies that are not installed
    or are otherwise unavailable at runtime.
    """


class PiiRedactionRuntimeError(PiiRedactionError):
    """
    Raised when redaction execution fails while sanitizing text.
    """
