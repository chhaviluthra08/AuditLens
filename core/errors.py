"""Custom exceptions for AuditLens."""

from typing import Optional


class AuditLensError(Exception):
    """Base exception class for AuditLens with user-facing messages and remedies."""

    def __init__(self, message: str, fix_hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.fix_hint = fix_hint or "Please review your input data and parameters."

    def __str__(self) -> str:
        if self.fix_hint:
            return f"{self.message} (Suggested fix: {self.fix_hint})"
        return self.message


class InvalidInputError(AuditLensError):
    """Raised when uploaded CSV fails schema, value range, or size checks."""

    pass


class ConformalFailure(AuditLensError):
    """Raised when conformal prediction cannot be computed or calibration set is invalid."""

    pass


class CapacityError(AuditLensError):
    """Raised when capacity constraints cannot be satisfied."""

    pass
