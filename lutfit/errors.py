"""Exceptions shared across the package (kept separate to avoid import cycles)."""


class Cancelled(Exception):
    """Raised when the user cancels a running fit."""
