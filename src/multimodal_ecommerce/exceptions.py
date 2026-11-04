"""Shared exception root for actionable e-commerce workflow failures."""


class EcommerceError(Exception):
    """Identify a business failure while preserving each subtype's standard base."""
