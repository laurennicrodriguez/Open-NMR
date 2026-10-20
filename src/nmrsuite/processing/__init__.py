"""Processing steps. Each takes an NMRData and returns a new one (never in place)."""

from nmrsuite.processing.digital_filter import remove_group_delay

__all__ = ["remove_group_delay"]
