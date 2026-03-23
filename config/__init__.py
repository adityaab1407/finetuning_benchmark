"""Configuration module for Financial Fine-Tuning Laboratory.

Exports the Settings class, singleton instance, and getter function
so consumers can simply do: from config import settings
"""

from config.settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
