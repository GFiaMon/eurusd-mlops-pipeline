"""
Utilities package for EUR/USD ML Pipeline.

Contains shared utilities used across the project:
- DataManager: Unified data access layer with S3 and local mirroring
"""

from .data_manager import DataManager

__all__ = ['DataManager']
