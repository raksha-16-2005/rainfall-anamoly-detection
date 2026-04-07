"""
Pytest configuration and fixtures.

Handles import path setup for parent directory modules.
"""

import sys
import os

# Add parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
