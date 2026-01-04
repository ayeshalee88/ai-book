"""
Configuration for pytest
"""
import sys
import os

# Add the backend directory to the path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))