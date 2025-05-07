"""
Integrated Electrical-Thermal Impedance Analysis System.

This package provides tools for simultaneous acquisition and analysis 
of electrical and thermal impedance data for comprehensive 
characterization of various systems.
"""

from ._version import __version__

from .analyzer import IntegratedImpedanceAnalyzer

__all__ = ['IntegratedImpedanceAnalyzer']