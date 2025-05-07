"""
Integrated Electrical-Thermal Impedance Analysis System.

This package provides tools for simultaneous acquisition and analysis of
electrical and thermal impedance data for comprehensive system characterization.
"""

from ._version import __version__

# Import key components
from .analyzer import IntegratedImpedanceAnalyzer
from .electrical_impedance import ElectricalImpedanceAnalyzer, MeasurementMode
from .thermal_impedance import ThermalImpedanceAnalyzer, ThermalStimulationMode
from .thermal_management import PCMThermalManager, PCMType, ThermalEnhancerType
from .ai_analyzer import AIAnalyzer, ImpedanceDataset

# Define package exports
__all__ = [
    'IntegratedImpedanceAnalyzer',
    'ElectricalImpedanceAnalyzer',
    'ThermalImpedanceAnalyzer',
    'PCMThermalManager',
    'AIAnalyzer',
    'ImpedanceDataset',
    'MeasurementMode',
    'ThermalStimulationMode',
    'PCMType',
    'ThermalEnhancerType',
]
