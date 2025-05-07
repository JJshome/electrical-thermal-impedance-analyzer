"""
Acquisition package for the Integrated Electrical-Thermal Impedance Analyzer

This package contains modules for acquiring electrical and thermal impedance
measurements and integrating them for comprehensive system analysis.

Modules:
- electrical_impedance: Classes for electrical impedance measurement
- thermal_impedance: Classes for thermal impedance measurement
- integrated_impedance_analyzer: Main class that integrates both methods

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

from .electrical_impedance import ElectricalImpedanceMeasurement
from .thermal_impedance import ThermalImpedanceMeasurement
from .integrated_impedance_analyzer import (
    IntegratedImpedanceAnalyzer,
    SystemType,
    CorrelationMethod
)

__all__ = [
    'ElectricalImpedanceMeasurement',
    'ThermalImpedanceMeasurement',
    'IntegratedImpedanceAnalyzer',
    'SystemType',
    'CorrelationMethod'
]
