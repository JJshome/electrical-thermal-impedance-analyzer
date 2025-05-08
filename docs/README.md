# Documentation

This directory contains documentation for the Integrated Electrical-Thermal Impedance Analysis System.

## Contents

- [System Overview](system_overview.md)
- [Hardware Documentation](hardware/README.md)
- [Software Documentation](software/README.md)
- [Application Examples](examples/README.md)
- [API Reference](api_reference.md)

## Getting Started

For a quick start guide to using the system, refer to the main [README.md](../README.md) file in the repository root.

## Technical Details

The system combines electrical impedance spectroscopy (EIS) and thermal impedance spectroscopy (TIS) to provide comprehensive characterization of various systems including energy storage devices, semiconductor components, and biological tissues.

### Electrical Impedance Spectroscopy (EIS)

The EIS module measures the electrical impedance of a system across a wide frequency range (0.1Hz to 500kHz). Key components include:

- Precision waveform generator
- Low-noise current source
- High-resolution voltage measurement
- Phase-sensitive detection

### Thermal Impedance Spectroscopy (TIS)

The TIS module measures the thermal impedance of a system by applying controlled thermal stimuli and measuring the temperature response. Key components include:

- Precision thermal stimulation (Peltier element)
- High-resolution temperature sensing
- Phase Change Material (PCM) thermal management
- Adaptive thermal control

### Integrated Signal Processing

The system integrates electrical and thermal impedance data to provide a comprehensive characterization of the system under test:

- Cross-domain correlation analysis
- Feature extraction
- Equivalent circuit modeling
- AI-based pattern recognition

## Advanced Topics

- [Calibration Procedures](advanced/calibration.md)
- [Custom Measurement Protocols](advanced/custom_protocols.md)
- [Data Analysis Techniques](advanced/data_analysis.md)
- [Machine Learning Models](advanced/ml_models.md)
