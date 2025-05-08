# Testing

This directory contains unit tests for the Integrated Electrical-Thermal Impedance Analysis System.

## Test Structure

The tests are organized to align with the structure of the software implementation:

- `test_impedance_analyzer.py` - Tests for the core `IntegratedImpedanceAnalyzer` class
- `test_electrical_module.py` - Tests for the electrical impedance measurement module
- `test_thermal_module.py` - Tests for the thermal impedance measurement module
- `test_signal_processing.py` - Tests for signal processing functions
- `test_analysis.py` - Tests for data analysis and equivalent circuit modeling
- `test_applications.py` - Tests for specific application implementations

## Running Tests

You can run all tests using pytest:

```bash
pytest tests/
```

Or run specific test files:

```bash
pytest tests/test_impedance_analyzer.py
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=impedance_analyzer tests/
```

## Continuous Integration

These tests are automatically run in the CI/CD pipeline whenever changes are pushed to the repository. The CI pipeline ensures that:

1. All tests pass
2. Code coverage remains above the threshold (currently 80%)
3. There are no linting errors or warnings

## Adding New Tests

When adding new functionality to the codebase, please also add corresponding tests. Follow these guidelines:

1. Create test methods that test a single concept
2. Use descriptive names that explain what is being tested
3. Include both positive tests (expected behavior) and negative tests (error handling)
4. Mock external dependencies when appropriate
5. Keep tests independent of each other

## Test Data

The `test_data/` subdirectory contains sample data files used for testing, including:

- Reference impedance spectra
- Calibration data
- Sample measurement data for various applications

Please do not modify these reference files unless the expected behavior of the system has changed.
