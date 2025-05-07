"""
Electrical Impedance Measurement Module

This module provides classes and functions for measuring electrical impedance
across a wide frequency range using various measurement techniques.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
import time
from enum import Enum
import logging

# Set up logging
logger = logging.getLogger(__name__)

class MeasurementMode(Enum):
    """Enumeration for different impedance measurement modes"""
    SINGLE_FREQUENCY = 1
    FREQUENCY_SWEEP = 2
    MULTI_FREQUENCY = 3
    ADAPTIVE_SWEEP = 4


class ElectrodesConfig(Enum):
    """Enumeration for electrode configurations"""
    TWO_ELECTRODE = 1
    FOUR_ELECTRODE = 2


class ElectricalImpedanceMeasurement:
    """
    Class for electrical impedance measurement control and data acquisition
    
    This class handles the configuration and execution of electrical impedance
    measurements across various frequencies, using different techniques.
    """
    
    def __init__(self, hardware_interface=None):
        """
        Initialize the electrical impedance measurement system
        
        Parameters
        ----------
        hardware_interface : object, optional
            Interface to the measurement hardware. If None, a simulation
            mode will be used.
        """
        self.hardware = hardware_interface
        self.is_simulation = hardware_interface is None
        
        # Default configuration
        self.config = {
            'min_frequency': 0.1,      # Hz
            'max_frequency': 500000,    # Hz
            'num_points': 50,
            'voltage_amplitude': 10e-3, # V
            'current_range': 100e-6,    # A
            'electrode_config': ElectrodesConfig.FOUR_ELECTRODE,
            'measurement_mode': MeasurementMode.FREQUENCY_SWEEP,
            'integration_time': 0.1,    # seconds per measurement
            'averaging': 1,             # number of measurements to average
        }
        
        self._calibration_data = None
        self._last_results = None
        
        if self.is_simulation:
            logger.info("Running in simulation mode - no hardware connected")
        else:
            logger.info(f"Connected to hardware: {self.hardware.get_info()}")

    def configure(self, **kwargs):
        """
        Configure the measurement parameters
        
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters to update
            
        Returns
        -------
        dict
            Current configuration after updates
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
                
        # Validate configuration
        self._validate_config()
        
        if not self.is_simulation and self.hardware is not None:
            self.hardware.configure(self.config)
            
        return self.config
    
    def _validate_config(self):
        """Validate the current configuration"""
        if self.config['min_frequency'] <= 0:
            raise ValueError("Minimum frequency must be greater than zero")
            
        if self.config['max_frequency'] <= self.config['min_frequency']:
            raise ValueError("Maximum frequency must be greater than minimum frequency")
            
        if self.config['num_points'] < 2:
            raise ValueError("Number of frequency points must be at least 2")
            
        if self.config['voltage_amplitude'] <= 0:
            raise ValueError("Voltage amplitude must be positive")
            
        if self.config['integration_time'] <= 0:
            raise ValueError("Integration time must be positive")
    
    def calibrate(self, calibration_standards=None):
        """
        Perform calibration using known standards
        
        Parameters
        ----------
        calibration_standards : list, optional
            List of calibration standard values with known impedances
            
        Returns
        -------
        bool
            True if calibration succeeded
        """
        if self.is_simulation:
            logger.info("Simulation mode: Using ideal calibration values")
            self._calibration_data = {
                'open_circuit': float('inf'),
                'short_circuit': 0.0,
                'reference_resistor': 1000.0,  # Ohms
                'timestamp': time.time(),
                'temperature': 25.0  # °C
            }
            return True
            
        if calibration_standards is None:
            logger.warning("No calibration standards provided, using defaults")
            calibration_standards = ['open', 'short', '1k']
            
        try:
            calibration_results = {}
            
            for standard in calibration_standards:
                logger.info(f"Measuring calibration standard: {standard}")
                
                if not self.is_simulation:
                    # Prompt user to connect the right standard
                    input(f"Please connect {standard} standard and press Enter...")
                
                # Perform measurement
                if standard == 'open':
                    # Open circuit measurement
                    if self.is_simulation:
                        result = {'impedance': 1e9, 'phase': -90}
                    else:
                        result = self.hardware.measure_calibration('open')
                    calibration_results['open_circuit'] = result
                
                elif standard == 'short':
                    # Short circuit measurement
                    if self.is_simulation:
                        result = {'impedance': 0.1, 'phase': 0}
                    else:
                        result = self.hardware.measure_calibration('short')
                    calibration_results['short_circuit'] = result
                
                elif standard == '1k':
                    # 1kΩ reference resistor
                    if self.is_simulation:
                        result = {'impedance': 1000, 'phase': 0}
                    else:
                        result = self.hardware.measure_calibration('1k')
                    calibration_results['reference_resistor'] = result
                    
                else:
                    # Custom standard
                    if self.is_simulation:
                        result = {'impedance': 1000, 'phase': 0}
                    else:
                        result = self.hardware.measure_calibration(standard)
                    calibration_results[f'custom_{standard}'] = result
            
            # Add metadata
            calibration_results['timestamp'] = time.time()
            if not self.is_simulation and hasattr(self.hardware, 'get_temperature'):
                calibration_results['temperature'] = self.hardware.get_temperature()
            else:
                calibration_results['temperature'] = 25.0  # °C
                
            self._calibration_data = calibration_results
            logger.info("Calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False
    
    def measure(self, frequencies=None):
        """
        Perform impedance measurements
        
        Parameters
        ----------
        frequencies : array-like, optional
            Specific frequencies to measure. If None, uses the configured 
            frequency range and number of points.
            
        Returns
        -------
        dict
            Measurement results containing frequencies, impedance magnitudes, 
            and phases
        """
        # Check if calibration has been performed
        if self._calibration_data is None:
            logger.warning("No calibration data available, results may be inaccurate")
            
        # Generate frequency points if not provided
        if frequencies is None:
            if self.config['measurement_mode'] == MeasurementMode.SINGLE_FREQUENCY:
                frequencies = [self.config['min_frequency']]
            else:
                frequencies = np.logspace(
                    np.log10(self.config['min_frequency']),
                    np.log10(self.config['max_frequency']),
                    self.config['num_points']
                )
        
        # Initialize results arrays
        impedance_mag = np.zeros_like(frequencies, dtype=np.float64)
        impedance_phase = np.zeros_like(frequencies, dtype=np.float64)
        
        # Perform measurements
        logger.info(f"Starting electrical impedance measurement with {len(frequencies)} frequency points")
        start_time = time.time()
        
        for i, freq in enumerate(frequencies):
            if self.is_simulation:
                # Simulation mode: generate synthetic data
                # Example: RC circuit with R=1kΩ, C=1µF
                R = 1000  # Ohms
                C = 1e-6  # Farads
                omega = 2 * np.pi * freq
                Z_real = R / (1 + (omega * R * C)**2)
                Z_imag = -omega * R**2 * C / (1 + (omega * R * C)**2)
                
                # Add some noise
                noise_level = 0.01  # 1%
                Z_real *= (1 + noise_level * (2 * np.random.random() - 1))
                Z_imag *= (1 + noise_level * (2 * np.random.random() - 1))
                
                # Calculate magnitude and phase
                mag = np.sqrt(Z_real**2 + Z_imag**2)
                phase = np.degrees(np.arctan2(Z_imag, Z_real))
                
            else:
                # Hardware mode: perform actual measurement
                measurement = self.hardware.measure_impedance(freq, 
                                                            self.config['voltage_amplitude'],
                                                            self.config['integration_time'],
                                                            self.config['averaging'])
                mag = measurement['magnitude']
                phase = measurement['phase']
            
            impedance_mag[i] = mag
            impedance_phase[i] = phase
            
            # Progress update for long measurements
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * len(frequencies) / i
                remaining = estimated_total - elapsed
                logger.info(f"Progress: {i}/{len(frequencies)} frequencies measured. "
                          f"Est. {remaining:.1f}s remaining")
        
        elapsed = time.time() - start_time
        logger.info(f"Measurement completed in {elapsed:.1f} seconds")
        
        # Store and return results
        results = {
            'frequencies': frequencies,
            'magnitude': impedance_mag,
            'phase': impedance_phase,
            'timestamp': time.time(),
            'config': self.config.copy()
        }
        
        # Calculate complex impedance
        results['real'] = impedance_mag * np.cos(np.radians(impedance_phase))
        results['imaginary'] = impedance_mag * np.sin(np.radians(impedance_phase))
        
        self._last_results = results
        return results
    
    def analyze_impedance(self, results=None):
        """
        Basic analysis of impedance measurement results
        
        Parameters
        ----------
        results : dict, optional
            Measurement results to analyze. If None, uses the last measurement.
            
        Returns
        -------
        dict
            Analysis results including key parameters
        """
        if results is None:
            if self._last_results is None:
                raise ValueError("No measurement results available for analysis")
            results = self._last_results
        
        # Extract data
        frequencies = results['frequencies']
        magnitude = results['magnitude']
        phase = results['phase']
        real = results['real']
        imaginary = results['imaginary']
        
        # Find minimum and maximum impedance magnitudes
        min_idx = np.argmin(magnitude)
        max_idx = np.argmax(magnitude)
        
        min_impedance = {
            'magnitude': magnitude[min_idx],
            'phase': phase[min_idx],
            'frequency': frequencies[min_idx]
        }
        
        max_impedance = {
            'magnitude': magnitude[max_idx],
            'phase': phase[max_idx],
            'frequency': frequencies[max_idx]
        }
        
        # Calculate impedance at specific frequencies (if measured)
        impedance_at_specific = {}
        target_frequencies = [10, 100, 1000, 10000, 100000]  # Hz
        
        for target in target_frequencies:
            # Find closest measured frequency
            idx = np.argmin(np.abs(frequencies - target))
            if abs(frequencies[idx] - target) / target < 0.1:  # Within 10%
                impedance_at_specific[f'{target}Hz'] = {
                    'magnitude': magnitude[idx],
                    'phase': phase[idx],
                    'frequency': frequencies[idx]
                }
        
        # Find crossover frequency (phase = 0)
        crossover_idx = None
        for i in range(len(phase) - 1):
            if phase[i] * phase[i+1] <= 0:  # Sign change
                crossover_idx = i if abs(phase[i]) < abs(phase[i+1]) else i+1
                break
                
        crossover_frequency = None
        if crossover_idx is not None:
            crossover_frequency = {
                'frequency': frequencies[crossover_idx],
                'magnitude': magnitude[crossover_idx],
                'phase': phase[crossover_idx]
            }
        
        # Analyze frequency dependency
        # Simple power law fit: |Z| ~ f^alpha
        try:
            mask = frequencies > 0  # Avoid log(0)
            log_freq = np.log10(frequencies[mask])
            log_mag = np.log10(magnitude[mask])
            
            # Linear fit in log-log space
            poly = np.polyfit(log_freq, log_mag, 1)
            alpha = poly[0]  # Slope gives the power law exponent
            
        except:
            alpha = None
        
        # Return analysis results
        analysis = {
            'min_impedance': min_impedance,
            'max_impedance': max_impedance,
            'impedance_at_specific': impedance_at_specific,
            'crossover_frequency': crossover_frequency,
            'frequency_dependency': {
                'alpha': alpha,  # Power law exponent
            }
        }
        
        return analysis

    def get_calibration_status(self):
        """
        Get information about the current calibration status
        
        Returns
        -------
        dict
            Calibration information
        """
        if self._calibration_data is None:
            return {
                'calibrated': False,
                'message': 'System not calibrated'
            }
        
        # Calculate time since calibration
        hours_since_cal = (time.time() - self._calibration_data['timestamp']) / 3600
        
        status = {
            'calibrated': True,
            'timestamp': self._calibration_data['timestamp'],
            'hours_since_calibration': hours_since_cal,
            'temperature_at_calibration': self._calibration_data['temperature']
        }
        
        # Add calibration validity warning if too old
        if hours_since_cal > 24:
            status['message'] = 'Calibration is more than 24 hours old'
        else:
            status['message'] = 'Calibration valid'
            
        return status


class AD5940ElectricalImpedance:
    """
    Implementation of electrical impedance measurement using AD5940
    
    This class provides a hardware abstraction layer for the 
    Analog Devices AD5940 impedance measurement chip.
    """
    
    def __init__(self, spi_device='/dev/spidev0.0', reset_pin=17):
        """
        Initialize the AD5940 hardware interface
        
        Parameters
        ----------
        spi_device : str
            SPI device path
        reset_pin : int
            GPIO pin number connected to AD5940 reset
        """
        self.spi_device = spi_device
        self.reset_pin = reset_pin
        self.initialized = False
        self.current_config = {}
        
        try:
            # This would normally initialize the hardware
            # For this implementation, we'll just log the attempt
            logger.info(f"Initializing AD5940 on {spi_device} with reset pin {reset_pin}")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize AD5940: {str(e)}")
            raise
    
    def get_info(self):
        """Get hardware information"""
        return {
            'name': 'AD5940 Impedance Analyzer',
            'spi_device': self.spi_device,
            'reset_pin': self.reset_pin,
            'initialized': self.initialized
        }
    
    def configure(self, config):
        """Configure the AD5940 hardware"""
        logger.info(f"Configuring AD5940 with parameters: {config}")
        self.current_config = config
        
        # In a real implementation, this would set up the AD5940 registers
        # For now, we'll just log the configuration
        
        return True
    
    def measure_impedance(self, frequency, amplitude, integration_time, averaging):
        """
        Measure impedance at a specific frequency
        
        Parameters
        ----------
        frequency : float
            Measurement frequency in Hz
        amplitude : float
            Excitation signal amplitude in Volts
        integration_time : float
            Measurement integration time in seconds
        averaging : int
            Number of measurements to average
            
        Returns
        -------
        dict
            Measurement results with magnitude and phase
        """
        logger.info(f"Measuring impedance at {frequency} Hz")
        
        # In a real implementation, this would communicate with the AD5940
        # For now, we'll return simulated values
        
        # Simulate an RC circuit with R=1kΩ, C=1µF
        R = 1000  # Ohms
        C = 1e-6  # Farads
        omega = 2 * np.pi * frequency
        Z_real = R / (1 + (omega * R * C)**2)
        Z_imag = -omega * R**2 * C / (1 + (omega * R * C)**2)
        
        # Add noise that decreases with longer integration time and more averaging
        noise_factor = 0.01 * np.sqrt(0.1 / integration_time / averaging)
        Z_real *= (1 + noise_factor * (2 * np.random.random() - 1))
        Z_imag *= (1 + noise_factor * (2 * np.random.random() - 1))
        
        # Calculate magnitude and phase
        magnitude = np.sqrt(Z_real**2 + Z_imag**2)
        phase = np.degrees(np.arctan2(Z_imag, Z_real))
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'real': Z_real,
            'imaginary': Z_imag,
            'frequency': frequency
        }
    
    def measure_calibration(self, standard_type):
        """
        Measure calibration standard
        
        Parameters
        ----------
        standard_type : str
            Type of calibration standard ('open', 'short', '1k', etc)
            
        Returns
        -------
        dict
            Calibration measurement results
        """
        logger.info(f"Measuring calibration standard: {standard_type}")
        
        # Simulate calibration measurements
        if standard_type == 'open':
            magnitude = 1e9  # Very high impedance
            phase = -90      # Capacitive
        elif standard_type == 'short':
            magnitude = 0.1  # Very low impedance
            phase = 0        # Resistive
        elif standard_type == '1k':
            magnitude = 1000 # 1kΩ reference
            phase = 0        # Purely resistive
        else:
            magnitude = 1000
            phase = 0
        
        # Add some noise
        magnitude *= (1 + 0.001 * (2 * np.random.random() - 1))
        phase += 0.1 * (2 * np.random.random() - 1)
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'standard_type': standard_type
        }
    
    def get_temperature(self):
        """Get the current temperature of the device"""
        # Simulate temperature measurement
        return 25.0 + 0.1 * (2 * np.random.random() - 1)  # 25°C ± 0.1°C


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a measurement instance in simulation mode
    eis = ElectricalImpedanceMeasurement()
    
    # Configure
    eis.configure(
        min_frequency=1,
        max_frequency=10000,
        num_points=20,
        voltage_amplitude=50e-3,
        averaging=3
    )
    
    # Calibrate
    eis.calibrate()
    
    # Measure
    results = eis.measure()
    
    # Analyze
    analysis = eis.analyze_impedance(results)
    
    # Print some results
    print("\nMeasurement Results:")
    print(f"Frequency range: {min(results['frequencies']):.1f} Hz to {max(results['frequencies']):.1f} Hz")
    print(f"Impedance range: {min(results['magnitude']):.1f} Ω to {max(results['magnitude']):.1f} Ω")
    
    print("\nAnalysis Results:")
    print(f"Minimum impedance: {analysis['min_impedance']['magnitude']:.1f} Ω at {analysis['min_impedance']['frequency']:.1f} Hz")
    print(f"Maximum impedance: {analysis['max_impedance']['magnitude']:.1f} Ω at {analysis['max_impedance']['frequency']:.1f} Hz")
    
    if analysis['crossover_frequency'] is not None:
        print(f"Crossover frequency: {analysis['crossover_frequency']['frequency']:.1f} Hz")
    
    if analysis['frequency_dependency']['alpha'] is not None:
        print(f"Frequency dependence: |Z| ~ f^{analysis['frequency_dependency']['alpha']:.3f}")
