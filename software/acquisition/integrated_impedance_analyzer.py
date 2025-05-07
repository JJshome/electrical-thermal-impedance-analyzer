"""
Integrated Electrical-Thermal Impedance Analyzer

This module provides the core implementation of the integrated electrical-thermal
impedance analysis system based on patented technology by Ucaretron Inc.

References:
    Barsoukov et al., "Thermal impedance spectroscopy for Li-ion batteries 
    using heat-pulse response analysis", Journal of Power Sources, 2002
    
    Patent: 열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
    (Integrated Electrical-Thermal Impedance Analysis System and Method)
    Inventor: Jihwan Jang (장지환)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegratedImpedanceAnalyzer")

class IntegratedImpedanceAnalyzer:
    """
    Integrated Electrical-Thermal Impedance Analyzer
    
    This class implements simultaneous acquisition and analysis of both 
    electrical and thermal impedance data.
    
    Key features:
    - Wide frequency range for electrical impedance (0.1Hz to 500kHz)
    - Thermal impedance measurement with heat pulse response (0.01Hz to 1Hz)
    - PCM-based thermal management for precise temperature control
    - Multi-frequency simultaneous measurement
    - Real-time processing and AI-based analysis
    """
    
    def __init__(self, use_advanced_models: bool = True, thermal_control: bool = True):
        """
        Initialize the Integrated Impedance Analyzer
        
        Parameters:
        -----------
        use_advanced_models : bool
            Whether to use advanced AI models for data analysis
        thermal_control : bool
            Whether to enable active thermal control using PCM
        """
        logger.info("Initializing Integrated Electrical-Thermal Impedance Analyzer")
        
        # Default configuration
        self.electrical_freq_range = (0.1, 100000)  # Hz
        self.thermal_freq_range = (0.01, 1)  # Hz
        self.voltage_amplitude = 10e-3  # V
        self.thermal_pulse_power = 100e-3  # W
        self.sampling_rate = 1e6  # Hz
        self.thermal_sampling_rate = 100  # Hz
        
        # System components
        self.use_advanced_models = use_advanced_models
        self.thermal_control = thermal_control
        self._initialize_hardware()
        
        # Storage for measurement results and calibration data
        self.calibration_data = None
        self.last_measurement = None
        self.model_parameters = None
        
        logger.info("Initialization complete")

    def _initialize_hardware(self):
        """
        Initialize hardware components
        
        In a real implementation, this would configure the hardware interfaces
        including ADCs, DACs, signal generators, temperature controllers, etc.
        """
        logger.info("Initializing hardware components")
        
        # Mock implementation for demonstration
        self.hardware_ready = True
        self.eis_module_status = "ready"
        self.tis_module_status = "ready"
        self.thermal_control_status = "ready" if self.thermal_control else "disabled"
        
        # In real implementation, initialize:
        # - ADC for electrical impedance
        # - DAC for signal generation
        # - Thermal pulse generator
        # - Temperature sensors
        # - PCM thermal management system
        # - FPGA for signal processing
        
        logger.info("Hardware initialization complete")
    
    def configure(self, 
                  electrical_freq_range: Optional[Tuple[float, float]] = None, 
                  thermal_freq_range: Optional[Tuple[float, float]] = None, 
                  voltage_amplitude: Optional[float] = None, 
                  thermal_pulse_power: Optional[float] = None):
        """
        Configure measurement parameters
        
        Parameters:
        -----------
        electrical_freq_range : tuple
            Frequency range for electrical impedance (min, max) in Hz
        thermal_freq_range : tuple
            Frequency range for thermal impedance (min, max) in Hz
        voltage_amplitude : float
            Amplitude of voltage stimulus for electrical impedance in V
        thermal_pulse_power : float
            Power of thermal pulse for thermal impedance in W
        """
        if electrical_freq_range:
            self.electrical_freq_range = electrical_freq_range
        if thermal_freq_range:
            self.thermal_freq_range = thermal_freq_range
        if voltage_amplitude:
            self.voltage_amplitude = voltage_amplitude
        if thermal_pulse_power:
            self.thermal_pulse_power = thermal_pulse_power
        
        logger.info("Measurement parameters configured:")
        logger.info(f"  Electrical frequency range: {self.electrical_freq_range} Hz")
        logger.info(f"  Thermal frequency range: {self.thermal_freq_range} Hz")
        logger.info(f"  Voltage amplitude: {self.voltage_amplitude:.2e} V")
        logger.info(f"  Thermal pulse power: {self.thermal_pulse_power:.2e} W")
    
    def calibrate(self):
        """
        Perform system calibration using reference standards
        
        This method calibrates both the electrical and thermal impedance 
        measurement systems using known reference impedances.
        
        Returns:
        --------
        bool
            True if calibration was successful, False otherwise
        """
        logger.info("Starting system calibration")
        
        try:
            # Calibrate electrical impedance
            logger.info("Calibrating electrical impedance measurement")
            e_cal_success = self._calibrate_electrical_impedance()
            
            # Calibrate thermal impedance
            logger.info("Calibrating thermal impedance measurement")
            t_cal_success = self._calibrate_thermal_impedance()
             
            # Store calibration data
            self.calibration_data = {
                'timestamp': datetime.now(),
                'electrical': {
                    'reference_values': [0.01, 0.1, 1.0, 10.0],  # Ohms
                    'measured_values': [0.0102, 0.0998, 1.003, 9.997],  # Ohms
                    'correction_factors': [0.9804, 1.0020, 0.9970, 1.0003]
                },
                'thermal': {
                    'reference_values': [1.0, 5.0, 10.0],  # K/W
                    'measured_values': [1.02, 5.03, 9.98],  # K/W
                    'correction_factors': [0.9804, 0.9940, 1.0020]
                }
            }
            
            calibration_success = e_cal_success and t_cal_success
            logger.info(f"Calibration {'successful' if calibration_success else 'failed'}")
            return calibration_success
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False
    
    def _calibrate_electrical_impedance(self):
        """
        Calibrate the electrical impedance measurement system
        
        In a real implementation, this would use reference resistors and capacitors
        to calibrate the measurement system at various frequencies.
        
        Returns:
        --------
        bool
            True if calibration was successful, False otherwise
        """
        # Mock implementation - in real system would use reference standards
        time.sleep(1)  # Simulate calibration time
        return True
    
    def _calibrate_thermal_impedance(self):
        """
        Calibrate the thermal impedance measurement system
        
        In a real implementation, this would use reference materials with known
        thermal properties to calibrate the measurement system.
        
        Returns:
        --------
        bool
            True if calibration was successful, False otherwise
        """
        # Mock implementation - in real system would use reference standards
        time.sleep(1.5)  # Simulate calibration time
        return True
        
    def measure(self, 
                target_system: Optional[Any] = None, 
                measurement_time: float = 60.0,
                apply_calibration: bool = True) -> Dict[str, Any]:
        """
        Perform integrated impedance measurements
        
        Parameters:
        -----------
        target_system : object
            The target system to measure (battery, tissue, etc.)
        measurement_time : float
            Maximum measurement time in seconds
        apply_calibration : bool
            Whether to apply calibration corrections to the measurements
            
        Returns:
        --------
        results : dict
            Dictionary containing measurement results
        """
        logger.info("Starting integrated impedance measurement")
        
        # Check if hardware is ready
        if not self.hardware_ready:
            logger.error("Hardware not ready, aborting measurement")
            return {'error': 'Hardware not ready'}
        
        # Configure PCM thermal management to maintain constant temperature
        if self.thermal_control:
            logger.info("Activating PCM thermal management")
            self._activate_thermal_control()
        
        try:
            # Generate measurement frequencies
            e_freqs = self._generate_frequency_array(
                self.electrical_freq_range[0],
                self.electrical_freq_range[1],
                50  # Number of points
            )
            
            t_freqs = self._generate_frequency_array(
                self.thermal_freq_range[0],
                self.thermal_freq_range[1],
                10  # Number of points
            )
            
            # In a real implementation, these steps would interface with hardware
            # Here we simulate measurement with synthetic data generation
            
            # Measure electrical impedance
            logger.info("Measuring electrical impedance")
            e_impedance = self._measure_electrical_impedance(e_freqs, target_system)
            
            # Measure thermal impedance
            logger.info("Measuring thermal impedance")
            t_impedance = self._measure_thermal_impedance(t_freqs, target_system)
            
            # Apply calibration corrections if available and requested
            if self.calibration_data is not None and apply_calibration:
                logger.info("Applying calibration corrections")
                e_impedance = self._apply_electrical_calibration(e_freqs, e_impedance)
                t_impedance = self._apply_thermal_calibration(t_freqs, t_impedance)
            
            # Compile results
            results = {
                'e_frequencies': e_freqs,
                't_frequencies': t_freqs,
                'e_impedance': e_impedance,
                't_impedance': t_impedance,
                'timestamp': datetime.now(),
                'configuration': {
                    'e_freq_range': self.electrical_freq_range,
                    't_freq_range': self.thermal_freq_range,
                    'voltage_amplitude': self.voltage_amplitude,
                    'thermal_pulse_power': self.thermal_pulse_power
                },
                'measurement_duration': measurement_time,
                'target_system_info': str(target_system) if target_system else "None"
            }
            
            # Store the results
            self.last_measurement = results
            
            logger.info("Measurement completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Measurement failed: {str(e)}")
            return {'error': str(e)}
        finally:
            # Deactivate thermal control
            if self.thermal_control:
                logger.info("Deactivating thermal control")
                self._deactivate_thermal_control()
    
    def _generate_frequency_array(self, f_min: float, f_max: float, num_points: int) -> np.ndarray:
        """
        Generate logarithmically spaced frequency array
        
        Parameters:
        -----------
        f_min : float
            Minimum frequency
        f_max : float
            Maximum frequency
        num_points : int
            Number of frequency points
            
        Returns:
        --------
        frequencies : ndarray
            Array of frequencies
        """
        return np.logspace(np.log10(f_min), np.log10(f_max), num_points)
    
    def _measure_electrical_impedance(self, 
                                     frequencies: np.ndarray, 
                                     target_system: Optional[Any] = None) -> np.ndarray:
        """
        Measure electrical impedance at specified frequencies
        
        In a real implementation, this would interface with hardware to apply
        voltage signals and measure current responses at each frequency.
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequencies to measure
        target_system : object
            The target system to measure
            
        Returns:
        --------
        impedance : ndarray
            Complex impedance array
        """
        # Simulate a typical impedance response (simplified Randles circuit)
        # Z = R_s + R_ct/(1 + jωC_dl) + Z_W
        
        # Mock parameters - these would be actual measured values in a real system
        R_s = 0.05  # Series resistance (Ohm)
        R_ct = 0.1  # Charge transfer resistance (Ohm)
        C_dl = 1.0  # Double layer capacitance (F)
        Z_W_mag = 0.05  # Warburg impedance magnitude (Ohm·s^(-1/2))
        Z_W_phase = np.pi/4  # Warburg impedance phase (radians)
        
        # If a target system is provided, adjust parameters based on its properties
        if target_system is not None:
            # Example of how parameters might be adjusted based on the target system
            if hasattr(target_system, 'internal_resistance'):
                R_s = target_system.internal_resistance
            
            if hasattr(target_system, 'capacitance'):
                C_dl = target_system.capacitance
        
        # Calculate impedance
        omega = 2 * np.pi * frequencies
        Z_dl = R_ct / (1 + 1j * omega * C_dl)
        Z_W = Z_W_mag * np.exp(1j * Z_W_phase) / np.sqrt(omega)
        Z = R_s + Z_dl + Z_W
        
        # Add noise to simulate real-world measurements
        noise_mag = 0.01 * np.abs(Z) * np.random.randn(len(frequencies))
        noise_phase = 0.02 * np.random.randn(len(frequencies))
        Z_noisy = Z * (1 + noise_mag) * np.exp(1j * noise_phase)
        
        return Z_noisy
    
    def _measure_thermal_impedance(self, 
                                  frequencies: np.ndarray, 
                                  target_system: Optional[Any] = None) -> np.ndarray:
        """
        Measure thermal impedance at specified frequencies
        
        In a real implementation, this would interface with hardware to apply
        heat pulses and measure temperature responses at each frequency.
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequencies to measure
        target_system : object
            The target system to measure
            
        Returns:
        --------
        impedance : ndarray
            Complex thermal impedance array
        """
        # Simulate thermal impedance response based on Barsoukov et al. 2002
        # Z_th(s) = (1/Cth) / (s + 1/(Rth*Cth))
        
        # Mock parameters - these would be actual measured values in a real system
        R_th = 2.0  # Thermal resistance (K/W)
        C_th = 1000.0  # Thermal capacitance (J/K)
        
        # If a target system is provided, adjust parameters based on its properties
        if target_system is not None:
            # Example of how parameters might be adjusted based on the target system
            if hasattr(target_system, 'thermal_resistance'):
                R_th = target_system.thermal_resistance
            
            if hasattr(target_system, 'thermal_capacitance'):
                C_th = target_system.thermal_capacitance
        
        # Calculate thermal impedance
        omega = 2 * np.pi * frequencies
        s = 1j * omega
        Z = (1/C_th) / (s + 1/(R_th*C_th))
        
        # Add noise to simulate real-world measurements
        noise_mag = 0.02 * np.abs(Z) * np.random.randn(len(frequencies))
        noise_phase = 0.03 * np.random.randn(len(frequencies))
        Z_noisy = Z * (1 + noise_mag) * np.exp(1j * noise_phase)
        
        return Z_noisy
    
    def _apply_electrical_calibration(self, 
                                     frequencies: np.ndarray, 
                                     impedance: np.ndarray) -> np.ndarray:
        """
        Apply calibration corrections to electrical impedance measurements
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequencies
        impedance : ndarray
            Measured impedance array
            
        Returns:
        --------
        corrected_impedance : ndarray
            Calibration-corrected impedance array
        """
        # In a real implementation, this would apply frequency-dependent
        # calibration factors to the measured impedance
        
        # Mock implementation using linear interpolation of correction factors
        correction_factors = self.calibration_data['electrical']['correction_factors']
        reference_values = self.calibration_data['electrical']['reference_values']
        
        # Simple linear correction - in a real system this would be more sophisticated
        # and frequency-dependent
        avg_correction = np.mean(correction_factors)
        corrected_impedance = impedance * avg_correction
        
        return corrected_impedance
    
    def _apply_thermal_calibration(self, 
                                  frequencies: np.ndarray, 
                                  impedance: np.ndarray) -> np.ndarray:
        """
        Apply calibration corrections to thermal impedance measurements
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequencies
        impedance : ndarray
            Measured impedance array
            
        Returns:
        --------
        corrected_impedance : ndarray
            Calibration-corrected impedance array
        """
        # In a real implementation, this would apply frequency-dependent
        # calibration factors to the measured impedance
        
        # Mock implementation using average correction factor
        correction_factors = self.calibration_data['thermal']['correction_factors']
        
        # Simple correction - in a real system this would be more sophisticated
        avg_correction = np.mean(correction_factors)
        corrected_impedance = impedance * avg_correction
        
        return corrected_impedance
    
    def _activate_thermal_control(self):
        """
        Activate PCM-based thermal management system
        
        This method controls the PCM system to maintain constant temperature
        during measurements.
        """
        # In a real implementation, this would interface with the PCM
        # thermal management hardware
        
        # Mock implementation
        logger.info("PCM thermal control activated")
        
    def _deactivate_thermal_control(self):
        """
        Deactivate PCM-based thermal management system
        """
        # In a real implementation, this would interface with the PCM
        # thermal management hardware
        
        # Mock implementation
        logger.info("PCM thermal control deactivated")
        
    def analyze(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze impedance data using integrated approach
        
        Parameters:
        -----------
        results : dict
            Dictionary containing measurement results. If None, the last 
            measurement results will be used.
            
        Returns:
        --------
        characteristics : dict
            Dictionary containing extracted system characteristics
        """
        logger.info("Analyzing integrated impedance data")
        
        # Use the provided results or the last measurement
        if results is None:
            if self.last_measurement is None:
                logger.error("No measurement data available for analysis")
                return {'error': 'No measurement data available'}
            results = self.last_measurement
        
        try:
            # Extract impedance data
            e_freqs = results['e_frequencies']
            t_freqs = results['t_frequencies']
            e_impedance = results['e_impedance']
            t_impedance = results['t_impedance']
            
            # Convert complex impedance to magnitude and phase
            e_magnitude = np.abs(e_impedance)
            e_phase = np.angle(e_impedance, deg=True)
            t_magnitude = np.abs(t_impedance)
            t_phase = np.angle(t_impedance, deg=True)
            
            # Separate real and imaginary parts
            e_real = np.real(e_impedance)
            e_imag = np.imag(e_impedance)
            t_real = np.real(t_impedance)
            t_imag = np.imag(t_impedance)
            
            # Fit electrical impedance model (simplified Randles model)
            e_params = self._fit_electrical_model(e_freqs, e_real, e_imag)
            
            # Fit thermal impedance model
            t_params = self._fit_thermal_model(t_freqs, t_real, t_imag)
            
            # Integrated analysis
            integrated_params = self._perform_integrated_analysis(e_params, t_params)
            
            # Create results dictionary
            characteristics = {
                'electrical_parameters': e_params,
                'thermal_parameters': t_params,
                'integrated_parameters': integrated_params
            }
            
            # Store model parameters
            self.model_parameters = characteristics
            
            logger.info("Analysis completed successfully")
            return characteristics
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _fit_electrical_model(self, 
                             frequencies: np.ndarray, 
                             real_part: np.ndarray, 
                             imag_part: np.ndarray) -> Dict[str, float]:
        """
        Fit electrical impedance data to equivalent circuit model
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequencies
        real_part : ndarray
            Real part of impedance
        imag_part : ndarray
            Imaginary part of impedance
            
        Returns:
        --------
        params : dict
            Dictionary of fitted model parameters
        """
        try:
            # Define electrical model function for curve fitting (real part only)
            def electrical_model_func(freq, R_s, R_ct, C_dl, Z_W_mag):
                omega = 2 * np.pi * freq
                Z_dl_real = R_ct / (1 + (omega * C_dl)**2)
                Z_W_real = Z_W_mag / np.sqrt(omega)
                Z_real = R_s + Z_dl_real + Z_W_real
                return Z_real
            
            # Initial parameter guesses
            p0 = [0.05, 0.1, 1.0, 0.05]
            
            # Fit real part of electrical impedance
            popt, _ = curve_fit(
                electrical_model_func,
                frequencies,
                real_part,
                p0=p0,
                bounds=([0, 0, 0, 0], [1.0, 1.0, 100.0, 1.0])
            )
            
            # Extract fitted parameters
            R_s, R_ct, C_dl, Z_W_mag = popt
            
            # Calculate total internal resistance
            R_total = R_s + R_ct
            
            # Calculate characteristic time constant
            tau = R_ct * C_dl
            
            return {
                'R_s': R_s,
                'R_ct': R_ct,
                'C_dl': C_dl,
                'Z_W_mag': Z_W_mag,
                'R_total': R_total,
                'tau': tau
            }
            
        except RuntimeError as e:
            logger.warning(f"Electrical model fitting failed: {str(e)}")
            # Return default values if fitting fails
            return {
                'R_s': 0.05,
                'R_ct': 0.1,
                'C_dl': 1.0,
                'Z_W_mag': 0.05,
                'R_total': 0.15,
                'tau': 0.1
            }
    
    def _fit_thermal_model(self, 
                          frequencies: np.ndarray, 
                          real_part: np.ndarray, 
                          imag_part: np.ndarray) -> Dict[str, float]:
        """
        Fit thermal impedance data to thermal model
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequencies
        real_part : ndarray
            Real part of impedance
        imag_part : ndarray
            Imaginary part of impedance
            
        Returns:
        --------
        params : dict
            Dictionary of fitted model parameters
        """
        try:
            # Define thermal model function for curve fitting (real part only)
            def thermal_model_func(freq, R_th, C_th):
                omega = 2 * np.pi * freq
                denominator = 1 + (omega * R_th * C_th)**2
                Z_real = R_th / denominator
                return Z_real
            
            # Initial parameter guesses
            p0 = [2.0, 1000.0]
            
            # Fit real part of thermal impedance
            popt, _ = curve_fit(
                thermal_model_func,
                frequencies,
                real_part,
                p0=p0,
                bounds=([0, 0], [100.0, 100000.0])
            )
            
            # Extract fitted parameters
            R_th, C_th = popt
            
            # Calculate thermal time constant
            thermal_time_constant = R_th * C_th
            
            # Calculate thermal diffusivity
            # This is a simplified approximation - in a real system,
            # thermal diffusivity would be calculated based on the
            # physical dimensions of the system
            thermal_diffusivity = 1.0 / (R_th * C_th)  # m²/s
            
            return {
                'R_th': R_th,
                'C_th': C_th,
                'thermal_time_constant': thermal_time_constant,
                'thermal_diffusivity': thermal_diffusivity
            }
            
        except RuntimeError as e:
            logger.warning(f"Thermal model fitting failed: {str(e)}")
            # Return default values if fitting fails
            return {
                'R_th': 2.0,
                'C_th': 1000.0,
                'thermal_time_constant': 2000.0,
                'thermal_diffusivity': 0.0005
            }
    
    def _perform_integrated_analysis(self, 
                                    e_params: Dict[str, float], 
                                    t_params: Dict[str, float]) -> Dict[str, float]:
        """
        Perform integrated analysis of electrical and thermal parameters
        
        This method combines the electrical and thermal parameters to extract
        higher-level system characteristics.
        
        Parameters:
        -----------
        e_params : dict
            Electrical model parameters
        t_params : dict
            Thermal model parameters
            
        Returns:
        --------
        integrated_params : dict
            Integrated system characteristics
        """
        # Extract parameters
        R_total = e_params['R_total']
        R_th = t_params['R_th']
        C_th = t_params['C_th']
        
        # Estimated power dissipation during operation
        estimated_power = R_total * (self.voltage_amplitude**2)
        
        # Temperature rise estimation (P * R_th)
        estimated_temp_rise = estimated_power * R_th
        
        # State of health estimation (simplified model)
        # In a real implementation, this would use a more sophisticated model
        # trained on reference data
        # Here we just use a simple heuristic based on R_total and R_th
        norm_R_total = R_total / 0.15  # Normalize by typical value
        norm_R_th = R_th / 2.0  # Normalize by typical value
        state_of_health = 100.0 * np.exp(-0.5 * (norm_R_total - 1.0)**2) * np.exp(-0.5 * (norm_R_th - 1.0)**2)
        
        # Thermal conductivity estimation (simplified)
        # In a real implementation, this would be based on the geometry
        # of the system and more sophisticated models
        thermal_conductivity = 1.0 / (R_th * 0.01)  # W/(m·K), assuming 1cm thickness
        
        # Heat capacity (directly from thermal model)
        heat_capacity = C_th  # J/K
        
        return {
            'state_of_health': state_of_health,
            'thermal_conductivity': thermal_conductivity,
            'heat_capacity': heat_capacity,
            'internal_resistance': R_total,
            'thermal_time_constant': t_params['thermal_time_constant'],
            'estimated_power_dissipation': estimated_power,
            'estimated_temperature_rise': estimated_temp_rise
        }
        
    def plot_impedance_spectra(self, results: Optional[Dict[str, Any]] = None) -> None:
        """
        Plot impedance spectra
        
        Parameters:
        -----------
        results : dict
            Dictionary containing measurement results. If None, the last 
            measurement results will be used.
        """
        # Use the provided results or the last measurement
        if results is None:
            if self.last_measurement is None:
                logger.error("No measurement data available for plotting")
                return
            results = self.last_measurement
            
        e_freqs = results['e_frequencies']
        t_freqs = results['t_frequencies']
        e_impedance = results['e_impedance']
        t_impedance = results['t_impedance']
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Electrical impedance - Bode plot
        axs[0, 0].loglog(e_freqs, np.abs(e_impedance), 'b-', linewidth=2)
        axs[0, 0].set_xlabel('Frequency (Hz)')
        axs[0, 0].set_ylabel('|Z| (Ω)')
        axs[0, 0].set_title('Electrical Impedance Magnitude')
        axs[0, 0].grid(True, which="both", ls="--")
        
        axs[0, 1].semilogx(e_freqs, np.angle(e_impedance, deg=True), 'b-', linewidth=2)
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Phase (degrees)')
        axs[0, 1].set_title('Electrical Impedance Phase')
        axs[0, 1].grid(True, which="both", ls="--")
        
        # Thermal impedance - Bode plot
        axs[1, 0].loglog(t_freqs, np.abs(t_impedance), 'r-', linewidth=2)
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('|Z| (K/W)')
        axs[1, 0].set_title('Thermal Impedance Magnitude')
        axs[1, 0].grid(True, which="both", ls="--")
        
        axs[1, 1].semilogx(t_freqs, np.angle(t_impedance, deg=True), 'r-', linewidth=2)
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Phase (degrees)')
        axs[1, 1].set_title('Thermal Impedance Phase')
        axs[1, 1].grid(True, which="both", ls="--")
        
        plt.tight_layout()
        plt.show()
        
        # Nyquist plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Electrical impedance - Nyquist plot
        axs[0].plot(np.real(e_impedance), -np.imag(e_impedance), 'bo-', linewidth=2)
        axs[0].set_xlabel('Re(Z) (Ω)')
        axs[0].set_ylabel('-Im(Z) (Ω)')
        axs[0].set_title('Electrical Impedance Nyquist Plot')
        axs[0].grid(True, ls="--")
        axs[0].axis('equal')  # Equal scaling for better visualization
        
        # Add frequency markers
        for i, f in enumerate(e_freqs):
            if i % 10 == 0:  # Mark every 10th frequency point
                axs[0].annotate(f'{f:.1f} Hz', 
                             (np.real(e_impedance)[i], -np.imag(e_impedance)[i]),
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center')
        
        # Thermal impedance - Nyquist plot
        axs[1].plot(np.real(t_impedance), -np.imag(t_impedance), 'ro-', linewidth=2)
        axs[1].set_xlabel('Re(Z) (K/W)')
        axs[1].set_ylabel('-Im(Z) (K/W)')
        axs[1].set_title('Thermal Impedance Nyquist Plot')
        axs[1].grid(True, ls="--")
        axs[1].axis('equal')  # Equal scaling for better visualization
        
        # Add frequency markers
        for i, f in enumerate(t_freqs):
            if i % 2 == 0:  # Mark every 2nd frequency point
                axs[1].annotate(f'{f:.3f} Hz', 
                             (np.real(t_impedance)[i], -np.imag(t_impedance)[i]),
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Integrated plot (Real part of both impedances, normalized)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Normalize impedances for comparison
        e_real_norm = np.real(e_impedance) / np.max(np.abs(np.real(e_impedance)))
        t_real_norm = np.real(t_impedance) / np.max(np.abs(np.real(t_impedance)))
        
        # Electrical impedance
        color = 'tab:blue'
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Normalized Electrical Re(Z)', color=color)
        ax1.semilogx(e_freqs, e_real_norm, color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Thermal impedance on secondary y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Normalized Thermal Re(Z)', color=color)
        ax2.semilogx(t_freqs, t_real_norm, color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legend and title
        plt.title('Integrated Electrical-Thermal Impedance Analysis')
        ax1.legend(['Electrical Impedance'], loc='upper left')
        ax2.legend(['Thermal Impedance'], loc='upper right')
        
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

        
# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure measurement parameters
    analyzer.configure(
        electrical_freq_range=(0.1, 100000),  # Hz
        thermal_freq_range=(0.01, 1),         # Hz
        voltage_amplitude=10e-3,              # V
        thermal_pulse_power=100e-3,           # W
    )
    
    # Calibrate the system
    analyzer.calibrate()
    
    # Perform measurements
    results = analyzer.measure()
    
    # Analyze the results
    characteristics = analyzer.analyze(results)
    
    # Print results
    print("\nExtracted Characteristics:")
    print("=========================")
    print(f"Electrical Parameters:")
    for key, value in characteristics['electrical_parameters'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nThermal Parameters:")
    for key, value in characteristics['thermal_parameters'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nIntegrated Parameters:")
    for key, value in characteristics['integrated_parameters'].items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize
    analyzer.plot_impedance_spectra(results)
