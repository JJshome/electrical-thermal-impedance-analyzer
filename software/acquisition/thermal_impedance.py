"""
Thermal Impedance Measurement Module

This module provides classes and functions for measuring thermal impedance
across a frequency range using various heat stimulation and temperature
measurement techniques.

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
import threading
from scipy import signal

# Set up logging
logger = logging.getLogger(__name__)


class HeatSource(Enum):
    """Enumeration for different heat sources"""
    PELTIER = 1
    RESISTIVE_HEATER = 2
    LASER = 3
    INFRARED_LED = 4


class TemperatureSensor(Enum):
    """Enumeration for temperature sensor types"""
    THERMISTOR = 1
    THERMOCOUPLE = 2
    RTD = 3
    INFRARED = 4
    THERMAL_CAMERA = 5


class MeasurementMode(Enum):
    """Enumeration for different thermal impedance measurement modes"""
    SINGLE_FREQUENCY = 1
    FREQUENCY_SWEEP = 2
    STEP_RESPONSE = 3
    ADAPTIVE_SWEEP = 4


class ThermalImpedanceMeasurement:
    """
    Class for thermal impedance measurement control and data acquisition
    
    This class handles the configuration and execution of thermal impedance
    measurements across various frequencies, using different heating and
    temperature sensing techniques.
    """
    
    def __init__(self, hardware_interface=None):
        """
        Initialize the thermal impedance measurement system
        
        Parameters
        ----------
        hardware_interface : object, optional
            Interface to the thermal measurement hardware. If None, a simulation
            mode will be used.
        """
        self.hardware = hardware_interface
        self.is_simulation = hardware_interface is None
        
        # Default configuration
        self.config = {
            'min_frequency': 0.01,      # Hz
            'max_frequency': 1.0,       # Hz
            'num_points': 10,
            'heat_amplitude': 100e-3,   # W
            'heat_source': HeatSource.PELTIER,
            'temperature_sensor': TemperatureSensor.RTD,
            'measurement_mode': MeasurementMode.FREQUENCY_SWEEP,
            'integration_cycles': 3,     # Number of cycles to measure per frequency
            'sampling_rate': 10,         # Hz
            'pcm_enabled': True,         # Whether to use Phase Change Material thermal management
            'pcm_temperature': 35.0,     # °C - target temperature for PCM
        }
        
        self._calibration_data = None
        self._last_results = None
        self._temperature_data = []
        self._stop_measurement = False
        
        if self.is_simulation:
            logger.info("Running in simulation mode - no hardware connected")
            # Set up simulation parameters
            self._sim_params = {
                'thermal_resistance': 10.0,    # K/W
                'thermal_capacitance': 5.0,    # J/K
                'ambient_temperature': 25.0,   # °C
                'noise_level': 0.02,          # °C
                'thermal_time_constant': 2.0,  # seconds
            }
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
            
        if self.config['heat_amplitude'] <= 0:
            raise ValueError("Heat stimulus amplitude must be positive")
            
        if self.config['sampling_rate'] <= 2 * self.config['max_frequency']:
            logger.warning(f"Sampling rate ({self.config['sampling_rate']} Hz) may be too low "
                         f"for maximum frequency ({self.config['max_frequency']} Hz)")
    
    def calibrate(self):
        """
        Perform thermal calibration
        
        Returns
        -------
        bool
            True if calibration succeeded
        """
        if self.is_simulation:
            logger.info("Simulation mode: Using ideal thermal calibration values")
            self._calibration_data = {
                'thermal_resistance_reference': 10.0,  # K/W
                'heat_source_efficiency': 0.95,
                'timestamp': time.time(),
                'ambient_temperature': 25.0  # °C
            }
            return True
            
        try:
            logger.info("Starting thermal calibration procedure")
            
            # In real hardware, this would:
            # 1. Measure ambient temperature
            # 2. Apply known power to the heat source
            # 3. Measure steady-state temperature rise
            # 4. Calculate thermal resistance reference
            # 5. Determine heat source efficiency
            
            if not self.is_simulation and self.hardware is not None:
                calibration_results = self.hardware.perform_thermal_calibration()
            else:
                # Simulate calibration results
                calibration_results = {
                    'thermal_resistance_reference': 10.0,  # K/W
                    'heat_source_efficiency': 0.95,
                    'ambient_temperature': 25.0  # °C
                }
            
            # Add metadata
            calibration_results['timestamp'] = time.time()
                
            self._calibration_data = calibration_results
            logger.info("Thermal calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Thermal calibration failed: {str(e)}")
            return False
    
    def _temperature_sampling_thread(self, duration, sampling_rate, start_time, results_list):
        """
        Thread function for temperature sampling
        
        Parameters
        ----------
        duration : float
            Duration of sampling in seconds
        sampling_rate : float
            Sampling rate in Hz
        start_time : float
            Reference start time
        results_list : list
            List to store temperature samples
        """
        sampling_interval = 1.0 / sampling_rate
        num_samples = int(duration * sampling_rate)
        
        self._stop_measurement = False
        
        for i in range(num_samples):
            if self._stop_measurement:
                logger.info("Temperature sampling stopped early")
                break
                
            # Calculate exact time for this sample
            target_time = start_time + i * sampling_interval
            current_time = time.time()
            
            # If we're ahead of schedule, wait until the right time
            if current_time < target_time:
                time.sleep(target_time - current_time)
            
            # Get the current temperature
            if self.is_simulation:
                temperature = self._simulate_temperature(time.time() - start_time)
            else:
                temperature = self.hardware.read_temperature()
                
            # Store the sample with timestamp
            results_list.append({
                'timestamp': time.time() - start_time,
                'temperature': temperature
            })
            
            # Progress update for long measurements
            if i % (10 * sampling_rate) == 0 and i > 0:  # Update every 10 seconds
                logger.info(f"Temperature sampling progress: {i}/{num_samples} samples")
    
    def _simulate_temperature(self, elapsed_time):
        """
        Simulate temperature response for a given elapsed time
        
        Parameters
        ----------
        elapsed_time : float
            Time since start of simulation in seconds
            
        Returns
        -------
        float
            Simulated temperature in °C
        """
        # Get simulation parameters
        R_th = self._sim_params['thermal_resistance']  # K/W
        C_th = self._sim_params['thermal_capacitance']  # J/K
        T_ambient = self._sim_params['ambient_temperature']  # °C
        noise_level = self._sim_params['noise_level']  # °C
        
        # Get heat input settings from configuration
        heat_amplitude = self.config['heat_amplitude']  # W
        
        # If sweeping frequency, calculate current frequency based on time
        if self.config['measurement_mode'] == MeasurementMode.FREQUENCY_SWEEP:
            # Use the middle frequency for simulation simplicity
            freq = np.sqrt(self.config['min_frequency'] * self.config['max_frequency'])
        else:
            freq = self.config['min_frequency']
        
        # Calculate temperature response based on RC thermal model
        # For sinusoidal input:
        #   T(t) = T_ambient + heat_amplitude * R_th * sin(2*pi*freq*t - phi) / sqrt(1 + (2*pi*freq*R_th*C_th)^2)
        #   where phi = arctan(2*pi*freq*R_th*C_th)
        
        omega = 2 * np.pi * freq
        tau = R_th * C_th  # thermal time constant
        
        # Phase lag due to thermal capacitance
        phi = np.arctan(omega * tau)
        
        # Amplitude attenuation due to thermal capacitance
        attenuation = 1.0 / np.sqrt(1 + (omega * tau)**2)
        
        # Calculate temperature
        delta_T = heat_amplitude * R_th * attenuation * np.sin(omega * elapsed_time - phi)
        
        # Add noise
        noise = noise_level * (2 * np.random.random() - 1)
        
        temperature = T_ambient + delta_T + noise
        
        return temperature
    
    def apply_heat_stimulus(self, frequency, amplitude, duration):
        """
        Apply sinusoidal heat stimulus
        
        Parameters
        ----------
        frequency : float
            Frequency of the heat stimulus in Hz
        amplitude : float
            Amplitude of the heat stimulus in Watts
        duration : float
            Duration of the stimulus in seconds
        
        Returns
        -------
        bool
            True if stimulus was successfully applied
        """
        if self.is_simulation:
            logger.info(f"Simulation: Applying heat stimulus at {frequency} Hz, {amplitude} W for {duration} s")
            return True
            
        try:
            if self.hardware is not None:
                return self.hardware.apply_heat_stimulus(frequency, amplitude, duration)
            return False
        except Exception as e:
            logger.error(f"Failed to apply heat stimulus: {str(e)}")
            return False
    
    def measure_single_frequency(self, frequency):
        """
        Measure thermal impedance at a single frequency
        
        Parameters
        ----------
        frequency : float
            Frequency in Hz
        
        Returns
        -------
        dict
            Measurement results for this frequency
        """
        logger.info(f"Measuring thermal impedance at {frequency} Hz")
        
        # Calculate required measurement duration based on frequency
        # We want to measure at least n complete cycles
        cycles = self.config['integration_cycles']
        duration = cycles / frequency
        
        # Set up temperature sampling
        sampling_rate = self.config['sampling_rate']
        self._temperature_data = []
        
        # Start temperature sampling in a separate thread
        start_time = time.time()
        sampling_thread = threading.Thread(
            target=self._temperature_sampling_thread,
            args=(duration, sampling_rate, start_time, self._temperature_data)
        )
        sampling_thread.start()
        
        # Apply heat stimulus
        heat_amplitude = self.config['heat_amplitude']
        self.apply_heat_stimulus(frequency, heat_amplitude, duration)
        
        # Wait for sampling to complete
        sampling_thread.join()
        
        # Process the temperature data
        if len(self._temperature_data) == 0:
            logger.error("No temperature data collected")
            return None
            
        # Extract time and temperature data
        times = np.array([sample['timestamp'] for sample in self._temperature_data])
        temperatures = np.array([sample['temperature'] for sample in self._temperature_data])
        
        # Calculate reference temperature (average of first few samples before heating effect)
        num_reference_samples = min(10, int(len(temperatures) * 0.1))
        if num_reference_samples > 0:
            reference_temperature = np.mean(temperatures[:num_reference_samples])
        else:
            reference_temperature = temperatures[0] if len(temperatures) > 0 else 25.0
        
        # Compute amplitude and phase using FFT or curve fitting
        try:
            # Detrend the temperature data to remove any linear drift
            temperatures_detrended = signal.detrend(temperatures)
            
            # Use FFT to find amplitude and phase at the stimulus frequency
            N = len(temperatures_detrended)
            T = times[-1] / (N - 1)  # Average sampling period
            
            # FFT
            yf = np.fft.rfft(temperatures_detrended)
            xf = np.fft.rfftfreq(N, T)
            
            # Find the bin closest to our stimulus frequency
            idx = np.argmin(np.abs(xf - frequency))
            
            # Get magnitude and phase
            magnitude = np.abs(yf[idx]) * 2 / N  # Scale factor for amplitude
            phase_rad = np.angle(yf[idx])
            
            # Convert phase to degrees and adjust as needed
            phase_deg = np.degrees(phase_rad)
            
            # Calculate thermal impedance
            # Thermal impedance magnitude = temperature amplitude / power amplitude
            thermal_impedance_magnitude = magnitude / heat_amplitude  # °C/W
            
            # Create result dictionary
            result = {
                'frequency': frequency,
                'magnitude': thermal_impedance_magnitude,
                'phase': phase_deg,
                'reference_temperature': reference_temperature,
                'temperature_amplitude': magnitude,
                'measurement_duration': duration,
                'num_samples': len(temperatures),
                'sampling_rate': sampling_rate,
                'timestamp': time.time()
            }
            
            # For debugging or advanced analysis, add raw time series data
            result['raw_data'] = {
                'times': times.tolist(),
                'temperatures': temperatures.tolist()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing thermal impedance data at {frequency} Hz: {str(e)}")
            return None

    def measure(self, frequencies=None):
        """
        Perform thermal impedance measurements
        
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
            logger.warning("No thermal calibration data available, results may be inaccurate")
            
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
        temperature_amplitudes = np.zeros_like(frequencies, dtype=np.float64)
        reference_temperatures = np.zeros_like(frequencies, dtype=np.float64)
        
        # Adjust PCM temperature if enabled
        if self.config['pcm_enabled'] and not self.is_simulation and self.hardware is not None:
            logger.info(f"Setting PCM temperature to {self.config['pcm_temperature']} °C")
            self.hardware.set_pcm_temperature(self.config['pcm_temperature'])
        
        # Perform measurements
        logger.info(f"Starting thermal impedance measurement with {len(frequencies)} frequency points")
        start_time = time.time()
        
        all_results = []  # Store detailed results for each frequency
        
        for i, freq in enumerate(frequencies):
            # Sort frequencies from highest to lowest for more efficient measurement
            # This minimizes the waiting time between measurements
            # (thermal systems have longer time constants at lower frequencies)
            
            # Measure at this frequency
            result = self.measure_single_frequency(freq)
            
            if result is not None:
                impedance_mag[i] = result['magnitude']
                impedance_phase[i] = result['phase']
                temperature_amplitudes[i] = result['temperature_amplitude']
                reference_temperatures[i] = result['reference_temperature']
                all_results.append(result)
            else:
                logger.warning(f"Failed to measure thermal impedance at {freq} Hz")
                impedance_mag[i] = np.nan
                impedance_phase[i] = np.nan
                temperature_amplitudes[i] = np.nan
                reference_temperatures[i] = np.nan
            
            # Progress update for long measurements
            elapsed = time.time() - start_time
            estimated_total = elapsed * len(frequencies) / (i + 1)
            remaining = estimated_total - elapsed
            logger.info(f"Progress: {i+1}/{len(frequencies)} frequencies measured. "
                      f"Est. {remaining:.1f}s remaining")
        
        elapsed = time.time() - start_time
        logger.info(f"Thermal measurement completed in {elapsed:.1f} seconds")
        
        # Store and return results
        results = {
            'frequencies': frequencies,
            'magnitude': impedance_mag,
            'phase': impedance_phase,
            'temperature_amplitudes': temperature_amplitudes,
            'reference_temperatures': reference_temperatures,
            'timestamp': time.time(),
            'config': self.config.copy(),
            'detailed_results': all_results
        }
        
        # Calculate complex impedance
        results['real'] = impedance_mag * np.cos(np.radians(impedance_phase))
        results['imaginary'] = impedance_mag * np.sin(np.radians(impedance_phase))
        
        self._last_results = results
        return results
    
    def analyze_thermal_impedance(self, results=None):
        """
        Basic analysis of thermal impedance measurement results
        
        Parameters
        ----------
        results : dict, optional
            Measurement results to analyze. If None, uses the last measurement.
            
        Returns
        -------
        dict
            Analysis results including key thermal parameters
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
        
        # Calculate thermal resistance (low-frequency asymptote)
        try:
            # Use the lowest measured frequencies (first 3 points or 10%, whichever is larger)
            n_low = max(3, int(len(frequencies) * 0.1))
            low_freq_idx = np.argsort(frequencies)[:n_low]
            thermal_resistance = np.mean(magnitude[low_freq_idx])  # K/W or °C/W
        except:
            thermal_resistance = np.nan
        
        # Find thermal time constant
        # The thermal time constant corresponds to the frequency where 
        # the phase is approximately -45 degrees
        try:
            # Find the index where phase is closest to -45 degrees
            idx = np.argmin(np.abs(phase - (-45)))
            thermal_time_constant = 1.0 / (2 * np.pi * frequencies[idx])  # seconds
        except:
            thermal_time_constant = np.nan
        
        # Estimate thermal capacitance
        try:
            thermal_capacitance = thermal_time_constant / thermal_resistance  # J/K
        except:
            thermal_capacitance = np.nan
        
        # Find cut-off frequency (3dB point)
        try:
            # 3dB drop is at ~0.707 of the DC value
            dc_value = thermal_resistance
            threshold = dc_value * 0.707
            
            # Find first frequency where magnitude drops below threshold
            for i in range(len(frequencies)):
                if magnitude[i] < threshold:
                    cutoff_frequency = frequencies[i]
                    break
            else:
                # If not found, use the highest frequency
                cutoff_frequency = frequencies[-1]
        except:
            cutoff_frequency = np.nan
        
        # Estimate number of significant thermal time constants (layers)
        try:
            # A simple approach is to look at the slope of the phase vs. frequency
            # Multiple time constants create a more gradual phase shift
            phase_data = phase[~np.isnan(phase)]
            if len(phase_data) > 3:
                phase_range = np.max(phase_data) - np.min(phase_data)
                num_time_constants = round(phase_range / 90)
                num_time_constants = max(1, num_time_constants)  # At least 1
            else:
                num_time_constants = 1
        except:
            num_time_constants = 1
            
        # Return analysis results
        analysis = {
            'thermal_resistance': thermal_resistance,  # K/W
            'thermal_time_constant': thermal_time_constant,  # seconds
            'thermal_capacitance': thermal_capacitance,  # J/K
            'cutoff_frequency': cutoff_frequency,  # Hz
            'estimated_thermal_layers': num_time_constants,
        }
        
        return analysis

    def get_calibration_status(self):
        """
        Get information about the current thermal calibration status
        
        Returns
        -------
        dict
            Calibration information
        """
        if self._calibration_data is None:
            return {
                'calibrated': False,
                'message': 'System not thermally calibrated'
            }
        
        # Calculate time since calibration
        hours_since_cal = (time.time() - self._calibration_data['timestamp']) / 3600
        
        status = {
            'calibrated': True,
            'timestamp': self._calibration_data['timestamp'],
            'hours_since_calibration': hours_since_cal,
            'ambient_temperature_at_calibration': self._calibration_data['ambient_temperature']
        }
        
        # Add calibration validity warning if too old
        if hours_since_cal > 24:
            status['message'] = 'Thermal calibration is more than 24 hours old'
        else:
            status['message'] = 'Thermal calibration valid'
            
        return status
    
    def stop_measurement(self):
        """Stop any ongoing measurements"""
        self._stop_measurement = True
        logger.info("Thermal impedance measurement stop requested")
        
        if not self.is_simulation and self.hardware is not None:
            self.hardware.stop_heat_stimulus()


class PeltierThermalController:
    """
    Implementation of thermal control using Peltier (TEC) elements
    
    This class provides a hardware abstraction layer for the 
    thermal control system based on Peltier elements with PCM
    (Phase Change Material) thermal management.
    """
    
    def __init__(self, i2c_device='/dev/i2c-1'):
        """
        Initialize the Peltier thermal controller
        
        Parameters
        ----------
        i2c_device : str
            I2C device path for temperature sensors and controller
        """
        self.i2c_device = i2c_device
        self.initialized = False
        self.current_config = {}
        self.current_temperature = 25.0
        self.pcm_temperature = 35.0
        self.heat_stimulus_active = False
        self.temperature_sensors = []
        
        try:
            # This would normally initialize the hardware
            # For this implementation, we'll just log the attempt
            logger.info(f"Initializing Peltier thermal controller on {i2c_device}")
            
            # Initialize temperature sensors
            self._init_temperature_sensors()
            
            # Initialize Peltier controller
            self._init_peltier_controller()
            
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Peltier thermal controller: {str(e)}")
            raise
    
    def _init_temperature_sensors(self):
        """Initialize temperature sensors"""
        # In a real implementation, this would initialize the temperature sensors
        # For now, we'll just simulate it
        self.temperature_sensors = [
            {'type': 'RTD', 'name': 'Surface', 'address': 0x48},
            {'type': 'RTD', 'name': 'Ambient', 'address': 0x49},
            {'type': 'Thermistor', 'name': 'PCM', 'address': 0x4A}
        ]
        logger.info(f"Initialized {len(self.temperature_sensors)} temperature sensors")
    
    def _init_peltier_controller(self):
        """Initialize Peltier controller"""
        # In a real implementation, this would initialize the Peltier controller
        # For now, we'll just simulate it
        logger.info("Initialized Peltier controller")
    
    def get_info(self):
        """Get hardware information"""
        return {
            'name': 'Peltier Thermal Controller',
            'i2c_device': self.i2c_device,
            'initialized': self.initialized,
            'sensors': self.temperature_sensors,
            'pcm_temperature': self.pcm_temperature
        }
    
    def configure(self, config):
        """Configure the thermal controller"""
        logger.info(f"Configuring thermal controller with parameters: {config}")
        self.current_config = config
        
        # Apply PCM temperature if specified
        if 'pcm_temperature' in config:
            self.set_pcm_temperature(config['pcm_temperature'])
        
        # In a real implementation, this would set up the hardware
        # For now, we'll just log the configuration
        
        return True
    
    def read_temperature(self):
        """
        Read the current temperature
        
        Returns
        -------
        float
            Temperature in °C
        """
        # In a real implementation, this would read from the actual sensors
        # For now, return the simulated temperature with small random fluctuations
        self.current_temperature += 0.01 * (2 * np.random.random() - 1)
        
        if self.heat_stimulus_active:
            # Simulate heating effect
            elapsed = time.time() - self.heat_stimulus_start_time
            # Simple first-order response
            freq = self.heat_stimulus_frequency
            amp = self.heat_stimulus_amplitude * 0.01  # Scale factor to get reasonable temperature changes
            temp_change = amp * np.sin(2 * np.pi * freq * elapsed)
            self.current_temperature += temp_change
        
        return self.current_temperature
    
    def set_pcm_temperature(self, temperature):
        """
        Set the PCM (Phase Change Material) temperature
        
        Parameters
        ----------
        temperature : float
            Target PCM temperature in °C
            
        Returns
        -------
        bool
            True if temperature was set successfully
        """
        logger.info(f"Setting PCM temperature to {temperature} °C")
        self.pcm_temperature = temperature
        
        # In a real implementation, this would control the PCM temperature
        # by adjusting the Peltier controller
        
        # Simulate gradual approach to target temperature
        if abs(self.current_temperature - temperature) > 0.5:
            # Move current temperature 10% closer to target
            self.current_temperature += 0.1 * (temperature - self.current_temperature)
        
        return True
    
    def apply_heat_stimulus(self, frequency, amplitude, duration):
        """
        Apply sinusoidal heat stimulus
        
        Parameters
        ----------
        frequency : float
            Frequency of the heat stimulus in Hz
        amplitude : float
            Amplitude of the heat stimulus in Watts
        duration : float
            Duration of the stimulus in seconds
            
        Returns
        -------
        bool
            True if stimulus was successfully applied
        """
        logger.info(f"Applying heat stimulus at {frequency} Hz, {amplitude} W for {duration} s")
        
        # Store stimulus parameters for temperature simulation
        self.heat_stimulus_active = True
        self.heat_stimulus_frequency = frequency
        self.heat_stimulus_amplitude = amplitude
        self.heat_stimulus_start_time = time.time()
        
        # In a real implementation, this would program the Peltier controller
        # to generate the sinusoidal heat pattern
        
        # Schedule stopping the stimulus after duration
        def stop_after_duration():
            time.sleep(duration)
            self.stop_heat_stimulus()
        
        threading.Thread(target=stop_after_duration, daemon=True).start()
        
        return True
    
    def stop_heat_stimulus(self):
        """Stop any active heat stimulus"""
        if self.heat_stimulus_active:
            logger.info("Stopping heat stimulus")
            self.heat_stimulus_active = False
        
        # In a real implementation, this would signal the Peltier controller
        # to stop the stimulus and return to PCM temperature maintenance
    
    def perform_thermal_calibration(self):
        """
        Perform thermal calibration routine
        
        Returns
        -------
        dict
            Calibration results
        """
        logger.info("Performing thermal calibration")
        
        # In a real implementation, this would:
        # 1. Measure ambient temperature
        # 2. Apply a known power to the Peltier
        # 3. Measure steady-state temperature rise
        # 4. Calculate thermal resistance
        # 5. Perform step response to characterize thermal time constant
        
        # Simulate calibration results
        return {
            'thermal_resistance_reference': 10.0 + 0.5 * (2 * np.random.random() - 1),  # K/W
            'heat_source_efficiency': 0.95 - 0.05 * np.random.random(),  # 0.90-0.95
            'ambient_temperature': self.read_temperature()
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a measurement instance in simulation mode
    tis = ThermalImpedanceMeasurement()
    
    # Configure
    tis.configure(
        min_frequency=0.01,
        max_frequency=0.5,
        num_points=5,
        heat_amplitude=0.2,  # W
        sampling_rate=20,    # Hz
        pcm_enabled=True,
        pcm_temperature=33.0  # °C
    )
    
    # Calibrate
    tis.calibrate()
    
    # Measure
    results = tis.measure()
    
    # Analyze
    analysis = tis.analyze_thermal_impedance(results)
    
    # Print some results
    print("\nThermal Measurement Results:")
    print(f"Frequency range: {min(results['frequencies']):.3f} Hz to {max(results['frequencies']):.3f} Hz")
    print(f"Thermal impedance range: {min(results['magnitude']):.2f} K/W to {max(results['magnitude']):.2f} K/W")
    
    print("\nThermal Analysis Results:")
    print(f"Thermal resistance: {analysis['thermal_resistance']:.2f} K/W")
    print(f"Thermal time constant: {analysis['thermal_time_constant']:.2f} seconds")
    print(f"Thermal capacitance: {analysis['thermal_capacitance']:.2f} J/K")
    print(f"Cutoff frequency: {analysis['cutoff_frequency']:.3f} Hz")
    print(f"Estimated thermal layers: {analysis['estimated_thermal_layers']}")
