"""
Thermal impedance module for the Integrated Electrical-Thermal Impedance Analysis System.

This module implements thermal impedance spectroscopy (TIS) for measuring
thermal transport properties across various frequency ranges.
"""

import numpy as np
import logging
import time
from typing import Dict, Tuple, List, Optional, Union, Any
from enum import Enum, auto
import threading

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThermalStimulationMode(Enum):
    """Enumeration of supported thermal stimulation modes."""
    SINE_WAVE = auto()       # Sinusoidal thermal stimulation
    SQUARE_WAVE = auto()     # Square wave thermal stimulation
    PULSE = auto()           # Single pulse thermal stimulation
    MULTI_SINE = auto()      # Multi-frequency sinusoidal stimulation
    CHIRP = auto()           # Frequency sweep stimulation


class ThermalImpedanceAnalyzer:
    """
    Thermal Impedance Analyzer.
    
    This class provides methods for measuring thermal impedance spectroscopy (TIS)
    data across a frequency range (0.001Hz to 1Hz).
    """
    
    def __init__(self, 
                frequency_range: Tuple[float, float] = (0.001, 1.0),
                thermal_power: float = 100e-3,  # W
                stimulation_mode: ThermalStimulationMode = ThermalStimulationMode.SINE_WAVE,
                device_id: Optional[str] = None):
        """
        Initialize the thermal impedance analyzer.
        
        Parameters
        ----------
        frequency_range : Tuple[float, float], optional
            Range of frequencies for measurement (Hz), default is (0.001, 1.0).
        thermal_power : float, optional
            Amplitude of the thermal power stimulus (W), default is 100e-3.
        stimulation_mode : ThermalStimulationMode, optional
            Thermal stimulation mode, default is ThermalStimulationMode.SINE_WAVE.
        device_id : str, optional
            Device ID for hardware connection, default is None.
        """
        # Validate frequency range
        if frequency_range[0] <= 0 or frequency_range[1] <= 0:
            raise ValueError("Frequencies must be positive")
        if frequency_range[0] >= frequency_range[1]:
            raise ValueError("Invalid frequency range")
        
        self.frequency_range = frequency_range
        self.thermal_power = thermal_power
        self.stimulation_mode = stimulation_mode
        self.device_id = device_id
        
        # Hardware connection status
        self._is_connected = False
        self._is_measuring = False
        self._measurement_thread = None
        self._stop_measurement = False
        
        # Measurement configuration
        self.num_frequencies = 50  # Number of frequencies in sweep
        self.frequency_spacing = 'log'  # 'log' or 'linear'
        self.cycles_per_frequency = 5  # Number of cycles at each frequency
        self.settle_time = 3  # Settle time in seconds before measurement
        
        # Temperature sensor configuration
        self.num_temp_sensors = 4  # Number of temperature sensors
        self.temp_sensor_rate = 100  # Samples per second
        self.temp_sensor_resolution = 0.01  # °C
        
        # Thermal stimulus generator configuration
        self.max_thermal_power = 5.0  # W
        self.thermal_response_time = 0.1  # s
        self.power_resolution = 0.001  # W
        
        # Precalculated frequency points
        self._frequency_points = self._calculate_frequency_points()
        
        # Calibration data
        self.calibration_data = {
            'date': None,
            'reference_values': {},
            'correction_factors': {}
        }
        
        # Connect to hardware if device_id is provided
        if device_id:
            self.connect()
            
        logger.info(f"Initialized Thermal Impedance Analyzer with frequency range "
                   f"{self.frequency_range[0]:.5f}Hz - {self.frequency_range[1]:.3f}Hz, "
                   f"thermal power {self.thermal_power*1000:.1f}mW, "
                   f"mode {self.stimulation_mode.name}")
    
    def connect(self) -> bool:
        """
        Connect to the thermal analyzer hardware.
        
        Returns
        -------
        bool
            True if connection was successful, False otherwise.
        """
        if self._is_connected:
            logger.warning("Already connected to hardware")
            return True
        
        try:
            # Hardware connection code would go here
            logger.info(f"Connecting to thermal analyzer hardware with device ID: {self.device_id}")
            
            # In a real implementation, we would:
            # 1. Initialize communication interface (USB, SPI, I2C, etc.)
            # 2. Check device identification and firmware version
            # 3. Initialize temperature sensors
            # 4. Initialize thermal stimulus generator
            # 5. Load calibration data
            
            # Simulate hardware connection
            time.sleep(0.5)  # Simulate connection time
            self._is_connected = True
            
            # Perform initial hardware setup
            self._setup_hardware()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to hardware: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the thermal analyzer hardware.
        
        Returns
        -------
        bool
            True if disconnection was successful, False otherwise.
        """
        if not self._is_connected:
            logger.warning("Not connected to hardware")
            return True
        
        try:
            # Hardware disconnection code would go here
            logger.info("Disconnecting from thermal analyzer hardware")
            
            # Stop any ongoing measurements
            if self._is_measuring:
                self.stop_measurement()
            
            # In a real implementation, we would:
            # 1. Disable thermal stimulus generator
            # 2. Power down temperature sensors
            # 3. Close communication channels
            # 4. Release resources
            
            # Simulate hardware disconnection
            time.sleep(0.3)  # Simulate disconnection time
            self._is_connected = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from hardware: {e}")
            return False
    
    def calibrate(self, calibration_type: str = 'thermal_reference') -> bool:
        """
        Perform system calibration.
        
        Parameters
        ----------
        calibration_type : str, optional
            Type of calibration to perform, default is 'thermal_reference'.
            Options: 'thermal_reference', 'sensor_calibration', 'power_calibration'.
            
        Returns
        -------
        bool
            True if calibration was successful, False otherwise.
        """
        if not self._is_connected:
            logger.error("Cannot calibrate without hardware connection")
            return False
        
        if self._is_measuring:
            logger.error("Cannot calibrate while measurement is in progress")
            return False
        
        logger.info(f"Starting {calibration_type} calibration")
        
        try:
            # Real calibration would involve:
            # 1. Using reference samples with known thermal properties
            # 2. Measuring their thermal response
            # 3. Calculating correction factors
            # 4. Storing calibration data
            
            if calibration_type == 'thermal_reference':
                # Thermal reference material calibration
                logger.info("Measuring thermal reference material")
                # self._measure_thermal_reference()
                
            elif calibration_type == 'sensor_calibration':
                # Temperature sensor calibration
                logger.info("Calibrating temperature sensors")
                # self._calibrate_temperature_sensors()
                
            elif calibration_type == 'power_calibration':
                # Thermal stimulus power calibration
                logger.info("Calibrating thermal stimulus power")
                # self._calibrate_thermal_power()
                
            else:
                logger.error(f"Unknown calibration type: {calibration_type}")
                return False
            
            # Simulate successful calibration
            time.sleep(2.0)  # Simulate calibration time
            
            # Update calibration data
            self.calibration_data['date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def configure(self, 
                 frequency_range: Optional[Tuple[float, float]] = None,
                 thermal_power: Optional[float] = None,
                 stimulation_mode: Optional[ThermalStimulationMode] = None,
                 num_frequencies: Optional[int] = None,
                 frequency_spacing: Optional[str] = None,
                 cycles_per_frequency: Optional[int] = None,
                 settle_time: Optional[float] = None) -> None:
        """
        Configure measurement parameters.
        
        Parameters
        ----------
        frequency_range : Tuple[float, float], optional
            Range of frequencies for measurement (Hz), default is None (no change).
        thermal_power : float, optional
            Amplitude of the thermal power stimulus (W), default is None (no change).
        stimulation_mode : ThermalStimulationMode, optional
            Thermal stimulation mode, default is None (no change).
        num_frequencies : int, optional
            Number of frequencies in sweep, default is None (no change).
        frequency_spacing : str, optional
            Frequency spacing ('log' or 'linear'), default is None (no change).
        cycles_per_frequency : int, optional
            Number of cycles at each frequency, default is None (no change).
        settle_time : float, optional
            Settle time in seconds before measurement, default is None (no change).
        """
        # Update parameters if provided
        if frequency_range is not None:
            # Validate frequency range
            if frequency_range[0] <= 0 or frequency_range[1] <= 0:
                raise ValueError("Frequencies must be positive")
            if frequency_range[0] >= frequency_range[1]:
                raise ValueError("Invalid frequency range")
                
            self.frequency_range = frequency_range
            logger.info(f"Frequency range updated to {frequency_range[0]:.5f}Hz - {frequency_range[1]:.3f}Hz")
        
        if thermal_power is not None:
            # Validate thermal power
            if thermal_power <= 0:
                raise ValueError("Thermal power must be positive")
            if thermal_power > self.max_thermal_power:
                logger.warning(f"Requested thermal power {thermal_power}W exceeds maximum ({self.max_thermal_power}W), capping at maximum")
                thermal_power = self.max_thermal_power
                
            self.thermal_power = thermal_power
            logger.info(f"Thermal power updated to {thermal_power*1000:.1f}mW")
        
        if stimulation_mode is not None:
            self.stimulation_mode = stimulation_mode
            logger.info(f"Stimulation mode updated to {stimulation_mode.name}")
        
        if num_frequencies is not None:
            # Validate number of frequencies
            if num_frequencies < 2:
                raise ValueError("Number of frequencies must be at least 2")
                
            self.num_frequencies = num_frequencies
            logger.info(f"Number of frequencies updated to {num_frequencies}")
        
        if frequency_spacing is not None:
            # Validate frequency spacing
            if frequency_spacing not in ['log', 'linear']:
                raise ValueError("Frequency spacing must be 'log' or 'linear'")
                
            self.frequency_spacing = frequency_spacing
            logger.info(f"Frequency spacing updated to {frequency_spacing}")
        
        if cycles_per_frequency is not None:
            # Validate cycles per frequency
            if cycles_per_frequency < 1:
                raise ValueError("Cycles per frequency must be at least 1")
                
            self.cycles_per_frequency = cycles_per_frequency
            logger.info(f"Cycles per frequency updated to {cycles_per_frequency}")
        
        if settle_time is not None:
            # Validate settle time
            if settle_time < 0:
                raise ValueError("Settle time cannot be negative")
                
            self.settle_time = settle_time
            logger.info(f"Settle time updated to {settle_time} seconds")
        
        # Recalculate frequency points if relevant parameters changed
        if frequency_range is not None or num_frequencies is not None or frequency_spacing is not None:
            self._frequency_points = self._calculate_frequency_points()
        
        # Apply configuration to hardware if connected
        if self._is_connected:
            self._apply_configuration()
    
    def measure(self, 
               frequencies: Optional[List[float]] = None,
               blocking: bool = True) -> Dict[str, Any]:
        """
        Perform thermal impedance measurement.
        
        Parameters
        ----------
        frequencies : List[float], optional
            Specific frequencies to measure (Hz), default is None (use precalculated points).
        blocking : bool, optional
            Whether to block until measurement is complete, default is True.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing measurement results.
        """
        if self._is_measuring:
            logger.warning("Measurement already in progress")
            return {'error': 'Measurement already in progress'}
        
        try:
            self._is_measuring = True
            
            # Check hardware connection
            if not self._is_connected and self.device_id:
                logger.info("Not connected to hardware, attempting to connect")
                self.connect()
            
            # Use provided frequencies or precalculated points
            if frequencies is not None:
                # Validate frequencies
                for freq in frequencies:
                    if freq <= 0:
                        raise ValueError(f"Invalid frequency: {freq}Hz")
                        
                measurement_freqs = frequencies
                logger.info(f"Using {len(measurement_freqs)} user-specified frequencies")
            else:
                measurement_freqs = self._frequency_points
                logger.info(f"Using {len(measurement_freqs)} precalculated frequencies")
            
            # Start measurement
            logger.info(f"Starting thermal impedance measurement in {self.stimulation_mode.name} mode")
            
            if blocking:
                # Perform measurement in current thread
                results = self._perform_measurement(measurement_freqs)
                self._is_measuring = False
                return results
            else:
                # Start measurement in a separate thread
                self._stop_measurement = False
                self._measurement_thread = threading.Thread(
                    target=self._measurement_thread_function,
                    args=(measurement_freqs,)
                )
                self._measurement_thread.daemon = True
                self._measurement_thread.start()
                
                return {'status': 'started', 'mode': self.stimulation_mode.name}
            
        except Exception as e:
            logger.error(f"Error during measurement: {e}")
            self._is_measuring = False
            return {'error': str(e)}
    
    def stop_measurement(self) -> bool:
        """
        Stop an ongoing measurement.
        
        Returns
        -------
        bool
            True if stopping was successful, False otherwise.
        """
        if not self._is_measuring:
            logger.warning("No measurement in progress")
            return True
        
        try:
            # Set stop flag for measurement thread
            self._stop_measurement = True
            
            # Wait for measurement thread to complete
            if self._measurement_thread is not None and self._measurement_thread.is_alive():
                self._measurement_thread.join(timeout=5.0)
                
            # Force stop if thread is still running
            if self._measurement_thread is not None and self._measurement_thread.is_alive():
                logger.warning("Measurement thread did not stop gracefully, forcing stop")
                
            # Reset measurement state
            self._is_measuring = False
            self._measurement_thread = None
            
            logger.info("Measurement stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop measurement: {e}")
            return False
    
    def get_measurement_status(self) -> Dict[str, Any]:
        """
        Get the status of the current measurement.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing measurement status.
        """
        status = {
            'is_measuring': self._is_measuring,
            'mode': self.stimulation_mode.name if self._is_measuring else None,
            'progress': 0.0
        }
        
        # Add more status information if available
        if hasattr(self, '_measurement_progress'):
            status['progress'] = self._measurement_progress
            
        return status
    
    def simulate_thermal_impedance(self, frequency: float, model_type: str = 'one_time_constant') -> complex:
        """
        Simulate thermal impedance for testing purposes.
        
        Parameters
        ----------
        frequency : float
            Frequency to simulate (Hz).
        model_type : str, optional
            Type of thermal model to use, default is 'one_time_constant'.
            Options: 'one_time_constant', 'two_time_constant', 'semi_infinite', 'layered'.
            
        Returns
        -------
        complex
            Simulated complex thermal impedance.
        """
        omega = 2 * np.pi * frequency
        
        if model_type == 'one_time_constant':
            # Simple RC thermal model: R_th + 1/(j*ω*C_th)
            R_th = 5.0    # Thermal resistance (K/W)
            C_th = 0.5    # Thermal capacitance (J/K)
            
            # Calculate thermal impedance
            Z_th = R_th / (1 + 1j * omega * R_th * C_th)
            
        elif model_type == 'two_time_constant':
            # Two-stage RC thermal model: R_th1 || C_th1 + R_th2 || C_th2
            R_th1 = 2.0   # First thermal resistance (K/W)
            C_th1 = 0.5   # First thermal capacitance (J/K)
            R_th2 = 3.0   # Second thermal resistance (K/W)
            C_th2 = 1.0   # Second thermal capacitance (J/K)
            
            # Calculate impedance components
            Z_th1 = R_th1 / (1 + 1j * omega * R_th1 * C_th1)
            Z_th2 = R_th2 / (1 + 1j * omega * R_th2 * C_th2)
            
            # Series combination
            Z_th = Z_th1 + Z_th2
            
        elif model_type == 'semi_infinite':
            # Semi-infinite medium model (thermal diffusion)
            alpha = 1e-6  # Thermal diffusivity (m²/s)
            k = 0.5       # Thermal conductivity (W/(m·K))
            
            # Thermal impedance with diffusion term
            Z_th = 1 / (k * np.sqrt(1j * omega / alpha))
            
        elif model_type == 'layered':
            # Layered material model (simplified)
            R_bulk = 2.0    # Bulk thermal resistance (K/W)
            R_interface = 1.0  # Interface thermal resistance (K/W)
            C_bulk = 0.8    # Bulk thermal capacitance (J/K)
            C_interface = 0.2  # Interface thermal capacitance (J/K)
            
            # Calculate impedance components
            Z_bulk = R_bulk / (1 + 1j * omega * R_bulk * C_bulk)
            Z_interface = R_interface / (1 + 1j * omega * R_interface * C_interface)
            
            # Series combination
            Z_th = Z_bulk + Z_interface
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add some random noise
        noise_factor = 0.03  # 3% noise
        Z_th += Z_th * noise_factor * (np.random.randn() + 1j * np.random.randn())
        
        return Z_th
    
    def _setup_hardware(self) -> None:
        """Set up hardware subsystems."""
        if not self._is_connected:
            return
            
        logger.debug("Setting up thermal measurement hardware")
        
        # In a real implementation, we would:
        # 1. Initialize temperature sensors
        # 2. Configure thermal stimulus generator
        # 3. Set up data acquisition parameters
        # 4. Initialize digital filters
        # 5. Set up synchronization and triggering
    
    def _apply_configuration(self) -> None:
        """Apply configuration to hardware."""
        if not self._is_connected:
            return
            
        logger.debug("Applying configuration to thermal measurement hardware")
        
        # In a real implementation, we would:
        # 1. Update temperature sensor sampling rates
        # 2. Configure thermal stimulus parameters
        # 3. Set up frequency-specific settings
        # 4. Configure stimulation mode-specific parameters
    
    def _calculate_frequency_points(self) -> List[float]:
        """
        Calculate frequency points for measurement.
        
        Returns
        -------
        List[float]
            List of frequency points.
        """
        if self.frequency_spacing == 'log':
            # Logarithmic spacing
            min_log = np.log10(self.frequency_range[0])
            max_log = np.log10(self.frequency_range[1])
            
            # Use log spacing for decade distribution
            freq_points = np.logspace(min_log, max_log, self.num_frequencies)
            
        else:
            # Linear spacing
            freq_points = np.linspace(self.frequency_range[0], self.frequency_range[1], self.num_frequencies)
        
        return freq_points.tolist()
    
    def _perform_measurement(self, frequencies: List[float]) -> Dict[str, Any]:
        """
        Perform the actual thermal impedance measurement.
        
        Parameters
        ----------
        frequencies : List[float]
            List of frequencies to measure.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing measurement results.
        """
        # Initialize results
        results = {
            'frequency': [],
            'real': [],
            'imag': [],
            'magnitude': [],
            'phase': [],
            'timestamp': time.time(),
            'duration': 0.0,
            'thermal_power': self.thermal_power
        }
        
        start_time = time.time()
        self._measurement_progress = 0.0
        
        # Sort frequencies from lowest to highest for efficiency
        # (thermal measurements are more efficient when going from low to high frequency)
        sorted_freqs = sorted(frequencies)
        
        # Perform measurement based on mode
        if self.stimulation_mode == ThermalStimulationMode.SINE_WAVE:
            # Sinusoidal thermal stimulation
            for i, freq in enumerate(sorted_freqs):
                # Check if measurement should be stopped
                if self._stop_measurement:
                    logger.info("Measurement stopped by user")
                    break
                
                # Calculate measurement time for this frequency
                # For thermal measurements, we need multiple cycles to reach steady state
                # Lower frequencies require more time
                cycles_needed = max(self.cycles_per_frequency, int(10.0 / freq))  # At least 10 seconds or specified cycles
                measurement_time = cycles_needed / freq + self.settle_time
                
                logger.debug(f"Measuring at {freq:.6f}Hz (cycle time: {1/freq:.2f}s, total time: {measurement_time:.2f}s)")
                
                # Simulate measurement at this frequency
                if self._is_connected:
                    # In a real implementation, this would:
                    # 1. Generate sinusoidal thermal stimulus at the specified frequency
                    # 2. Wait for thermal steady state to be reached
                    # 3. Measure the temperature response
                    # 4. Calculate thermal impedance
                    
                    # Simulate measurement delay (proportional to measurement time)
                    # In reality, we would be acquiring data during this time
                    time.sleep(min(0.2, measurement_time / 50))  # Cap at 0.2s for simulation
                    
                    # For now, use simulated data
                    impedance = self.simulate_thermal_impedance(freq)
                else:
                    # Generate simulated data for software-only demonstration
                    impedance = self.simulate_thermal_impedance(freq)
                
                # Store results
                results['frequency'].append(freq)
                results['real'].append(impedance.real)
                results['imag'].append(impedance.imag)
                results['magnitude'].append(abs(impedance))
                results['phase'].append(np.angle(impedance, deg=True))
                
                # Update progress
                self._measurement_progress = (i + 1) / len(sorted_freqs)
                
                # Log progress periodically
                if (i + 1) % 5 == 0 or (i + 1) == len(sorted_freqs):
                    logger.debug(f"Thermal measurement progress: {self._measurement_progress:.1%}")
        
        elif self.stimulation_mode == ThermalStimulationMode.MULTI_SINE:
            # Multi-sine thermal stimulation
            
            # In a real implementation, this would:
            # 1. Generate a multi-sine thermal stimulus with carefully selected frequencies
            # 2. Apply the stimulus to the sample
            # 3. Measure the temperature response
            # 4. Perform Fourier analysis to extract the response at each frequency
            # 5. Calculate thermal impedance at each frequency
            
            # For now, simulate multi-sine measurement
            logger.info(f"Simulating multi-sine thermal measurement with {len(frequencies)} frequencies")
            
            # Multi-sine only works well with a limited number of frequencies
            # Group frequencies into batches if there are too many
            batch_size = 5  # Maximum frequencies per multi-sine batch
            
            # Process frequencies in batches
            for batch_start in range(0, len(sorted_freqs), batch_size):
                # Check if measurement should be stopped
                if self._stop_measurement:
                    logger.info("Measurement stopped by user")
                    break
                
                batch_end = min(batch_start + batch_size, len(sorted_freqs))
                batch_freqs = sorted_freqs[batch_start:batch_end]
                
                # Determine measurement time for this batch
                min_freq = min(batch_freqs)
                cycles_needed = max(self.cycles_per_frequency, int(10.0 / min_freq))
                measurement_time = cycles_needed / min_freq + self.settle_time
                
                logger.debug(f"Measuring batch of {len(batch_freqs)} frequencies, min freq: {min_freq:.6f}Hz")
                
                # Simulate measurement delay
                time.sleep(min(0.3, measurement_time / 30))
                
                # Generate simulated data for the batch
                for freq in batch_freqs:
                    impedance = self.simulate_thermal_impedance(freq)
                    
                    # Store results
                    results['frequency'].append(freq)
                    results['real'].append(impedance.real)
                    results['imag'].append(impedance.imag)
                    results['magnitude'].append(abs(impedance))
                    results['phase'].append(np.angle(impedance, deg=True))
                
                # Update progress
                self._measurement_progress = min(1.0, batch_end / len(sorted_freqs))
                logger.debug(f"Thermal measurement progress: {self._measurement_progress:.1%}")
        
        elif self.stimulation_mode == ThermalStimulationMode.CHIRP:
            # Chirp (frequency sweep) thermal stimulation
            
            # In a real implementation, this would:
            # 1. Generate a chirp signal that sweeps through the frequency range
            # 2. Apply the stimulus to the sample
            # 3. Measure the temperature response
            # 4. Perform time-frequency analysis to extract the frequency response
            # 5. Calculate thermal impedance at multiple frequencies
            
            # For now, simulate chirp measurement
            logger.info(f"Simulating chirp thermal measurement")
            
            # Calculate total measurement time
            min_freq = min(sorted_freqs)
            max_freq = max(sorted_freqs)
            # Time needed to complete one sweep
            sweep_time = 30.0  # Fixed time for chirp measurement
            
            # Simulate measurement delay
            time.sleep(min(0.5, sweep_time / 60))
            
            # Generate simulated data
            for freq in sorted_freqs:
                impedance = self.simulate_thermal_impedance(freq)
                
                # Store results
                results['frequency'].append(freq)
                results['real'].append(impedance.real)
                results['imag'].append(impedance.imag)
                results['magnitude'].append(abs(impedance))
                results['phase'].append(np.angle(impedance, deg=True))
            
            self._measurement_progress = 1.0
            
        elif self.stimulation_mode == ThermalStimulationMode.SQUARE_WAVE:
            # Square wave thermal stimulation
            
            # Similar to sine wave but using square wave stimulus
            for i, freq in enumerate(sorted_freqs):
                # Check if measurement should be stopped
                if self._stop_measurement:
                    logger.info("Measurement stopped by user")
                    break
                
                # Calculate measurement time
                cycles_needed = max(self.cycles_per_frequency, int(10.0 / freq))
                measurement_time = cycles_needed / freq + self.settle_time
                
                logger.debug(f"Measuring at {freq:.6f}Hz with square wave")
                
                # Simulate measurement delay
                time.sleep(min(0.2, measurement_time / 50))
                
                # Generate simulated data
                # For square wave, adjust the impedance calculation to account for harmonics
                impedance = self.simulate_thermal_impedance(freq, model_type='one_time_constant')
                
                # Apply square wave adjustment factor (simplified)
                # Square waves have additional harmonic content
                harmonic_factor = 1.0 - 0.1j  # Simplified adjustment
                impedance *= harmonic_factor
                
                # Store results
                results['frequency'].append(freq)
                results['real'].append(impedance.real)
                results['imag'].append(impedance.imag)
                results['magnitude'].append(abs(impedance))
                results['phase'].append(np.angle(impedance, deg=True))
                
                # Update progress
                self._measurement_progress = (i + 1) / len(sorted_freqs)
                
                if (i + 1) % 5 == 0 or (i + 1) == len(sorted_freqs):
                    logger.debug(f"Thermal measurement progress: {self._measurement_progress:.1%}")
        
        elif self.stimulation_mode == ThermalStimulationMode.PULSE:
            # Single pulse thermal stimulation
            
            # In a real implementation, this would:
            # 1. Apply a single thermal pulse to the sample
            # 2. Measure the temperature response over time
            # 3. Perform Fourier transform to obtain frequency domain response
            # 4. Calculate thermal impedance at multiple frequencies
            
            # For pulse measurements, we don't need to measure each frequency individually
            logger.info(f"Performing pulse thermal measurement")
            
            # Calculate measurement time
            # For pulse measurement, time is based on lowest frequency
            min_freq = min(sorted_freqs)
            measurement_time = 3.0 / min_freq  # Need at least 3x the period of lowest frequency
            
            # Simulate measurement delay
            time.sleep(min(0.5, measurement_time / 30))
            
            # Generate simulated data for all frequencies
            for freq in sorted_freqs:
                impedance = self.simulate_thermal_impedance(freq)
                
                # Store results
                results['frequency'].append(freq)
                results['real'].append(impedance.real)
                results['imag'].append(impedance.imag)
                results['magnitude'].append(abs(impedance))
                results['phase'].append(np.angle(impedance, deg=True))
            
            self._measurement_progress = 1.0
            
        # Calculate measurement duration
        results['duration'] = time.time() - start_time
        
        # Calculate temperature amplitude from thermal impedance and power
        results['temperature_amplitude'] = [mag * self.thermal_power for mag in results['magnitude']]
        
        logger.info(f"Thermal measurement completed in {results['duration']:.2f} seconds with {len(results['frequency'])} points")
        
        return results
    
    def _measurement_thread_function(self, frequencies: List[float]) -> None:
        """
        Thread function for non-blocking measurement.
        
        Parameters
        ----------
        frequencies : List[float]
            List of frequencies to measure.
        """
        try:
            # Perform measurement
            results = self._perform_measurement(frequencies)
            
            # Store results for later retrieval
            self._last_measurement_results = results
            
            # Set measurement complete
            self._is_measuring = False
            
            logger.info("Thermal measurement thread completed successfully")
            
        except Exception as e:
            logger.error(f"Error in thermal measurement thread: {e}")
            self._is_measuring = False
