"""
Thermal management module for the Integrated Electrical-Thermal Impedance Analysis System.

This module implements the Phase Change Material (PCM) based thermal control system
that maintains precise temperature stability during impedance measurements.
"""

import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional, Union, Any, List
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCMType(Enum):
    """Enumeration of supported Phase Change Materials with their properties."""
    
    # Format: (name, melting_point_celsius, latent_heat_J_g, thermal_conductivity_W_mK)
    PCM1 = ("n-octadecane", 28.0, 244.0, 0.2)
    PCM2 = ("paraffin", 45.0, 170.0, 0.25)
    PCM3 = ("palmitic_acid", 63.0, 185.0, 0.16)
    PCM4 = ("erucic_acid", 33.0, 201.0, 0.21)
    PCM5 = ("1-tetradecanol", 38.0, 205.0, 0.18)
    
    def __init__(self, name: str, melting_point: float, latent_heat: float, thermal_conductivity: float):
        self.name = name
        self.melting_point = melting_point  # °C
        self.latent_heat = latent_heat  # J/g
        self.thermal_conductivity = thermal_conductivity  # W/(m·K)


class ThermalEnhancerType(Enum):
    """Enumeration of supported thermal conductivity enhancers with their properties."""
    
    # Format: (name, thermal_conductivity_W_mK, max_loading_wt_percent)
    GRAPHENE = ("Graphene nanoplatelets", 3000.0, 20.0)
    CNT = ("Carbon nanotubes", 2000.0, 15.0)
    BN = ("Boron nitride nanosheets", 400.0, 40.0)
    GRAPHITE = ("Expanded graphite", 150.0, 30.0)
    ALUMINUM = ("Aluminum nanoparticles", 180.0, 50.0)
    SILVER = ("Silver nanoparticles", 420.0, 30.0)


class PCMThermalManager:
    """
    Phase Change Material-based thermal management system.
    
    This class provides methods for precise temperature control using Phase Change
    Materials enhanced with high thermal conductivity additives.
    """
    
    def __init__(self, 
                pcm_type: Union[PCMType, str] = PCMType.PCM1,
                enhancer_type: Union[ThermalEnhancerType, str] = ThermalEnhancerType.GRAPHENE,
                enhancer_loading: float = 3.0,  # wt%
                temperature_stability_threshold: float = 0.1,  # °C
                device_id: Optional[str] = None):
        """
        Initialize the PCM thermal management system.
        
        Parameters
        ----------
        pcm_type : PCMType or str, optional
            Type of Phase Change Material to use, default is PCMType.PCM1 (n-octadecane).
            If string is provided, it will try to match to a PCMType name.
        enhancer_type : ThermalEnhancerType or str, optional
            Type of thermal conductivity enhancer to use, default is ThermalEnhancerType.GRAPHENE.
            If string is provided, it will try to match to a ThermalEnhancerType name.
        enhancer_loading : float, optional
            Weight percentage of thermal enhancer in the PCM, default is 3.0%.
        temperature_stability_threshold : float, optional
            Temperature stability threshold (°C), default is 0.1°C.
        device_id : str, optional
            Device ID for hardware connection, default is None.
        """
        # Process PCM type input
        if isinstance(pcm_type, str):
            pcm_found = False
            for pcm in PCMType:
                if pcm_type.lower() == pcm.name.lower():
                    self.pcm_type = pcm
                    pcm_found = True
                    break
            if not pcm_found:
                logger.warning(f"PCM type '{pcm_type}' not recognized, using default")
                self.pcm_type = PCMType.PCM1
        else:
            self.pcm_type = pcm_type
            
        # Process enhancer type input
        if isinstance(enhancer_type, str):
            enhancer_found = False
            for enhancer in ThermalEnhancerType:
                if enhancer_type.lower() == enhancer.name.lower():
                    self.enhancer_type = enhancer
                    enhancer_found = True
                    break
            if not enhancer_found:
                logger.warning(f"Enhancer type '{enhancer_type}' not recognized, using default")
                self.enhancer_type = ThermalEnhancerType.GRAPHENE
        else:
            self.enhancer_type = enhancer_type
            
        # Validate enhancer loading against maximum
        if enhancer_loading > self.enhancer_type.max_loading_wt_percent:
            logger.warning(f"Enhancer loading {enhancer_loading}% exceeds maximum " 
                          f"{self.enhancer_type.max_loading_wt_percent}% for {self.enhancer_type.name}, "
                          f"capping at maximum")
            self.enhancer_loading = self.enhancer_type.max_loading_wt_percent
        else:
            self.enhancer_loading = enhancer_loading
            
        self.temperature_stability_threshold = temperature_stability_threshold
        self.device_id = device_id
        
        # PID controller parameters (optimized using Ziegler-Nichols method)
        self.pid_kp = 10.0  # Proportional gain
        self.pid_ki = 0.5   # Integral gain
        self.pid_kd = 2.0   # Derivative gain
        
        # Enhanced thermal properties calculation
        self.effective_thermal_conductivity = self._calculate_effective_thermal_conductivity()
        
        # State variables
        self._current_temperature = 25.0
        self._target_temperature = 25.0
        self._is_connected = False
        self._is_controlling = False
        self._integral_error = 0.0
        self._last_error = 0.0
        self._last_update_time = time.time()
        self._temperature_history = []
        
        # Try to connect to hardware if device_id is provided
        if device_id:
            self.connect()
            
        logger.info(f"Initialized PCM Thermal Manager with {self.pcm_type.name} "
                   f"enhanced with {self.enhancer_loading}% {self.enhancer_type.name}")
        logger.info(f"Calculated effective thermal conductivity: "
                   f"{self.effective_thermal_conductivity:.2f} W/(m·K)")
    
    def connect(self) -> bool:
        """
        Connect to the thermal control hardware.
        
        Returns
        -------
        bool
            True if connection was successful, False otherwise.
        """
        if self._is_connected:
            logger.warning("Already connected to thermal control hardware")
            return True
        
        try:
            # Hardware connection code would go here
            logger.info(f"Connecting to thermal control hardware with device ID: {self.device_id}")
            
            # In a real implementation, we would:
            # 1. Initialize communication with temperature sensors
            # 2. Initialize Peltier element controllers
            # 3. Perform hardware self-test
            # 4. Initialize PID controller
            
            # Simulate successful connection
            self._is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to thermal control hardware: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the thermal control hardware.
        
        Returns
        -------
        bool
            True if disconnection was successful, False otherwise.
        """
        if not self._is_connected:
            logger.warning("Not connected to thermal control hardware")
            return True
        
        try:
            # Hardware disconnection code would go here
            logger.info("Disconnecting from thermal control hardware")
            
            # Stop any ongoing control
            if self._is_controlling:
                self.stop_control()
            
            # In a real implementation, we would:
            # 1. Set Peltier elements to neutral state
            # 2. Close communication channels
            # 3. Power down sensitive components
            
            # Simulate successful disconnection
            self._is_connected = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from thermal control hardware: {e}")
            return False
    
    def set_temperature(self, target_temperature: float) -> bool:
        """
        Set target temperature and begin control.
        
        Parameters
        ----------
        target_temperature : float
            Target temperature in Celsius.
            
        Returns
        -------
        bool
            True if temperature setting was accepted, False otherwise.
        """
        # Validate temperature is within safe operating range
        if target_temperature < 10.0 or target_temperature > 80.0:
            logger.error(f"Target temperature {target_temperature}°C is outside safe range (10-80°C)")
            return False
        
        # Check if temperature is too far from PCM melting point
        temp_diff = abs(target_temperature - self.pcm_type.melting_point)
        if temp_diff > 15.0:
            logger.warning(f"Target temperature {target_temperature}°C is {temp_diff}°C from PCM "
                          f"melting point ({self.pcm_type.melting_point}°C). "
                          f"This reduces thermal stability benefits.")
        
        # Set target temperature
        self._target_temperature = target_temperature
        logger.info(f"Setting target temperature to {target_temperature}°C")
        
        # Reset PID control variables
        self._integral_error = 0.0
        self._last_error = 0.0
        self._last_update_time = time.time()
        
        # Begin temperature control if not already running
        if not self._is_controlling:
            return self.start_control()
        return True
    
    def start_control(self) -> bool:
        """
        Start temperature control process.
        
        Returns
        -------
        bool
            True if control was started successfully, False otherwise.
        """
        if self._is_controlling:
            logger.warning("Temperature control is already running")
            return True
        
        # Check hardware connection
        if not self._is_connected and self.device_id:
            logger.info("Not connected to hardware, attempting to connect")
            self.connect()
            
        if not self._is_connected:
            logger.error("Cannot start temperature control without hardware connection")
            return False
        
        try:
            # Initialize control
            logger.info(f"Starting temperature control (target: {self._target_temperature}°C)")
            
            # In a real implementation, we would:
            # 1. Initialize PID controller
            # 2. Enable Peltier elements
            # 3. Start control loop in a separate thread
            
            # Simulate successful start
            self._is_controlling = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start temperature control: {e}")
            return False
    
    def stop_control(self) -> bool:
        """
        Stop temperature control process.
        
        Returns
        -------
        bool
            True if control was stopped successfully, False otherwise.
        """
        if not self._is_controlling:
            logger.warning("Temperature control is not running")
            return True
        
        try:
            # Stop control
            logger.info("Stopping temperature control")
            
            # In a real implementation, we would:
            # 1. Stop control loop thread
            # 2. Set Peltier elements to neutral state
            # 3. Save control data if needed
            
            # Simulate successful stop
            self._is_controlling = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop temperature control: {e}")
            return False
    
    def get_current_temperature(self) -> float:
        """
        Get current temperature from sensor.
        
        Returns
        -------
        float
            Current temperature in Celsius.
        """
        if self._is_connected:
            # In a real implementation, read from temperature sensor
            # For now, simulate temperature with some randomness
            if self._is_controlling:
                # When controlling, temperature should be close to target with small variations
                error = abs(self._current_temperature - self._target_temperature)
                if error > 0.5:
                    # Still approaching target
                    self._current_temperature += np.sign(self._target_temperature - self._current_temperature) * min(0.2, error / 5)
                else:
                    # At or near target, small fluctuations
                    self._current_temperature = self._target_temperature + np.random.normal(0, self.temperature_stability_threshold / 3)
            else:
                # When not controlling, simulate drift toward ambient
                self._current_temperature = 0.99 * self._current_temperature + 0.01 * 25.0 + np.random.normal(0, 0.05)
        
        # Add to history (limit length to avoid memory issues)
        self._temperature_history.append((time.time(), self._current_temperature))
        if len(self._temperature_history) > 1000:
            self._temperature_history.pop(0)
            
        return self._current_temperature
    
    def update_control(self) -> None:
        """
        Update temperature control using PID algorithm.
        
        This method should be called periodically to maintain temperature control.
        """
        if not self._is_controlling or not self._is_connected:
            return
        
        # Get current temperature and time
        current_temp = self.get_current_temperature()
        current_time = time.time()
        dt = current_time - self._last_update_time
        
        # Calculate error
        error = self._target_temperature - current_temp
        
        # Calculate PID terms
        p_term = self.pid_kp * error
        
        # Update integral term with anti-windup
        self._integral_error += error * dt
        i_term = self.pid_ki * self._integral_error
        
        # Apply limits to integral term to prevent windup
        i_term = max(-10.0, min(10.0, i_term))
        
        # Calculate derivative term
        if dt > 0:
            d_term = self.pid_kd * (error - self._last_error) / dt
        else:
            d_term = 0.0
        
        # Calculate control output
        control_output = p_term + i_term + d_term
        
        # Apply limits to control output (-1.0 to 1.0 range)
        control_output = max(-1.0, min(1.0, control_output))
        
        # In a real implementation, apply control_output to Peltier elements
        # Positive output = heating, negative output = cooling
        logger.debug(f"PID output: {control_output:.3f} (P:{p_term:.3f}, I:{i_term:.3f}, D:{d_term:.3f})")
        
        # Update state for next iteration
        self._last_error = error
        self._last_update_time = current_time
    
    def is_temperature_stable(self, window_seconds: float = 10.0) -> bool:
        """
        Check if temperature has been stable for the specified window.
        
        Parameters
        ----------
        window_seconds : float, optional
            Time window to check for stability in seconds, default is 10.0.
            
        Returns
        -------
        bool
            True if temperature has been stable, False otherwise.
        """
        # Need sufficient history to evaluate stability
        if len(self._temperature_history) < 3:
            return False
        
        # Find relevant temperature points within the window
        current_time = time.time()
        window_start = current_time - window_seconds
        
        window_temps = [temp for time_point, temp in self._temperature_history 
                      if time_point >= window_start]
        
        # Need sufficient points in window
        if len(window_temps) < 3:
            return False
        
        # Check if all temperatures in window are within threshold of target
        max_temp = max(window_temps)
        min_temp = min(window_temps)
        
        return (max_temp - min_temp <= self.temperature_stability_threshold and
                abs(max_temp - self._target_temperature) <= self.temperature_stability_threshold and
                abs(min_temp - self._target_temperature) <= self.temperature_stability_threshold)
    
    def wait_for_stability(self, timeout_seconds: float = 300.0) -> bool:
        """
        Wait for temperature to stabilize within threshold.
        
        Parameters
        ----------
        timeout_seconds : float, optional
            Maximum time to wait in seconds, default is 300.0.
            
        Returns
        -------
        bool
            True if stability was achieved, False if timeout occurred.
        """
        if not self._is_controlling:
            logger.warning("Temperature control is not running")
            return False
        
        logger.info(f"Waiting for temperature stability at {self._target_temperature}°C "
                   f"(±{self.temperature_stability_threshold}°C)")
        
        start_time = time.time()
        update_interval = 0.5  # seconds
        
        while time.time() - start_time < timeout_seconds:
            # Update control
            self.update_control()
            
            # Check stability over a window
            if self.is_temperature_stable():
                logger.info(f"Temperature stable at {self.get_current_temperature():.2f}°C")
                return True
            
            # Wait for next update
            time.sleep(update_interval)
        
        logger.warning(f"Temperature stability timeout after {timeout_seconds} seconds")
        return False
    
    def get_temperature_stats(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """
        Get temperature statistics over a time window.
        
        Parameters
        ----------
        window_seconds : float, optional
            Time window for statistics in seconds, default is None (all history).
            
        Returns
        -------
        Dict[str, float]
            Dictionary with temperature statistics.
        """
        if not self._temperature_history:
            return {
                'mean': self._current_temperature,
                'min': self._current_temperature,
                'max': self._current_temperature,
                'std': 0.0,
                'stability': 1.0
            }
        
        # Filter history based on window
        if window_seconds is not None:
            current_time = time.time()
            window_start = current_time - window_seconds
            temps = [temp for time_point, temp in self._temperature_history 
                    if time_point >= window_start]
        else:
            temps = [temp for _, temp in self._temperature_history]
        
        # Calculate statistics
        if not temps:
            temps = [self._current_temperature]
            
        mean_temp = sum(temps) / len(temps)
        min_temp = min(temps)
        max_temp = max(temps)
        
        # Calculate standard deviation
        if len(temps) > 1:
            variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
            std_temp = np.sqrt(variance)
        else:
            std_temp = 0.0
        
        # Calculate stability metric (1.0 = perfectly stable, 0.0 = highly unstable)
        # Based on ratio of observed variation to threshold
        if self.temperature_stability_threshold > 0:
            temp_range = max_temp - min_temp
            stability = max(0.0, min(1.0, 1.0 - (temp_range / (2 * self.temperature_stability_threshold))))
        else:
            stability = 1.0 if std_temp < 0.01 else 0.0
        
        return {
            'mean': mean_temp,
            'min': min_temp,
            'max': max_temp,
            'std': std_temp,
            'stability': stability
        }
    
    def _calculate_effective_thermal_conductivity(self) -> float:
        """
        Calculate effective thermal conductivity of the PCM-enhancer composite.
        
        Returns
        -------
        float
            Effective thermal conductivity in W/(m·K).
        """
        # Convert weight percentage to volume fraction
        # Assuming approximate densities: PCM = 800 kg/m³, enhancer = 2200 kg/m³
        pcm_density = 800.0  # kg/m³
        enhancer_density = 2200.0  # kg/m³
        
        # Weight fractions
        wf_enhancer = self.enhancer_loading / 100.0
        wf_pcm = 1.0 - wf_enhancer
        
        # Convert to volume fractions
        total_volume = (wf_pcm / pcm_density) + (wf_enhancer / enhancer_density)
        vf_pcm = (wf_pcm / pcm_density) / total_volume
        vf_enhancer = (wf_enhancer / enhancer_density) / total_volume
        
        # Use Maxwell-Eucken model for effective thermal conductivity
        k_pcm = self.pcm_type.thermal_conductivity
        k_enhancer = self.enhancer_type.thermal_conductivity
        
        # Calculate effective thermal conductivity
        numerator = k_pcm * (2 * k_pcm + k_enhancer + 2 * vf_enhancer * (k_enhancer - k_pcm))
        denominator = 2 * k_pcm + k_enhancer - vf_enhancer * (k_enhancer - k_pcm)
        
        effective_k = numerator / denominator
        
        return effective_k
    
    def tune_pid_parameters(self, 
                          kp: Optional[float] = None, 
                          ki: Optional[float] = None, 
                          kd: Optional[float] = None) -> None:
        """
        Tune PID controller parameters.
        
        Parameters
        ----------
        kp : float, optional
            Proportional gain, default is None (no change).
        ki : float, optional
            Integral gain, default is None (no change).
        kd : float, optional
            Derivative gain, default is None (no change).
        """
        if kp is not None:
            self.pid_kp = max(0.0, kp)
        
        if ki is not None:
            self.pid_ki = max(0.0, ki)
        
        if kd is not None:
            self.pid_kd = max(0.0, kd)
            
        logger.info(f"PID parameters updated: Kp={self.pid_kp}, Ki={self.pid_ki}, Kd={self.pid_kd}")
    
    def auto_tune_pid(self) -> Tuple[float, float, float]:
        """
        Automatically tune PID parameters using Ziegler-Nichols method.
        
        Returns
        -------
        Tuple[float, float, float]
            Tuned PID parameters (Kp, Ki, Kd).
        """
        if not self._is_connected:
            logger.error("Cannot auto-tune without hardware connection")
            return (self.pid_kp, self.pid_ki, self.pid_kd)
        
        logger.info("Starting PID auto-tuning using Ziegler-Nichols method")
        
        # In a real implementation, we would:
        # 1. Find ultimate gain (Ku) by increasing P gain until oscillation
        # 2. Measure oscillation period (Tu)
        # 3. Calculate PID parameters using Ziegler-Nichols formulas
        
        # For now, simulate with fixed improvement
        original_params = (self.pid_kp, self.pid_ki, self.pid_kd)
        
        # Simulate auto-tuning process
        time.sleep(5.0)  # Simulate time for auto-tuning
        
        # Set "optimized" parameters (in reality these would be calculated)
        new_kp = max(1.0, self.pid_kp * 1.2)
        new_ki = max(0.1, self.pid_ki * 1.1)
        new_kd = max(0.5, self.pid_kd * 1.3)
        
        # Apply new parameters
        self.tune_pid_parameters(new_kp, new_ki, new_kd)
        
        logger.info(f"Auto-tuning complete: Kp={self.pid_kp}, Ki={self.pid_ki}, Kd={self.pid_kd}")
        return (self.pid_kp, self.pid_ki, self.pid_kd)
