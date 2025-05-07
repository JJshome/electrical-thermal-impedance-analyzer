"""
Thermal PCM Controller for the Integrated Electrical-Thermal Impedance Analyzer

This module provides hardware abstraction for controlling Peltier elements and temperature
sensors in a PCM-based thermal management system. It interfaces with the thermal_management
module to provide the actual hardware control.

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import time
import numpy as np
import threading
import logging
from enum import Enum
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ControllerType(Enum):
    """Enumeration for different thermal controller hardware types."""
    ARDUINO = 1      # Arduino-based controller
    CUSTOM_DAQ = 2   # Custom DAQ board
    PI_BASED = 3     # Raspberry Pi based controller
    SIMULATION = 4   # Simulation for development


class SensorType(Enum):
    """Enumeration for different temperature sensor types."""
    PT100 = 1      # Platinum resistance thermometer
    THERMOCOUPLE = 2  # Thermocouple (type K by default)
    THERMISTOR = 3  # NTC thermistor
    INFRARED = 4   # Non-contact infrared sensor
    DIGITAL = 5    # Digital temperature sensor (e.g., DS18B20)


class PeltierController:
    """
    Controller for Peltier thermal elements
    
    This class provides hardware interfaces for controlling Peltier elements
    and reading temperature sensors, with support for various hardware platforms.
    """
    
    def __init__(self, controller_type=ControllerType.ARDUINO, 
                sensor_type=SensorType.PT100, port=None, baudrate=115200,
                simulation_mode=False):
        """
        Initialize the Peltier controller.
        
        Parameters
        ----------
        controller_type : ControllerType, optional
            Type of controller hardware
        sensor_type : SensorType, optional
            Type of temperature sensor
        port : str, optional
            Serial port for communication with controller
        baudrate : int, optional
            Serial baudrate
        simulation_mode : bool, optional
            Whether to run in simulation mode (True) or with real hardware (False)
        """
        self.controller_type = controller_type
        self.sensor_type = sensor_type
        self.port = port
        self.baudrate = baudrate
        self.simulation_mode = simulation_mode
        
        # Hardware connection
        self.serial_conn = None
        
        # System state
        self.connected = False
        self.temperature = 25.0  # Current temperature in °C
        self.peltier_power = 0.0  # Current Peltier power (-1.0 to 1.0)
        self.max_power_watts = 10.0  # Maximum power in watts
        
        # Sensor calibration parameters
        self.sensor_offset = 0.0
        self.sensor_gain = 1.0
        
        # Performance metrics
        self.temperature_stability = 0.0  # Standard deviation
        self.response_time = 0.0  # Time to reach 90% of setpoint
        
        # Connect to hardware if not in simulation mode
        if not simulation_mode:
            self.connect()
        else:
            logger.info("Running in simulation mode, hardware connection skipped")
            self.connected = True
    
    def connect(self):
        """Connect to the controller hardware."""
        if self.simulation_mode:
            logger.info("Simulation mode active, skipping hardware connection")
            self.connected = True
            return True
            
        try:
            if self.controller_type == ControllerType.ARDUINO:
                # If port not specified, try to find Arduino
                if self.port is None:
                    ports = list(serial.tools.list_ports.comports())
                    for p in ports:
                        if 'Arduino' in p.description:
                            self.port = p.device
                            break
                
                if self.port is None:
                    logger.error("Arduino not found. Please specify port manually.")
                    return False
                
                # Connect to Arduino
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino to reset
                
                # Send initialization command
                self.serial_conn.write(b'INIT\n')
                response = self.serial_conn.readline().decode('utf-8').strip()
                
                if response == 'READY':
                    self.connected = True
                    logger.info(f"Connected to Arduino controller on {self.port}")
                    
                    # Get controller capabilities
                    self.serial_conn.write(b'CAPABILITIES\n')
                    capabilities = self.serial_conn.readline().decode('utf-8').strip()
                    logger.info(f"Controller capabilities: {capabilities}")
                    
                    return True
                else:
                    logger.error(f"Failed to initialize Arduino controller. Response: {response}")
                    return False
            
            elif self.controller_type == ControllerType.CUSTOM_DAQ:
                # Implementation for custom DAQ board
                logger.info("Connecting to custom DAQ board...")
                # TODO: Implement custom DAQ communication
                self.connected = True
                return True
            
            elif self.controller_type == ControllerType.PI_BASED:
                # Implementation for Raspberry Pi based controller
                logger.info("Connecting to Raspberry Pi based controller...")
                # TODO: Implement Raspberry Pi GPIO control
                self.connected = True
                return True
            
            else:
                logger.error(f"Unsupported controller type: {self.controller_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to controller: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the controller hardware."""
        if self.simulation_mode:
            logger.info("Simulation mode active, no hardware to disconnect")
            self.connected = False
            return True
            
        try:
            if self.connected:
                if self.controller_type == ControllerType.ARDUINO:
                    # Set Peltier power to 0 before disconnecting
                    self.set_peltier_power(0.0)
                    time.sleep(0.5)
                    
                    if self.serial_conn:
                        self.serial_conn.close()
                
                elif self.controller_type == ControllerType.CUSTOM_DAQ:
                    # Implementation for custom DAQ board
                    # TODO: Implement custom DAQ disconnect
                    pass
                
                elif self.controller_type == ControllerType.PI_BASED:
                    # Implementation for Raspberry Pi based controller
                    # TODO: Implement Raspberry Pi GPIO cleanup
                    pass
                
                self.connected = False
                logger.info("Disconnected from controller")
                return True
            else:
                logger.warning("Not connected to controller")
                return False
                
        except Exception as e:
            logger.error(f"Error disconnecting from controller: {e}")
            return False
    
    def set_peltier_power(self, power):
        """
        Set power to the Peltier element.
        
        Parameters
        ----------
        power : float
            Power level from -1.0 (full cooling) to 1.0 (full heating)
        
        Returns
        -------
        bool
            Success flag
        """
        # Bound power between -1.0 and 1.0
        power = max(min(power, 1.0), -1.0)
        
        if self.simulation_mode:
            # In simulation mode, just update the power value
            self.peltier_power = power
            return True
            
        try:
            if not self.connected:
                logger.warning("Not connected to controller")
                return False
                
            if self.controller_type == ControllerType.ARDUINO:
                # Scale power to 0-255 for Arduino PWM
                # 0-127: Cooling (0: max cooling, 127: no cooling)
                # 128-255: Heating (128: no heating, 255: max heating)
                
                if power < 0:
                    # Cooling (0 to 127)
                    pwm_value = int(127 * (1 + power))
                else:
                    # Heating (128 to 255)
                    pwm_value = int(128 + 127 * power)
                
                # Send command to Arduino
                command = f'PWM:{pwm_value}\n'
                self.serial_conn.write(command.encode('utf-8'))
                
                # Check response
                response = self.serial_conn.readline().decode('utf-8').strip()
                
                if response == 'OK':
                    self.peltier_power = power
                    return True
                else:
                    logger.error(f"Failed to set Peltier power. Response: {response}")
                    return False
            
            elif self.controller_type == ControllerType.CUSTOM_DAQ:
                # Implementation for custom DAQ board
                # TODO: Implement custom DAQ power control
                self.peltier_power = power
                return True
            
            elif self.controller_type == ControllerType.PI_BASED:
                # Implementation for Raspberry Pi based controller
                # TODO: Implement Raspberry Pi GPIO PWM control
                self.peltier_power = power
                return True
            
            else:
                logger.error(f"Unsupported controller type: {self.controller_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting Peltier power: {e}")
            return False
    
    def read_temperature(self):
        """
        Read temperature from the sensor.
        
        Returns
        -------
        float
            Temperature in °C
        """
        if self.simulation_mode:
            # Simulate temperature reading with some noise
            self.temperature += self.peltier_power * 0.1
            self.temperature += np.random.normal(0, 0.05)
            return self.temperature
            
        try:
            if not self.connected:
                logger.warning("Not connected to controller")
                return None
                
            if self.controller_type == ControllerType.ARDUINO:
                # Send command to Arduino
                self.serial_conn.write(b'TEMP\n')
                
                # Read response
                response = self.serial_conn.readline().decode('utf-8').strip()
                
                try:
                    temp_raw = float(response)
                    
                    # Apply calibration
                    self.temperature = temp_raw * self.sensor_gain + self.sensor_offset
                    
                    return self.temperature
                except ValueError:
                    logger.error(f"Invalid temperature reading: {response}")
                    return None
            
            elif self.controller_type == ControllerType.CUSTOM_DAQ:
                # Implementation for custom DAQ board
                # TODO: Implement custom DAQ temperature reading
                self.temperature = 25.0 + np.random.normal(0, 0.1)
                return self.temperature
            
            elif self.controller_type == ControllerType.PI_BASED:
                # Implementation for Raspberry Pi based controller
                # TODO: Implement Raspberry Pi temperature sensor reading
                self.temperature = 25.0 + np.random.normal(0, 0.1)
                return self.temperature
            
            else:
                logger.error(f"Unsupported controller type: {self.controller_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading temperature: {e}")
            return None
    
    def calibrate_sensor(self, reference_temps=[0.0, 25.0, 100.0]):
        """
        Calibrate the temperature sensor.
        
        Parameters
        ----------
        reference_temps : list, optional
            Reference temperatures for calibration
        
        Returns
        -------
        dict
            Calibration parameters
        """
        logger.info("Starting sensor calibration...")
        
        if self.simulation_mode:
            logger.info("Simulation mode active, skipping calibration")
            return {'offset': 0.0, 'gain': 1.0}
        
        try:
            if not self.connected:
                logger.warning("Not connected to controller")
                return None
            
            # For simplicity, we'll implement a 2-point calibration
            # Assumes the first point is ice water (0°C) and the second is boiling water (100°C)
            
            if len(reference_temps) < 2:
                logger.error("At least 2 reference temperatures are needed for calibration")
                return None
            
            # List to store raw readings
            raw_readings = []
            
            for i, ref_temp in enumerate(reference_temps):
                logger.info(f"Place sensor in reference temperature {ref_temp}°C")
                input("Press Enter when ready...")
                
                # Take multiple readings and average
                readings = []
                for _ in range(10):
                    if self.controller_type == ControllerType.ARDUINO:
                        self.serial_conn.write(b'TEMP_RAW\n')
                        response = self.serial_conn.readline().decode('utf-8').strip()
                        readings.append(float(response))
                    else:
                        # For other controller types, implement appropriate reading method
                        readings.append(25.0 + np.random.normal(0, 0.1))
                    
                    time.sleep(0.1)
                
                raw_reading = np.mean(readings)
                raw_readings.append(raw_reading)
                logger.info(f"Raw reading at {ref_temp}°C: {raw_reading}")
            
            # Calculate calibration parameters
            # For 2-point calibration
            x1, x2 = raw_readings[0], raw_readings[-1]
            y1, y2 = reference_temps[0], reference_temps[-1]
            
            gain = (y2 - y1) / (x2 - x1)
            offset = y1 - gain * x1
            
            self.sensor_gain = gain
            self.sensor_offset = offset
            
            logger.info(f"Calibration completed: gain = {gain}, offset = {offset}")
            
            return {'gain': gain, 'offset': offset}
            
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            return None
    
    def get_stability_metrics(self, duration=60.0, target_temp=25.0):
        """
        Measure temperature stability over a period of time.
        
        Parameters
        ----------
        duration : float, optional
            Measurement duration in seconds
        target_temp : float, optional
            Target temperature
        
        Returns
        -------
        dict
            Stability metrics
        """
        logger.info(f"Measuring temperature stability at {target_temp}°C for {duration} seconds...")
        
        # Lists to store measurements
        temperatures = []
        timestamps = []
        
        # Set to target temperature
        self.set_peltier_power(0.1)  # Small heating power to reach target
        
        # Wait for temperature to stabilize at target
        start_time = time.time()
        while time.time() - start_time < 60.0:  # 60 second pre-stabilization
            temp = self.read_temperature()
            
            if abs(temp - target_temp) < 0.5:
                break
                
            if temp < target_temp:
                self.set_peltier_power(0.2)  # Heat
            else:
                self.set_peltier_power(-0.2)  # Cool
                
            time.sleep(1.0)
        
        # Start measurement
        logger.info("Starting stability measurement...")
        start_time = time.time()
        while time.time() - start_time < duration:
            temp = self.read_temperature()
            timestamp = time.time() - start_time
            
            temperatures.append(temp)
            timestamps.append(timestamp)
            
            # Simple control to maintain target temperature
            if temp < target_temp - 0.1:
                self.set_peltier_power(0.1)  # Heat
            elif temp > target_temp + 0.1:
                self.set_peltier_power(-0.1)  # Cool
            else:
                self.set_peltier_power(0.0)  # Maintain
                
            time.sleep(1.0)
        
        # Calculate metrics
        temperatures = np.array(temperatures)
        stability = np.std(temperatures)
        max_deviation = np.max(np.abs(temperatures - target_temp))
        
        # Store results
        self.temperature_stability = stability
        
        # Return to zero power
        self.set_peltier_power(0.0)
        
        logger.info(f"Stability measurement completed: std dev = {stability:.4f}°C, max deviation = {max_deviation:.4f}°C")
        
        return {
            'mean': np.mean(temperatures),
            'std_dev': stability,
            'max_deviation': max_deviation,
            'temperatures': temperatures,
            'timestamps': timestamps
        }
    
    def measure_response_time(self, step_size=5.0, direction='heating'):
        """
        Measure the response time of the thermal system.
        
        Parameters
        ----------
        step_size : float, optional
            Temperature step size in °C
        direction : str, optional
            'heating' or 'cooling'
        
        Returns
        -------
        dict
            Response time metrics
        """
        logger.info(f"Measuring {direction} response time with {step_size}°C step...")
        
        # Determine initial and target temperatures
        if direction == 'heating':
            initial_temp = 25.0
            target_temp = initial_temp + step_size
            initial_power = -0.5  # Cool to ensure we start below initial_temp
            step_power = 1.0  # Full heating power for step
        else:  # cooling
            initial_temp = 30.0
            target_temp = initial_temp - step_size
            initial_power = 0.5  # Heat to ensure we start above initial_temp
            step_power = -1.0  # Full cooling power for step
        
        # Lists to store measurements
        temperatures = []
        timestamps = []
        
        # Set to initial temperature
        self.set_peltier_power(initial_power)
        
        # Wait for temperature to stabilize at initial temperature
        start_time = time.time()
        while time.time() - start_time < 120.0:  # 2 minute max wait time
            temp = self.read_temperature()
            
            if direction == 'heating' and temp <= initial_temp:
                break
            elif direction == 'cooling' and temp >= initial_temp:
                break
                
            time.sleep(1.0)
        
        # Let temperature stabilize at initial point
        logger.info(f"Stabilizing at initial temperature {initial_temp}°C...")
        self.set_peltier_power(0.0)
        time.sleep(30.0)  # 30 second stabilization
        
        # Record initial temperature
        initial_measured = self.read_temperature()
        
        # Start measurement and apply step
        logger.info(f"Applying {direction} step to target {target_temp}°C...")
        start_time = time.time()
        self.set_peltier_power(step_power)
        
        # Measure until we reach 95% of step or timeout
        target_threshold = initial_measured + 0.95 * (target_temp - initial_measured) if direction == 'heating' else initial_measured - 0.95 * (initial_measured - target_temp)
        timeout = start_time + 300.0  # 5 minute timeout
        
        while time.time() < timeout:
            temp = self.read_temperature()
            timestamp = time.time() - start_time
            
            temperatures.append(temp)
            timestamps.append(timestamp)
            
            if direction == 'heating' and temp >= target_threshold:
                break
            elif direction == 'cooling' and temp <= target_threshold:
                break
                
            time.sleep(0.5)
        
        # Calculate response time (time to reach 90% of step)
        target_90pct = initial_measured + 0.9 * (target_temp - initial_measured) if direction == 'heating' else initial_measured - 0.9 * (initial_measured - target_temp)
        
        temperatures = np.array(temperatures)
        timestamps = np.array(timestamps)
        
        # Find the time at which temperature crossed 90% threshold
        if direction == 'heating':
            indices = np.where(temperatures >= target_90pct)[0]
        else:
            indices = np.where(temperatures <= target_90pct)[0]
            
        if len(indices) > 0:
            response_time = timestamps[indices[0]]
        else:
            response_time = None
            logger.warning("Failed to reach 90% of target within timeout")
        
        # Store results
        self.response_time = response_time if response_time is not None else float('inf')
        
        # Return to zero power
        self.set_peltier_power(0.0)
        
        logger.info(f"Response time measurement completed: t90 = {response_time:.2f} seconds")
        
        return {
            'initial_temp': initial_measured,
            'target_temp': target_temp,
            'response_time_90pct': response_time,
            'temperatures': temperatures.tolist(),
            'timestamps': timestamps.tolist(),
            'direction': direction
        }
    
    def plot_response_data(self, response_data, figsize=(10, 6)):
        """
        Plot response time measurement data.
        
        Parameters
        ----------
        response_data : dict
            Response data from measure_response_time
        figsize : tuple, optional
            Figure size
        
        Returns
        -------
        tuple
            Figure and axes objects
        """
        if not response_data:
            logger.warning("No response data to plot")
            return None, None
        
        temps = np.array(response_data['temperatures'])
        times = np.array(response_data['timestamps'])
        
        initial_temp = response_data['initial_temp']
        target_temp = response_data['target_temp']
        t90 = response_data['response_time_90pct']
        
        # 90% threshold line
        threshold_90pct = initial_temp + 0.9 * (target_temp - initial_temp) if response_data['direction'] == 'heating' else initial_temp - 0.9 * (initial_temp - target_temp)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot temperature curve
        ax.plot(times, temps, 'b-', linewidth=2, label='Temperature')
        
        # Plot initial, target, and 90% threshold lines
        ax.axhline(y=initial_temp, color='g', linestyle='--', label='Initial')
        ax.axhline(y=target_temp, color='r', linestyle='--', label='Target')
        ax.axhline(y=threshold_90pct, color='m', linestyle=':', label='90% Threshold')
        
        # Plot response time
        if t90 is not None:
            ax.axvline(x=t90, color='k', linestyle='--', label=f'90% Response Time: {t90:.2f}s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{response_data["direction"].capitalize()} Response Time Measurement')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig, ax


class MultiZoneThermalController:
    """
    Multi-zone thermal controller for complex thermal management
    
    This class provides control for multiple Peltier elements
    to create thermal gradients or maintain multiple test zones
    at different temperatures.
    """
    
    def __init__(self, num_zones=2, controller_type=ControllerType.ARDUINO,
                 sensor_type=SensorType.PT100, port=None, baudrate=115200,
                 simulation_mode=False):
        """
        Initialize the multi-zone thermal controller.
        
        Parameters
        ----------
        num_zones : int, optional
            Number of thermal zones
        controller_type : ControllerType, optional
            Type of controller hardware
        sensor_type : SensorType, optional
            Type of temperature sensor
        port : str, optional
            Serial port for communication with controller
        baudrate : int, optional
            Serial baudrate
        simulation_mode : bool, optional
            Whether to run in simulation mode
        """
        self.num_zones = num_zones
        self.simulation_mode = simulation_mode
        
        # Create controllers for each zone
        self.controllers = []
        for i in range(num_zones):
            controller = PeltierController(
                controller_type=controller_type,
                sensor_type=sensor_type,
                port=port,
                baudrate=baudrate,
                simulation_mode=simulation_mode
            )
            self.controllers.append(controller)
        
        # Zone settings
        self.zone_targets = [25.0] * num_zones
        self.zone_powers = [0.0] * num_zones
        self.zone_temperatures = [25.0] * num_zones
        
        # Thermal coupling matrix (how zones affect each other)
        # Default: Identity matrix (no coupling)
        self.coupling_matrix = np.eye(num_zones)
        
        logger.info(f"Initialized multi-zone thermal controller with {num_zones} zones")
    
    def connect_all(self):
        """
        Connect to all zone controllers.
        
        Returns
        -------
        bool
            Success flag
        """
        success = True
        for i, controller in enumerate(self.controllers):
            if not controller.connect():
                logger.error(f"Failed to connect to controller for zone {i}")
                success = False
        
        return success
    
    def disconnect_all(self):
        """
        Disconnect from all zone controllers.
        
        Returns
        -------
        bool
            Success flag
        """
        success = True
        for i, controller in enumerate(self.controllers):
            if not controller.disconnect():
                logger.error(f"Failed to disconnect from controller for zone {i}")
                success = False
        
        return success
    
    def set_zone_temperature(self, zone_idx, target_temp):
        """
        Set target temperature for a specific zone.
        
        Parameters
        ----------
        zone_idx : int
            Zone index
        target_temp : float
            Target temperature in °C
        
        Returns
        -------
        bool
            Success flag
        """
        if zone_idx < 0 or zone_idx >= self.num_zones:
            logger.error(f"Invalid zone index: {zone_idx}")
            return False
        
        self.zone_targets[zone_idx] = target_temp
        logger.info(f"Set zone {zone_idx} target temperature to {target_temp}°C")
        return True
    
    def set_all_zone_temperatures(self, temperatures):
        """
        Set target temperatures for all zones.
        
        Parameters
        ----------
        temperatures : list
            List of target temperatures in °C
        
        Returns
        -------
        bool
            Success flag
        """
        if len(temperatures) != self.num_zones:
            logger.error(f"Expected {self.num_zones} temperatures, got {len(temperatures)}")
            return False
        
        for i, temp in enumerate(temperatures):
            self.zone_targets[i] = temp
        
        logger.info(f"Set all zone temperatures: {temperatures}")
        return True
    
    def set_thermal_gradient(self, start_temp, end_temp):
        """
        Set a linear thermal gradient across zones.
        
        Parameters
        ----------
        start_temp : float
            Starting temperature in °C (zone 0)
        end_temp : float
            Ending temperature in °C (last zone)
        
        Returns
        -------
        bool
            Success flag
        """
        if self.num_zones < 2:
            logger.error("At least 2 zones are required for thermal gradient")
            return False
        
        # Calculate temperature for each zone
        temps = np.linspace(start_temp, end_temp, self.num_zones)
        
        # Set temperatures for each zone
        for i, temp in enumerate(temps):
            self.zone_targets[i] = temp
        
        logger.info(f"Set thermal gradient from {start_temp}°C to {end_temp}°C")
        return True
    
    def set_custom_thermal_profile(self, profile):
        """
        Set a custom thermal profile across zones.
        
        Parameters
        ----------
        profile : list
            List of target temperatures in °C for each zone
        
        Returns
        -------
        bool
            Success flag
        """
        if len(profile) != self.num_zones:
            logger.error(f"Expected {self.num_zones} temperatures, got {len(profile)}")
            return False
        
        # Set temperatures for each zone
        for i, temp in enumerate(profile):
            self.zone_targets[i] = temp
        
        logger.info(f"Set custom thermal profile: {profile}")
        return True
    
    def set_thermal_coupling_matrix(self, matrix):
        """
        Set thermal coupling matrix between zones.
        
        Parameters
        ----------
        matrix : numpy.ndarray
            Thermal coupling matrix (NxN)
        
        Returns
        -------
        bool
            Success flag
        """
        if matrix.shape != (self.num_zones, self.num_zones):
            logger.error(f"Expected {self.num_zones}x{self.num_zones} matrix, got {matrix.shape}")
            return False
        
        self.coupling_matrix = matrix
        logger.info("Set thermal coupling matrix")
        return True
    
    def update_temperatures(self):
        """
        Update temperature readings from all zones.
        
        Returns
        -------
        list
            List of current temperatures in °C
        """
        for i, controller in enumerate(self.controllers):
            temp = controller.read_temperature()
            if temp is not None:
                self.zone_temperatures[i] = temp
        
        return self.zone_temperatures
    
    def update_control(self):
        """
        Update thermal control for all zones.
        
        Returns
        -------
        list
            List of applied power levels
        """
        # Update temperature readings
        self.update_temperatures()
        
        # Calculate power levels considering thermal coupling
        errors = np.array(self.zone_targets) - np.array(self.zone_temperatures)
        
        # Simple proportional control
        Kp = 0.2
        powers_raw = Kp * errors
        
        # Apply coupling matrix to account for thermal interaction
        powers_coupled = np.linalg.solve(self.coupling_matrix, powers_raw)
        
        # Clip powers to valid range
        powers_clipped = np.clip(powers_coupled, -1.0, 1.0)
        
        # Apply power to each controller
        for i, controller in enumerate(self.controllers):
            controller.set_peltier_power(powers_clipped[i])
            self.zone_powers[i] = powers_clipped[i]
        
        return self.zone_powers
    
    def run_control_loop(self, update_interval=1.0, max_duration=None):
        """
        Run continuous control loop for all zones.
        
        Parameters
        ----------
        update_interval : float, optional
            Control loop interval in seconds
        max_duration : float, optional
            Maximum duration in seconds (None for continuous)
        
        Returns
        -------
        bool
            Success flag
        """
        logger.info("Starting multi-zone thermal control loop")
        
        start_time = time.time()
        running = True
        
        try:
            while running:
                # Update control
                self.update_control()
                
                # Check if max duration reached
                if max_duration is not None and time.time() - start_time >= max_duration:
                    running = False
                
                # Sleep for update interval
                time.sleep(update_interval)
                
            # Set all powers to zero when done
            for controller in self.controllers:
                controller.set_peltier_power(0.0)
                
            logger.info("Multi-zone thermal control loop completed")
            return True
            
        except KeyboardInterrupt:
            logger.info("Multi-zone thermal control loop interrupted")
            # Set all powers to zero
            for controller in self.controllers:
                controller.set_peltier_power(0.0)
            return True
            
        except Exception as e:
            logger.error(f"Error in multi-zone thermal control loop: {e}")
            # Set all powers to zero
            for controller in self.controllers:
                controller.set_peltier_power(0.0)
            return False
    
    def get_zone_status(self, zone_idx):
        """
        Get status of a specific zone.
        
        Parameters
        ----------
        zone_idx : int
            Zone index
        
        Returns
        -------
        dict
            Zone status
        """
        if zone_idx < 0 or zone_idx >= self.num_zones:
            logger.error(f"Invalid zone index: {zone_idx}")
            return None
        
        return {
            'temperature': self.zone_temperatures[zone_idx],
            'target': self.zone_targets[zone_idx],
            'power': self.zone_powers[zone_idx],
            'error': self.zone_targets[zone_idx] - self.zone_temperatures[zone_idx]
        }
    
    def get_all_zone_status(self):
        """
        Get status of all zones.
        
        Returns
        -------
        list
            List of zone status dictionaries
        """
        return [self.get_zone_status(i) for i in range(self.num_zones)]
    
    def plot_zone_temperatures(self, duration=60.0, interval=1.0, figsize=(12, 8)):
        """
        Plot temperatures of all zones over time.
        
        Parameters
        ----------
        duration : float, optional
            Monitoring duration in seconds
        interval : float, optional
            Sampling interval in seconds
        figsize : tuple, optional
            Figure size
        
        Returns
        -------
        tuple
            Figure and axes objects
        """
        logger.info(f"Monitoring zone temperatures for {duration} seconds...")
        
        # Lists to store data
        timestamps = []
        temp_data = [[] for _ in range(self.num_zones)]
        power_data = [[] for _ in range(self.num_zones)]
        
        # Start monitoring
        start_time = time.time()
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            # Update control and record data
            self.update_control()
            
            timestamps.append(current_time)
            for i in range(self.num_zones):
                temp_data[i].append(self.zone_temperatures[i])
                power_data[i].append(self.zone_powers[i])
            
            # Sleep for interval
            time.sleep(interval)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Temperature plot
        for i in range(self.num_zones):
            ax1.plot(timestamps, temp_data[i], label=f'Zone {i}')
        
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Multi-Zone Temperature Monitoring')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Power plot
        for i in range(self.num_zones):
            ax2.plot(timestamps, power_data[i], label=f'Zone {i}')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power (-1 to 1)')
        ax2.set_title('Zone Power Levels')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2)
"""