"""
Thermal management module for the Integrated Electrical-Thermal Impedance Analyzer

This module implements thermal control and monitoring using Phase Change Materials (PCM)
for precise temperature control during impedance measurements. The thermal management
system ensures stable temperature conditions for accurate impedance analysis.

Based on the methodology described in the patent:
Integrated Electrical-Thermal Impedance Analysis System and Method

Author: Jee Hwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
import time
import threading
import logging
from enum import Enum
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThermalMode(Enum):
    """Enumeration for different thermal control modes."""
    CONSTANT = 1      # Maintain constant temperature
    PULSE = 2         # Apply thermal pulse
    SINUSOIDAL = 3    # Apply sinusoidal temperature variation
    STEP = 4          # Apply step temperature change
    RAMP = 5          # Apply temperature ramp
    ADAPTIVE = 6      # Adaptive temperature control based on measurement


class PCMType(Enum):
    """Enumeration for different PCM types."""
    OCTADECANE = 1        # n-octadecane: Tm = 28°C, Latent heat = 200-240 J/g
    PARAFFIN = 2          # Paraffin wax: Tm = 45-65°C, Latent heat = 170-200 J/g 
    PALMITIC_ACID = 3     # Palmitic acid: Tm = 63°C, Latent heat = 185-200 J/g
    EICOSANE = 4          # n-eicosane: Tm = 35-37°C, Latent heat = 240-250 J/g
    SALT_HYDRATE = 5      # Salt hydrate: Tm varies, Latent heat = 150-300 J/g
    CUSTOM = 6            # Custom PCM with user-defined properties


class ThermalManager:
    """
    Thermal management system using Phase Change Materials (PCM)
    
    This class provides temperature control and monitoring for
    impedance measurements, ensuring stable thermal conditions
    and applying precise thermal stimuli when needed.
    """
    
    def __init__(self, pcm_type=PCMType.EICOSANE, mode=ThermalMode.CONSTANT,
                target_temp=25.0, control_interval=0.1, simulation_mode=True):
        """
        Initialize the thermal management system.
        
        Parameters
        ----------
        pcm_type : PCMType, optional
            Type of Phase Change Material to use
        mode : ThermalMode, optional
            Temperature control mode
        target_temp : float, optional
            Target temperature in °C
        control_interval : float, optional
            Control loop interval in seconds
        simulation_mode : bool, optional
            Whether to run in simulation mode (True) or with real hardware (False)
        """
        self.pcm_type = pcm_type
        self.mode = mode
        self.target_temp = target_temp
        self.control_interval = control_interval
        self.simulation_mode = simulation_mode
        
        # PCM properties
        self.pcm_properties = self._get_pcm_properties(pcm_type)
        
        # Temperature sensor data
        self.current_temp = target_temp
        self.temp_history = []
        self.time_history = []
        self.start_time = None
        
        # Control parameters
        self.running = False
        self.control_thread = None
        self.pid_params = {'Kp': 5.0, 'Ki': 0.5, 'Kd': 1.0}
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        # Peltier device parameters
        self.max_power = 10.0  # W
        self.current_power = 0.0  # W
        self.power_history = []
        
        # Pulse parameters
        self.pulse_amplitude = 1.0  # °C
        self.pulse_duration = 10.0  # s
        self.pulse_active = False
        
        # Sinusoidal parameters
        self.sin_amplitude = 0.5  # °C
        self.sin_frequency = 0.1  # Hz
        self.sin_phase = 0.0  # rad
        
        # Step parameters
        self.step_size = 5.0  # °C
        self.step_start_time = None
        
        # Ramp parameters
        self.ramp_rate = 0.1  # °C/s
        self.ramp_target = target_temp + 5.0  # °C
        
        # Thermal model parameters (used in simulation mode)
        self.thermal_model = {
            'ambient_temp': 25.0,  # °C
            'thermal_resistance': 5.0,  # K/W (device to ambient)
            'thermal_capacitance': 50.0,  # J/K (device thermal mass)
            'pcm_mass': 10.0,  # g
            'pcm_specific_heat_solid': self.pcm_properties['specific_heat_solid'],  # J/(g*K)
            'pcm_specific_heat_liquid': self.pcm_properties['specific_heat_liquid'],  # J/(g*K)
            'pcm_latent_heat': self.pcm_properties['latent_heat'],  # J/g
            'pcm_melting_temp': self.pcm_properties['melting_temp'],  # °C
            'pcm_phase_range': self.pcm_properties['phase_range'],  # °C
            'pcm_thermal_conductivity': self.pcm_properties['thermal_conductivity'],  # W/(m*K)
            'pcm_state': 0.0  # 0.0 = solid, 1.0 = liquid, in-between = partially melted
        }

        logger.info(f"Thermal management initialized with {pcm_type.name} PCM")
        logger.info(f"Target temperature: {target_temp}°C, Mode: {mode.name}")
    
    def _get_pcm_properties(self, pcm_type):
        """
        Get properties of the selected PCM.
        
        Parameters
        ----------
        pcm_type : PCMType
            Type of Phase Change Material
        
        Returns
        -------
        dict
            PCM properties
        """
        properties = {}
        
        if pcm_type == PCMType.OCTADECANE:
            properties = {
                'melting_temp': 28.0,  # °C
                'latent_heat': 240.0,  # J/g
                'specific_heat_solid': 2.0,  # J/(g*K)
                'specific_heat_liquid': 2.3,  # J/(g*K)
                'thermal_conductivity': 0.2,  # W/(m*K)
                'density_solid': 0.85,  # g/cm³
                'density_liquid': 0.78,  # g/cm³
                'phase_range': 1.0,  # °C
                'description': 'n-octadecane'
            }
        
        elif pcm_type == PCMType.PARAFFIN:
            properties = {
                'melting_temp': 45.0,  # °C (can vary)
                'latent_heat': 200.0,  # J/g
                'specific_heat_solid': 2.1,  # J/(g*K)
                'specific_heat_liquid': 2.4,  # J/(g*K)
                'thermal_conductivity': 0.25,  # W/(m*K)
                'density_solid': 0.88,  # g/cm³
                'density_liquid': 0.76,  # g/cm³
                'phase_range': 8.0,  # °C
                'description': 'Paraffin wax'
            }
        
        elif pcm_type == PCMType.PALMITIC_ACID:
            properties = {
                'melting_temp': 63.0,  # °C
                'latent_heat': 190.0,  # J/g
                'specific_heat_solid': 1.9,  # J/(g*K)
                'specific_heat_liquid': 2.2,  # J/(g*K)
                'thermal_conductivity': 0.16,  # W/(m*K)
                'density_solid': 0.94,  # g/cm³
                'density_liquid': 0.85,  # g/cm³
                'phase_range': 2.0,  # °C
                'description': 'Palmitic acid'
            }
        
        elif pcm_type == PCMType.EICOSANE:
            properties = {
                'melting_temp': 36.0,  # °C
                'latent_heat': 245.0,  # J/g
                'specific_heat_solid': 2.0,  # J/(g*K)
                'specific_heat_liquid': 2.3,  # J/(g*K)
                'thermal_conductivity': 0.23,  # W/(m*K)
                'density_solid': 0.84,  # g/cm³
                'density_liquid': 0.78,  # g/cm³
                'phase_range': 1.5,  # °C
                'description': 'n-eicosane'
            }
        
        elif pcm_type == PCMType.SALT_HYDRATE:
            properties = {
                'melting_temp': 50.0,  # °C (can vary widely)
                'latent_heat': 250.0,  # J/g
                'specific_heat_solid': 1.5,  # J/(g*K)
                'specific_heat_liquid': 2.0,  # J/(g*K)
                'thermal_conductivity': 0.6,  # W/(m*K)
                'density_solid': 1.7,  # g/cm³
                'density_liquid': 1.5,  # g/cm³
                'phase_range': 3.0,  # °C
                'description': 'Salt hydrate'
            }
        
        elif pcm_type == PCMType.CUSTOM:
            # Default values for custom PCM
            properties = {
                'melting_temp': 35.0,  # °C
                'latent_heat': 200.0,  # J/g
                'specific_heat_solid': 2.0,  # J/(g*K)
                'specific_heat_liquid': 2.3,  # J/(g*K)
                'thermal_conductivity': 0.3,  # W/(m*K)
                'density_solid': 0.9,  # g/cm³
                'density_liquid': 0.85,  # g/cm³
                'phase_range': 2.0,  # °C
                'description': 'Custom PCM'
            }
        
        return properties
    
    def set_custom_pcm_properties(self, properties):
        """
        Set properties for a custom PCM.
        
        Parameters
        ----------
        properties : dict
            Dictionary of PCM properties
        """
        if self.pcm_type != PCMType.CUSTOM:
            logger.warning("Setting custom properties for non-custom PCM type")
            self.pcm_type = PCMType.CUSTOM
        
        self.pcm_properties.update(properties)
        
        # Update thermal model
        self.thermal_model['pcm_specific_heat_solid'] = self.pcm_properties['specific_heat_solid']
        self.thermal_model['pcm_specific_heat_liquid'] = self.pcm_properties['specific_heat_liquid']
        self.thermal_model['pcm_latent_heat'] = self.pcm_properties['latent_heat']
        self.thermal_model['pcm_melting_temp'] = self.pcm_properties['melting_temp']
        self.thermal_model['pcm_phase_range'] = self.pcm_properties['phase_range']
        self.thermal_model['pcm_thermal_conductivity'] = self.pcm_properties['thermal_conductivity']
        
        logger.info("Custom PCM properties updated")
    
    def set_mode(self, mode, params=None):
        """
        Set the thermal control mode and parameters.
        
        Parameters
        ----------
        mode : ThermalMode
            Temperature control mode
        params : dict, optional
            Mode-specific parameters
        """
        self.mode = mode
        
        if params is not None:
            if mode == ThermalMode.CONSTANT:
                if 'target_temp' in params:
                    self.target_temp = params['target_temp']
            
            elif mode == ThermalMode.PULSE:
                if 'amplitude' in params:
                    self.pulse_amplitude = params['amplitude']
                if 'duration' in params:
                    self.pulse_duration = params['duration']
            
            elif mode == ThermalMode.SINUSOIDAL:
                if 'amplitude' in params:
                    self.sin_amplitude = params['amplitude']
                if 'frequency' in params:
                    self.sin_frequency = params['frequency']
                if 'phase' in params:
                    self.sin_phase = params['phase']
            
            elif mode == ThermalMode.STEP:
                if 'step_size' in params:
                    self.step_size = params['step_size']
                if 'target_temp' in params:
                    self.target_temp = params['target_temp']
            
            elif mode == ThermalMode.RAMP:
                if 'rate' in params:
                    self.ramp_rate = params['rate']
                if 'target_temp' in params:
                    self.ramp_target = params['target_temp']
            
            elif mode == ThermalMode.ADAPTIVE:
                if 'pid_params' in params:
                    self.pid_params = params['pid_params']
        
        logger.info(f"Thermal control mode set to {mode.name}")
    
    def start(self):
        """Start thermal control in a separate thread."""
        if self.running:
            logger.warning("Thermal control is already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.temp_history = []
        self.time_history = []
        self.power_history = []
        
        # Reset control parameters
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("Thermal control started")
    
    def stop(self):
        """Stop thermal control."""
        if not self.running:
            logger.warning("Thermal control is not running")
            return
        
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        # Set zero power
        self._set_peltier_power(0.0)
        
        logger.info("Thermal control stopped")
    
    def _control_loop(self):
        """Main control loop for thermal management."""
        while self.running:
            try:
                # Get current time
                current_time = time.time() - self.start_time
                
                # Read current temperature
                if self.simulation_mode:
                    # Simulate temperature response
                    self._simulate_temperature()
                else:
                    # Read from hardware
                    self._read_temperature()
                
                # Calculate target temperature based on mode
                target = self._calculate_target_temperature(current_time)
                
                # Calculate control output
                power = self._calculate_control_output(target)
                
                # Apply control to Peltier device
                self._set_peltier_power(power)
                
                # Record data
                self.temp_history.append(self.current_temp)
                self.time_history.append(current_time)
                self.power_history.append(self.current_power)
                
                # Sleep for control interval
                time.sleep(self.control_interval)
            
            except Exception as e:
                logger.error(f"Error in thermal control loop: {e}")
                self.running = False
                break
    
    def _calculate_target_temperature(self, current_time):
        """
        Calculate the target temperature based on the current mode.
        
        Parameters
        ----------
        current_time : float
            Current time in seconds since start
        
        Returns
        -------
        float
            Target temperature in °C
        """
        if self.mode == ThermalMode.CONSTANT:
            return self.target_temp
        
        elif self.mode == ThermalMode.PULSE:
            if not self.pulse_active:
                # Start a new pulse
                self.pulse_active = True
                self.pulse_start_time = current_time
            
            # Check if pulse is still active
            if current_time - self.pulse_start_time < self.pulse_duration:
                return self.target_temp + self.pulse_amplitude
            else:
                self.pulse_active = False
                return self.target_temp
        
        elif self.mode == ThermalMode.SINUSOIDAL:
            # Calculate sinusoidal temperature variation
            return self.target_temp + self.sin_amplitude * np.sin(
                2 * np.pi * self.sin_frequency * current_time + self.sin_phase)
        
        elif self.mode == ThermalMode.STEP:
            if self.step_start_time is None:
                self.step_start_time = current_time
                return self.target_temp
            else:
                return self.target_temp + self.step_size
        
        elif self.mode == ThermalMode.RAMP:
            # Calculate temperature ramp
            ramp_temp = self.target_temp + self.ramp_rate * current_time
            if self.ramp_rate > 0:
                # Heating ramp
                return min(ramp_temp, self.ramp_target)
            else:
                # Cooling ramp
                return max(ramp_temp, self.ramp_target)
        
        elif self.mode == ThermalMode.ADAPTIVE:
            # In adaptive mode, target temp is dynamically adjusted
            # based on the measurement requirements
            return self.target_temp
        
        else:
            logger.warning(f"Unknown mode: {self.mode}")
            return self.target_temp
    
    def _calculate_control_output(self, target_temp):
        """
        Calculate control output using PID algorithm.
        
        Parameters
        ----------
        target_temp : float
            Target temperature in °C
        
        Returns
        -------
        float
            Control output (power to Peltier device)
        """
        # Calculate error
        error = target_temp - self.current_temp
        
        # PID terms
        p_term = self.pid_params['Kp'] * error
        
        # Integral term with anti-windup
        self.error_integral += error * self.control_interval
        if self.error_integral > 10.0:
            self.error_integral = 10.0
        elif self.error_integral < -10.0:
            self.error_integral = -10.0
        i_term = self.pid_params['Ki'] * self.error_integral
        
        # Derivative term
        if self.prev_error is not None:
            d_term = self.pid_params['Kd'] * (error - self.prev_error) / self.control_interval
        else:
            d_term = 0.0
        self.prev_error = error
        
        # Sum PID terms
        output = p_term + i_term + d_term
        
        # Limit output
        if output > self.max_power:
            output = self.max_power
        elif output < -self.max_power:
            output = -self.max_power
        
        return output
    
    def _set_peltier_power(self, power):
        """
        Set power to the Peltier device.
        
        Parameters
        ----------
        power : float
            Power in watts (-ve: cooling, +ve: heating)
        """
        if self.simulation_mode:
            # Update current power for simulation
            self.current_power = power
        else:
            # Send command to hardware
            # This would be implemented for the specific hardware interface
            self.current_power = power
            logger.debug(f"Setting Peltier power to {power:.2f} W")
    
    def _read_temperature(self):
        """Read current temperature from sensor."""
        if self.simulation_mode:
            # Temperature is updated in _simulate_temperature
            pass
        else:
            # Read from hardware
            # This would be implemented for the specific hardware interface
            # For now, we'll just simulate a noisy reading
            self.current_temp = self.current_temp + np.random.normal(0, 0.05)
            logger.debug(f"Read temperature: {self.current_temp:.2f}°C")
    
    def _simulate_temperature(self):
        """Simulate temperature response to applied power."""
        # Time step
        dt = self.control_interval
        
        # Get current power and temperature
        power = self.current_power
        temp = self.current_temp
        
        # Get PCM and thermal model parameters
        ambient_temp = self.thermal_model['ambient_temp']
        thermal_resistance = self.thermal_model['thermal_resistance']
        thermal_capacitance = self.thermal_model['thermal_capacitance']
        pcm_mass = self.thermal_model['pcm_mass']
        pcm_melting_temp = self.thermal_model['pcm_melting_temp']
        pcm_phase_range = self.thermal_model['pcm_phase_range']
        pcm_latent_heat = self.thermal_model['pcm_latent_heat']
        pcm_state = self.thermal_model['pcm_state']
        
        # Calculate heat flows
        heat_flow_power = power  # W
        heat_flow_ambient = (ambient_temp - temp) / thermal_resistance  # W
        
        # Check if in phase change region
        phase_change_lower = pcm_melting_temp - pcm_phase_range / 2
        phase_change_upper = pcm_melting_temp + pcm_phase_range / 2
        
        if phase_change_lower <= temp <= phase_change_upper:
            # In phase change region
            # Calculate how much of the heat goes into phase change
            # vs. temperature change
            
            # Heat capacity during phase change (including latent heat)
            effective_heat_capacity = thermal_capacitance + \
                                    pcm_mass * pcm_latent_heat / pcm_phase_range
            
            # Update temperature
            delta_temp = (heat_flow_power + heat_flow_ambient) * dt / effective_heat_capacity
            
            # Update PCM state
            old_state = pcm_state
            if delta_temp > 0:
                # Heating - melting
                pcm_state += delta_temp / pcm_phase_range
                if pcm_state > 1.0:
                    pcm_state = 1.0
            else:
                # Cooling - solidifying
                pcm_state += delta_temp / pcm_phase_range
                if pcm_state < 0.0:
                    pcm_state = 0.0
            
            # Apply temperature change
            if old_state == 0.0 and pcm_state == 0.0:
                # Fully solid
                effective_heat_capacity = thermal_capacitance + \
                                        pcm_mass * self.thermal_model['pcm_specific_heat_solid']
                delta_temp = (heat_flow_power + heat_flow_ambient) * dt / effective_heat_capacity
            elif old_state == 1.0 and pcm_state == 1.0:
                # Fully liquid
                effective_heat_capacity = thermal_capacitance + \
                                        pcm_mass * self.thermal_model['pcm_specific_heat_liquid']
                delta_temp = (heat_flow_power + heat_flow_ambient) * dt / effective_heat_capacity
            else:
                # Partial phase change
                delta_temp = (pcm_state - old_state) * pcm_phase_range
        
        else:
            # Not in phase change region
            if temp < phase_change_lower:
                # Solid phase
                specific_heat = self.thermal_model['pcm_specific_heat_solid']
                pcm_state = 0.0
            else:
                # Liquid phase
                specific_heat = self.thermal_model['pcm_specific_heat_liquid']
                pcm_state = 1.0
            
            # Effective heat capacity
            effective_heat_capacity = thermal_capacitance + pcm_mass * specific_heat
            
            # Update temperature
            delta_temp = (heat_flow_power + heat_flow_ambient) * dt / effective_heat_capacity
        
        # Update temperature
        self.current_temp += delta_temp
        
        # Add some noise
        self.current_temp += np.random.normal(0, 0.02)
        
        # Update thermal model
        self.thermal_model['pcm_state'] = pcm_state
    
    def get_current_data(self):
        """
        Get current thermal data.
        
        Returns
        -------
        dict
            Current thermal data
        """
        return {
            'temperature': self.current_temp,
            'power': self.current_power,
            'time': time.time() - self.start_time if self.start_time else 0.0,
            'target': self._calculate_target_temperature(time.time() - self.start_time if self.start_time else 0.0),
            'pcm_state': self.thermal_model['pcm_state'] if self.simulation_mode else None,
            'mode': self.mode.name
        }
    
    def get_history_data(self):
        """
        Get historical thermal data.
        
        Returns
        -------
        dict
            Historical thermal data
        """
        return {
            'temperature': self.temp_history,
            'time': self.time_history,
            'power': self.power_history
        }
    
    def plot_temperature_history(self, figsize=(10, 6)):
        """
        Plot temperature history.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size
        
        Returns
        -------
        tuple
            Figure and axes objects
        """
        if not self.temp_history:
            logger.warning("No temperature history available")
            return None, None
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot temperature
        ax1.plot(self.time_history, self.temp_history, 'b-', label='Temperature')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (°C)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Plot power on second y-axis
        ax2 = ax1.twinx()
        ax2.plot(self.time_history, self.power_history, 'r-', label='Power')
        ax2.set_ylabel('Power (W)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        title = f"Temperature Control - Mode: {self.mode.name}"
        if self.mode == ThermalMode.CONSTANT:
            title += f", Target: {self.target_temp:.1f}°C"
        plt.title(title)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def optimize_pid(self, target_temp, duration=60.0, method='BFGS'):
        """
        Optimize PID parameters for given target temperature.
        
        Parameters
        ----------
        target_temp : float
            Target temperature in °C
        duration : float, optional
            Duration of optimization simulation in seconds
        method : str, optional
            Optimization method
        
        Returns
        -------
        dict
            Optimized PID parameters
        """
        logger.info("Starting PID parameter optimization")
        
        # Set target temperature and mode
        original_mode = self.mode
        original_target = self.target_temp
        self.set_mode(ThermalMode.CONSTANT, {'target_temp': target_temp})
        
        # Define cost function for optimization
        def cost_function(params):
            # Unpack parameters
            Kp, Ki, Kd = params
            
            # Set PID parameters
            self.pid_params = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
            
            # Reset simulation
            self.current_temp = original_target
            self.error_integral = 0.0
            self.prev_error = 0.0
            self.thermal_model['pcm_state'] = 0.5  # Assume mid-state
            
            # Simulate for given duration
            temp_history = []
            time_points = np.arange(0, duration, self.control_interval)
            
            for t in time_points:
                # Calculate control output
                target = self._calculate_target_temperature(t)
                power = self._calculate_control_output(target)
                self._set_peltier_power(power)
                
                # Simulate temperature response
                self._simulate_temperature()
                
                # Record temperature
                temp_history.append(self.current_temp)
            
            # Calculate error metrics
            errors = np.array(temp_history) - target_temp
            ise = np.sum(errors**2)  # Integral of squared error
            
            # Penalize large oscillations
            oscillation_penalty = np.sum(np.abs(np.diff(errors)))
            
            # Penalize extreme control outputs
            control_penalty = np.sum(np.abs(np.array(self.power_history) / self.max_power))
            
            # Total cost
            cost = ise + 0.1 * oscillation_penalty + 0.05 * control_penalty
            
            # Reset control and histories
            self.power_history = []
            
            logger.debug(f"Params: {params}, Cost: {cost}")
            return cost
        
        # Initial guess
        x0 = [self.pid_params['Kp'], self.pid_params['Ki'], self.pid_params['Kd']]
        
        # Parameter bounds
        bounds = [(0.1, 20.0), (0.0, 5.0), (0.0, 10.0)]
        
        # Run optimization
        result = minimize(cost_function, x0, method=method, bounds=bounds)
        
        if result.success:
            optimized_params = {
                'Kp': result.x[0],
                'Ki': result.x[1],
                'Kd': result.x[2]
            }
            
            # Set optimized parameters
            self.pid_params = optimized_params
            
            logger.info(f"PID optimization successful: {optimized_params}")
        else:
            logger.warning(f"PID optimization failed: {result.message}")
            optimized_params = self.pid_params
        
        # Restore original mode and target
        self.set_mode(original_mode, {'target_temp': original_target})
        
        return optimized_params
    
    def apply_thermal_stimulus(self, stimulus_type, params):
        """
        Apply thermal stimulus for impedance measurement.
        
        Parameters
        ----------
        stimulus_type : str
            Type of stimulus ('pulse', 'sine', 'step', 'ramp')
        params : dict
            Stimulus parameters
        
        Returns
        -------
        bool
            Success flag
        """
        # Map stimulus type to thermal mode
        mode_map = {
            'pulse': ThermalMode.PULSE,
            'sine': ThermalMode.SINUSOIDAL,
            'step': ThermalMode.STEP,
            'ramp': ThermalMode.RAMP
        }
        
        if stimulus_type not in mode_map:
            logger.error(f"Unknown stimulus type: {stimulus_type}")
            return False
        
        # Set mode and parameters
        self.set_mode(mode_map[stimulus_type], params)
        
        # Start thermal control if not running
        if not self.running:
            self.start()
        
        logger.info(f"Applied {stimulus_type} thermal stimulus with params: {params}")
        return True
    
    def set_pcm_type(self, pcm_type):
        """
        Set the PCM type.
        
        Parameters
        ----------
        pcm_type : PCMType
            Type of Phase Change Material
        """
        self.pcm_type = pcm_type
        self.pcm_properties = self._get_pcm_properties(pcm_type)
        
        # Update thermal model
        self.thermal_model['pcm_specific_heat_solid'] = self.pcm_properties['specific_heat_solid']
        self.thermal_model['pcm_specific_heat_liquid'] = self.pcm_properties['specific_heat_liquid']
        self.thermal_model['pcm_latent_heat'] = self.pcm_properties['latent_heat']
        self.thermal_model['pcm_melting_temp'] = self.pcm_properties['melting_temp']
        self.thermal_model['pcm_phase_range'] = self.pcm_properties['phase_range']
        self.thermal_model['pcm_thermal_conductivity'] = self.pcm_properties['thermal_conductivity']
        
        logger.info(f"PCM type set to {pcm_type.name}")
    
    def get_pcm_properties(self):
        """
        Get current PCM properties.
        
        Returns
        -------
        dict
            PCM properties
        """
        return self.pcm_properties.copy()


class ThermalStimulator:
    """
    Thermal stimulator for impedance measurements.
    
    This class provides specialized thermal stimuli specifically
    designed for thermal impedance spectroscopy.
    """
    
    def __init__(self, thermal_manager):
        """
        Initialize the thermal stimulator.
        
        Parameters
        ----------
        thermal_manager : ThermalManager
            Thermal management system
        """
        self.thermal_manager = thermal_manager
        self.base_temp = thermal_manager.target_temp
        self.stimulus_profiles = []
    
    def create_frequency_sweep(self, freq_min=0.001, freq_max=1.0, 
                              num_points=10, amplitude=0.5):
        """
        Create a frequency sweep profile for thermal impedance spectroscopy.
        
        Parameters
        ----------
        freq_min : float, optional
            Minimum frequency in Hz
        freq_max : float, optional
            Maximum frequency in Hz
        num_points : int, optional
            Number of frequency points (logarithmically spaced)
        amplitude : float, optional
            Temperature amplitude in °C
        
        Returns
        -------
        list
            Frequency sweep profile
        """
        # Create logarithmically spaced frequency points
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_points)
        
        # Create profile
        profile = []
        for freq in frequencies:
            profile.append({
                'type': 'sine',
                'frequency': freq,
                'amplitude': amplitude,
                'duration': max(5 / freq, 10)  # At least 5 cycles or 10 seconds
            })
        
        self.stimulus_profiles = profile
        logger.info(f"Created frequency sweep with {num_points} points from {freq_min} to {freq_max} Hz")
        return profile
    
    def create_step_profile(self, step_sizes=[0.5, 1.0, 2.0], durations=None):
        """
        Create a step profile for thermal step response measurements.
        
        Parameters
        ----------
        step_sizes : list, optional
            List of step sizes in °C
        durations : list, optional
            List of step durations in seconds (default: 60s for each step)
        
        Returns
        -------
        list
            Step profile
        """
        if durations is None:
            durations = [60.0] * len(step_sizes)
        
        if len(step_sizes) != len(durations):
            logger.warning("step_sizes and durations must have the same length")
            return None
        
        # Create profile
        profile = []
        for size, duration in zip(step_sizes, durations):
            profile.append({
                'type': 'step',
                'step_size': size,
                'duration': duration
            })
        
        self.stimulus_profiles = profile
        logger.info(f"Created step profile with {len(step_sizes)} steps")
        return profile
    
    def create_pulse_profile(self, amplitudes=[1.0, 2.0], durations=[5.0, 10.0]):
        """
        Create a pulse profile for thermal pulse response measurements.
        
        Parameters
        ----------
        amplitudes : list, optional
            List of pulse amplitudes in °C
        durations : list, optional
            List of pulse durations in seconds
        
        Returns
        -------
        list
            Pulse profile
        """
        if len(amplitudes) != len(durations):
            logger.warning("amplitudes and durations must have the same length")
            return None
        
        # Create profile
        profile = []
        for amp, duration in zip(amplitudes, durations):
            profile.append({
                'type': 'pulse',
                'amplitude': amp,
                'duration': duration
            })
        
        self.stimulus_profiles = profile
        logger.info(f"Created pulse profile with {len(amplitudes)} pulses")
        return profile
    
    def run_stimulus_sequence(self, wait_time=30.0, callback=None):
        """
        Run a sequence of thermal stimuli.
        
        Parameters
        ----------
        wait_time : float, optional
            Wait time between stimuli in seconds
        callback : callable, optional
            Callback function to be called after each stimulus
        
        Returns
        -------
        bool
            Success flag
        """
        if not self.stimulus_profiles:
            logger.warning("No stimulus profiles defined")
            return False
        
        # Store original mode and target
        original_mode = self.thermal_manager.mode
        original_target = self.thermal_manager.target_temp
        
        # Ensure thermal controller is running
        if not self.thermal_manager.running:
            self.thermal_manager.start()
        
        # Set constant temperature and wait for stabilization
        self.thermal_manager.set_mode(ThermalMode.CONSTANT, {'target_temp': self.base_temp})
        logger.info(f"Waiting for temperature to stabilize at {self.base_temp}°C")
        time.sleep(wait_time)
        
        # Run each stimulus
        for i, profile in enumerate(self.stimulus_profiles):
            logger.info(f"Running stimulus {i+1}/{len(self.stimulus_profiles)}: {profile['type']}")
            
            stimulus_type = profile.pop('type')
            duration = profile.pop('duration')
            
            # Apply stimulus
            self.thermal_manager.apply_thermal_stimulus(stimulus_type, profile)
            
            # Wait for stimulus duration
            time.sleep(duration)
            
            # Return to base temperature and wait for stabilization
            self.thermal_manager.set_mode(ThermalMode.CONSTANT, {'target_temp': self.base_temp})
            time.sleep(wait_time)
            
            # Call callback if provided
            if callback is not None:
                callback(i, self.thermal_manager.get_history_data())
        
        # Restore original mode and target
        self.thermal_manager.set_mode(original_mode, {'target_temp': original_target})
        
        logger.info("Stimulus sequence completed")
        return True


class PCMMixture:
    """
    Class for designing PCM mixtures with enhanced thermal properties.
    
    This class allows creating custom PCM mixtures with additives like
    carbon nanotubes, graphene, or metal particles to enhance thermal
    conductivity and other properties.
    """
    
    def __init__(self, base_pcm_type=PCMType.EICOSANE):
        """
        Initialize PCM mixture designer.
        
        Parameters
        ----------
        base_pcm_type : PCMType, optional
            Base PCM type
        """
        self.base_pcm_type = base_pcm_type
        self.base_properties = self._get_base_properties(base_pcm_type)
        self.additives = []
        self.mixture_properties = self.base_properties.copy()
        
        logger.info(f"PCM mixture designer initialized with {base_pcm_type.name}")
    
    def _get_base_properties(self, pcm_type):
        """
        Get properties of the base PCM.
        
        Parameters
        ----------
        pcm_type : PCMType
            Base PCM type
        
        Returns
        -------
        dict
            Base PCM properties
        """
        # Create a temporary ThermalManager to get PCM properties
        temp_manager = ThermalManager(pcm_type=pcm_type)
        return temp_manager.get_pcm_properties()
    
    def add_carbon_nanotubes(self, weight_percent=3.0):
        """
        Add carbon nanotubes to enhance thermal conductivity.
        
        Parameters
        ----------
        weight_percent : float, optional
            Weight percentage of carbon nanotubes
        
        Returns
        -------
        dict
            Updated mixture properties
        """
        if weight_percent <= 0 or weight_percent >= 100:
            logger.warning("Weight percentage must be between 0 and 100")
            return self.mixture_properties
        
        # Add to additives list
        self.additives.append({
            'type': 'carbon_nanotubes',
            'weight_percent': weight_percent
        })
        
        # Update properties
        # Thermal conductivity enhancement based on literature
        # (approximate model)
        k_base = self.mixture_properties['thermal_conductivity']
        k_cnt = 3000.0  # W/(m*K), approximate for carbon nanotubes
        
        # Simple effective medium approximation
        vol_frac = weight_percent / 100 * (self.base_properties['density_solid'] / 1.3)  # Approx. CNT density of 1.3 g/cm³
        k_eff = k_base * (1 + 3 * vol_frac * (k_cnt - k_base) / (k_cnt + 2 * k_base))
        
        # Maximum practical enhancement factor (limited by percolation and dispersion)
        max_enhancement = 5.0  # typical max enhancement for well-dispersed CNTs
        if k_eff / k_base > max_enhancement:
            k_eff = k_base * max_enhancement
        
        self.mixture_properties['thermal_conductivity'] = k_eff
        
        # Slight reduction in latent heat
        latent_heat_reduction = 1.0 - 0.01 * weight_percent  # 1% reduction per wt%
        self.mixture_properties['latent_heat'] *= latent_heat_reduction
        
        # Slight increase in density
        density_increase = 1.0 + 0.005 * weight_percent  # 0.5% increase per wt%
        self.mixture_properties['density_solid'] *= density_increase
        self.mixture_properties['density_liquid'] *= density_increase
        
        # Update description
        self.mixture_properties['description'] = f"{self.base_properties['description']} + {weight_percent}wt% Carbon Nanotubes"
        
        logger.info(f"Added {weight_percent}wt% carbon nanotubes to PCM mixture")
        logger.info(f"Thermal conductivity increased from {k_base:.2f} to {k_eff:.2f} W/(m*K)")
        
        return self.mixture_properties
    
    def add_graphene(self, weight_percent=2.0):
        """
        Add graphene to enhance thermal conductivity.
        
        Parameters
        ----------
        weight_percent : float, optional
            Weight percentage of graphene
        
        Returns
        -------
        dict
            Updated mixture properties
        """
        if weight_percent <= 0 or weight_percent >= 100:
            logger.warning("Weight percentage must be between 0 and 100")
            return self.mixture_properties
        
        # Add to additives list
        self.additives.append({
            'type': 'graphene',
            'weight_percent': weight_percent
        })
        
        # Update properties
        # Thermal conductivity enhancement based on literature
        k_base = self.mixture_properties['thermal_conductivity']
        k_graphene = 5000.0  # W/(m*K), approximate for graphene
        
        # Simple effective medium approximation
        vol_frac = weight_percent / 100 * (self.base_properties['density_solid'] / 2.2)  # Approx. graphene density of 2.2 g/cm³
        k_eff = k_base * (1 + 3 * vol_frac * (k_graphene - k_base) / (k_graphene + 2 * k_base))
        
        # Maximum practical enhancement factor (limited by percolation and dispersion)
        max_enhancement = 8.0  # typical max enhancement for well-dispersed graphene
        if k_eff / k_base > max_enhancement:
            k_eff = k_base * max_enhancement
        
        self.mixture_properties['thermal_conductivity'] = k_eff
        
        # Slight reduction in latent heat
        latent_heat_reduction = 1.0 - 0.008 * weight_percent  # 0.8% reduction per wt%
        self.mixture_properties['latent_heat'] *= latent_heat_reduction
        
        # Slight increase in density
        density_increase = 1.0 + 0.004 * weight_percent  # 0.4% increase per wt%
        self.mixture_properties['density_solid'] *= density_increase
        self.mixture_properties['density_liquid'] *= density_increase
        
        # Update description
        self.mixture_properties['description'] = f"{self.base_properties['description']} + {weight_percent}wt% Graphene"
        
        logger.info(f"Added {weight_percent}wt% graphene to PCM mixture")
        logger.info(f"Thermal conductivity increased from {k_base:.2f} to {k_eff:.2f} W/(m*K)")
        
        return self.mixture_properties
    
    def add_metal_particles(self, metal_type='aluminum', weight_percent=5.0):
        """
        Add metal particles to enhance thermal conductivity.
        
        Parameters
        ----------
        metal_type : str, optional
            Type of metal ('aluminum', 'copper', 'silver')
        weight_percent : float, optional
            Weight percentage of metal particles
        
        Returns
        -------
        dict
            Updated mixture properties
        """
        if weight_percent <= 0 or weight_percent >= 100:
            logger.warning("Weight percentage must be between 0 and 100")
            return self.mixture_properties
        
        # Metal properties
        metal_properties = {
            'aluminum': {
                'thermal_conductivity': 237.0,  # W/(m*K)
                'density': 2.7  # g/cm³
            },
            'copper': {
                'thermal_conductivity': 401.0,  # W/(m*K)
                'density': 8.96  # g/cm³
            },
            'silver': {
                'thermal_conductivity': 429.0,  # W/(m*K)
                'density': 10.49  # g/cm³
            }
        }
        
        if metal_type not in metal_properties:
            logger.warning(f"Unknown metal type: {metal_type}")
            return self.mixture_properties
        
        # Add to additives list
        self.additives.append({
            'type': f'{metal_type}_particles',
            'weight_percent': weight_percent
        })
        
        # Update properties
        # Thermal conductivity enhancement based on literature
        k_base = self.mixture_properties['thermal_conductivity']
        k_metal = metal_properties[metal_type]['thermal_conductivity']
        
        # Simple effective medium approximation
        vol_frac = weight_percent / 100 * (self.base_properties['density_solid'] / metal_properties[metal_type]['density'])
        k_eff = k_base * (1 + 3 * vol_frac * (k_metal - k_base) / (k_metal + 2 * k_base))
        
        # Maximum practical enhancement factor (limited by percolation and dispersion)
        max_enhancement = 3.0  # typical max enhancement for metal particles
        if k_eff / k_base > max_enhancement:
            k_eff = k_base * max_enhancement
        
        self.mixture_properties['thermal_conductivity'] = k_eff
        
        # Reduction in latent heat
        latent_heat_reduction = 1.0 - 0.015 * weight_percent  # 1.5% reduction per wt%
        self.mixture_properties['latent_heat'] *= latent_heat_reduction
        
        # Increase in density
        density_increase = 1.0 + 0.02 * weight_percent  # 2% increase per wt%
        self.mixture_properties['density_solid'] *= density_increase
        self.mixture_properties['density_liquid'] *= density_increase
        
        # Update description
        self.mixture_properties['description'] = f"{self.base_properties['description']} + {weight_percent}wt% {metal_type.capitalize()} Particles"
        
        logger.info(f"Added {weight_percent}wt% {metal_type} particles to PCM mixture")
        logger.info(f"Thermal conductivity increased from {k_base:.2f} to {k_eff:.2f} W/(m*K)")
        
        return self.mixture_properties
    
    def adjust_melting_point(self, target_temp):
        """
        Adjust the melting point of the PCM mixture.
        
        Parameters
        ----------
        target_temp : float
            Target melting temperature in °C
        
        Returns
        -------
        dict
            Updated mixture properties
        """
        original_temp = self.mixture_properties['melting_temp']
        
        if abs(target_temp - original_temp) > 15:
            logger.warning(f"Requested temperature adjustment ({target_temp}°C) is too large")
            logger.warning("Consider using a different base PCM")
            return self.mixture_properties
        
        # Add adjustment to additives list
        self.additives.append({
            'type': 'melting_point_adjuster',
            'target_temp': target_temp
        })
        
        # Update properties
        self.mixture_properties['melting_temp'] = target_temp
        
        # Update description
        self.mixture_properties['description'] += f", melting point adjusted to {target_temp}°C"
        
        logger.info(f"Adjusted melting point from {original_temp}°C to {target_temp}°C")
        
        return self.mixture_properties
    
    def get_mixture_properties(self):
        """
        Get current mixture properties.
        
        Returns
        -------
        dict
            Mixture properties
        """
        return self.mixture_properties.copy()
    
    def get_additives(self):
        """
        Get list of additives.
        
        Returns
        -------
        list
            List of additives
        """
        return self.additives.copy()
    
    def create_custom_pcm(self):
        """
        Create a custom PCM type with the current mixture properties.
        
        Returns
        -------
        dict
            Custom PCM properties for use with ThermalManager
        """
        return self.mixture_properties.copy()"""