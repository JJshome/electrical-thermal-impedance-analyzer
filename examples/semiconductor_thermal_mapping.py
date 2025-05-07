"""
Semiconductor Thermal Mapping Example

This example demonstrates how to use the Integrated Electrical-Thermal Impedance Analyzer
for semiconductor thermal mapping and fault detection.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Reference work:
Blackburn, D.L., "Temperature measurements of semiconductor devices-a review",
Semiconductor Thermal Measurement and Management Symposium, 2004
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Add parent directory to path to import the analyzer module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from software.acquisition.integrated_impedance_analyzer import IntegratedImpedanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SemiconductorThermalMapping")

class SemiconductorDevice:
    """
    Simplified semiconductor device model for demonstration purposes
    
    This class simulates various types of semiconductor devices with
    different thermal and electrical characteristics, including fault conditions.
    """
    
    # Define device types and their base parameters
    DEVICE_TYPES = {
        'cpu': {
            'thermal_conductivity': 150,      # W/(m·K) - bulk silicon
            'electrical_conductivity': 1e3,    # S/m
            'specific_heat': 700,             # J/(kg·K)
            'density': 2330,                  # kg/m³
            'die_size': (10e-3, 10e-3, 0.5e-3),  # m (length, width, thickness)
            'power_density': 100e6,           # W/m³
            'number_of_cores': 8,
            'transistor_count': 5e9,
            'feature_size': 7e-9,             # 7 nm
            'max_frequency': 4e9,             # 4 GHz
            'max_temperature': 100,           # °C
        },
        'gpu': {
            'thermal_conductivity': 150,      # W/(m·K) - bulk silicon
            'electrical_conductivity': 1e3,    # S/m
            'specific_heat': 700,             # J/(kg·K)
            'density': 2330,                  # kg/m³
            'die_size': (15e-3, 15e-3, 0.5e-3),  # m (length, width, thickness)
            'power_density': 120e6,           # W/m³
            'number_of_cores': 3840,          # CUDA cores
            'transistor_count': 10e9,
            'feature_size': 7e-9,             # 7 nm
            'max_frequency': 2e9,             # 2 GHz
            'max_temperature': 95,            # °C
        },
        'memory': {
            'thermal_conductivity': 150,      # W/(m·K) - bulk silicon
            'electrical_conductivity': 1e3,    # S/m
            'specific_heat': 700,             # J/(kg·K)
            'density': 2330,                  # kg/m³
            'die_size': (12e-3, 6e-3, 0.4e-3),  # m (length, width, thickness)
            'power_density': 30e6,            # W/m³
            'transistor_count': 8e9,
            'feature_size': 10e-9,            # 10 nm
            'max_frequency': 3.2e9,           # 3.2 GHz
            'max_temperature': 85,            # °C
        },
        'power_device': {
            'thermal_conductivity': 450,      # W/(m·K) - SiC
            'electrical_conductivity': 5e2,    # S/m
            'specific_heat': 750,             # J/(kg·K)
            'density': 3210,                  # kg/m³
            'die_size': (5e-3, 5e-3, 1e-3),   # m (length, width, thickness)
            'power_density': 200e6,           # W/m³
            'max_blocking_voltage': 1200,     # V
            'max_current_density': 250,       # A/cm²
            'feature_size': 1e-6,             # 1 μm
            'max_frequency': 100e3,           # 100 kHz
            'max_temperature': 150,           # °C
        }
    }
    
    # Define fault types and their characteristics
    FAULT_TYPES = {
        'none': {
            'description': 'No fault',
            'thermal_effect': 1.0,  # Multiplier for thermal conductivity
            'electrical_effect': 1.0,  # Multiplier for electrical conductivity
            'location': None,  # No specific location
        },
        'hotspot': {
            'description': 'Thermal hotspot due to excessive power density',
            'thermal_effect': 0.7,  # Reduced thermal conductivity
            'electrical_effect': 1.0,  # No change in electrical conductivity
            'location': 'random',  # Will be set randomly
        },
        'crack': {
            'description': 'Die crack or packaging damage',
            'thermal_effect': 0.4,  # Significantly reduced thermal conductivity
            'electrical_effect': 0.6,  # Reduced electrical conductivity
            'location': 'edge',  # Always at the edge of the die
        },
        'delamination': {
            'description': 'Delamination between die and package',
            'thermal_effect': 0.2,  # Severely reduced thermal conductivity
            'electrical_effect': 1.0,  # No change in electrical conductivity
            'location': 'corner',  # Always at a corner
        },
        'void': {
            'description': 'Void in die attach material',
            'thermal_effect': 0.1,  # Very poor thermal conductivity
            'electrical_effect': 1.0,  # No change in electrical conductivity
            'location': 'center',  # Usually near the center
        }
    }
    
    def __init__(self, 
                 device_type='cpu',
                 operating_power=50.0,        # W
                 ambient_temperature=25.0,    # °C
                 fault_type='none',
                 grid_size=(10, 10)):         # Spatial resolution for thermal mapping
        """
        Initialize SemiconductorDevice model
        
        Parameters:
        -----------
        device_type : str
            Type of device ('cpu', 'gpu', 'memory', 'power_device')
        operating_power : float
            Device operating power in watts
        ambient_temperature : float
            Ambient temperature in °C
        fault_type : str
            Type of fault ('none', 'hotspot', 'crack', 'delamination', 'void')
        grid_size : tuple
            Resolution for thermal and electrical mapping (rows, cols)
        """
        self.device_type = device_type
        self.operating_power = operating_power
        self.ambient_temperature = ambient_temperature
        self.fault_type = fault_type
        self.grid_size = grid_size
        
        # Get base parameters for this device type
        if device_type not in self.DEVICE_TYPES:
            raise ValueError(f"Unknown device type: {device_type}")
        self.base_params = self.DEVICE_TYPES[device_type].copy()
        
        # Get fault characteristics
        if fault_type not in self.FAULT_TYPES:
            raise ValueError(f"Unknown fault type: {fault_type}")
        self.fault_params = self.FAULT_TYPES[fault_type].copy()
        
        # Set fault location based on fault type
        self._set_fault_location()
        
        # Initialize temperature and impedance maps
        self._initialize_maps()
        
        # Update parameters based on initial state
        self._update_parameters()
        
    def _set_fault_location(self):
        """Set the location of the fault based on fault type"""
        rows, cols = self.grid_size
        
        if self.fault_type == 'none':
            self.fault_location = None
        elif self.fault_params['location'] == 'random':
            # Random location excluding the edges
            r = np.random.randint(1, rows-1)
            c = np.random.randint(1, cols-1)
            self.fault_location = (r, c)
        elif self.fault_params['location'] == 'edge':
            # Random edge location
            if np.random.choice([True, False]):  # Horizontal or vertical edge
                r = np.random.choice([0, rows-1])
                c = np.random.randint(0, cols)
            else:
                r = np.random.randint(0, rows)
                c = np.random.choice([0, cols-1])
            self.fault_location = (r, c)
        elif self.fault_params['location'] == 'corner':
            # Random corner
            r = np.random.choice([0, rows-1])
            c = np.random.choice([0, cols-1])
            self.fault_location = (r, c)
        elif self.fault_params['location'] == 'center':
            # Near center
            center_r, center_c = rows // 2, cols // 2
            r = center_r + np.random.randint(-1, 2)
            c = center_c + np.random.randint(-1, 2)
            self.fault_location = (r, c)
            
    def _initialize_maps(self):
        """Initialize thermal and electrical mapping arrays"""
        rows, cols = self.grid_size
        
        # Physical dimensions from die size
        length, width, _ = self.base_params['die_size']
        self.dx = length / cols
        self.dy = width / rows
        
        # Initialize maps
        self.thermal_conductivity_map = np.ones((rows, cols)) * self.base_params['thermal_conductivity']
        self.electrical_conductivity_map = np.ones((rows, cols)) * self.base_params['electrical_conductivity']
        self.power_density_map = np.ones((rows, cols)) * self.base_params['power_density']
        self.temperature_map = np.ones((rows, cols)) * self.ambient_temperature
        
        # Initialize impedance maps
        self.electrical_impedance_map = np.zeros((rows, cols), dtype=complex)
        self.thermal_impedance_map = np.zeros((rows, cols), dtype=complex)
        
        # Apply fault effects
        if self.fault_type != 'none':
            self._apply_fault_effects()
    
    def _apply_fault_effects(self):
        """Apply the effects of the fault to the maps"""
        if self.fault_location is not None:
            r, c = self.fault_location
            rows, cols = self.grid_size
            
            # Apply fault directly at fault location
            self.thermal_conductivity_map[r, c] *= self.fault_params['thermal_effect']
            self.electrical_conductivity_map[r, c] *= self.fault_params['electrical_effect']
            
            # Create a fault influence map (fault has maximum effect at center, diminishing with distance)
            for i in range(rows):
                for j in range(cols):
                    # Calculate distance from fault (normalized to 0-1 range)
                    distance = np.sqrt(((i - r) / rows)**2 + ((j - c) / cols)**2)
                    
                    # Apply diminishing effect based on distance
                    if distance > 0:  # Don't modify the center again
                        thermal_factor = 1.0 - (1.0 - self.fault_params['thermal_effect']) * np.exp(-5 * distance)
                        electrical_factor = 1.0 - (1.0 - self.fault_params['electrical_effect']) * np.exp(-5 * distance)
                        
                        self.thermal_conductivity_map[i, j] *= thermal_factor
                        self.electrical_conductivity_map[i, j] *= electrical_factor
    
    def _update_parameters(self):
        """
        Update device parameters based on current state
        
        This method simulates heat distribution and calculates electrical
        and thermal impedances across the device.
        """
        rows, cols = self.grid_size
        
        # Scale power density based on operating power
        total_volume = self.base_params['die_size'][0] * self.base_params['die_size'][1] * self.base_params['die_size'][2]
        normalized_power = self.operating_power / (self.base_params['power_density'] * total_volume)
        
        # Create power density distribution (cores/functional blocks)
        if self.device_type == 'cpu' or self.device_type == 'gpu':
            # Simulate cores/compute units as hotspots
            num_cores = min(self.base_params.get('number_of_cores', 1), rows * cols // 4)
            
            # Reset power density map
            self.power_density_map = np.ones((rows, cols)) * self.base_params['power_density'] * 0.2  # Base power
            
            # Randomly place cores/compute units
            if self.device_type == 'cpu':
                # CPUs typically have fewer, larger cores
                core_positions = []
                for _ in range(self.base_params.get('number_of_cores', 8)):
                    r = np.random.randint(1, rows-1)
                    c = np.random.randint(1, cols-1)
                    core_positions.append((r, c))
                    
                for r, c in core_positions:
                    # Apply a Gaussian distribution for each core
                    for i in range(rows):
                        for j in range(cols):
                            distance = np.sqrt(((i - r) / 2)**2 + ((j - c) / 2)**2)
                            core_power = np.exp(-distance**2) * self.base_params['power_density'] * 2
                            self.power_density_map[i, j] += core_power / len(core_positions)
            else:
                # GPUs have many small compute units
                for _ in range(min(100, num_cores)):  # Cap at 100 for computational reasons
                    r = np.random.randint(0, rows)
                    c = np.random.randint(0, cols)
                    self.power_density_map[r, c] = self.base_params['power_density'] * 1.5
                    
        elif self.device_type == 'memory':
            # Memory typically has more uniform power distribution with bank structure
            bank_size = max(1, min(rows, cols) // 4)
            for i in range(0, rows, bank_size):
                for j in range(0, cols, bank_size):
                    bank_power = np.random.uniform(0.8, 1.2) * self.base_params['power_density']
                    for bi in range(bank_size):
                        for bj in range(bank_size):
                            if i+bi < rows and j+bj < cols:
                                self.power_density_map[i+bi, j+bj] = bank_power
                                
        elif self.device_type == 'power_device':
            # Power devices typically have edge effects with current crowding
            center_r, center_c = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    # Higher power density near edges (current crowding)
                    edge_distance = min(i, j, rows-i-1, cols-j-1) / min(rows, cols)
                    center_distance = np.sqrt(((i - center_r) / rows)**2 + ((j - center_c) / cols)**2)
                    
                    # Combine effects: higher at center and edges
                    power_factor = 1.0 + 0.5 * np.exp(-5 * center_distance) + 0.3 * np.exp(-10 * edge_distance)
                    self.power_density_map[i, j] = self.base_params['power_density'] * power_factor
        
        # Scale to match total operating power
        self.power_density_map *= normalized_power
        
        # Calculate temperature distribution (simplified heat equation)
        # In real implementation, this would use a more sophisticated thermal model
        self._calculate_temperature_distribution()
        
        # Calculate electrical and thermal impedance maps
        self._calculate_impedance_maps()
        
        # Calculate overall device characteristics
        self._calculate_device_characteristics()
        
    def _calculate_temperature_distribution(self):
        """
        Calculate temperature distribution using simplified heat equation solver
        
        This uses a basic finite difference method for demonstration purposes.
        A real implementation would use a more sophisticated thermal model.
        """
        rows, cols = self.grid_size
        
        # Initialize temperature map to ambient
        self.temperature_map = np.ones((rows, cols)) * self.ambient_temperature
        
        # Material properties
        rho = self.base_params['density']  # kg/m³
        cp = self.base_params['specific_heat']  # J/(kg·K)
        
        # Time stepping parameters
        dt = 0.01  # Time step [s]
        max_time = 5.0  # Simulate until steady state [s]
        alpha_min = np.min(self.thermal_conductivity_map) / (rho * cp)  # Thermal diffusivity
        
        # Stability condition for explicit method
        dx_min = min(self.dx, self.dy)
        dt = min(dt, 0.25 * dx_min**2 / alpha_min)  # CFL condition
        
        # Simplified steady-state solver
        # In a real implementation, use an efficient sparse matrix solver
        for t in np.arange(0, max_time, dt):
            temperature_new = self.temperature_map.copy()
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    k = self.thermal_conductivity_map[i, j]
                    q = self.power_density_map[i, j]
                    
                    # Conduction from neighboring cells (5-point stencil)
                    conduction_x = (self.temperature_map[i+1, j] - 2*self.temperature_map[i, j] + self.temperature_map[i-1, j]) / self.dx**2
                    conduction_y = (self.temperature_map[i, j+1] - 2*self.temperature_map[i, j] + self.temperature_map[i, j-1]) / self.dy**2
                    
                    # Update temperature
                    dT_dt = (k * (conduction_x + conduction_y) + q) / (rho * cp)
                    temperature_new[i, j] = self.temperature_map[i, j] + dt * dT_dt
            
            # Apply boundary conditions
            temperature_new[0, :] = self.ambient_temperature
            temperature_new[-1, :] = self.ambient_temperature
            temperature_new[:, 0] = self.ambient_temperature
            temperature_new[:, -1] = self.ambient_temperature
            
            # Update temperature map
            self.temperature_map = temperature_new
            
            # Check for convergence
            if np.max(np.abs(temperature_new - self.temperature_map)) < 0.01:
                break
                
    def _calculate_impedance_maps(self):
        """
        Calculate electrical and thermal impedance maps
        
        This is a simplified model for demonstration. A real implementation
        would use more sophisticated methods for impedance calculation.
        """
        rows, cols = self.grid_size
        thickness = self.base_params['die_size'][2]  # m
        
        # Calculate electrical impedance map
        for i in range(rows):
            for j in range(cols):
                # Simplified electrical impedance model (RC network)
                conductivity = self.electrical_conductivity_map[i, j]  # S/m
                resistance = thickness / (conductivity * self.dx * self.dy)  # Ohm
                
                # Temperature effect on resistance
                temperature_factor = 1.0 + 0.004 * (self.temperature_map[i, j] - self.ambient_temperature)
                resistance *= temperature_factor
                
                # Simple RC model with frequency-dependent phase
                capacitance = 1e-12  # F (typical parasitic capacitance)
                reactance = -1j / (2 * np.pi * 1e3 * capacitance)  # Xc at 1kHz
                
                self.electrical_impedance_map[i, j] = resistance + reactance
        
        # Calculate thermal impedance map
        for i in range(rows):
            for j in range(cols):
                # Simplified thermal impedance model
                conductivity = self.thermal_conductivity_map[i, j]  # W/(m·K)
                resistance = thickness / (conductivity * self.dx * self.dy)  # K/W
                
                # Simplified thermal capacitance
                rho = self.base_params['density']  # kg/m³
                cp = self.base_params['specific_heat']  # J/(kg·K)
                volume = self.dx * self.dy * thickness  # m³
                capacitance = rho * cp * volume  # J/K
                
                # Simple RC model with frequency-dependent phase
                # Using 0.1 Hz as reference frequency for thermal response
                reactance = -1j / (2 * np.pi * 0.1 * capacitance)
                
                self.thermal_impedance_map[i, j] = resistance + reactance
                
    def _calculate_device_characteristics(self):
        """Calculate overall device characteristics from the impedance maps"""
        # Average temperature
        self.average_temperature = np.mean(self.temperature_map)
        
        # Maximum temperature
        self.max_temperature = np.max(self.temperature_map)
        self.max_temperature_location = np.unravel_index(
            np.argmax(self.temperature_map), self.temperature_map.shape
        )
        
        # Total thermal resistance (junction-to-ambient)
        self.thermal_resistance_ja = (self.max_temperature - self.ambient_temperature) / self.operating_power
        
        # Average electrical resistance
        self.average_electrical_resistance = np.mean(np.real(self.electrical_impedance_map))
        
        # Average thermal impedance
        self.average_thermal_impedance = np.mean(np.real(self.thermal_impedance_map))
        
        # Calculate thermal uniformity (standard deviation of temperature)
        self.thermal_uniformity = np.std(self.temperature_map)
        
        # Detect thermal hotspots (areas above 90% of the max temperature rise)
        temp_rise = self.temperature_map - self.ambient_temperature
        max_rise = np.max(temp_rise)
        hotspot_threshold = 0.9 * max_rise
        self.hotspot_map = temp_rise > hotspot_threshold
        self.hotspot_count = np.sum(self.hotspot_map)
        
        # Calculate junction-to-case thermal resistance (simplified)
        self.thermal_resistance_jc = self.thermal_resistance_ja * 0.7  # Typical ratio
        
        # Electrical-thermal correlation (simplified)
        flattened_e = np.real(self.electrical_impedance_map).flatten()
        flattened_t = np.real(self.thermal_impedance_map).flatten()
        self.e_t_correlation = np.corrcoef(flattened_e, flattened_t)[0, 1]
        
    def set_operating_power(self, power):
        """
        Set device operating power
        
        Parameters:
        -----------
        power : float
            Operating power in watts
        """
        self.operating_power = power
        self._update_parameters()
        logger.info(f"Operating power set to {self.operating_power:.1f} W")
        
    def set_ambient_temperature(self, temperature):
        """
        Set ambient temperature
        
        Parameters:
        -----------
        temperature : float
            Ambient temperature in °C
        """
        self.ambient_temperature = temperature
        self._update_parameters()
        logger.info(f"Ambient temperature set to {self.ambient_temperature:.1f}°C")
        
    def set_fault(self, fault_type):
        """
        Set fault type
        
        Parameters:
        -----------
        fault_type : str
            Type of fault ('none', 'hotspot', 'crack', 'delamination', 'void')
        """
        if fault_type not in self.FAULT_TYPES:
            raise ValueError(f"Unknown fault type: {fault_type}")
            
        self.fault_type = fault_type
        self.fault_params = self.FAULT_TYPES[fault_type].copy()
        
        # Reset maps
        self._initialize_maps()
        self._update_parameters()
        
        logger.info(f"Fault type set to '{self.fault_type}'")
        if self.fault_location is not None:
            logger.info(f"Fault location: {self.fault_location}")
            
    def __str__(self):
        """String representation of device state"""
        return (f"{self.device_type.upper()} Device:\n"
                f"  Operating Power: {self.operating_power:.1f} W\n"
                f"  Ambient Temperature: {self.ambient_temperature:.1f}°C\n"
                f"  Fault Type: {self.fault_type}\n"
                f"  Temperature Statistics:\n"
                f"    Average: {self.average_temperature:.1f}°C\n"
                f"    Maximum: {self.max_temperature:.1f}°C at {self.max_temperature_location}\n"
                f"    Uniformity (Std Dev): {self.thermal_uniformity:.1f}°C\n"
                f"    Hotspot Count: {self.hotspot_count}\n"
                f"  Thermal Characteristics:\n"
                f"    R_ja: {self.thermal_resistance_ja:.2f} K/W\n"
                f"    R_jc: {self.thermal_resistance_jc:.2f} K/W\n"
                f"  Electrical-Thermal Correlation: {self.e_t_correlation:.2f}")


def perform_device_analysis(device, analyzer, plot_results=True):
    """
    Perform comprehensive device analysis using integrated impedance analyzer
    
    Parameters:
    -----------
    device : SemiconductorDevice
        Device to analyze
    analyzer : IntegratedImpedanceAnalyzer
        Configured impedance analyzer
    plot_results : bool
        Whether to plot results
        
    Returns:
    --------
    analysis_results : dict
        Dictionary containing analysis results
    """
    logger.info(f"Starting comprehensive analysis of {device.device_type} device")
    
    # Measure impedance
    logger.info("Measuring impedance spectra")
    measurement_results = analyzer.measure(target_system=device)
    
    # Analyze results
    logger.info("Analyzing impedance data")
    characteristics = analyzer.analyze(measurement_results)
    
    # Extract key parameters
    e_params = characteristics['electrical_parameters']
    t_params = characteristics['thermal_parameters']
    i_params = characteristics['integrated_parameters']
    
    # Calculate device-specific metrics
    # Estimate maximum allowable power based on thermal characteristics
    max_temp = device.base_params['max_temperature']
    power_limit_thermal = (max_temp - device.ambient_temperature) / t_params['R_th']
    
    # Calculate thermal efficiency (actual vs ideal heat dissipation)
    ideal_thermal_resistance = device.base_params['die_size'][2] / (
        device.base_params['thermal_conductivity'] * 
        device.base_params['die_size'][0] * 
        device.base_params['die_size'][1]
    )
    thermal_efficiency = ideal_thermal_resistance / t_params['R_th'] * 100.0
    
    # Calculate electrical efficiency (based on impedance)
    electrical_efficiency = 100.0 / (1.0 + e_params['R_total'] * device.operating_power / 1.0)  # Assuming 1V operation
    
    # Detect fault probability and type
    if device.fault_type == 'none':
        fault_probability = i_params['anomaly_score'] * 100
        
        # Determine most likely fault type based on characteristics
        if t_params['thermal_uniformity'] > 0.3:  # High non-uniformity
            if i_params['e_t_correlation'] < 0.5:
                predicted_fault = 'hotspot'
            else:
                predicted_fault = 'crack'
        elif t_params['R_th'] > 1.5 * ideal_thermal_resistance:  # High thermal resistance
            if t_params['thermal_uniformity'] < 0.2:
                predicted_fault = 'delamination'
            else:
                predicted_fault = 'void'
        else:
            predicted_fault = 'none'
    else:
        # For demonstration, when we know the fault
        fault_probability = 95.0
        predicted_fault = device.fault_type
    
    # Combine all results
    analysis_results = {
        'timestamp': datetime.now(),
        'device_info': {
            'type': device.device_type,
            'operating_power': device.operating_power,
            'ambient_temperature': device.ambient_temperature,
            'fault_type': device.fault_type,
            'fault_location': device.fault_location
        },
        'temperature_map': device.temperature_map.copy(),
        'electrical_impedance_map': device.electrical_impedance_map.copy(),
        'thermal_impedance_map': device.thermal_impedance_map.copy(),
        'measured_characteristics': characteristics,
        'derived_metrics': {
            'power_limit_thermal': power_limit_thermal,
            'thermal_efficiency': thermal_efficiency,
            'electrical_efficiency': electrical_efficiency,
            'fault_probability': fault_probability,
            'predicted_fault': predicted_fault
        }
    }
    
    # Print summary
    print("\nDevice Analysis Results:")
    print("=======================")
    print(f"Device: {device.device_type.upper()}, Power: {device.operating_power:.1f}W")
    print(f"Ambient Temperature: {device.ambient_temperature:.1f}°C")
    print(f"Actual Fault: {device.fault_type}")
    
    print("\nTemperature Profile:")
    print(f"  Average Temperature: {device.average_temperature:.1f}°C")
    print(f"  Maximum Temperature: {device.max_temperature:.1f}°C")
    print(f"  Junction-to-Ambient Thermal Resistance: {t_params['R_th']:.2f} K/W")
    print(f"  Thermal Uniformity: {t_params['thermal_uniformity']:.2f}°C")
    
    print("\nElectrical Characteristics:")
    print(f"  Total Resistance: {e_params['R_total']:.4f} Ohm")
    print(f"  Characteristic Frequency: {e_params['characteristic_frequency']:.1f} Hz")
    
    print("\nFault Analysis:")
    print(f"  Fault Probability: {fault_probability:.1f}%")
    print(f"  Predicted Fault Type: {predicted_fault}")
    print(f"  Electrical-Thermal Correlation: {i_params['electrical_thermal_correlation']:.2f}")
    
    print("\nOperational Limits:")
    print(f"  Maximum Allowable Power (Thermal): {power_limit_thermal:.1f} W")
    print(f"  Thermal Efficiency: {thermal_efficiency:.1f}%")
    print(f"  Electrical Efficiency: {electrical_efficiency:.1f}%")
    
    # Plot if requested
    if plot_results:
        analyzer.plot_impedance_spectra(measurement_results)
        plot_device_characteristics(analysis_results)
    
    return analysis_results


def plot_device_characteristics(analysis_results):
    """
    Create visualizations of device characteristics
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results
    """
    # Extract data
    device_info = analysis_results['device_info']
    temperature_map = analysis_results['temperature_map']
    e_impedance_map = analysis_results['electrical_impedance_map']
    t_impedance_map = analysis_results['thermal_impedance_map']
    e_params = analysis_results['measured_characteristics']['electrical_parameters']
    t_params = analysis_results['measured_characteristics']['thermal_parameters']
    i_params = analysis_results['measured_characteristics']['integrated_parameters']
    derived = analysis_results['derived_metrics']
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Device Characterization: {device_info['type'].upper()}", fontsize=16)
    
    # Plot 1: Temperature Map
    ax1 = axs[0, 0]
    im1 = ax1.imshow(temperature_map, cmap='hot', interpolation='bilinear')
    ax1.set_title('Temperature Distribution (°C)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1)
    
    # Mark fault location if present
    if device_info['fault_location'] is not None:
        r, c = device_info['fault_location']
        ax1.plot(c, r, 'ko', markersize=10, markerfacecolor='none')
        ax1.text(c+0.5, r+0.5, 'Fault', color='white', fontweight='bold')
    
    # Plot 2: Thermal Impedance Map
    ax2 = axs[0, 1]
    # Use real part of thermal impedance
    thermal_impedance_real = np.real(t_impedance_map)
    im2 = ax2.imshow(thermal_impedance_real, cmap='viridis', interpolation='bilinear')
    ax2.set_title('Thermal Resistance (K/W)')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Electrical-Thermal Correlation
    ax3 = axs[1, 0]
    # Flatten maps
    e_flat = np.real(e_impedance_map).flatten()
    t_flat = np.real(t_impedance_map).flatten()
    
    # Create scatter plot
    scatter = ax3.scatter(e_flat, t_flat, c=temperature_map.flatten(), cmap='hot', alpha=0.7)
    ax3.set_xlabel('Electrical Resistance (Ohm)')
    ax3.set_ylabel('Thermal Resistance (K/W)')
    ax3.set_title('Electrical-Thermal Correlation')
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='Temperature (°C)')
    
    # Add correlation text
    corr = i_params['electrical_thermal_correlation']
    ax3.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax3.transAxes,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Plot 4: Fault Analysis Dashboard
    ax4 = axs[1, 1]
    
    # Create bar chart for fault probability
    if derived['predicted_fault'] == 'none':
        fault_label = 'No Fault'
        fault_color = 'green'
    else:
        fault_label = derived['predicted_fault'].capitalize()
        fault_color = 'red'
    
    ax4.bar(['Fault Probability'], [derived['fault_probability']], color=fault_color)
    ax4.set_ylabel('Probability (%)')
    ax4.set_ylim(0, 100)
    ax4.axhline(y=50, color='gray', linestyle='--')
    ax4.set_title('Fault Analysis')
    
    # Add fault type text
    ax4.text(0.5, 0.5, f"Detected: {fault_label}\nActual: {device_info['fault_type'].capitalize()}",
             transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add operational metrics
    metrics_text = (f"Thermal Limit: {derived['power_limit_thermal']:.1f}W\n"
                    f"Current Power: {device_info['operating_power']:.1f}W\n"
                    f"Thermal Efficiency: {derived['thermal_efficiency']:.1f}%\n"
                    f"Max Temperature: {np.max(temperature_map):.1f}°C")
    
    ax4.text(0.5, 0.2, metrics_text, transform=ax4.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def power_sweep_study(device, analyzer, plot=True):
    """
    Conduct a power sweep study
    
    This function simulates device behavior across different power levels
    and analyzes the thermal and electrical characteristics.
    
    Parameters:
    -----------
    device : SemiconductorDevice
        Device to analyze
    analyzer : IntegratedImpedanceAnalyzer
        Configured impedance analyzer
    plot : bool
        Whether to plot the results
        
    Returns:
    --------
    study_data : pd.DataFrame
        DataFrame containing study results
    """
    logger.info("Starting power sweep study")
    
    # Define power levels to test (as percentage of device max power)
    max_power = (device.base_params['max_temperature'] - device.ambient_temperature) / 0.5  # Estimate based on typical Rth
    power_levels = np.linspace(0.1, 1.0, 10) * max_power
    
    # Storage for results
    results = []
    
    # Perform measurements for each power level
    for power in power_levels:
        logger.info(f"Testing power level: {power:.1f} W")
        
        # Set device power
        device.set_operating_power(power)
        
        # Analyze device
        analysis = perform_device_analysis(device, analyzer, plot_results=False)
        
        # Store results
        results.append({
            'power': power,
            'max_temperature': np.max(analysis['temperature_map']),
            'avg_temperature': np.mean(analysis['temperature_map']),
            'thermal_uniformity': analysis['measured_characteristics']['thermal_parameters']['thermal_uniformity'],
            'thermal_resistance': analysis['measured_characteristics']['thermal_parameters']['R_th'],
            'electrical_resistance': analysis['measured_characteristics']['electrical_parameters']['R_total'],
            'electrical_thermal_correlation': analysis['measured_characteristics']['integrated_parameters']['electrical_thermal_correlation'],
            'thermal_efficiency': analysis['derived_metrics']['thermal_efficiency']
        })
        
        logger.info(f"Power {power:.1f}W: Max Temp = {results[-1]['max_temperature']:.1f}°C, "
                   f"R_th = {results[-1]['thermal_resistance']:.2f} K/W")
    
    # Convert to DataFrame
    study_data = pd.DataFrame(results)
    
    # Plot results
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Temperature vs Power
        ax1 = axs[0, 0]
        ax1.plot(study_data['power'], study_data['max_temperature'], 'ro-', label='Maximum')
        ax1.plot(study_data['power'], study_data['avg_temperature'], 'bo-', label='Average')
        ax1.set_xlabel('Power (W)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature vs Power')
        ax1.axhline(y=device.base_params['max_temperature'], color='r', linestyle='--', 
                   label=f"Max Limit ({device.base_params['max_temperature']}°C)")
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Thermal Resistance vs Power
        ax2 = axs[0, 1]
        ax2.plot(study_data['power'], study_data['thermal_resistance'], 'go-')
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Thermal Resistance (K/W)')
        ax2.set_title('Thermal Resistance vs Power')
        ax2.grid(True)
        
        # Add thermal uniformity as second axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(study_data['power'], study_data['thermal_uniformity'], 'mo-', label='Uniformity')
        ax2_twin.set_ylabel('Thermal Uniformity (°C)', color='m')
        ax2_twin.tick_params(axis='y', labelcolor='m')
        
        # Plot 3: Electrical Resistance vs Power
        ax3 = axs[1, 0]
        ax3.plot(study_data['power'], study_data['electrical_resistance'], 'ko-')
        ax3.set_xlabel('Power (W)')
        ax3.set_ylabel('Electrical Resistance (Ohm)')
        ax3.set_title('Electrical Resistance vs Power')
        ax3.grid(True)
        
        # Plot 4: Efficiency vs Power
        ax4 = axs[1, 1]
        ax4.plot(study_data['power'], study_data['thermal_efficiency'], 'co-')
        ax4.set_xlabel('Power (W)')
        ax4.set_ylabel('Thermal Efficiency (%)')
        ax4.set_title('Thermal Efficiency vs Power')
        ax4.grid(True)
        
        # Add correlation as second axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(study_data['power'], study_data['electrical_thermal_correlation'], 'yo-')
        ax4_twin.set_ylabel('E-T Correlation', color='y')
        ax4_twin.tick_params(axis='y', labelcolor='y')
        
        plt.tight_layout()
        plt.show()
        
        # Create animation of temperature map evolution
        create_temperature_evolution_animation(device, power_levels)
    
    return study_data


def fault_detection_study(device, analyzer, plot=True):
    """
    Conduct a fault detection study
    
    This function simulates and analyzes different fault types
    to evaluate detection capabilities.
    
    Parameters:
    -----------
    device : SemiconductorDevice
        Device to analyze (will be modified with different faults)
    analyzer : IntegratedImpedanceAnalyzer
        Configured impedance analyzer
    plot : bool
        Whether to plot the results
        
    Returns:
    --------
    study_data : pd.DataFrame
        DataFrame containing study results
    """
    logger.info("Starting fault detection study")
    
    # Define fault types to test
    fault_types = ['none', 'hotspot', 'crack', 'delamination', 'void']
    
    # Storage for results
    results = []
    all_temp_maps = {}
    
    # Perform measurements for each fault type
    for fault_type in fault_types:
        logger.info(f"Testing fault type: {fault_type}")
        
        # Set device fault
        device.set_fault(fault_type)
        
        # Analyze device
        analysis = perform_device_analysis(device, analyzer, plot_results=False)
        
        # Store results
        results.append({
            'fault_type': fault_type,
            'max_temperature': np.max(analysis['temperature_map']),
            'thermal_uniformity': analysis['measured_characteristics']['thermal_parameters']['thermal_uniformity'],
            'thermal_resistance': analysis['measured_characteristics']['thermal_parameters']['R_th'],
            'electrical_resistance': analysis['measured_characteristics']['electrical_parameters']['R_total'],
            'electrical_thermal_correlation': analysis['measured_characteristics']['integrated_parameters']['electrical_thermal_correlation'],
            'predicted_fault': analysis['derived_metrics']['predicted_fault'],
            'fault_probability': analysis['derived_metrics']['fault_probability']
        })
        
        # Store temperature map for later visualization
        all_temp_maps[fault_type] = analysis['temperature_map'].copy()
        
        logger.info(f"Fault {fault_type}: "
                   f"Max Temp = {results[-1]['max_temperature']:.1f}°C, "
                   f"Detected as {results[-1]['predicted_fault']} "
                   f"with {results[-1]['fault_probability']:.1f}% probability")
    
    # Convert to DataFrame
    study_data = pd.DataFrame(results)
    
    # Calculate confusion matrix (for demonstration)
    true_faults = study_data['fault_type'].values
    predicted_faults = study_data['predicted_fault'].values
    
    num_correct = sum(1 for t, p in zip(true_faults, predicted_faults) if t == p)
    accuracy = num_correct / len(true_faults) * 100
    
    # Plot results
    if plot:
        # Create a large figure for comparison
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle("Fault Detection Comparison", fontsize=16)
        
        # Plot temperature maps for each fault type
        for i, fault_type in enumerate(fault_types):
            ax = fig.add_subplot(2, 3, i+1)
            im = ax.imshow(all_temp_maps[fault_type], cmap='hot', interpolation='bilinear')
            ax.set_title(f"{fault_type.capitalize()}")
            plt.colorbar(im, ax=ax)
            
            # Add detection result text
            detection_result = study_data.loc[study_data['fault_type'] == fault_type, 'predicted_fault'].values[0]
            probability = study_data.loc[study_data['fault_type'] == fault_type, 'fault_probability'].values[0]
            
            if detection_result == fault_type:
                result_color = 'green'
            else:
                result_color = 'red'
                
            ax.text(0.5, -0.15, f"Detected: {detection_result.capitalize()} ({probability:.0f}%)",
                   transform=ax.transAxes, ha='center', color=result_color, fontweight='bold')
        
        # Add overall accuracy text
        ax_text = fig.add_subplot(2, 3, 6)
        ax_text.axis('off')
        text = (f"Overall Accuracy: {accuracy:.1f}%\n\n"
                f"Detection Characteristics:\n"
                f"• Hotspot: Temperature non-uniformity\n"
                f"• Crack: Thermal & electrical impedance\n"
                f"• Delamination: High thermal resistance\n"
                f"• Void: Localized thermal barrier")
        
        ax_text.text(0.1, 0.5, text, va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
        
        # Also plot radar chart comparison of key parameters
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Define parameters to plot
        params = ['thermal_resistance', 'electrical_resistance', 
                 'thermal_uniformity', 'electrical_thermal_correlation', 'max_temperature']
        
        # Normalize parameters for radar chart
        normalized_data = {}
        for param in params:
            min_val = study_data[param].min()
            max_val = study_data[param].max()
            if max_val > min_val:
                normalized_data[param] = (study_data[param] - min_val) / (max_val - min_val)
            else:
                normalized_data[param] = study_data[param] / max_val
        
        # Number of variables
        N = len(params)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Labels for each axis
        labels = ['Thermal\nResistance', 'Electrical\nResistance', 
                 'Thermal\nUniformity', 'E-T\nCorrelation', 'Max\nTemperature']
        
        # Set up the plot
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        # Plot each fault type
        for fault_type in fault_types:
            values = [normalized_data[param].loc[study_data['fault_type'] == fault_type].values[0] for param in params]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=fault_type.capitalize())
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Fault Characteristics Comparison", size=15, y=1.1)
        plt.tight_layout()
        plt.show()
    
    return study_data


def create_temperature_evolution_animation(device, power_levels):
    """
    Create an animation showing temperature map evolution with power
    
    Parameters:
    -----------
    device : SemiconductorDevice
        Device to analyze
    power_levels : array
        Array of power levels to simulate
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Temperature Distribution Evolution')
    
    # Initialize with ambient temperature map
    device.set_operating_power(0.0)
    init_temp_map = device.temperature_map.copy()
    
    # Set color map limits based on max possible temperature
    vmin = device.ambient_temperature
    vmax = device.base_params['max_temperature']
    
    # Create initial image
    im = ax.imshow(init_temp_map, cmap='hot', interpolation='bilinear', 
                  vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    # Text for power level
    power_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(facecolor='black', alpha=0.5))
    
    # Max temperature text
    temp_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', color='white',
                       bbox=dict(facecolor='black', alpha=0.5))
    
    def init():
        """Initialize animation"""
        im.set_array(init_temp_map)
        power_text.set_text('Power: 0.0 W')
        temp_text.set_text(f'Max Temp: {device.ambient_temperature:.1f}°C')
        return [im, power_text, temp_text]
    
    def update(frame):
        """Update function for animation"""
        power = power_levels[frame]
        
        # Update device power
        device.set_operating_power(power)
        
        # Update image
        im.set_array(device.temperature_map)
        
        # Update text
        power_text.set_text(f'Power: {power:.1f} W')
        temp_text.set_text(f'Max Temp: {np.max(device.temperature_map):.1f}°C')
        
        return [im, power_text, temp_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=range(len(power_levels)),
        init_func=init, blit=True, interval=500
    )
    
    plt.tight_layout()
    plt.show()
    
    # Reset device to original power
    device.set_operating_power(power_levels[0])


if __name__ == "__main__":
    # Example 1: Single device analysis
    print("\n=== Example 1: Single Device Analysis ===")
    device = SemiconductorDevice(device_type='cpu', operating_power=50.0, 
                              ambient_temperature=25.0, fault_type='none')
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure analyzer
    analyzer.configure(
        electrical_freq_range=(100, 1000000),   # Hz
        thermal_freq_range=(0.001, 1),          # Hz
        voltage_amplitude=0.1,                  # V
        thermal_pulse_power=0.5,                # W
    )
    
    # Calibrate analyzer
    analyzer.calibrate()
    
    # Perform analysis
    analysis_results = perform_device_analysis(device, analyzer)
    
    # Example 2: Power sweep study
    print("\n=== Example 2: Power Sweep Study ===")
    power_data = power_sweep_study(device, analyzer, plot=True)
    
    # Example 3: Fault detection study
    print("\n=== Example 3: Fault Detection Study ===")
    fault_data = fault_detection_study(device, analyzer, plot=True)
    
    print("\nAnalysis complete. Check the plots for visual results.")
