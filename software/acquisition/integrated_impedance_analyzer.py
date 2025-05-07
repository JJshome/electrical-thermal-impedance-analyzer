"""
Integrated Electrical-Thermal Impedance Analyzer

This is the main module that integrates both electrical and thermal impedance
measurements to provide comprehensive characterization of various systems.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
import time
import logging
import threading
import json
import os
from enum import Enum, auto
from datetime import datetime

# Import the individual measurement modules
from .electrical_impedance import ElectricalImpedanceMeasurement, MeasurementMode as ElectricalMode
from .thermal_impedance import ThermalImpedanceMeasurement, MeasurementMode as ThermalMode

# Set up logging
logger = logging.getLogger(__name__)


class SystemType(Enum):
    """Enumeration for different system types to be analyzed"""
    BATTERY = auto()
    SEMICONDUCTOR = auto()
    BIOLOGICAL_TISSUE = auto()
    FUEL_CELL = auto()
    MATERIALS = auto()
    CUSTOM = auto()


class CorrelationMethod(Enum):
    """Enumeration for different correlation methods between electrical and thermal impedance"""
    DIRECT = auto()          # Direct correlation of measurements
    CROSS_SPECTRUM = auto()  # Cross-spectral density analysis
    MUTUAL_INFORMATION = auto()  # Information theory based correlation
    PHASE_COHERENCE = auto()  # Phase coherence between electrical and thermal responses
    GRANGER_CAUSALITY = auto()  # Causality analysis


class IntegratedImpedanceAnalyzer:
    """
    Main class for the integrated electrical-thermal impedance analyzer
    
    This class coordinates measurements from both electrical and thermal
    impedance systems and provides integrated analysis capabilities.
    """
    
    def __init__(self, 
                 electrical_hardware=None, 
                 thermal_hardware=None,
                 system_type=SystemType.CUSTOM):
        """
        Initialize the integrated impedance analyzer
        
        Parameters
        ----------
        electrical_hardware : object, optional
            Hardware interface for electrical impedance measurements
        thermal_hardware : object, optional
            Hardware interface for thermal impedance measurements
        system_type : SystemType, optional
            Type of system being analyzed, for optimized parameter selection
        """
        self.system_type = system_type
        
        # Initialize the individual measurement systems
        self.electrical = ElectricalImpedanceMeasurement(electrical_hardware)
        self.thermal = ThermalImpedanceMeasurement(thermal_hardware)
        
        # Default configuration
        self.config = {
            'electrical_freq_range': (0.1, 100000),  # Hz
            'thermal_freq_range': (0.01, 1),         # Hz
            'electrical_num_points': 50,
            'thermal_num_points': 10,
            'voltage_amplitude': 10e-3,              # V
            'thermal_pulse_power': 100e-3,           # W
            'correlation_method': CorrelationMethod.DIRECT,
            'simultaneous_measurement': False,       # Whether to measure electrical and thermal in parallel
            'stabilization_time': 30,                # Seconds to wait for thermal stabilization
            'auto_calibration': True,                # Whether to calibrate before measurement
            'save_raw_data': True,                   # Whether to save raw measurement data
            'result_directory': './results',         # Directory for saving results
            'pcm_enabled': True,                     # Whether to use PCM thermal management
            'pcm_temperature': 35.0,                 # °C - target temperature for PCM
        }
        
        # Set system-specific default configurations
        self._set_system_specific_config()
        
        # Storage for measurement results
        self._last_results = None
        self._measurement_in_progress = False
        
        logger.info(f"Initialized integrated electrical-thermal impedance analyzer for {system_type.name} applications")
    
    def _set_system_specific_config(self):
        """Set configuration parameters specific to the system type"""
        if self.system_type == SystemType.BATTERY:
            # Batteries: Focus on lower frequencies for electrical, relevant thermal freq range
            self.config.update({
                'electrical_freq_range': (0.01, 10000),  # Hz (extended low for diffusion)
                'thermal_freq_range': (0.001, 0.1),      # Hz (slower thermal processes)
                'voltage_amplitude': 5e-3,               # V (lower to avoid disturbing battery)
                'thermal_pulse_power': 50e-3,            # W (lower to minimize temperature rise)
                'pcm_temperature': 25.0,                 # °C (room temperature)
            })
            
        elif self.system_type == SystemType.SEMICONDUCTOR:
            # Semiconductors: Higher electrical frequencies, faster thermal response
            self.config.update({
                'electrical_freq_range': (1000, 500000),  # Hz (higher for semiconductors)
                'thermal_freq_range': (0.05, 5),          # Hz (faster thermal processes)
                'voltage_amplitude': 50e-3,               # V (higher for better SNR)
                'thermal_pulse_power': 200e-3,            # W (higher for better thermal response)
                'pcm_temperature': 30.0,                  # °C (device operating temperature)
            })
            
        elif self.system_type == SystemType.BIOLOGICAL_TISSUE:
            # Biological tissue: Focus on tissue-specific frequencies, low power
            self.config.update({
                'electrical_freq_range': (10, 100000),    # Hz (bioimpedance range)
                'thermal_freq_range': (0.01, 0.5),        # Hz (blood flow influence)
                'voltage_amplitude': 1e-3,                # V (very low to ensure safety)
                'thermal_pulse_power': 10e-3,             # W (minimal thermal stimulus)
                'pcm_temperature': 33.0,                  # °C (near body temperature)
            })
            
        elif self.system_type == SystemType.FUEL_CELL:
            # Fuel cells: Focus on electrochemical processes
            self.config.update({
                'electrical_freq_range': (0.01, 50000),   # Hz (wide range for reactions)
                'thermal_freq_range': (0.001, 0.2),       # Hz (thermal management)
                'voltage_amplitude': 10e-3,               # V (moderate)
                'thermal_pulse_power': 100e-3,            # W (modest heat input)
                'pcm_temperature': 60.0,                  # °C (typical PEMFC temp)
            })
            
        elif self.system_type == SystemType.MATERIALS:
            # Materials characterization: Wide ranges
            self.config.update({
                'electrical_freq_range': (0.1, 500000),   # Hz (full range)
                'thermal_freq_range': (0.001, 1.0),       # Hz (wide thermal range)
                'voltage_amplitude': 100e-3,              # V (higher for better SNR)
                'thermal_pulse_power': 500e-3,            # W (higher for materials)
                'pcm_temperature': 25.0,                  # °C (room temperature)
            })
    
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
        
        # Update electrical and thermal subsystems with relevant parameters
        self._update_subsystem_configs()
        
        return self.config
    
    def _update_subsystem_configs(self):
        """Update the configurations of electrical and thermal subsystems"""
        # Configure electrical subsystem
        self.electrical.configure(
            min_frequency=self.config['electrical_freq_range'][0],
            max_frequency=self.config['electrical_freq_range'][1],
            num_points=self.config['electrical_num_points'],
            voltage_amplitude=self.config['voltage_amplitude'],
        )
        
        # Configure thermal subsystem
        self.thermal.configure(
            min_frequency=self.config['thermal_freq_range'][0],
            max_frequency=self.config['thermal_freq_range'][1],
            num_points=self.config['thermal_num_points'],
            heat_amplitude=self.config['thermal_pulse_power'],
            pcm_enabled=self.config['pcm_enabled'],
            pcm_temperature=self.config['pcm_temperature'],
        )
    
    def calibrate(self):
        """
        Perform calibration of both electrical and thermal systems
        
        Returns
        -------
        bool
            True if both calibrations succeeded
        """
        logger.info("Starting integrated calibration sequence")
        
        # Calibrate electrical system
        logger.info("Calibrating electrical impedance system...")
        electrical_cal_success = self.electrical.calibrate()
        
        if not electrical_cal_success:
            logger.error("Electrical impedance calibration failed")
            return False
        
        # Calibrate thermal system
        logger.info("Calibrating thermal impedance system...")
        thermal_cal_success = self.thermal.calibrate()
        
        if not thermal_cal_success:
            logger.error("Thermal impedance calibration failed")
            return False
        
        logger.info("Integrated calibration completed successfully")
        return True
    
    def measure(self, frequencies_electrical=None, frequencies_thermal=None):
        """
        Perform integrated electrical and thermal impedance measurements
        
        Parameters
        ----------
        frequencies_electrical : array-like, optional
            Specific frequencies for electrical measurements. If None, uses the configured range.
        frequencies_thermal : array-like, optional
            Specific frequencies for thermal measurements. If None, uses the configured range.
            
        Returns
        -------
        dict
            Integrated measurement results
        """
        if self._measurement_in_progress:
            logger.warning("A measurement is already in progress")
            return None
        
        self._measurement_in_progress = True
        start_time = time.time()
        
        try:
            # Auto-calibration if enabled
            if self.config['auto_calibration']:
                logger.info("Auto-calibration enabled, performing calibration...")
                self.calibrate()
            
            # Create results directory if needed
            if self.config['save_raw_data']:
                os.makedirs(self.config['result_directory'], exist_ok=True)
            
            # Allow system to thermally stabilize if PCM is enabled
            if self.config['pcm_enabled']:
                logger.info(f"Waiting {self.config['stabilization_time']} seconds for thermal stabilization...")
                time.sleep(self.config['stabilization_time'])
            
            # Perform measurements (either sequential or simultaneous)
            if self.config['simultaneous_measurement']:
                results_electrical, results_thermal = self._perform_simultaneous_measurements(
                    frequencies_electrical, frequencies_thermal)
            else:
                results_electrical, results_thermal = self._perform_sequential_measurements(
                    frequencies_electrical, frequencies_thermal)
            
            # Combine results
            integrated_results = self._integrate_results(results_electrical, results_thermal)
            
            # Save results if configured
            if self.config['save_raw_data']:
                self._save_results(integrated_results)
            
            # Store last results
            self._last_results = integrated_results
            
            elapsed = time.time() - start_time
            logger.info(f"Integrated measurement completed in {elapsed:.1f} seconds")
            
            return integrated_results
            
        finally:
            self._measurement_in_progress = False
    
    def _perform_sequential_measurements(self, frequencies_electrical, frequencies_thermal):
        """
        Perform electrical and thermal measurements sequentially
        
        Parameters
        ----------
        frequencies_electrical : array-like or None
            Frequencies for electrical measurements
        frequencies_thermal : array-like or None
            Frequencies for thermal measurements
            
        Returns
        -------
        tuple
            (electrical_results, thermal_results)
        """
        logger.info("Performing sequential electrical and thermal measurements")
        
        # First perform electrical impedance measurements
        logger.info("Starting electrical impedance measurements...")
        results_electrical = self.electrical.measure(frequencies_electrical)
        
        # Then perform thermal impedance measurements
        logger.info("Starting thermal impedance measurements...")
        results_thermal = self.thermal.measure(frequencies_thermal)
        
        return results_electrical, results_thermal
    
    def _perform_simultaneous_measurements(self, frequencies_electrical, frequencies_thermal):
        """
        Perform electrical and thermal measurements simultaneously using threads
        
        Parameters
        ----------
        frequencies_electrical : array-like or None
            Frequencies for electrical measurements
        frequencies_thermal : array-like or None
            Frequencies for thermal measurements
            
        Returns
        -------
        tuple
            (electrical_results, thermal_results)
        """
        logger.info("Performing simultaneous electrical and thermal measurements")
        
        # Storage for results
        results_electrical = [None]
        results_thermal = [None]
        
        # Define measurement threads
        def electrical_measurement_thread():
            results_electrical[0] = self.electrical.measure(frequencies_electrical)
            
        def thermal_measurement_thread():
            results_thermal[0] = self.thermal.measure(frequencies_thermal)
        
        # Start both measurement threads
        thread_electrical = threading.Thread(target=electrical_measurement_thread)
        thread_thermal = threading.Thread(target=thermal_measurement_thread)
        
        thread_electrical.start()
        thread_thermal.start()
        
        # Wait for both to complete
        thread_electrical.join()
        thread_thermal.join()
        
        return results_electrical[0], results_thermal[0]
    
    def _integrate_results(self, results_electrical, results_thermal):
        """
        Integrate electrical and thermal measurement results
        
        Parameters
        ----------
        results_electrical : dict
            Electrical impedance measurement results
        results_thermal : dict
            Thermal impedance measurement results
            
        Returns
        -------
        dict
            Integrated results
        """
        # Create base integrated results dictionary
        integrated_results = {
            'timestamp': time.time(),
            'config': self.config.copy(),
            'electrical': results_electrical,
            'thermal': results_thermal,
            'correlation': {}
        }
        
        # Calculate correlations based on selected method
        correlation_method = self.config['correlation_method']
        
        if correlation_method == CorrelationMethod.DIRECT:
            integrated_results['correlation'] = self._calculate_direct_correlation(
                results_electrical, results_thermal)
                
        elif correlation_method == CorrelationMethod.CROSS_SPECTRUM:
            integrated_results['correlation'] = self._calculate_cross_spectrum(
                results_electrical, results_thermal)
                
        elif correlation_method == CorrelationMethod.MUTUAL_INFORMATION:
            integrated_results['correlation'] = self._calculate_mutual_information(
                results_electrical, results_thermal)
                
        elif correlation_method == CorrelationMethod.PHASE_COHERENCE:
            integrated_results['correlation'] = self._calculate_phase_coherence(
                results_electrical, results_thermal)
                
        elif correlation_method == CorrelationMethod.GRANGER_CAUSALITY:
            integrated_results['correlation'] = self._calculate_granger_causality(
                results_electrical, results_thermal)
        
        return integrated_results
    
    def _calculate_direct_correlation(self, results_electrical, results_thermal):
        """
        Calculate direct correlation between electrical and thermal impedance
        
        This method looks for direct relationships between magnitudes and phases
        at similar time scales (frequency ranges).
        
        Parameters
        ----------
        results_electrical : dict
            Electrical impedance measurement results
        results_thermal : dict
            Thermal impedance measurement results
            
        Returns
        -------
        dict
            Correlation analysis results
        """
        # Initialize correlation results
        correlation = {
            'method': 'direct',
            'correlation_factors': [],
            'matched_points': []
        }
        
        # Get data
        e_freqs = results_electrical['frequencies']
        e_mag = results_electrical['magnitude']
        e_phase = results_electrical['phase']
        
        t_freqs = results_thermal['frequencies']
        t_mag = results_thermal['magnitude']
        t_phase = results_thermal['phase']
        
        # Find overlapping frequency ranges (based on thermal dynamics)
        # For each thermal frequency, find the electrical frequency that's closest
        # to a scaled value (thermal processes typically have slower time constants)
        # Scaling factor: ratio of thermal to electrical time constants
        scaling_factor = 1000  # Example: electrical processes ~1000x faster than thermal
        
        for i, t_freq in enumerate(t_freqs):
            # Find equivalent electrical frequency (scaled)
            e_equiv_freq = t_freq * scaling_factor
            
            # Find closest electrical frequency measured
            idx = np.argmin(np.abs(e_freqs - e_equiv_freq))
            
            # Store matched point
            matched_point = {
                'thermal_frequency': t_freq,
                'electrical_frequency': e_freqs[idx],
                'thermal_magnitude': t_mag[i],
                'electrical_magnitude': e_mag[idx],
                'thermal_phase': t_phase[i],
                'electrical_phase': e_phase[idx],
                'frequency_scale_factor': e_freqs[idx] / t_freq
            }
            
            correlation['matched_points'].append(matched_point)
        
        # Calculate correlation coefficients for magnitudes and phases
        if len(correlation['matched_points']) > 1:
            thermal_mags = [point['thermal_magnitude'] for point in correlation['matched_points']]
            electrical_mags = [point['electrical_magnitude'] for point in correlation['matched_points']]
            
            thermal_phases = [point['thermal_phase'] for point in correlation['matched_points']]
            electrical_phases = [point['electrical_phase'] for point in correlation['matched_points']]
            
            try:
                # Calculate Pearson correlation coefficients
                mag_corr = np.corrcoef(thermal_mags, electrical_mags)[0, 1]
                phase_corr = np.corrcoef(thermal_phases, electrical_phases)[0, 1]
                
                correlation['correlation_factors'].append({
                    'type': 'magnitude',
                    'correlation_coefficient': mag_corr
                })
                
                correlation['correlation_factors'].append({
                    'type': 'phase',
                    'correlation_coefficient': phase_corr
                })
                
                # Overall correlation (magnitude-based)
                correlation['overall_correlation'] = mag_corr
                
            except:
                logger.warning("Failed to calculate correlation coefficients")
                correlation['overall_correlation'] = None
        else:
            logger.warning("Not enough matched points for correlation analysis")
            correlation['overall_correlation'] = None
        
        return correlation
    
    def _calculate_cross_spectrum(self, results_electrical, results_thermal):
        """
        Calculate cross-spectral density between electrical and thermal impedance
        
        Parameters
        ----------
        results_electrical : dict
            Electrical impedance measurement results
        results_thermal : dict
            Thermal impedance measurement results
            
        Returns
        -------
        dict
            Cross-spectral analysis results
        """
        # This implementation is simplified - a real implementation would
        # require resampling to common frequency points and proper spectral analysis
        
        correlation = {
            'method': 'cross_spectrum',
            'message': 'Cross-spectral density analysis not fully implemented',
            'overall_correlation': None
        }
        
        # In a real implementation, would calculate proper cross-spectral density
        # and coherence functions
        
        return correlation
    
    def _calculate_mutual_information(self, results_electrical, results_thermal):
        """
        Calculate mutual information between electrical and thermal impedance
        
        Parameters
        ----------
        results_electrical : dict
            Electrical impedance measurement results
        results_thermal : dict
            Thermal impedance measurement results
            
        Returns
        -------
        dict
            Mutual information analysis results
        """
        # This is a placeholder - real implementation would use information
        # theory metrics to quantify relationships
        
        correlation = {
            'method': 'mutual_information',
            'message': 'Mutual information analysis not fully implemented',
            'overall_correlation': None
        }
        
        return correlation
    
    def _calculate_phase_coherence(self, results_electrical, results_thermal):
        """
        Calculate phase coherence between electrical and thermal impedance
        
        Parameters
        ----------
        results_electrical : dict
            Electrical impedance measurement results
        results_thermal : dict
            Thermal impedance measurement results
            
        Returns
        -------
        dict
            Phase coherence analysis results
        """
        # Placeholder for phase coherence calculation
        
        correlation = {
            'method': 'phase_coherence',
            'message': 'Phase coherence analysis not fully implemented',
            'overall_correlation': None
        }
        
        return correlation
    
    def _calculate_granger_causality(self, results_electrical, results_thermal):
        """
        Calculate Granger causality between electrical and thermal impedance
        
        Parameters
        ----------
        results_electrical : dict
            Electrical impedance measurement results
        results_thermal : dict
            Thermal impedance measurement results
            
        Returns
        -------
        dict
            Granger causality analysis results
        """
        # Placeholder for Granger causality analysis
        
        correlation = {
            'method': 'granger_causality',
            'message': 'Granger causality analysis not fully implemented',
            'overall_correlation': None
        }
        
        return correlation
    
    def _save_results(self, results):
        """
        Save measurement results to file
        
        Parameters
        ----------
        results : dict
            Measurement results to save
        """
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.system_type.name.lower()}_impedance_{timestamp}.json"
            filepath = os.path.join(self.config['result_directory'], filename)
            
            # Create a copy of results for serialization
            results_for_save = {}
            
            # Convert numpy arrays to lists for JSON serialization
            for key, value in results.items():
                if key == 'electrical' or key == 'thermal':
                    results_for_save[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            results_for_save[key][subkey] = subvalue.tolist()
                        else:
                            results_for_save[key][subkey] = subvalue
                else:
                    results_for_save[key] = value
            
            # Convert enum values to strings
            if 'config' in results_for_save:
                config_copy = results_for_save['config'].copy()
                for key, value in config_copy.items():
                    if isinstance(value, Enum):
                        results_for_save['config'][key] = value.name
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results_for_save, f, indent=2)
                
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def analyze(self, results=None):
        """
        Perform integrated analysis of measurement results
        
        Parameters
        ----------
        results : dict, optional
            Measurement results to analyze. If None, uses the last measurement.
            
        Returns
        -------
        dict
            Analysis results
        """
        if results is None:
            if self._last_results is None:
                raise ValueError("No measurement results available for analysis")
            results = self._last_results
        
        # Initialize analysis results
        analysis = {
            'timestamp': time.time(),
            'system_type': self.system_type.name,
            'electrical_analysis': {},
            'thermal_analysis': {},
            'integrated_analysis': {},
            'system_parameters': {}
        }
        
        # Perform individual analyses
        analysis['electrical_analysis'] = self.electrical.analyze_impedance(results['electrical'])
        analysis['thermal_analysis'] = self.thermal.analyze_thermal_impedance(results['thermal'])
        
        # Perform integrated analysis based on system type
        if self.system_type == SystemType.BATTERY:
            analysis['integrated_analysis'] = self._analyze_battery(results)
        elif self.system_type == SystemType.SEMICONDUCTOR:
            analysis['integrated_analysis'] = self._analyze_semiconductor(results)
        elif self.system_type == SystemType.BIOLOGICAL_TISSUE:
            analysis['integrated_analysis'] = self._analyze_biological_tissue(results)
        elif self.system_type == SystemType.FUEL_CELL:
            analysis['integrated_analysis'] = self._analyze_fuel_cell(results)
        elif self.system_type == SystemType.MATERIALS:
            analysis['integrated_analysis'] = self._analyze_materials(results)
        else:
            # Generic analysis for custom systems
            analysis['integrated_analysis'] = self._analyze_custom(results)
        
        # Extract key system parameters
        analysis['system_parameters'] = self._extract_system_parameters(
            analysis['electrical_analysis'], 
            analysis['thermal_analysis'],
            analysis['integrated_analysis']
        )
        
        return analysis
    
    def _analyze_battery(self, results):
        """
        Analyze battery-specific parameters from impedance measurements
        
        Parameters
        ----------
        results : dict
            Integrated measurement results
            
        Returns
        -------
        dict
            Battery-specific analysis
        """
        # Initialize battery analysis
        battery_analysis = {
            'state_of_charge': None,
            'state_of_health': None,
            'internal_resistance': None,
            'charge_transfer_resistance': None,
            'diffusion_coefficient': None,
            'thermal_resistance': None,
            'thermal_runaway_risk': None
        }
        
        # Extract electrical parameters
        e_results = results['electrical']
        e_freq = e_results['frequencies']
        e_mag = e_results['magnitude']
        e_phase = e_results['phase']
        
        # Extract thermal parameters
        t_results = results['thermal']
        t_freq = t_results['frequencies']
        t_mag = t_results['magnitude']
        
        # Find internal resistance (high-frequency impedance)
        high_freq_idx = np.argmax(e_freq)
        battery_analysis['internal_resistance'] = e_mag[high_freq_idx]
        
        # Estimate charge transfer resistance from mid-frequency region
        # In a real implementation, this would use equivalent circuit fitting
        mid_freq = 100  # Hz, typical for charge transfer processes
        mid_freq_idx = np.argmin(np.abs(e_freq - mid_freq))
        battery_analysis['charge_transfer_resistance'] = e_mag[mid_freq_idx] - battery_analysis['internal_resistance']
        
        # Estimate diffusion coefficient from low-frequency response
        # In a real implementation, this would use Warburg impedance fitting
        battery_analysis['diffusion_coefficient'] = 1e-10  # Placeholder
        
        # Thermal resistance from thermal impedance
        battery_analysis['thermal_resistance'] = np.mean(t_mag[:3])  # Average of lowest frequencies
        
        # Calculate state of health based on internal resistance and thermal properties
        # Simplified model: higher resistance and thermal impedance = lower SOH
        r_ratio = battery_analysis['internal_resistance'] / 0.01  # Normalized to typical new cell
        t_r_ratio = battery_analysis['thermal_resistance'] / 10  # Normalized to typical new cell
        
        # SOH estimation (0-100%), simplified model
        battery_analysis['state_of_health'] = max(0, min(100, 100 * (2 - r_ratio - 0.5 * t_r_ratio)))
        
        # Thermal runaway risk assessment based on thermal impedance pattern
        # Lower thermal impedance + high internal resistance = higher risk
        risk_factor = (1.0 / battery_analysis['thermal_resistance']) * r_ratio
        
        if risk_factor < 0.5:
            battery_analysis['thermal_runaway_risk'] = 'Low'
        elif risk_factor < 1.0:
            battery_analysis['thermal_runaway_risk'] = 'Medium'
        else:
            battery_analysis['thermal_runaway_risk'] = 'High'
        
        return battery_analysis
    
    def _analyze_semiconductor(self, results):
        """
        Analyze semiconductor-specific parameters from impedance measurements
        
        Parameters
        ----------
        results : dict
            Integrated measurement results
            
        Returns
        -------
        dict
            Semiconductor-specific analysis
        """
        # Initialize semiconductor analysis
        semiconductor_analysis = {
            'junction_resistance': None,
            'parasitic_capacitance': None,
            'thermal_resistance': None,
            'thermal_capacitance': None,
            'hotspot_detection': None,
            'thermal_conductivity': None
        }
        
        # Extract electrical parameters
        e_results = results['electrical']
        e_freq = e_results['frequencies']
        e_mag = e_results['magnitude']
        e_phase = e_results['phase']
        
        # Extract thermal parameters
        t_results = results['thermal']
        t_mag = t_results['magnitude']
        
        # Estimate junction resistance from high-frequency electrical impedance
        high_freq_idx = np.argmax(e_freq)
        semiconductor_analysis['junction_resistance'] = e_mag[high_freq_idx]
        
        # Estimate parasitic capacitance
        # In a real implementation, this would use equivalent circuit fitting
        high_freq = 1e6  # Hz
        high_freq_idx = np.argmin(np.abs(e_freq - high_freq))
        phase_at_high_freq = e_phase[high_freq_idx]
        
        # Simplified capacitance calculation from impedance phase
        # C = -1/(2*pi*f*Z*sin(phase)) if phase is negative
        if phase_at_high_freq < 0:
            semiconductor_analysis['parasitic_capacitance'] = \
                -1.0 / (2 * np.pi * e_freq[high_freq_idx] * e_mag[high_freq_idx] * np.sin(np.radians(phase_at_high_freq)))
        else:
            semiconductor_analysis['parasitic_capacitance'] = 1e-12  # Placeholder
        
        # Extract thermal parameters directly from thermal analysis
        semiconductor_analysis['thermal_resistance'] = np.mean(t_mag[:3])  # K/W
        
        # Simplified thermal capacitance calculation
        # In a real implementation, this would use thermal equivalent circuit fitting
        semiconductor_analysis['thermal_capacitance'] = 0.01  # J/K, placeholder
        
        # Thermal conductivity estimation (inverse of thermal resistance * thickness)
        # Assuming a 1mm thick device with 1cm² area
        thickness = 0.001  # m
        area = 0.0001      # m²
        if semiconductor_analysis['thermal_resistance'] > 0:
            semiconductor_analysis['thermal_conductivity'] = \
                thickness / (semiconductor_analysis['thermal_resistance'] * area)  # W/(m·K)
        else:
            semiconductor_analysis['thermal_conductivity'] = None
        
        # Hotspot detection based on thermal uniformity
        # In a real implementation, this would analyze spatial thermal data
        semiconductor_analysis['hotspot_detection'] = 'None detected'  # Placeholder
        
        return semiconductor_analysis
    
    def _analyze_biological_tissue(self, results):
        """
        Analyze biological tissue parameters from impedance measurements
        
        Parameters
        ----------
        results : dict
            Integrated measurement results
            
        Returns
        -------
        dict
            Tissue-specific analysis
        """
        # Initialize tissue analysis
        tissue_analysis = {
            'tissue_type': None,  # e.g., muscle, fat, epithelial
            'hydration_level': None,
            'perfusion_index': None,
            'cell_density': None,
            'extracellular_resistance': None,
            'intracellular_resistance': None,
            'membrane_capacitance': None,
            'thermal_diffusivity': None
        }
        
        # Extract electrical parameters
        e_results = results['electrical']
        e_freq = e_results['frequencies']
        e_mag = e_results['magnitude']
        e_phase = e_results['phase']
        
        # Extract thermal parameters
        t_results = results['thermal']
        t_mag = t_results['magnitude']
        
        # Find Cole-Cole center frequency (frequency with most negative phase)
        center_idx = np.argmin(e_phase)
        center_freq = e_freq[center_idx]
        
        # Estimate extracellular and intracellular resistances
        # Based on simplified Cole-Cole model
        low_freq_idx = np.argmin(e_freq)
        high_freq_idx = np.argmax(e_freq)
        
        r_low = e_mag[low_freq_idx]
        r_high = e_mag[high_freq_idx]
        
        tissue_analysis['extracellular_resistance'] = r_high  # R∞ (high frequency asymptote)
        tissue_analysis['intracellular_resistance'] = r_low - r_high  # R0 - R∞
        
        # Estimate membrane capacitance from center frequency
        # Cm = 1/(2*pi*fc*(R0-R∞)*R∞/(R0))
        if center_freq > 0 and r_low > r_high > 0:
            tissue_analysis['membrane_capacitance'] = \
                1.0 / (2 * np.pi * center_freq * (r_low - r_high) * r_high / r_low)
        else:
            tissue_analysis['membrane_capacitance'] = None
        
        # Thermal diffusivity from thermal time constant
        # α = x²/τ where x is a characteristic length
        # Assuming 1cm penetration depth
        depth = 0.01  # m
        
        # Find thermal time constant from thermal impedance
        if len(t_mag) > 1 and len(t_results['frequencies']) > 1:
            # Simplified: use frequency where thermal impedance drops to ~70% of max
            t_max = np.max(t_mag)
            cutoff_idx = np.argmin(np.abs(t_mag - 0.7 * t_max))
            if cutoff_idx < len(t_results['frequencies']):
                thermal_time_constant = 1.0 / (2 * np.pi * t_results['frequencies'][cutoff_idx])
                
                # Calculate diffusivity
                if thermal_time_constant > 0:
                    tissue_analysis['thermal_diffusivity'] = depth**2 / thermal_time_constant  # m²/s
        
        # Estimate hydration level based on electrical properties
        # Higher extracellular resistance often indicates lower hydration
        if tissue_analysis['extracellular_resistance'] is not None:
            # Simplified model: normalize to a reference value
            r_ext_ref = 50  # Ohm, reference for normal hydration
            hydration_factor = r_ext_ref / tissue_analysis['extracellular_resistance']
            tissue_analysis['hydration_level'] = min(100, max(0, 100 * hydration_factor))  # 0-100%
        
        # Estimate perfusion index from thermal properties
        # Higher perfusion = faster heat dissipation
        if tissue_analysis['thermal_diffusivity'] is not None:
            # Simplified model: normalize to a reference value
            diffusivity_ref = 1.5e-7  # m²/s, typical for well-perfused tissue
            perfusion_factor = tissue_analysis['thermal_diffusivity'] / diffusivity_ref
            tissue_analysis['perfusion_index'] = min(100, max(0, 100 * perfusion_factor))  # 0-100%
        
        # Estimate tissue type based on electrical and thermal properties
        # This is highly simplified - real tissue classification would use
        # trained models with reference data
        if tissue_analysis['extracellular_resistance'] is not None and \
           tissue_analysis['intracellular_resistance'] is not None:
            
            r_ratio = tissue_analysis['intracellular_resistance'] / tissue_analysis['extracellular_resistance']
            
            if r_ratio > 2.0:
                tissue_analysis['tissue_type'] = 'Muscle'
            elif r_ratio < 0.5:
                tissue_analysis['tissue_type'] = 'Fat'
            else:
                tissue_analysis['tissue_type'] = 'Epithelial'
        
        return tissue_analysis
    
    def _analyze_fuel_cell(self, results):
        """
        Analyze fuel cell parameters from impedance measurements
        
        Parameters
        ----------
        results : dict
            Integrated measurement results
            
        Returns
        -------
        dict
            Fuel cell-specific analysis
        """
        # Simplified placeholder for fuel cell analysis
        fuel_cell_analysis = {
            'membrane_resistance': None,
            'charge_transfer_resistance': None,
            'mass_transport_resistance': None,
            'thermal_resistance': None,
            'water_management_index': None,
            'performance_degradation': None
        }
        
        # In a real implementation, this would analyze the complex
        # electrochemical and thermal processes in fuel cells
        
        return fuel_cell_analysis
    
    def _analyze_materials(self, results):
        """
        Analyze material properties from impedance measurements
        
        Parameters
        ----------
        results : dict
            Integrated measurement results
            
        Returns
        -------
        dict
            Materials-specific analysis
        """
        # Simplified placeholder for materials analysis
        materials_analysis = {
            'electrical_conductivity': None,
            'dielectric_constant': None,
            'thermal_conductivity': None,
            'thermal_diffusivity': None,
            'specific_heat_capacity': None,
            'phase_transitions': None
        }
        
        # In a real implementation, this would extract material
        # properties from the impedance spectra
        
        return materials_analysis
    
    def _analyze_custom(self, results):
        """
        Generic analysis for custom system types
        
        Parameters
        ----------
        results : dict
            Integrated measurement results
            
        Returns
        -------
        dict
            Generic analysis results
        """
        # Initialize generic analysis
        custom_analysis = {
            'electrical_characteristics': {},
            'thermal_characteristics': {},
            'correlation_strength': None
        }
        
        # Basic electrical characteristics from frequency response
        e_results = results['electrical']
        
        if 'frequencies' in e_results and len(e_results['frequencies']) > 0:
            custom_analysis['electrical_characteristics'] = {
                'min_impedance': min(e_results['magnitude']),
                'max_impedance': max(e_results['magnitude']),
                'frequency_dependency': None  # Would be calculated from slope
            }
        
        # Basic thermal characteristics
        t_results = results['thermal']
        
        if 'frequencies' in t_results and len(t_results['frequencies']) > 0:
            custom_analysis['thermal_characteristics'] = {
                'thermal_resistance': np.mean(t_results['magnitude'][:3]) if len(t_results['magnitude']) >= 3 else None,
                'thermal_response_time': None  # Would be calculated from time constant
            }
        
        # Correlation between electrical and thermal properties
        if 'correlation' in results and 'overall_correlation' in results['correlation']:
            custom_analysis['correlation_strength'] = results['correlation']['overall_correlation']
        
        return custom_analysis
    
    def _extract_system_parameters(self, electrical_analysis, thermal_analysis, integrated_analysis):
        """
        Extract key system parameters from the analysis results
        
        Parameters
        ----------
        electrical_analysis : dict
            Results of electrical impedance analysis
        thermal_analysis : dict
            Results of thermal impedance analysis
        integrated_analysis : dict
            Results of integrated analysis
            
        Returns
        -------
        dict
            Key system parameters
        """
        # Extract the most important parameters based on system type
        if self.system_type == SystemType.BATTERY:
            return {
                'state_of_health': integrated_analysis.get('state_of_health'),
                'internal_resistance': integrated_analysis.get('internal_resistance'),
                'thermal_resistance': integrated_analysis.get('thermal_resistance'),
                'thermal_runaway_risk': integrated_analysis.get('thermal_runaway_risk')
            }
            
        elif self.system_type == SystemType.SEMICONDUCTOR:
            return {
                'junction_resistance': integrated_analysis.get('junction_resistance'),
                'thermal_resistance': integrated_analysis.get('thermal_resistance'),
                'thermal_conductivity': integrated_analysis.get('thermal_conductivity'),
                'hotspot_detection': integrated_analysis.get('hotspot_detection')
            }
            
        elif self.system_type == SystemType.BIOLOGICAL_TISSUE:
            return {
                'tissue_type': integrated_analysis.get('tissue_type'),
                'hydration_level': integrated_analysis.get('hydration_level'),
                'perfusion_index': integrated_analysis.get('perfusion_index')
            }
            
        elif self.system_type == SystemType.FUEL_CELL:
            return {
                'membrane_resistance': integrated_analysis.get('membrane_resistance'),
                'thermal_resistance': integrated_analysis.get('thermal_resistance'),
                'water_management_index': integrated_analysis.get('water_management_index'),
                'performance_degradation': integrated_analysis.get('performance_degradation')
            }
            
        elif self.system_type == SystemType.MATERIALS:
            return {
                'electrical_conductivity': integrated_analysis.get('electrical_conductivity'),
                'thermal_conductivity': integrated_analysis.get('thermal_conductivity'),
                'specific_heat_capacity': integrated_analysis.get('specific_heat_capacity')
            }
            
        else:  # CUSTOM
            # Return a generic set of parameters
            return {
                'electrical_impedance': electrical_analysis.get('min_impedance'),
                'thermal_resistance': thermal_analysis.get('thermal_resistance'),
                'correlation_strength': integrated_analysis.get('correlation_strength')
            }
    
    def plot_impedance_spectra(self, results=None):
        """
        Plot electrical and thermal impedance spectra together
        
        Parameters
        ----------
        results : dict, optional
            Results to plot. If None, uses the last measurement results.
            
        Returns
        -------
        str
            Information about the plot (in an actual implementation, would return figure handles)
        """
        if results is None:
            if self._last_results is None:
                raise ValueError("No measurement results available for plotting")
            results = self._last_results
        
        # In an actual implementation, this would create plots using matplotlib
        # or another plotting library
        
        return "Plotting would be implemented with matplotlib in a full implementation"
    
    def get_system_status(self):
        """
        Get the status of the measurement system
        
        Returns
        -------
        dict
            System status information
        """
        # Get calibration status
        electrical_cal = self.electrical.get_calibration_status()
        thermal_cal = self.thermal.get_calibration_status()
        
        # Combine into overall status
        status = {
            'system_type': self.system_type.name,
            'measurement_in_progress': self._measurement_in_progress,
            'electrical_calibration': electrical_cal,
            'thermal_calibration': thermal_cal,
            'last_measurement_time': None
        }
        
        if self._last_results is not None:
            status['last_measurement_time'] = self._last_results.get('timestamp')
        
        return status


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create an integrated analyzer for battery analysis
    analyzer = IntegratedImpedanceAnalyzer(system_type=SystemType.BATTERY)
    
    # Configure
    analyzer.configure(
        electrical_freq_range=(0.1, 10000),
        thermal_freq_range=(0.01, 0.5),
        electrical_num_points=20,
        thermal_num_points=5,
        voltage_amplitude=5e-3,
        thermal_pulse_power=50e-3,
        simultaneous_measurement=False,
        stabilization_time=5,  # Shortened for example
        save_raw_data=False
    )
    
    # Measure
    results = analyzer.measure()
    
    # Analyze
    analysis = analyzer.analyze(results)
    
    # Print some results
    print("\nIntegrated Measurement Results:")
    print("Electrical Impedance Range:")
    print(f"  {min(results['electrical']['magnitude']):.2f} Ω to {max(results['electrical']['magnitude']):.2f} Ω")
    print("Thermal Impedance Range:")
    print(f"  {min(results['thermal']['magnitude']):.2f} K/W to {max(results['thermal']['magnitude']):.2f} K/W")
    
    print("\nBattery Analysis Results:")
    print(f"State of Health: {analysis['integrated_analysis']['state_of_health']:.1f}%")
    print(f"Internal Resistance: {analysis['integrated_analysis']['internal_resistance']:.3f} Ω")
    print(f"Thermal Resistance: {analysis['integrated_analysis']['thermal_resistance']:.2f} K/W")
    print(f"Thermal Runaway Risk: {analysis['integrated_analysis']['thermal_runaway_risk']}")
