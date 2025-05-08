#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Battery Health Monitoring Application Example

This example demonstrates how to use the integrated electrical-thermal impedance
analyzer system for battery health monitoring and state-of-health (SOH) estimation.
The system combines electrical impedance spectroscopy (EIS) and thermal impedance
spectroscopy (TIS) to provide comprehensive characterization of battery cells.

The implementation is based on the research and technology developed by Ucaretron Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from datetime import datetime, timedelta
from impedance_analyzer import IntegratedImpedanceAnalyzer
from sklearn.preprocessing import StandardScaler
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

# Constants
EIS_FREQUENCIES = np.logspace(-1, 5, 30)  # 0.1Hz to 100kHz, 30 points
TIS_FREQUENCIES = np.logspace(-2, 0, 15)  # 0.01Hz to 1Hz, 15 points
VOLTAGE_AMPLITUDE = 10e-3  # V (small signal to avoid disturbing battery)
THERMAL_PULSE_POWER = 100e-3  # W
REFERENCE_TEMPERATURE = 25.0  # °C

# Battery types
BATTERY_TYPES = {
    'li_ion_18650': {
        'nominal_capacity': 3000,  # mAh
        'nominal_voltage': 3.7,    # V
        'internal_resistance_new': 30e-3,  # Ohm
        'internal_resistance_end': 60e-3,  # Ohm
        'thermal_resistance_new': 2.0,     # K/W
        'thermal_resistance_end': 3.5,     # K/W
    },
    'li_ion_21700': {
        'nominal_capacity': 4800,  # mAh
        'nominal_voltage': 3.7,    # V
        'internal_resistance_new': 25e-3,  # Ohm
        'internal_resistance_end': 50e-3,  # Ohm
        'thermal_resistance_new': 1.8,     # K/W
        'thermal_resistance_end': 3.0,     # K/W
    },
    'lifepo4': {
        'nominal_capacity': 2500,  # mAh
        'nominal_voltage': 3.2,    # V
        'internal_resistance_new': 40e-3,  # Ohm
        'internal_resistance_end': 80e-3,  # Ohm
        'thermal_resistance_new': 2.2,     # K/W
        'thermal_resistance_end': 3.8,     # K/W
    }
}


class BatteryHealthMonitoring:
    """Battery Health Monitoring using Integrated Impedance Analysis"""
    
    def __init__(self, battery_type='li_ion_18650'):
        """
        Initialize the battery health monitoring system
        
        Args:
            battery_type: Type of battery to monitor (default: li_ion_18650)
        """
        # Validate battery type
        if battery_type not in BATTERY_TYPES:
            raise ValueError(f"Unknown battery type: {battery_type}. "
                           f"Available types: {list(BATTERY_TYPES.keys())}")
        
        self.battery_type = battery_type
        self.battery_params = BATTERY_TYPES[battery_type]
        
        # Create the integrated impedance analyzer
        self.analyzer = IntegratedImpedanceAnalyzer()
        
        # Configure measurement parameters
        self.analyzer.configure(
            electrical_freq_range=(EIS_FREQUENCIES[0], EIS_FREQUENCIES[-1]),
            thermal_freq_range=(TIS_FREQUENCIES[0], TIS_FREQUENCIES[-1]),
            voltage_amplitude=VOLTAGE_AMPLITUDE,
            thermal_pulse_power=THERMAL_PULSE_POWER,
        )
        
        # Additional parameters specific to battery monitoring
        self.analyzer.set_advanced_parameters(
            integration_time=0.5,        # seconds
            averages=5,                  # number of measurements to average
            pcm_control=True,            # enable PCM temperature control
            target_temperature=REFERENCE_TEMPERATURE,  # °C (reference temperature)
            electrode_config="4-wire",   # 4-wire configuration for accuracy
            adaptive_sampling=True,      # adjust sampling based on changes
        )
        
        # Storage for battery measurements
        self.measurements = []
        self.timestamps = []
        self.soc_values = []
        self.soh_values = []
        self.temperature_values = []
        self.internal_resistance_values = []
        self.thermal_resistance_values = []
        
        # Reference data for new battery (will be populated on first run)
        self.reference_eis = None
        self.reference_tis = None
        
        # Equivalent circuit parameters
        self.ecm_parameters = {
            'Rs': [],    # Series resistance [Ohm]
            'Rct': [],   # Charge transfer resistance [Ohm]
            'Cdl': [],   # Double layer capacitance [F]
            'Rw': [],    # Warburg impedance parameter [Ohm·s^(-1/2)]
            'Rt': [],    # Thermal resistance [K/W]
            'Ct': [],    # Thermal capacitance [J/K]
        }
        
        # Battery cycle history
        self.cycle_count = 0
        self.cycle_data = []
        
        print(f"Battery Health Monitoring initialized for {battery_type}")
        print(f"Nominal capacity: {self.battery_params['nominal_capacity']} mAh")
        print(f"Nominal voltage: {self.battery_params['nominal_voltage']} V")
    
    def perform_reference_measurement(self, soc=1.0, temperature=REFERENCE_TEMPERATURE):
        """
        Perform a reference measurement on a new battery
        
        Args:
            soc: State of charge (0.0-1.0)
            temperature: Battery temperature in °C
            
        Returns:
            Dictionary with reference measurement data
        """
        print(f"Performing reference measurement at SoC={soc:.2f}, T={temperature}°C...")
        
        # Set the temperature control
        self.analyzer.set_advanced_parameters(
            target_temperature=temperature
        )
        
        # Perform the measurement
        # In a real implementation, this would use actual hardware
        # For demonstration, we simulate the measurement
        measurement = self._simulate_measurement(
            soc=soc,
            temperature=temperature,
            soh=1.0,  # New battery
            cycle_count=0
        )
        
        # Store as reference
        self.reference_eis = measurement['electrical_impedance']
        self.reference_tis = measurement['thermal_impedance']
        
        print("Reference measurement completed and stored.")
        return measurement
    
    def measure_battery(self, soc, temperature=None, cycle_count=None):
        """
        Measure the battery and analyze its health
        
        Args:
            soc: State of charge (0.0-1.0)
            temperature: Battery temperature in °C (optional)
            cycle_count: Battery cycle count (optional)
            
        Returns:
            Dictionary with battery health metrics
        """
        if temperature is None:
            temperature = REFERENCE_TEMPERATURE
        
        if cycle_count is not None:
            self.cycle_count = cycle_count
            
        print(f"Measuring battery at SoC={soc:.2f}, T={temperature}°C, Cycle={self.cycle_count}...")
        
        # Check if we have a reference measurement
        if self.reference_eis is None or self.reference_tis is None:
            print("No reference data available. Performing reference measurement first.")
            self.perform_reference_measurement(soc, REFERENCE_TEMPERATURE)
        
        # Set the temperature control if needed
        if temperature != self.analyzer.get_advanced_parameters()['target_temperature']:
            self.analyzer.set_advanced_parameters(
                target_temperature=temperature
            )
        
        # Get the current time
        timestamp = datetime.now()
        
        # Perform the measurement
        # In a real implementation, this would use actual hardware
        # For demonstration, we simulate the measurement
        measurement = self._simulate_measurement(
            soc=soc,
            temperature=temperature,
            soh=self._estimate_soh_from_cycle_count(self.cycle_count),
            cycle_count=self.cycle_count
        )
        
        # Store measurement data
        self.measurements.append(measurement)
        self.timestamps.append(timestamp)
        self.soc_values.append(soc)
        self.temperature_values.append(temperature)
        
        # Extract equivalent circuit parameters
        ecm_params = self._extract_equivalent_circuit_parameters(measurement)
        
        # Update ECM parameter history
        for param, value in ecm_params.items():
            self.ecm_parameters[param].append(value)
        
        # Estimate SOH based on impedance
        soh = self._estimate_soh(measurement, ecm_params)
        self.soh_values.append(soh)
        
        # Store internal and thermal resistance
        self.internal_resistance_values.append(ecm_params['Rs'] + ecm_params['Rct'])
        self.thermal_resistance_values.append(ecm_params['Rt'])
        
        # Prepare results
        results = {
            'timestamp': timestamp,
            'soc': soc,
            'temperature': temperature,
            'cycle_count': self.cycle_count,
            'soh': soh,
            'internal_resistance': ecm_params['Rs'] + ecm_params['Rct'],
            'thermal_resistance': ecm_params['Rt'],
            'ecm_parameters': ecm_params,
            'measurement': measurement
        }
        
        # Store cycle data if this is a new measurement cycle
        if len(self.cycle_data) == 0 or self.cycle_data[-1]['cycle_count'] != self.cycle_count:
            self.cycle_data.append({
                'cycle_count': self.cycle_count,
                'soh': soh,
                'internal_resistance': ecm_params['Rs'] + ecm_params['Rct'],
                'thermal_resistance': ecm_params['Rt'],
                'timestamp': timestamp
            })
        
        print(f"Measurement completed. Estimated SoH: {soh:.1f}%")
        return results
    
    def _simulate_measurement(self, soc, temperature, soh, cycle_count):
        """
        Simulate battery impedance measurements
        
        In a real implementation, this would be replaced with actual measurements
        from the impedance analyzer hardware
        
        Args:
            soc: State of charge (0.0-1.0)
            temperature: Battery temperature in °C
            soh: State of health (0.0-1.0)
            cycle_count: Battery cycle count
            
        Returns:
            Simulated measurement data
        """
        # Base parameters for a typical Li-ion battery
        # These will be modified based on SOC, temperature, and SOH
        r_s_base = self.battery_params['internal_resistance_new']  # Series resistance
        r_ct_base = 0.02  # Charge transfer resistance
        c_dl_base = 0.5   # Double layer capacitance
        r_w_base = 0.015  # Warburg coefficient
        r_t_base = self.battery_params['thermal_resistance_new']  # Thermal resistance
        c_t_base = 80.0   # Thermal capacitance
        
        # Adjust parameters based on SOC
        # Series resistance increases at very low and very high SOC
        soc_factor_rs = 1.0 + 0.2 * (1.0 - np.exp(-5.0 * (1.0 - soc))) + 0.3 * (1.0 - np.exp(-10.0 * soc))
        # Charge transfer resistance increases at low SOC
        soc_factor_rct = 1.0 + 1.5 * (1.0 - np.exp(-3.0 * (1.0 - soc)))
        # Double layer capacitance decreases at low SOC
        soc_factor_cdl = 1.0 - 0.3 * (1.0 - np.exp(-5.0 * (1.0 - soc)))
        # Warburg impedance increases at very low and very high SOC
        soc_factor_rw = 1.0 + 0.5 * (1.0 - np.exp(-5.0 * (1.0 - soc))) + 0.2 * (1.0 - np.exp(-5.0 * soc))
        
        # Adjust parameters based on temperature
        # Arrhenius-like temperature dependence
        temp_factor_rs = np.exp(3000 * (1/298.15 - 1/(temperature + 273.15)))
        temp_factor_rct = np.exp(5000 * (1/298.15 - 1/(temperature + 273.15)))
        temp_factor_cdl = np.exp(1000 * (1/(temperature + 273.15) - 1/298.15))
        temp_factor_rt = 1.0 + 0.01 * (REFERENCE_TEMPERATURE - temperature)
        temp_factor_ct = 1.0 - 0.005 * (REFERENCE_TEMPERATURE - temperature)
        
        # Adjust parameters based on SOH (aging effects)
        # Linear degradation model for simplicity
        aging_factor = 1.0 + (1.0 - soh) * 1.5
        
        # Additional randomness to simulate measurement noise
        noise_factor = 1.0 + 0.05 * (np.random.random() - 0.5)
        
        # Combine all factors
        r_s = r_s_base * soc_factor_rs * temp_factor_rs * aging_factor * noise_factor
        r_ct = r_ct_base * soc_factor_rct * temp_factor_rct * aging_factor * noise_factor
        c_dl = c_dl_base * soc_factor_cdl * temp_factor_cdl / aging_factor * noise_factor
        r_w = r_w_base * soc_factor_rw * aging_factor * noise_factor
        r_t = r_t_base * temp_factor_rt * aging_factor * noise_factor
        c_t = c_t_base * temp_factor_ct / aging_factor * noise_factor
        
        # Generate electrical impedance data
        eis_real = []
        eis_imag = []
        
        for freq in EIS_FREQUENCIES:
            omega = 2 * np.pi * freq
            # Series resistance contribution
            z_real = r_s
            z_imag = 0
            
            # Charge transfer and double layer capacitance (parallel RC)
            z_rc_real = r_ct / (1 + (omega * r_ct * c_dl)**2)
            z_rc_imag = -omega * (r_ct**2) * c_dl / (1 + (omega * r_ct * c_dl)**2)
            
            # Warburg impedance (diffusion)
            z_w_real = r_w / np.sqrt(omega)
            z_w_imag = -r_w / np.sqrt(omega)
            
            # Combine impedances
            z_real += z_rc_real + z_w_real
            z_imag += z_rc_imag + z_w_imag
            
            # Add some random noise
            z_real += 0.001 * r_s * (np.random.random() - 0.5)
            z_imag += 0.001 * r_s * (np.random.random() - 0.5)
            
            eis_real.append(z_real)
            eis_imag.append(z_imag)
        
        # Generate thermal impedance data
        tis_real = []
        tis_imag = []
        
        for freq in TIS_FREQUENCIES:
            omega = 2 * np.pi * freq
            # Simple RC thermal model
            z_real = r_t / (1 + (omega * r_t * c_t)**2)
            z_imag = -omega * (r_t**2) * c_t / (1 + (omega * r_t * c_t)**2)
            
            # Add some random noise
            z_real += 0.002 * r_t * (np.random.random() - 0.5)
            z_imag += 0.002 * r_t * (np.random.random() - 0.5)
            
            tis_real.append(z_real)
            tis_imag.append(z_imag)
        
        # Create measurement structure
        measurement = {
            'electrical_impedance': {
                'frequency': np.array(EIS_FREQUENCIES),
                'real': np.array(eis_real),
                'imaginary': np.array(eis_imag)
            },
            'thermal_impedance': {
                'frequency': np.array(TIS_FREQUENCIES),
                'real': np.array(tis_real),
                'imaginary': np.array(tis_imag)
            },
            'metadata': {
                'soc': soc,
                'temperature': temperature,
                'cycle_count': cycle_count,
                'actual_soh': soh,  # For validation purposes
                'parameters': {
                    'Rs': r_s,
                    'Rct': r_ct,
                    'Cdl': c_dl,
                    'Rw': r_w,
                    'Rt': r_t,
                    'Ct': c_t
                }
            }
        }
        
        return measurement
    
    def _extract_equivalent_circuit_parameters(self, measurement):
        """
        Extract equivalent circuit parameters from impedance measurements
        
        Args:
            measurement: Impedance measurement data
            
        Returns:
            Dictionary with extracted parameters
        """
        # Extract data
        eis_freq = measurement['electrical_impedance']['frequency']
        eis_real = measurement['electrical_impedance']['real']
        eis_imag = measurement['electrical_impedance']['imaginary']
        
        tis_freq = measurement['thermal_impedance']['frequency']
        tis_real = measurement['thermal_impedance']['real']
        tis_imag = measurement['thermal_impedance']['imaginary']
        
        # Define objective function for electrical impedance fitting
        def eis_residuals(params, freq, z_real, z_imag):
            r_s, r_ct, c_dl, r_w = params
            z_model_real = []
            z_model_imag = []
            
            for f in freq:
                omega = 2 * np.pi * f
                # Series resistance
                z_r = r_s
                z_i = 0
                
                # Charge transfer and double layer (parallel RC)
                z_rc_r = r_ct / (1 + (omega * r_ct * c_dl)**2)
                z_rc_i = -omega * (r_ct**2) * c_dl / (1 + (omega * r_ct * c_dl)**2)
                
                # Warburg impedance (diffusion)
                z_w_r = r_w / np.sqrt(omega)
                z_w_i = -r_w / np.sqrt(omega)
                
                # Combine impedances
                z_r += z_rc_r + z_w_r
                z_i += z_rc_i + z_w_i
                
                z_model_real.append(z_r)
                z_model_imag.append(z_i)
            
            residuals = np.concatenate([
                (np.array(z_model_real) - np.array(z_real)) / np.max(np.abs(z_real)),
                (np.array(z_model_imag) - np.array(z_imag)) / np.max(np.abs(z_imag))
            ])
            
            return residuals
        
        # Define objective function for thermal impedance fitting
        def tis_residuals(params, freq, z_real, z_imag):
            r_t, c_t = params
            z_model_real = []
            z_model_imag = []
            
            for f in freq:
                omega = 2 * np.pi * f
                # Simple RC thermal model
                z_r = r_t / (1 + (omega * r_t * c_t)**2)
                z_i = -omega * (r_t**2) * c_t / (1 + (omega * r_t * c_t)**2)
                
                z_model_real.append(z_r)
                z_model_imag.append(z_i)
            
            residuals = np.concatenate([
                (np.array(z_model_real) - np.array(z_real)) / np.max(np.abs(z_real)),
                (np.array(z_model_imag) - np.array(z_imag)) / np.max(np.abs(z_imag))
            ])
            
            return residuals
        
        # Initial parameter guesses
        p0_eis = [0.03, 0.02, 0.5, 0.015]  # [Rs, Rct, Cdl, Rw]
        p0_tis = [2.0, 80.0]  # [Rt, Ct]
        
        # Fit the models
        try:
            result_eis = opt.least_squares(
                eis_residuals, p0_eis, 
                args=(eis_freq, eis_real, eis_imag),
                bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
            )
            
            result_tis = opt.least_squares(
                tis_residuals, p0_tis, 
                args=(tis_freq, tis_real, tis_imag),
                bounds=([0, 0], [np.inf, np.inf])
            )
            
            # Extract fitted parameters
            r_s, r_ct, c_dl, r_w = result_eis.x
            r_t, c_t = result_tis.x
            
        except Exception as e:
            # Fallback to initial parameters if fitting fails
            print(f"Fitting failed: {e}. Using initial parameter estimates.")
            r_s, r_ct, c_dl, r_w = p0_eis
            r_t, c_t = p0_tis
        
        # Return parameters
        return {
            'Rs': r_s,
            'Rct': r_ct,
            'Cdl': c_dl,
            'Rw': r_w,
            'Rt': r_t,
            'Ct': c_t
        }
    
    def _estimate_soh(self, measurement, ecm_params=None):
        """
        Estimate battery state of health based on impedance measurements
        
        Args:
            measurement: Impedance measurement data
            ecm_params: Pre-extracted equivalent circuit parameters (optional)
            
        Returns:
            State of health (0-100%)
        """
        if ecm_params is None:
            ecm_params = self._extract_equivalent_circuit_parameters(measurement)
        
        # Calculate internal resistance (series + charge transfer)
        r_internal = ecm_params['Rs'] + ecm_params['Rct']
        
        # Calculate thermal resistance
        r_thermal = ecm_params['Rt']
        
        # Get reference values for new and end-of-life batteries
        r_internal_new = self.battery_params['internal_resistance_new']
        r_internal_eol = self.battery_params['internal_resistance_end']
        
        r_thermal_new = self.battery_params['thermal_resistance_new']
        r_thermal_eol = self.battery_params['thermal_resistance_end']
        
        # Calculate SOH based on internal resistance (higher resistance = lower SOH)
        # Linear interpolation between new and end-of-life values
        soh_r = 100 * (r_internal_eol - r_internal) / (r_internal_eol - r_internal_new)
        
        # Calculate SOH based on thermal resistance (higher resistance = lower SOH)
        soh_t = 100 * (r_thermal_eol - r_thermal) / (r_thermal_eol - r_thermal_new)
        
        # Combine electrical and thermal SOH estimates
        # Weight more heavily toward electrical SOH (80% electrical, 20% thermal)
        soh = 0.8 * soh_r + 0.2 * soh_t
        
        # Ensure SOH is in valid range [0, 100]
        soh = max(0, min(100, soh))
        
        return soh
    
    def _estimate_soh_from_cycle_count(self, cycle_count):
        """
        Estimate SOH based on cycle count - simplified model for simulation
        
        Args:
            cycle_count: Battery cycle count
            
        Returns:
            Estimated SOH (0.0-1.0)
        """
        # Simplified model: linear degradation until 80% SOH at 1000 cycles,
        # then accelerated degradation
        if cycle_count <= 1000:
            soh = 1.0 - 0.0002 * cycle_count  # Linear degradation: 0.02% per cycle
        else:
            soh = 0.8 - 0.0004 * (cycle_count - 1000)  # Accelerated degradation
        
        # Ensure SOH is in valid range [0, 1]
        soh = max(0, min(1, soh))
        return soh
    
    def simulate_aging(self, cycles_to_add, measurements_per_cycle=1):
        """
        Simulate battery aging by adding cycles and taking measurements
        
        Args:
            cycles_to_add: Number of cycles to simulate
            measurements_per_cycle: Number of measurements to take per cycle
            
        Returns:
            None
        """
        print(f"Simulating {cycles_to_add} aging cycles...")
        
        # Store initial cycle count
        initial_cycle = self.cycle_count
        
        # Perform measurements at different cycle counts
        for i in range(cycles_to_add):
            # Update cycle count
            self.cycle_count = initial_cycle + i + 1
            
            # Take measurements at different SOC values within the cycle
            for j in range(measurements_per_cycle):
                soc = 1.0 - j / max(1, measurements_per_cycle - 1) * 0.8  # SOC from 1.0 to 0.2
                
                # Randomize temperature slightly to simulate real conditions
                temperature = REFERENCE_TEMPERATURE + np.random.normal(0, 2)
                
                # Perform measurement
                self.measure_battery(soc, temperature)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == cycles_to_add - 1:
                print(f"Completed {i + 1}/{cycles_to_add} cycles. Current SOH: {self.soh_values[-1]:.1f}%")
    
    def plot_impedance_spectra(self, measurement_index=-1):
        """
        Plot electrical and thermal impedance spectra for a measurement
        
        Args:
            measurement_index: Index of measurement to plot (-1 for most recent)
            
        Returns:
            None
        """
        if not self.measurements:
            print("No measurements available.")
            return
        
        # Get measurement data
        measurement = self.measurements[measurement_index]
        timestamp = self.timestamps[measurement_index]
        soc = self.soc_values[measurement_index]
        soh = self.soh_values[measurement_index]
        
        # Extract impedance data
        eis_freq = measurement['electrical_impedance']['frequency']
        eis_real = measurement['electrical_impedance']['real']
        eis_imag = measurement['electrical_impedance']['imaginary']
        eis_mag = np.sqrt(eis_real**2 + eis_imag**2)
        eis_phase = np.arctan2(eis_imag, eis_real) * 180 / np.pi
        
        tis_freq = measurement['thermal_impedance']['frequency']
        tis_real = measurement['thermal_impedance']['real']
        tis_imag = measurement['thermal_impedance']['imaginary']
        tis_mag = np.sqrt(tis_real**2 + tis_imag**2)
        tis_phase = np.arctan2(tis_imag, tis_real) * 180 / np.pi
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Plot Nyquist plot for EIS
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(eis_real, -eis_imag, 'o-', color='blue')
        ax1.set_xlabel('Real Impedance (Ω)')
        ax1.set_ylabel('-Imaginary Impedance (Ω)')
        ax1.set_title('Electrical Impedance Nyquist Plot')
        ax1.grid(True)
        
        # Add some frequency markers
        for i, f in enumerate(eis_freq):
            if i % 5 == 0:  # Add marker every 5th frequency point
                ax1.annotate(f'{f:.1f} Hz', (eis_real[i], -eis_imag[i]),
                           xytext=(5, 5), textcoords='offset points')
        
        # Plot Bode plot for EIS (magnitude)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.loglog(eis_freq, eis_mag, 'o-', color='blue')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('|Z| (Ω)')
        ax2.set_title('Electrical Impedance Magnitude')
        ax2.grid(True, which='both')
        
        # Plot Bode plot for EIS (phase)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.semilogx(eis_freq, eis_phase, 'o-', color='blue')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Phase (degrees)')
        ax3.set_title('Electrical Impedance Phase')
        ax3.grid(True)
        
        # Plot Nyquist plot for TIS
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(tis_real, -tis_imag, 'o-', color='red')
        ax4.set_xlabel('Real Thermal Impedance (K/W)')
        ax4.set_ylabel('-Imaginary Thermal Impedance (K/W)')
        ax4.set_title('Thermal Impedance Nyquist Plot')
        ax4.grid(True)
        
        # Add some frequency markers
        for i, f in enumerate(tis_freq):
            if i % 3 == 0:  # Add marker every 3rd frequency point
                ax4.annotate(f'{f:.3f} Hz', (tis_real[i], -tis_imag[i]),
                           xytext=(5, 5), textcoords='offset points')
        
        # Plot Bode plot for TIS (magnitude)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.loglog(tis_freq, tis_mag, 'o-', color='red')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('|Z| (K/W)')
        ax5.set_title('Thermal Impedance Magnitude')
        ax5.grid(True, which='both')
        
        # Plot battery parameters
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create a text summary
        params_text = [
            f"Measurement Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Battery Type: {self.battery_type}",
            f"State of Charge: {soc:.2f}",
            f"State of Health: {soh:.1f}%",
            f"Cycle Count: {measurement['metadata']['cycle_count']}",
            f"Temperature: {measurement['metadata']['temperature']:.1f}°C",
            "",
            "Equivalent Circuit Parameters:",
            f"Series Resistance (Rs): {self.ecm_parameters['Rs'][measurement_index]:.6f} Ω",
            f"Charge Transfer Resistance (Rct): {self.ecm_parameters['Rct'][measurement_index]:.6f} Ω",
            f"Double Layer Capacitance (Cdl): {self.ecm_parameters['Cdl'][measurement_index]:.6f} F",
            f"Warburg Coefficient (Rw): {self.ecm_parameters['Rw'][measurement_index]:.6f} Ω·s^(-1/2)",
            f"Thermal Resistance (Rt): {self.ecm_parameters['Rt'][measurement_index]:.6f} K/W",
            f"Thermal Capacitance (Ct): {self.ecm_parameters['Ct'][measurement_index]:.6f} J/K",
            "",
            f"Total Internal Resistance: {self.internal_resistance_values[measurement_index]:.6f} Ω",
            f"Relative to New: {(self.internal_resistance_values[measurement_index]/self.battery_params['internal_resistance_new']-1)*100:.1f}% increase"
        ]
        
        # Display summary text
        plt.text(0.05, 0.95, "\n".join(params_text), transform=ax6.transAxes, 
                 fontsize=10, va='top')
        
        # Add main title
        plt.suptitle(f"Battery Impedance Analysis - SoH: {soh:.1f}%", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()
    
    def plot_aging_trends(self):
        """
        Plot aging trends from multiple measurements
        
        Returns:
            None
        """
        if len(self.soh_values) < 2:
            print("Not enough measurements for trend analysis.")
            return
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot SOH vs. cycle count
        cycles = [m['metadata']['cycle_count'] for m in self.measurements]
        
        # Filter data to unique cycle measurements (use cycle_data for cleaner plots)
        cycle_numbers = [data['cycle_count'] for data in self.cycle_data]
        cycle_soh = [data['soh'] for data in self.cycle_data]
        cycle_r_internal = [data['internal_resistance'] for data in self.cycle_data]
        cycle_r_thermal = [data['thermal_resistance'] for data in self.cycle_data]
        
        # Plot SOH vs cycle count
        axs[0, 0].plot(cycle_numbers, cycle_soh, 'o-', color='blue')
        axs[0, 0].set_xlabel('Cycle Count')
        axs[0, 0].set_ylabel('State of Health (%)')
        axs[0, 0].set_title('SOH vs. Cycle Count')
        axs[0, 0].grid(True)
        
        # Add 80% SOH line (typical end-of-life threshold)
        axs[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.7)
        axs[0, 0].text(min(cycle_numbers), 80, '80% EOL Threshold', va='bottom', ha='left', color='r')
        
        # Plot internal resistance vs cycle count
        axs[0, 1].plot(cycle_numbers, cycle_r_internal, 'o-', color='green')
        axs[0, 1].set_xlabel('Cycle Count')
        axs[0, 1].set_ylabel('Internal Resistance (Ω)')
        axs[0, 1].set_title('Internal Resistance vs. Cycle Count')
        axs[0, 1].grid(True)
        
        # Calculate percent change in internal resistance
        r_internal_new = self.battery_params['internal_resistance_new']
        r_percent = [(r/r_internal_new-1)*100 for r in cycle_r_internal]
        
        # Plot internal resistance percentage change
        axs[1, 0].plot(cycle_numbers, r_percent, 'o-', color='purple')
        axs[1, 0].set_xlabel('Cycle Count')
        axs[1, 0].set_ylabel('Internal Resistance Increase (%)')
        axs[1, 0].set_title('Internal Resistance Change vs. Cycle Count')
        axs[1, 0].grid(True)
        
        # Plot thermal resistance vs cycle count
        axs[1, 1].plot(cycle_numbers, cycle_r_thermal, 'o-', color='red')
        axs[1, 1].set_xlabel('Cycle Count')
        axs[1, 1].set_ylabel('Thermal Resistance (K/W)')
        axs[1, 1].set_title('Thermal Resistance vs. Cycle Count')
        axs[1, 1].grid(True)
        
        # Add main title
        plt.suptitle(f"Battery Aging Analysis - {self.battery_type}", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    def predict_remaining_life(self):
        """
        Predict remaining useful life of the battery
        
        Returns:
            Dictionary with remaining life predictions
        """
        if len(self.cycle_data) < 2:
            print("Not enough measurement cycles for prediction.")
            return None
        
        # Extract data from unique cycle measurements
        cycle_numbers = np.array([data['cycle_count'] for data in self.cycle_data])
        cycle_soh = np.array([data['soh'] for data in self.cycle_data])
        
        # Simple linear regression for SOH trend
        if len(cycle_numbers) >= 2:
            # Use polynomial fitting to model aging
            # 2nd order polynomial seems to work well for most battery aging profiles
            p = np.polyfit(cycle_numbers, cycle_soh, 2)
            
            # Create a function to predict SOH
            soh_prediction = np.poly1d(p)
            
            # Calculate cycles to reach 80% SOH (typical EOL threshold)
            # Solve quadratic equation: p[0]*x^2 + p[1]*x + p[2] - 80 = 0
            if p[0] != 0:  # If it's truly quadratic
                # Calculate discriminant
                discriminant = p[1]**2 - 4*p[0]*(p[2]-80)
                if discriminant >= 0:
                    # Two solutions
                    x1 = (-p[1] + np.sqrt(discriminant)) / (2*p[0])
                    x2 = (-p[1] - np.sqrt(discriminant)) / (2*p[0])
                    # Choose the solution that is greater than the current cycle
                    # and makes sense in context
                    current_cycle = cycle_numbers[-1]
                    if x1 > current_cycle and soh_prediction(x1) < cycle_soh[-1]:
                        cycles_to_eol = x1 - current_cycle
                    elif x2 > current_cycle and soh_prediction(x2) < cycle_soh[-1]:
                        cycles_to_eol = x2 - current_cycle
                    else:
                        # No valid solution, fall back to linear extrapolation
                        slope = (cycle_soh[-1] - cycle_soh[0]) / (cycle_numbers[-1] - cycle_numbers[0])
                        if slope < 0:  # Only if SOH is decreasing
                            cycles_to_eol = (80 - cycle_soh[-1]) / slope
                        else:
                            cycles_to_eol = float('inf')
                else:
                    # No real solutions, fall back to linear extrapolation
                    slope = (cycle_soh[-1] - cycle_soh[0]) / (cycle_numbers[-1] - cycle_numbers[0])
                    if slope < 0:  # Only if SOH is decreasing
                        cycles_to_eol = (80 - cycle_soh[-1]) / slope
                    else:
                        cycles_to_eol = float('inf')
            else:
                # Linear case
                slope = p[1]
                if slope < 0:  # Only if SOH is decreasing
                    cycles_to_eol = (80 - cycle_soh[-1]) / slope
                else:
                    cycles_to_eol = float('inf')
            
            # Ensure cycles_to_eol is positive and finite
            cycles_to_eol = max(0, cycles_to_eol)
            if not np.isfinite(cycles_to_eol):
                cycles_to_eol = float('inf')
            
            # Generate prediction curve for plotting
            future_cycles = np.linspace(cycle_numbers[0], cycle_numbers[-1] + cycles_to_eol * 1.2, 100)
            future_soh = soh_prediction(future_cycles)
            
            # Calculate coefficients of determination (R-squared)
            soh_fit = soh_prediction(cycle_numbers)
            ss_total = np.sum((cycle_soh - np.mean(cycle_soh))**2)
            ss_residual = np.sum((cycle_soh - soh_fit)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            # Plot prediction
            plt.figure(figsize=(12, 8))
            plt.plot(cycle_numbers, cycle_soh, 'o-', label='Measured SOH')
            plt.plot(future_cycles, future_soh, '--', label='Predicted SOH Trend')
            plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% EOL Threshold')
            
            # Mark predicted EOL point
            if np.isfinite(cycles_to_eol):
                eol_cycle = cycle_numbers[-1] + cycles_to_eol
                plt.plot([eol_cycle], [80], 'ro', markersize=10)
                plt.annotate(f'Predicted EOL: {eol_cycle:.0f} cycles',
                           xy=(eol_cycle, 80), xytext=(-80, -30),
                           textcoords='offset points', fontsize=12,
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            plt.xlabel('Cycle Count')
            plt.ylabel('State of Health (%)')
            plt.title('Battery State of Health Prediction')
            plt.grid(True)
            plt.legend()
            
            # Add text box with prediction details
            textstr = '\n'.join((
                f'Current SOH: {cycle_soh[-1]:.1f}%',
                f'Current Cycle: {cycle_numbers[-1]}',
                f'Estimated Cycles to 80% SOH: {cycles_to_eol:.0f}',
                f'Model R-squared: {r_squared:.3f}'
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.show()
            
            # Return prediction results
            prediction_results = {
                'current_soh': cycle_soh[-1],
                'current_cycle': cycle_numbers[-1],
                'cycles_to_eol': cycles_to_eol,
                'predicted_eol_cycle': cycle_numbers[-1] + cycles_to_eol,
                'r_squared': r_squared,
                'model_coefficients': p.tolist()
            }
            
            print(f"Current SOH: {cycle_soh[-1]:.1f}%")
            print(f"Current Cycle: {cycle_numbers[-1]}")
            print(f"Estimated Cycles to 80% SOH: {cycles_to_eol:.0f}")
            print(f"Prediction Model R-squared: {r_squared:.3f}")
            
            return prediction_results
        else:
            print("Not enough data points for prediction.")
            return None
    
    def export_data(self, filename):
        """
        Export battery measurement data to CSV
        
        Args:
            filename: Output filename
            
        Returns:
            None
        """
        if not self.measurements:
            print("No measurement data to export.")
            return
        
        # Create a DataFrame with basic measurement data
        data = {
            'timestamp': self.timestamps,
            'cycle_count': [m['metadata']['cycle_count'] for m in self.measurements],
            'soc': self.soc_values,
            'temperature': self.temperature_values,
            'soh': self.soh_values,
            'internal_resistance': self.internal_resistance_values,
            'thermal_resistance': self.thermal_resistance_values
        }
        
        # Add ECM parameters
        for param, values in self.ecm_parameters.items():
            if values:  # Only add if we have values
                data[f'ecm_{param}'] = values
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Battery measurement data exported to {filename}")
        
        # Also export cycle data
        if self.cycle_data:
            cycle_df = pd.DataFrame(self.cycle_data)
            cycle_filename = filename.replace('.csv', '_cycles.csv')
            cycle_df.to_csv(cycle_filename, index=False)
            print(f"Battery cycle data exported to {cycle_filename}")


def main():
    """Main function to demonstrate the battery health monitoring system"""
    # Create an instance of the battery health monitoring system
    battery_monitor = BatteryHealthMonitoring(battery_type='li_ion_18650')
    
    # Perform a reference measurement at full charge
    battery_monitor.perform_reference_measurement(soc=1.0)
    
    # Simulate battery aging
    battery_monitor.simulate_aging(cycles_to_add=200, measurements_per_cycle=2)
    
    # Plot impedance spectra for the latest measurement
    battery_monitor.plot_impedance_spectra()
    
    # Plot aging trends
    battery_monitor.plot_aging_trends()
    
    # Predict remaining useful life
    battery_monitor.predict_remaining_life()
    
    # Export the data
    battery_monitor.export_data("battery_monitoring_data.csv")


if __name__ == "__main__":
    main()
