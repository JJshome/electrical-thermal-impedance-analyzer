"""
Battery Analysis Example

This example demonstrates how to use the Integrated Electrical-Thermal Impedance Analyzer
for battery health assessment and thermal characterization.

Based on the methodology described in:
Barsoukov et al., "Thermal impedance spectroscopy for Li-ion batteries 
using heat-pulse response analysis", Journal of Power Sources, 2002

Patent: 열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
logger = logging.getLogger("BatteryAnalysis")

class LithiumIonBattery:
    """
    Simplified Li-ion battery model for demonstration purposes
    
    This class simulates a Li-ion battery with aging and temperature-dependent
    characteristics.
    """
    
    def __init__(self, 
                 capacity=3000,  # mAh
                 cycle_count=0,
                 temperature=25.0,  # °C
                 soc=1.0,  # State of Charge (0-1)
                 age=0.0):  # Years
        """
        Initialize LithiumIonBattery model
        
        Parameters:
        -----------
        capacity : float
            Nominal capacity in mAh
        cycle_count : int
            Number of charge-discharge cycles
        temperature : float
            Battery temperature in °C
        soc : float
            State of charge (0-1)
        age : float
            Calendar age in years
        """
        self.nominal_capacity = capacity  # mAh
        self.cycle_count = cycle_count
        self.temperature = temperature  # °C
        self.soc = soc  # State of Charge (0-1)
        self.age = age  # Years
        
        # Internal parameters
        self.internal_resistance = 0.05  # Ohm
        self.charge_transfer_resistance = 0.1  # Ohm
        self.capacitance = 1.0  # F
        self.thermal_resistance = 2.0  # K/W
        self.thermal_capacitance = 1000.0  # J/K
        
        # Update parameters based on initial state
        self._update_parameters()
        
    def _update_parameters(self):
        """
        Update battery parameters based on current state
        
        This method simulates how battery parameters change with aging,
        temperature, and state of charge.
        """
        # Capacity degradation with cycling and age
        cycle_factor = 1.0 - 0.0002 * self.cycle_count
        age_factor = 1.0 - 0.05 * self.age
        self.capacity = self.nominal_capacity * cycle_factor * age_factor
        
        # Temperature effects on resistance
        temp_factor = np.exp(-0.03 * (self.temperature - 25.0))  # Reference at 25°C
        
        # SOC effects on resistance and capacitance
        soc_factor = 1.0 + 0.5 * (1.0 - self.soc)**2
        
        # Update electrical parameters
        self.internal_resistance = 0.05 * soc_factor * temp_factor * (1.0 + 0.01 * self.cycle_count + 0.05 * self.age)
        self.charge_transfer_resistance = 0.1 * soc_factor * temp_factor * (1.0 + 0.02 * self.cycle_count + 0.1 * self.age)
        self.capacitance = 1.0 / (soc_factor * (1.0 + 0.005 * self.cycle_count + 0.02 * self.age))
        
        # Update thermal parameters
        self.thermal_resistance = 2.0 * (1.0 + 0.01 * self.cycle_count + 0.03 * self.age)
        self.thermal_capacitance = 1000.0 * (1.0 - 0.005 * self.cycle_count - 0.01 * self.age)
    
    def cycle(self, num_cycles=1):
        """
        Simulate battery cycling
        
        Parameters:
        -----------
        num_cycles : int
            Number of cycles to simulate
        """
        self.cycle_count += num_cycles
        self._update_parameters()
        logger.info(f"Performed {num_cycles} cycles. New cycle count: {self.cycle_count}")
        
    def set_temperature(self, temperature):
        """
        Set battery temperature
        
        Parameters:
        -----------
        temperature : float
            Battery temperature in °C
        """
        self.temperature = temperature
        self._update_parameters()
        logger.info(f"Temperature set to {temperature}°C")
        
    def set_soc(self, soc):
        """
        Set battery state of charge
        
        Parameters:
        -----------
        soc : float
            State of charge (0-1)
        """
        self.soc = max(0.0, min(1.0, soc))
        self._update_parameters()
        logger.info(f"SOC set to {self.soc*100:.1f}%")
        
    def age_battery(self, years=0.1):
        """
        Simulate battery aging
        
        Parameters:
        -----------
        years : float
            Years of aging to simulate
        """
        self.age += years
        self._update_parameters()
        logger.info(f"Aged battery by {years} years. New age: {self.age:.1f} years")
        
    def get_state_of_health(self):
        """
        Calculate state of health
        
        Returns:
        --------
        soh : float
            State of health (0-100%)
        """
        return 100.0 * self.capacity / self.nominal_capacity
        
    def __str__(self):
        """String representation of battery state"""
        return (f"Li-ion Battery: {self.nominal_capacity} mAh, {self.cycle_count} cycles, {self.age:.1f} years\n"
                f"  Current Capacity: {self.capacity:.1f} mAh\n"
                f"  State of Charge: {self.soc*100:.1f}%\n"
                f"  Temperature: {self.temperature:.1f}°C\n"
                f"  State of Health: {self.get_state_of_health():.1f}%\n"
                f"  Internal Resistance: {self.internal_resistance:.4f} Ohm\n"
                f"  Thermal Resistance: {self.thermal_resistance:.2f} K/W")


def perform_battery_analysis(battery, analyzer, plot_results=True):
    """
    Perform comprehensive battery analysis using integrated impedance analyzer
    
    Parameters:
    -----------
    battery : LithiumIonBattery
        Battery to analyze
    analyzer : IntegratedImpedanceAnalyzer
        Configured impedance analyzer
    plot_results : bool
        Whether to plot results
        
    Returns:
    --------
    analysis_results : dict
        Dictionary containing analysis results
    """
    logger.info("Starting comprehensive battery analysis")
    
    # Measure impedance
    logger.info("Measuring impedance spectra")
    measurement_results = analyzer.measure(target_system=battery)
    
    # Analyze results
    logger.info("Analyzing impedance data")
    characteristics = analyzer.analyze(measurement_results)
    
    # Extract key parameters
    e_params = characteristics['electrical_parameters']
    t_params = characteristics['thermal_parameters']
    i_params = characteristics['integrated_parameters']
    
    # Calculate additional battery-specific metrics
    remaining_capacity = battery.nominal_capacity * i_params['state_of_health'] / 100.0
    
    # Estimated power dissipation during 1C discharge
    discharge_current = battery.nominal_capacity / 1000.0  # A (1C rate)
    power_dissipation = e_params['R_total'] * (discharge_current ** 2)  # W
    
    # Temperature rise during continuous 1C discharge
    temp_rise_1c = power_dissipation * t_params['R_th']  # K
    
    # Calculate maximum safe continuous discharge rate based on thermal limits
    # Assuming max allowed temperature rise is 10°C
    max_temp_rise = 10.0  # °C
    max_power = max_temp_rise / t_params['R_th']  # W
    max_current = np.sqrt(max_power / e_params['R_total'])  # A
    max_c_rate = max_current / (battery.nominal_capacity / 1000.0)  # C
    
    # Combine all results
    analysis_results = {
        'timestamp': datetime.now(),
        'battery_info': {
            'nominal_capacity': battery.nominal_capacity,
            'cycle_count': battery.cycle_count,
            'age': battery.age,
            'temperature': battery.temperature,
            'soc': battery.soc,
            'actual_capacity': battery.capacity
        },
        'measured_characteristics': characteristics,
        'derived_metrics': {
            'state_of_health': i_params['state_of_health'],
            'remaining_capacity': remaining_capacity,
            'power_dissipation_1c': power_dissipation,
            'temp_rise_1c': temp_rise_1c,
            'max_safe_c_rate': max_c_rate
        }
    }
    
    # Print summary
    print("\nBattery Analysis Results:")
    print("========================")
    print(f"Battery: {battery.nominal_capacity} mAh, {battery.cycle_count} cycles, {battery.age:.1f} years")
    print(f"Temperature: {battery.temperature:.1f}°C, SOC: {battery.soc*100:.1f}%")
    print("\nHealth Assessment:")
    print(f"  State of Health: {i_params['state_of_health']:.1f}%")
    print(f"  Remaining Capacity: {remaining_capacity:.1f} mAh")
    print(f"  Internal Resistance: {e_params['R_total']:.4f} Ohm")
    
    print("\nThermal Characteristics:")
    print(f"  Thermal Resistance: {t_params['R_th']:.2f} K/W")
    print(f"  Thermal Capacitance: {t_params['C_th']:.1f} J/K")
    print(f"  Thermal Time Constant: {t_params['thermal_time_constant']:.1f} s")
    
    print("\nOperational Limits:")
    print(f"  Power Dissipation at 1C: {power_dissipation:.2f} W")
    print(f"  Temperature Rise at 1C: {temp_rise_1c:.1f}°C")
    print(f"  Maximum Safe Continuous Discharge Rate: {max_c_rate:.1f}C")
    
    # Plot if requested
    if plot_results:
        analyzer.plot_impedance_spectra(measurement_results)
        plot_battery_characteristics(analysis_results)
    
    return analysis_results


def plot_battery_characteristics(analysis_results):
    """
    Create visualizations of battery characteristics
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results
    """
    # Extract data
    battery_info = analysis_results['battery_info']
    e_params = analysis_results['measured_characteristics']['electrical_parameters']
    t_params = analysis_results['measured_characteristics']['thermal_parameters']
    i_params = analysis_results['measured_characteristics']['integrated_parameters']
    derived = analysis_results['derived_metrics']
    
    # Plot 1: Battery Health and Performance
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # SOH and Remaining Capacity
    ax1 = axs[0, 0]
    ax1.bar(['State of Health', 'Remaining Capacity'], 
           [derived['state_of_health'], 
            derived['remaining_capacity']/battery_info['nominal_capacity']*100], 
           color=['blue', 'green'])
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Battery Health')
    
    # Resistance components
    ax2 = axs[0, 1]
    resistances = [e_params['R_s'], e_params['R_ct'], e_params['Z_W_mag']]
    labels = ['Series Resistance', 'Charge Transfer\nResistance', 'Diffusion\nResistance']
    ax2.bar(labels, resistances, color=['lightblue', 'skyblue', 'steelblue'])
    ax2.set_ylabel('Resistance (Ohm)')
    ax2.set_title('Resistance Components')
    
    # Thermal characteristics
    ax3 = axs[1, 0]
    ax3.bar(['Thermal Resistance', 'Thermal Capacitance', 'Thermal Time Constant'],
           [t_params['R_th']/2.0, t_params['C_th']/1000.0, t_params['thermal_time_constant']/2000.0],
           color=['salmon', 'tomato', 'red'])
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Thermal Characteristics')
    ax3.text(0, -0.15, f"R_th = {t_params['R_th']:.2f} K/W", transform=ax3.transData)
    ax3.text(1, -0.15, f"C_th = {t_params['C_th']:.0f} J/K", transform=ax3.transData)
    ax3.text(2, -0.15, f"τ = {t_params['thermal_time_constant']:.0f} s", transform=ax3.transData)
    
    # Operational limits
    ax4 = axs[1, 1]
    c_rates = np.array([0.5, 1.0, 2.0, derived['max_safe_c_rate']])
    power = e_params['R_total'] * (c_rates * battery_info['nominal_capacity']/1000.0)**2
    temp_rise = power * t_params['R_th']
    
    ax4_twin = ax4.twinx()
    bars = ax4.bar(c_rates.astype(str), power, color='lightblue', alpha=0.7)
    ax4.set_xlabel('Discharge Rate (C)')
    ax4.set_ylabel('Power Dissipation (W)', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    
    ax4_twin.plot(range(len(c_rates)), temp_rise, 'ro-', linewidth=2)
    ax4_twin.set_ylabel('Temperature Rise (°C)', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4_twin.set_ylim(0, 12)
    ax4_twin.axhline(y=10, color='red', linestyle='--', alpha=0.5)
    ax4_twin.text(len(c_rates)-1.5, 10.2, 'Safety Limit', color='red')
    
    ax4.set_title('Operational Limits')
    ax4.set_xticks(range(len(c_rates)))
    ax4.set_xticklabels([f"{cr}C" for cr in c_rates])
    
    plt.tight_layout()
    plt.show()


def battery_aging_study(cycles=100, plot=True):
    """
    Conduct a battery aging study
    
    This function simulates battery aging through cycling and performs
    impedance analysis at regular intervals.
    
    Parameters:
    -----------
    cycles : int
        Total number of cycles to simulate
    plot : bool
        Whether to plot the aging trends
        
    Returns:
    --------
    aging_data : pd.DataFrame
        DataFrame containing aging data
    """
    logger.info(f"Starting battery aging study for {cycles} cycles")
    
    # Initialize battery and analyzer
    battery = LithiumIonBattery(capacity=3000, cycle_count=0, age=0.0)
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure analyzer
    analyzer.configure(
        electrical_freq_range=(0.1, 10000),  # Hz
        thermal_freq_range=(0.01, 1),        # Hz
        voltage_amplitude=10e-3,             # V
        thermal_pulse_power=100e-3,          # W
    )
    
    # Calibrate analyzer
    analyzer.calibrate()
    
    # Define measurement intervals
    measurement_cycles = np.unique(np.concatenate([
        [0],  # Initial state
        np.arange(10, cycles, 10),  # Every 10 cycles
        [cycles]  # Final state
    ]))
    
    # Storage for results
    results = []
    
    # Perform aging and measurements
    for cycle in range(cycles + 1):
        if cycle > 0:
            # Cycle the battery
            battery.cycle(1)
            
            # Add some random temperature variation (normal operation)
            temp_variation = np.random.normal(0, 1)
            battery.set_temperature(25.0 + temp_variation)
            
            # Vary SOC (simulate usage patterns)
            soc_variation = 0.7 + 0.3 * np.random.random()
            battery.set_soc(soc_variation)
        
        # Perform measurement at specified intervals
        if cycle in measurement_cycles:
            logger.info(f"Performing measurement at cycle {cycle}")
            
            # Set to standard conditions for consistent measurements
            battery.set_temperature(25.0)
            battery.set_soc(0.5)
            
            # Analyze battery
            analysis = perform_battery_analysis(battery, analyzer, plot_results=False)
            
            # Store results
            results.append({
                'cycle': cycle,
                'age': battery.age,
                'soh': analysis['derived_metrics']['state_of_health'],
                'capacity': analysis['derived_metrics']['remaining_capacity'],
                'internal_resistance': analysis['measured_characteristics']['electrical_parameters']['R_total'],
                'thermal_resistance': analysis['measured_characteristics']['thermal_parameters']['R_th'],
                'thermal_time_constant': analysis['measured_characteristics']['thermal_parameters']['thermal_time_constant']
            })
            
            logger.info(f"Cycle {cycle}: SOH = {results[-1]['soh']:.1f}%, R = {results[-1]['internal_resistance']:.4f} Ohm")
    
    # Convert to DataFrame
    aging_data = pd.DataFrame(results)
    
    # Plot results if requested
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Capacity fade
        axs[0, 0].plot(aging_data['cycle'], aging_data['capacity'], 'bo-')
        axs[0, 0].set_xlabel('Cycle Number')
        axs[0, 0].set_ylabel('Capacity (mAh)')
        axs[0, 0].set_title('Capacity Fade')
        axs[0, 0].grid(True)
        
        # State of Health
        axs[0, 1].plot(aging_data['cycle'], aging_data['soh'], 'go-')
        axs[0, 1].set_xlabel('Cycle Number')
        axs[0, 1].set_ylabel('State of Health (%)')
        axs[0, 1].set_title('State of Health')
        axs[0, 1].grid(True)
        
        # Internal Resistance Increase
        axs[1, 0].plot(aging_data['cycle'], aging_data['internal_resistance'], 'ro-')
        axs[1, 0].set_xlabel('Cycle Number')
        axs[1, 0].set_ylabel('Internal Resistance (Ohm)')
        axs[1, 0].set_title('Internal Resistance Growth')
        axs[1, 0].grid(True)
        
        # Thermal Parameters
        ax1 = axs[1, 1]
        ax2 = ax1.twinx()
        
        ax1.plot(aging_data['cycle'], aging_data['thermal_resistance'], 'mo-')
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Thermal Resistance (K/W)', color='m')
        ax1.tick_params(axis='y', labelcolor='m')
        
        ax2.plot(aging_data['cycle'], aging_data['thermal_time_constant'], 'co-')
        ax2.set_ylabel('Thermal Time Constant (s)', color='c')
        ax2.tick_params(axis='y', labelcolor='c')
        
        axs[1, 1].set_title('Thermal Parameters')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Create animation of impedance spectra evolution
        create_impedance_evolution_animation(battery, analyzer, aging_data)
    
    return aging_data


def create_impedance_evolution_animation(battery, analyzer, aging_data):
    """
    Create an animation showing the evolution of impedance spectra with aging
    
    Parameters:
    -----------
    battery : LithiumIonBattery
        Battery model
    analyzer : IntegratedImpedanceAnalyzer
        Impedance analyzer
    aging_data : pd.DataFrame
        Aging data
    """
    # Create a copy of the battery for animation
    anim_battery = LithiumIonBattery(
        capacity=battery.nominal_capacity,
        cycle_count=0,
        age=0.0
    )
    
    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Electrical impedance - Nyquist plot
    axs[0].set_xlabel('Re(Z) (Ω)')
    axs[0].set_ylabel('-Im(Z) (Ω)')
    axs[0].set_title('Electrical Impedance Evolution')
    axs[0].grid(True, ls="--")
    
    # Thermal impedance - Nyquist plot
    axs[1].set_xlabel('Re(Z) (K/W)')
    axs[1].set_ylabel('-Im(Z) (K/W)')
    axs[1].set_title('Thermal Impedance Evolution')
    axs[1].grid(True, ls="--")
    
    # Initialize lines
    e_line, = axs[0].plot([], [], 'bo-', linewidth=1.5, label='Current')
    t_line, = axs[1].plot([], [], 'ro-', linewidth=1.5, label='Current')
    
    # Reference lines (initial state)
    e_ref_line, = axs[0].plot([], [], 'b--', linewidth=1, alpha=0.5, label='Initial')
    t_ref_line, = axs[1].plot([], [], 'r--', linewidth=1, alpha=0.5, label='Initial')
    
    # Text for cycle count
    cycle_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
    
    # Add legends
    axs[0].legend()
    axs[1].legend()
    
    # Generate frequencies
    e_freqs = analyzer._generate_frequency_array(0.1, 10000, 50)
    t_freqs = analyzer._generate_frequency_array(0.01, 1, 10)
    
    # Get initial data
    initial_e_impedance = analyzer._measure_electrical_impedance(e_freqs, anim_battery)
    initial_t_impedance = analyzer._measure_thermal_impedance(t_freqs, anim_battery)
    
    # Set x and y limits
    e_real_max = np.max(np.real(initial_e_impedance)) * 3
    e_imag_max = np.max(-np.imag(initial_e_impedance)) * 3
    t_real_max = np.max(np.real(initial_t_impedance)) * 3
    t_imag_max = np.max(-np.imag(initial_t_impedance)) * 3
    
    axs[0].set_xlim(0, e_real_max)
    axs[0].set_ylim(0, e_imag_max)
    axs[1].set_xlim(0, t_real_max)
    axs[1].set_ylim(0, t_imag_max)
    
    # Update reference lines
    e_ref_line.set_data(np.real(initial_e_impedance), -np.imag(initial_e_impedance))
    t_ref_line.set_data(np.real(initial_t_impedance), -np.imag(initial_t_impedance))
    
    def init():
        """Initialize animation"""
        e_line.set_data([], [])
        t_line.set_data([], [])
        cycle_text.set_text('')
        return e_line, t_line, cycle_text
    
    def update(frame):
        """Update function for animation"""
        cycle = aging_data['cycle'].iloc[frame]
        
        # Update battery state
        anim_battery.cycle(cycle - anim_battery.cycle_count)
        
        # Measure impedance
        e_impedance = analyzer._measure_electrical_impedance(e_freqs, anim_battery)
        t_impedance = analyzer._measure_thermal_impedance(t_freqs, anim_battery)
        
        # Update data
        e_line.set_data(np.real(e_impedance), -np.imag(e_impedance))
        t_line.set_data(np.real(t_impedance), -np.imag(t_impedance))
        
        # Update text
        cycle_text.set_text(f'Cycle: {cycle}, SOH: {aging_data["soh"].iloc[frame]:.1f}%')
        
        return e_line, t_line, cycle_text
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=range(len(aging_data)),
        init_func=init, blit=True, interval=500
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example 1: Single battery analysis
    print("\n=== Example 1: Single Battery Analysis ===")
    battery = LithiumIonBattery(capacity=3000, cycle_count=50, temperature=25, soc=0.5)
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure analyzer
    analyzer.configure(
        electrical_freq_range=(0.1, 10000),  # Hz
        thermal_freq_range=(0.01, 1),        # Hz
        voltage_amplitude=10e-3,             # V
        thermal_pulse_power=100e-3,          # W
    )
    
    # Calibrate analyzer
    analyzer.calibrate()
    
    # Perform analysis
    analysis_results = perform_battery_analysis(battery, analyzer)
    
    # Example 2: Battery aging study
    print("\n=== Example 2: Battery Aging Study ===")
    aging_data = battery_aging_study(cycles=100, plot=True)
    
    print("\nAnalysis complete. Check the plots for visual results.")
