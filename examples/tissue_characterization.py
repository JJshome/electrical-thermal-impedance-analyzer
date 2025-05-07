"""
Tissue Characterization Example

This example demonstrates how to use the Integrated Electrical-Thermal Impedance Analyzer
for tissue characterization and hydration monitoring.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Reference work:
Kyle UG et al., "Bioelectrical impedance analysis--part I: review of principles 
and methods", Clinical Nutrition, 2004
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
logger = logging.getLogger("TissueCharacterization")

class BiologicalTissue:
    """
    Simplified biological tissue model for demonstration purposes
    
    This class simulates various types of biological tissues with
    different properties and responses to environmental changes.
    """
    
    # Define tissue types and their base parameters
    TISSUE_TYPES = {
        'skin': {
            'intracellular_resistivity': 4.0,  # Ohm·m
            'extracellular_resistivity': 2.5,  # Ohm·m
            'membrane_capacitance': 5e-6,     # F/m²
            'thermal_conductivity': 0.3,      # W/(m·K)
            'specific_heat': 3500,            # J/(kg·K)
            'density': 1100,                  # kg/m³
            'blood_perfusion_rate': 0.002,    # (m³/s)/m³
            'hydration_sensitivity': 0.8,     # Relative factor
            'temperature_sensitivity': 0.03,  # Per °C
        },
        'muscle': {
            'intracellular_resistivity': 3.0,  # Ohm·m
            'extracellular_resistivity': 2.0,  # Ohm·m
            'membrane_capacitance': 8e-6,     # F/m²
            'thermal_conductivity': 0.5,      # W/(m·K)
            'specific_heat': 3800,            # J/(kg·K)
            'density': 1050,                  # kg/m³
            'blood_perfusion_rate': 0.008,    # (m³/s)/m³
            'hydration_sensitivity': 0.6,     # Relative factor
            'temperature_sensitivity': 0.02,  # Per °C
        },
        'fat': {
            'intracellular_resistivity': 6.0,  # Ohm·m
            'extracellular_resistivity': 5.0,  # Ohm·m
            'membrane_capacitance': 2e-6,     # F/m²
            'thermal_conductivity': 0.2,      # W/(m·K)
            'specific_heat': 2300,            # J/(kg·K)
            'density': 950,                   # kg/m³
            'blood_perfusion_rate': 0.001,    # (m³/s)/m³
            'hydration_sensitivity': 0.3,     # Relative factor
            'temperature_sensitivity': 0.01,  # Per °C
        },
        'bone': {
            'intracellular_resistivity': 10.0,  # Ohm·m
            'extracellular_resistivity': 8.0,   # Ohm·m
            'membrane_capacitance': 1e-6,      # F/m²
            'thermal_conductivity': 0.4,       # W/(m·K)
            'specific_heat': 1700,             # J/(kg·K)
            'density': 1800,                   # kg/m³
            'blood_perfusion_rate': 0.0003,    # (m³/s)/m³
            'hydration_sensitivity': 0.1,      # Relative factor
            'temperature_sensitivity': 0.005,  # Per °C
        }
    }
    
    def __init__(self, 
                 tissue_type='skin',
                 thickness=0.01,        # m
                 area=0.0025,           # m²
                 hydration=0.7,         # 0-1 scale
                 temperature=37.0,      # °C
                 age=30,                # years
                 pathological=False):   # healthy vs. pathological
        """
        Initialize BiologicalTissue model
        
        Parameters:
        -----------
        tissue_type : str
            Type of tissue ('skin', 'muscle', 'fat', 'bone')
        thickness : float
            Tissue thickness in meters
        area : float
            Tissue surface area in square meters
        hydration : float
            Tissue hydration level (0-1)
        temperature : float
            Tissue temperature in °C
        age : int
            Age in years
        pathological : bool
            Whether tissue has pathological conditions
        """
        self.tissue_type = tissue_type
        self.thickness = thickness
        self.area = area
        self.hydration = hydration
        self.temperature = temperature
        self.age = age
        self.pathological = pathological
        
        # Get base parameters for this tissue type
        if tissue_type not in self.TISSUE_TYPES:
            raise ValueError(f"Unknown tissue type: {tissue_type}")
        self.base_params = self.TISSUE_TYPES[tissue_type].copy()
        
        # Update parameters based on initial state
        self._update_parameters()
        
    def _update_parameters(self):
        """
        Update tissue parameters based on current state
        
        This method simulates how tissue parameters change with hydration,
        temperature, age, and pathological conditions.
        """
        # Normalize hydration effect (centered at 0.7 which is typical)
        hydration_factor = 1.0 + self.base_params['hydration_sensitivity'] * (self.hydration - 0.7) / 0.3
        
        # Temperature effect (reference at 37°C)
        temp_factor = 1.0 - self.base_params['temperature_sensitivity'] * (self.temperature - 37.0)
        
        # Age effect (reference at 30 years)
        age_factor = 1.0 + 0.01 * (self.age - 30) / 10
        
        # Pathological condition effect
        if self.pathological:
            pathology_factor = 1.3  # 30% increase in resistivity
            thermal_pathology_factor = 0.8  # 20% decrease in thermal parameters
        else:
            pathology_factor = 1.0
            thermal_pathology_factor = 1.0
        
        # Calculate electrical parameters
        self.intracellular_resistivity = self.base_params['intracellular_resistivity'] * pathology_factor / hydration_factor * temp_factor * age_factor
        self.extracellular_resistivity = self.base_params['extracellular_resistivity'] * pathology_factor / (hydration_factor ** 2) * temp_factor * age_factor
        self.membrane_capacitance = self.base_params['membrane_capacitance'] * hydration_factor / age_factor
        
        # Calculate thermal parameters
        self.thermal_conductivity = self.base_params['thermal_conductivity'] * hydration_factor * thermal_pathology_factor
        self.specific_heat = self.base_params['specific_heat'] * hydration_factor * thermal_pathology_factor
        self.density = self.base_params['density'] * (1.0 + 0.1 * (self.hydration - 0.7))
        self.blood_perfusion_rate = self.base_params['blood_perfusion_rate'] * thermal_pathology_factor * (1.0 + 0.5 * (self.temperature - 37.0) / 5.0)
        
        # Calculate derived parameters
        # Electrical
        self.intracellular_resistance = self.intracellular_resistivity * self.thickness / self.area
        self.extracellular_resistance = self.extracellular_resistivity * self.thickness / self.area
        self.membrane_capacitance_total = self.membrane_capacitance * self.area / self.thickness
        
        # Thermal
        self.thermal_resistance = self.thickness / (self.thermal_conductivity * self.area)
        self.thermal_capacitance = self.specific_heat * self.density * self.area * self.thickness
        self.thermal_time_constant = self.thermal_resistance * self.thermal_capacitance
        
    def set_hydration(self, hydration):
        """
        Set tissue hydration level
        
        Parameters:
        -----------
        hydration : float
            Hydration level (0-1)
        """
        self.hydration = max(0.0, min(1.0, hydration))
        self._update_parameters()
        logger.info(f"Hydration set to {self.hydration:.2f}")
        
    def set_temperature(self, temperature):
        """
        Set tissue temperature
        
        Parameters:
        -----------
        temperature : float
            Temperature in °C
        """
        self.temperature = temperature
        self._update_parameters()
        logger.info(f"Temperature set to {self.temperature:.1f}°C")
        
    def set_pathological(self, pathological):
        """
        Set pathological condition
        
        Parameters:
        -----------
        pathological : bool
            Whether tissue has pathological conditions
        """
        self.pathological = pathological
        self._update_parameters()
        logger.info(f"Pathological condition set to {self.pathological}")
        
    def __str__(self):
        """String representation of tissue state"""
        return (f"{self.tissue_type.capitalize()} Tissue:\n"
                f"  Dimensions: {self.thickness*1000:.1f} mm thickness, {self.area*10000:.1f} cm² area\n"
                f"  Hydration: {self.hydration*100:.1f}%\n"
                f"  Temperature: {self.temperature:.1f}°C\n"
                f"  Age: {self.age} years\n"
                f"  Pathological: {self.pathological}\n"
                f"  Electrical Properties:\n"
                f"    Intracellular Resistance: {self.intracellular_resistance:.2f} Ω\n"
                f"    Extracellular Resistance: {self.extracellular_resistance:.2f} Ω\n"
                f"    Membrane Capacitance: {self.membrane_capacitance_total*1e6:.2f} µF\n"
                f"  Thermal Properties:\n"
                f"    Thermal Resistance: {self.thermal_resistance:.2f} K/W\n"
                f"    Thermal Capacitance: {self.thermal_capacitance:.2f} J/K\n"
                f"    Thermal Time Constant: {self.thermal_time_constant:.2f} s")


def perform_tissue_analysis(tissue, analyzer, plot_results=True):
    """
    Perform comprehensive tissue analysis using integrated impedance analyzer
    
    Parameters:
    -----------
    tissue : BiologicalTissue
        Tissue to analyze
    analyzer : IntegratedImpedanceAnalyzer
        Configured impedance analyzer
    plot_results : bool
        Whether to plot results
        
    Returns:
    --------
    analysis_results : dict
        Dictionary containing analysis results
    """
    logger.info(f"Starting comprehensive analysis of {tissue.tissue_type} tissue")
    
    # Measure impedance
    logger.info("Measuring impedance spectra")
    measurement_results = analyzer.measure(target_system=tissue)
    
    # Analyze results
    logger.info("Analyzing impedance data")
    characteristics = analyzer.analyze(measurement_results)
    
    # Extract key parameters
    e_params = characteristics['electrical_parameters']
    t_params = characteristics['thermal_parameters']
    i_params = characteristics['integrated_parameters']
    
    # Calculate tissue-specific metrics
    # Total body water estimation using bioelectrical impedance analysis
    # Based on simplified equations from literature
    
    # For demonstration - these are simplified versions of actual equations
    if tissue.tissue_type == 'skin':
        # Estimate hydration from electrical impedance
        estimated_hydration = 0.7 * (100.0 / e_params['R_total']) * tissue.area
        
        # Estimate blood perfusion from thermal parameters
        estimated_perfusion = 0.002 * t_params['thermal_diffusivity'] / 1e-7
        
        # Tissue health index (simplified demonstration)
        tissue_health_index = 100 * (1.0 - abs(i_params['electrical_thermal_correlation'] - 0.8))
    else:
        # Different equations for other tissue types
        estimated_hydration = 0.6 * (70.0 / e_params['R_total']) * tissue.area
        estimated_perfusion = 0.005 * t_params['thermal_diffusivity'] / 1e-7
        tissue_health_index = 100 * (1.0 - abs(i_params['electrical_thermal_correlation'] - 0.7))
    
    # Combine all results
    analysis_results = {
        'timestamp': datetime.now(),
        'tissue_info': {
            'type': tissue.tissue_type,
            'thickness': tissue.thickness,
            'area': tissue.area,
            'hydration': tissue.hydration,
            'temperature': tissue.temperature,
            'age': tissue.age,
            'pathological': tissue.pathological
        },
        'measured_characteristics': characteristics,
        'derived_metrics': {
            'estimated_hydration': estimated_hydration,
            'estimated_perfusion': estimated_perfusion,
            'tissue_health_index': tissue_health_index
        }
    }
    
    # Print summary
    print("\nTissue Analysis Results:")
    print("========================")
    print(f"Tissue: {tissue.tissue_type.capitalize()}, {tissue.thickness*1000:.1f} mm thickness")
    print(f"Condition: {'Pathological' if tissue.pathological else 'Healthy'}, Age: {tissue.age} years")
    print(f"Temperature: {tissue.temperature:.1f}°C, Actual Hydration: {tissue.hydration*100:.1f}%")
    
    print("\nElectrical Characteristics:")
    print(f"  Intracellular Resistance (Ri): {e_params['R_i']:.2f} Ω")
    print(f"  Extracellular Resistance (Re): {e_params['R_e']:.2f} Ω")
    print(f"  Membrane Capacitance (Cm): {e_params['C_m']*1e6:.2f} µF")
    print(f"  Characteristic Frequency (fc): {e_params['characteristic_frequency']:.1f} Hz")
    
    print("\nThermal Characteristics:")
    print(f"  Thermal Resistance (Rth): {t_params['R_th']:.2f} K/W")
    print(f"  Thermal Capacitance (Cth): {t_params['C_th']:.2f} J/K")
    print(f"  Thermal Time Constant (τ): {t_params['thermal_time_constant']:.2f} s")
    print(f"  Thermal Diffusivity: {t_params['thermal_diffusivity']*1e7:.2f} × 10⁻⁷ m²/s")
    
    print("\nIntegrated Analysis:")
    print(f"  Electrical-Thermal Correlation: {i_params['electrical_thermal_correlation']:.2f}")
    print(f"  Estimated Hydration: {estimated_hydration*100:.1f}%")
    print(f"  Estimated Blood Perfusion: {estimated_perfusion:.4f} (m³/s)/m³")
    print(f"  Tissue Health Index: {tissue_health_index:.1f}/100")
    
    # Plot if requested
    if plot_results:
        analyzer.plot_impedance_spectra(measurement_results)
        plot_tissue_characteristics(analysis_results)
    
    return analysis_results


def plot_tissue_characteristics(analysis_results):
    """
    Create visualizations of tissue characteristics
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results
    """
    # Extract data
    tissue_info = analysis_results['tissue_info']
    e_params = analysis_results['measured_characteristics']['electrical_parameters']
    t_params = analysis_results['measured_characteristics']['thermal_parameters']
    i_params = analysis_results['measured_characteristics']['integrated_parameters']
    derived = analysis_results['derived_metrics']
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Tissue Characterization: {tissue_info['type'].capitalize()}", fontsize=16)
    
    # Plot 1: Electrical Model Parameters
    ax1 = axs[0, 0]
    labels = ['R_i (Intracellular)', 'R_e (Extracellular)', 'R_total (Total)']
    values = [e_params['R_i'], e_params['R_e'], e_params['R_total']]
    colors = ['lightcoral', 'indianred', 'darkred']
    
    bars = ax1.bar(labels, values, color=colors)
    ax1.set_ylabel('Resistance (Ohm)')
    ax1.set_title('Tissue Electrical Properties')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add capacitance annotation
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Capacitance (µF)', color='blue')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    ax1_twin.bar(['C_m (Membrane)'], [e_params['C_m']*1e6], color='royalblue', alpha=0.7)
    ax1_twin.set_ylim(0, e_params['C_m']*1e6*2)
    
    # Plot 2: Cole-Cole Plot (Simplified representation)
    ax2 = axs[0, 1]
    
    # Generate some points for the Cole-Cole plot
    num_points = 50
    omega = np.logspace(-2, 5, num_points)
    R_inf = e_params['R_e']
    R_0 = e_params['R_e'] + e_params['R_i']
    tau = 1 / (2 * np.pi * e_params['characteristic_frequency'])
    alpha = 0.8  # Typical value for biological tissues
    
    # Cole-Cole equation
    Z_real = R_inf + (R_0 - R_inf) / (1 + (omega * tau)**(1-alpha) * np.sin(np.pi*alpha/2))
    Z_imag = -(R_0 - R_inf) * (omega * tau)**(1-alpha) * np.cos(np.pi*alpha/2) / (1 + (omega * tau)**(1-alpha) * np.sin(np.pi*alpha/2))
    
    # Plot the semicircle
    ax2.plot(Z_real, Z_imag, 'b.-')
    ax2.set_xlabel('Re(Z) (Ohm)')
    ax2.set_ylabel('-Im(Z) (Ohm)')
    ax2.set_title('Tissue Electrical Impedance (Cole-Cole Plot)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Mark characteristic frequency
    idx_characteristic = np.argmin(np.abs(omega - 2*np.pi*e_params['characteristic_frequency']))
    ax2.plot(Z_real[idx_characteristic], Z_imag[idx_characteristic], 'ro', ms=8)
    ax2.annotate(f'{e_params["characteristic_frequency"]:.1f} Hz', 
                 xy=(Z_real[idx_characteristic], Z_imag[idx_characteristic]),
                 xytext=(10, -10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Plot 3: Thermal Properties
    ax3 = axs[1, 0]
    
    # Generate a step response for thermal model
    time = np.linspace(0, 5*t_params['thermal_time_constant'], 1000)
    temp_response = 1 - np.exp(-time / t_params['thermal_time_constant'])
    
    ax3.plot(time, temp_response, 'r-')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Normalized Temperature Response')
    ax3.set_title('Thermal Step Response')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Mark time constant
    ax3.axvline(t_params['thermal_time_constant'], color='gray', linestyle='--')
    ax3.axhline(1-1/np.e, color='gray', linestyle='--')
    ax3.plot(t_params['thermal_time_constant'], 1-1/np.e, 'ko')
    ax3.annotate(f'τ = {t_params["thermal_time_constant"]:.2f} s', 
                 xy=(t_params['thermal_time_constant'], 1-1/np.e),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Plot 4: Tissue Health Dashboard
    ax4 = axs[1, 1]
    
    # Create a radar chart (polar plot) for tissue health metrics
    metrics = ['Hydration', 'Perfusion', 'Structure', 'Metabolism', 'Health Index']
    N = len(metrics)
    
    # Compute values (normalized to 0-1)
    actual_hydration = tissue_info['hydration']
    estimated_hydration = derived['estimated_hydration']
    hydration_accuracy = 1 - abs(actual_hydration - estimated_hydration) / actual_hydration
    
    values = [
        actual_hydration,  # Hydration
        derived['estimated_perfusion'] / 0.01,  # Perfusion (normalized to 0-1)
        1 - abs(e_params['R_i'] / e_params['R_e'] - 2) / 2,  # Structure (optimal ratio is ~2)
        i_params['electrical_thermal_correlation'],  # Metabolism (correlation)
        derived['tissue_health_index'] / 100  # Health Index
    ]
    
    # Angle of each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values += values[:1]  # Close the polygon
    angles += angles[:1]
    
    # Plot the radar chart
    ax4.set_theta_offset(np.pi / 2)  # Start from top
    ax4.set_theta_direction(-1)  # Clockwise
    ax4.set_thetagrids(np.degrees(angles[:-1]), metrics)
    
    ax4.plot(angles, values, 'b-', linewidth=2)
    ax4.fill(angles, values, 'b', alpha=0.1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Tissue Health Metrics')
    
    # Add pathological status indicator
    if tissue_info['pathological']:
        ax4.text(0.5, -0.1, 'PATHOLOGICAL CONDITION DETECTED', 
                 transform=ax4.transAxes, ha='center', 
                 color='red', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def hydration_monitoring_study():
    """
    Conduct a tissue hydration monitoring study
    
    This function simulates changing hydration levels in skin tissue
    and measures how the impedance characteristics change.
    
    Returns:
    --------
    study_data : pd.DataFrame
        DataFrame containing study results
    """
    logger.info("Starting tissue hydration monitoring study")
    
    # Initialize tissue model and analyzer
    tissue = BiologicalTissue(tissue_type='skin', hydration=0.7)
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure analyzer
    analyzer.configure(
        electrical_freq_range=(0.1, 100000),  # Hz
        thermal_freq_range=(0.01, 1),         # Hz
        voltage_amplitude=5e-3,               # V
        thermal_pulse_power=50e-3,            # W
    )
    
    # Calibrate analyzer
    analyzer.calibrate()
    
    # Define hydration levels to test
    hydration_levels = np.linspace(0.4, 0.9, 11)  # From dehydrated to over-hydrated
    
    # Storage for results
    results = []
    
    # Perform measurements for each hydration level
    for hydration in hydration_levels:
        logger.info(f"Testing hydration level: {hydration:.2f}")
        
        # Set tissue hydration
        tissue.set_hydration(hydration)
        
        # Analyze tissue
        analysis = perform_tissue_analysis(tissue, analyzer, plot_results=False)
        
        # Store results
        results.append({
            'hydration_actual': hydration,
            'hydration_estimated': analysis['derived_metrics']['estimated_hydration'],
            'R_total': analysis['measured_characteristics']['electrical_parameters']['R_total'],
            'R_i': analysis['measured_characteristics']['electrical_parameters']['R_i'],
            'R_e': analysis['measured_characteristics']['electrical_parameters']['R_e'],
            'C_m': analysis['measured_characteristics']['electrical_parameters']['C_m'],
            'characteristic_frequency': analysis['measured_characteristics']['electrical_parameters']['characteristic_frequency'],
            'R_th': analysis['measured_characteristics']['thermal_parameters']['R_th'],
            'C_th': analysis['measured_characteristics']['thermal_parameters']['C_th'],
            'thermal_time_constant': analysis['measured_characteristics']['thermal_parameters']['thermal_time_constant'],
            'electrical_thermal_correlation': analysis['measured_characteristics']['integrated_parameters']['electrical_thermal_correlation'],
            'tissue_health_index': analysis['derived_metrics']['tissue_health_index']
        })
        
        logger.info(f"Hydration {hydration:.2f}: "
                   f"Estimated: {results[-1]['hydration_estimated']:.2f}, "
                   f"R_total: {results[-1]['R_total']:.2f} Ohm")
    
    # Convert to DataFrame
    study_data = pd.DataFrame(results)
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Hydration estimation accuracy
    ax1 = axs[0, 0]
    ax1.plot(study_data['hydration_actual'], study_data['hydration_estimated'], 'bo-')
    ax1.plot([0.4, 0.9], [0.4, 0.9], 'k--')  # Ideal line
    ax1.set_xlabel('Actual Hydration')
    ax1.set_ylabel('Estimated Hydration')
    ax1.set_title('Hydration Estimation Accuracy')
    ax1.grid(True)
    
    # Calculate error statistics
    absolute_error = np.abs(study_data['hydration_estimated'] - study_data['hydration_actual'])
    mean_abs_error = absolute_error.mean()
    max_error = absolute_error.max()
    
    # Add error statistics to the plot
    ax1.text(0.05, 0.95, f'Mean Absolute Error: {mean_abs_error:.3f}\nMax Error: {max_error:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Electrical parameters vs hydration
    ax2 = axs[0, 1]
    ax2.plot(study_data['hydration_actual'], study_data['R_total'], 'ro-', label='Total Resistance')
    ax2.set_xlabel('Hydration Level')
    ax2.set_ylabel('Resistance (Ohm)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(study_data['hydration_actual'], study_data['C_m']*1e6, 'bo-', label='Membrane Capacitance')
    ax2_twin.set_ylabel('Capacitance (µF)', color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')
    
    ax2.set_title('Electrical Parameters vs Hydration')
    ax2.grid(True)
    
    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Plot 3: Thermal parameters vs hydration
    ax3 = axs[1, 0]
    ax3.plot(study_data['hydration_actual'], study_data['R_th'], 'mo-', label='Thermal Resistance')
    ax3.set_xlabel('Hydration Level')
    ax3.set_ylabel('Thermal Resistance (K/W)', color='m')
    ax3.tick_params(axis='y', labelcolor='m')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(study_data['hydration_actual'], study_data['thermal_time_constant'], 'co-', label='Thermal Time Constant')
    ax3_twin.set_ylabel('Time Constant (s)', color='c')
    ax3_twin.tick_params(axis='y', labelcolor='c')
    
    ax3.set_title('Thermal Parameters vs Hydration')
    ax3.grid(True)
    
    # Add legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Plot 4: Correlation plot
    ax4 = axs[1, 1]
    ax4.plot(study_data['hydration_actual'], study_data['electrical_thermal_correlation'], 'go-')
    ax4.set_xlabel('Hydration Level')
    ax4.set_ylabel('Electrical-Thermal Correlation')
    ax4.set_title('Integrated Analysis Correlation')
    ax4.grid(True)
    
    # Add health index contour
    ax4_twin = ax4.twinx()
    ax4_twin.plot(study_data['hydration_actual'], study_data['tissue_health_index'], 'ko--', alpha=0.6)
    ax4_twin.set_ylabel('Tissue Health Index', color='k')
    ax4_twin.tick_params(axis='y', labelcolor='k')
    
    plt.tight_layout()
    plt.show()
    
    return study_data


def tissue_comparison_study():
    """
    Compare different tissue types using the integrated impedance analysis
    
    This function analyzes different tissue types and compares their
    electrical and thermal impedance characteristics.
    """
    logger.info("Starting tissue comparison study")
    
    # Initialize analyzer
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure analyzer
    analyzer.configure(
        electrical_freq_range=(0.1, 100000),  # Hz
        thermal_freq_range=(0.01, 1),         # Hz
        voltage_amplitude=5e-3,               # V
        thermal_pulse_power=50e-3,            # W
    )
    
    # Calibrate analyzer
    analyzer.calibrate()
    
    # Define tissue types to analyze
    tissue_types = ['skin', 'muscle', 'fat', 'bone']
    
    # Storage for results
    e_spectra = {}
    t_spectra = {}
    characteristics = {}
    
    # Create figure for plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Tissue Type Comparison: Electrical & Thermal Impedance", fontsize=16)
    
    # Define colors for different tissue types
    colors = {
        'skin': 'royalblue',
        'muscle': 'firebrick',
        'fat': 'forestgreen',
        'bone': 'darkorange'
    }
    
    # Analyze each tissue type
    for tissue_type in tissue_types:
        logger.info(f"Analyzing {tissue_type} tissue")
        
        # Create tissue model
        tissue = BiologicalTissue(tissue_type=tissue_type, hydration=0.7, temperature=37.0)
        
        # Measure impedance
        results = analyzer.measure(target_system=tissue)
        
        # Analyze results
        chars = analyzer.analyze(results)
        
        # Store for later use
        e_spectra[tissue_type] = results['electrical_impedance']
        t_spectra[tissue_type] = results['thermal_impedance']
        characteristics[tissue_type] = chars
        
        # Extract data for plotting
        e_freq = results['electrical_freq']
        e_impedance = results['electrical_impedance']
        t_freq = results['thermal_freq']
        t_impedance = results['thermal_impedance']
        
        # Plot 1: Electrical Impedance Bode Plot (Magnitude)
        ax1 = axs[0, 0]
        ax1.loglog(e_freq, np.abs(e_impedance), '.-', color=colors[tissue_type], label=tissue_type.capitalize())
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('|Z| (Ohm)')
        ax1.set_title('Electrical Impedance Magnitude')
        ax1.grid(True, which="both", ls="--")
        ax1.legend()
        
        # Plot 2: Electrical Impedance Bode Plot (Phase)
        ax2 = axs[0, 1]
        ax2.semilogx(e_freq, np.angle(e_impedance, deg=True), '.-', color=colors[tissue_type], label=tissue_type.capitalize())
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Electrical Impedance Phase')
        ax2.grid(True, which="both", ls="--")
        ax2.legend()
        
        # Plot 3: Thermal Impedance Bode Plot (Magnitude)
        ax3 = axs[1, 0]
        ax3.loglog(t_freq, np.abs(t_impedance), '.-', color=colors[tissue_type], label=tissue_type.capitalize())
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('|Z| (K/W)')
        ax3.set_title('Thermal Impedance Magnitude')
        ax3.grid(True, which="both", ls="--")
        ax3.legend()
        
        # Plot 4: Thermal Impedance Bode Plot (Phase)
        ax4 = axs[1, 1]
        ax4.semilogx(t_freq, np.angle(t_impedance, deg=True), '.-', color=colors[tissue_type], label=tissue_type.capitalize())
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Phase (degrees)')
        ax4.set_title('Thermal Impedance Phase')
        ax4.grid(True, which="both", ls="--")
        ax4.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Create a comparison table
    print("\nTissue Comparison Summary:")
    print("==========================")
    print(f"{'Tissue Type':<10} | {'R_total (Ω)':<12} | {'C_m (µF)':<10} | {'f_c (Hz)':<10} | {'R_th (K/W)':<12} | {'τ_th (s)':<10} | {'E-T Corr':<8}")
    print("-" * 80)
    
    for tissue_type in tissue_types:
        chars = characteristics[tissue_type]
        e_params = chars['electrical_parameters']
        t_params = chars['thermal_parameters']
        i_params = chars['integrated_parameters']
        
        print(f"{tissue_type.capitalize():<10} | "
              f"{e_params['R_total']:<12.2f} | "
              f"{e_params['C_m']*1e6:<10.2f} | "
              f"{e_params['characteristic_frequency']:<10.1f} | "
              f"{t_params['R_th']:<12.2f} | "
              f"{t_params['thermal_time_constant']:<10.1f} | "
              f"{i_params['electrical_thermal_correlation']:<8.2f}")
    
    # Create a figure for comparison of key parameters
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tissue Type Comparison: Key Parameters", fontsize=16)
    
    # Prepare data for bar plots
    tissue_labels = [t.capitalize() for t in tissue_types]
    r_total_values = [characteristics[t]['electrical_parameters']['R_total'] for t in tissue_types]
    c_m_values = [characteristics[t]['electrical_parameters']['C_m']*1e6 for t in tissue_types]
    r_th_values = [characteristics[t]['thermal_parameters']['R_th'] for t in tissue_types]
    tau_th_values = [characteristics[t]['thermal_parameters']['thermal_time_constant'] for t in tissue_types]
    correlation_values = [characteristics[t]['integrated_parameters']['electrical_thermal_correlation'] for t in tissue_types]
    
    # Plot 1: Electrical Resistance
    ax1 = axs[0, 0]
    bars = ax1.bar(tissue_labels, r_total_values, color=[colors[t] for t in tissue_types])
    ax1.set_ylabel('Resistance (Ohm)')
    ax1.set_title('Total Electrical Resistance')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot 2: Membrane Capacitance
    ax2 = axs[0, 1]
    bars = ax2.bar(tissue_labels, c_m_values, color=[colors[t] for t in tissue_types])
    ax2.set_ylabel('Capacitance (µF)')
    ax2.set_title('Membrane Capacitance')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot 3: Thermal Parameters
    ax3 = axs[1, 0]
    ax3.bar(tissue_labels, r_th_values, color=[colors[t] for t in tissue_types], alpha=0.7, label='Thermal Resistance')
    ax3.set_ylabel('Thermal Resistance (K/W)')
    ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(tissue_labels, tau_th_values, 'ko-', linewidth=2, label='Time Constant')
    ax3_twin.set_ylabel('Thermal Time Constant (s)')
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax3.set_title('Thermal Parameters')
    
    # Plot 4: Electrical-Thermal Correlation
    ax4 = axs[1, 1]
    bars = ax4.bar(tissue_labels, correlation_values, color=[colors[t] for t in tissue_types])
    ax4.set_ylabel('Correlation Coefficient')
    ax4.set_title('Electrical-Thermal Correlation')
    ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    return characteristics


if __name__ == "__main__":
    # Example 1: Single tissue analysis
    print("\n=== Example 1: Single Tissue Analysis ===")
    tissue = BiologicalTissue(tissue_type='skin', thickness=0.01, area=0.0025, 
                            hydration=0.7, temperature=37.0, age=30)
    analyzer = IntegratedImpedanceAnalyzer()
    
    # Configure analyzer
    analyzer.configure(
        electrical_freq_range=(0.1, 100000),  # Hz
        thermal_freq_range=(0.01, 1),         # Hz
        voltage_amplitude=5e-3,               # V
        thermal_pulse_power=50e-3,            # W
    )
    
    # Calibrate analyzer
    analyzer.calibrate()
    
    # Perform analysis
    analysis_results = perform_tissue_analysis(tissue, analyzer)
    
    # Example 2: Hydration monitoring study
    print("\n=== Example 2: Hydration Monitoring Study ===")
    hydration_data = hydration_monitoring_study()
    
    # Example 3: Tissue type comparison
    print("\n=== Example 3: Tissue Type Comparison ===")
    comparison_results = tissue_comparison_study()
    
    print("\nAnalysis complete. Check the plots for visual results.")
