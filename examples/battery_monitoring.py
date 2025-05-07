#!/usr/bin/env python3
"""
Battery Health Monitoring Example

This example demonstrates how to use the Integrated Electrical-Thermal Impedance
Analysis System for battery state-of-health monitoring and prediction.

The system measures both electrical and thermal impedance of a lithium-ion battery
cell, then uses AI analysis to predict its remaining capacity and health status.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from impedance_analyzer import (
    IntegratedImpedanceAnalyzer,
    ElectricalImpedanceAnalyzer,
    ThermalImpedanceAnalyzer,
    PCMThermalManager,
    AIAnalyzer,
    ImpedanceDataset,
    MeasurementMode,
    ThermalStimulationMode
)


def run_battery_monitoring(battery_id, save_results=True, plot_results=True, device_id=None):
    """
    Run a battery health monitoring test on a specific battery.
    
    Parameters
    ----------
    battery_id : str
        Identifier for the battery being tested.
    save_results : bool, optional
        Whether to save results to a file, default is True.
    plot_results : bool, optional
        Whether to plot results, default is True.
    device_id : str, optional
        Device ID for hardware connection, default is None.
    """
    print(f"Starting battery health monitoring for battery: {battery_id}")
    
    # Initialize the integrated analyzer
    analyzer = IntegratedImpedanceAnalyzer(
        thermal_control=True,
        pcm_type="pcm1",  # Using n-octadecane (PCM1) for this temperature range
        device_id=device_id
    )
    
    # Configure measurement parameters optimized for battery characterization
    analyzer.configure(
        electrical_freq_range=(0.1, 100000),  # Hz
        thermal_freq_range=(0.01, 1),        # Hz
        voltage_amplitude=5e-3,              # V (small to prevent battery disturbance)
        thermal_pulse_power=100e-3           # W
    )
    
    # Set target temperature for measurement
    # 25°C is standard for battery testing
    target_temperature = 25.0
    
    try:
        # Perform measurement
        print(f"Measuring battery at {target_temperature}°C...")
        results = analyzer.measure(target_temperature=target_temperature)
        
        if not results or 'error' in results:
            print(f"Error during measurement: {results.get('error', 'Unknown error')}")
            return
        
        print(f"Measurement completed in {results['measurement_time']:.2f} seconds")
        
        # Analyze the results
        print("Analyzing battery health...")
        ai_analyzer = AIAnalyzer(model_type='conv_lstm')
        analysis_results = ai_analyzer.analyze(results)
        
        # Extract battery health parameters
        electrical_params = analysis_results.get('traditional_analysis', {}).get('electrical_parameters', {})
        thermal_params = analysis_results.get('traditional_analysis', {}).get('thermal_parameters', {})
        combined_params = analysis_results.get('traditional_analysis', {}).get('combined_analysis', {})
        
        # Extract AI-based predictions
        ai_results = analysis_results.get('ai_analysis', {})
        performance_metrics = ai_results.get('performance_metrics', {})
        recommendations = ai_results.get('recommendations', [])
        
        # Display results
        print("\n----- Battery Health Analysis Results -----")
        print(f"Battery ID: {battery_id}")
        print(f"Measurement Date: {datetime.fromtimestamp(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nElectrical Parameters:")
        for key, data in electrical_params.items():
            print(f"  {key}: {data['value']:.3f} {data['unit']}")
        
        print("\nThermal Parameters:")
        for key, data in thermal_params.items():
            print(f"  {key}: {data['value']:.3f} {data['unit']}")
        
        print("\nCombined Analysis:")
        for key, data in combined_params.items():
            print(f"  {key}: {data['value']:.3f} {data['unit']}")
        
        print("\nPerformance Metrics:")
        for key, data in performance_metrics.items():
            print(f"  {key}: {data['value']:.1f} {data['unit']}")
        
        print("\nRecommendations:")
        for recommendation in recommendations:
            print(f"  - {recommendation}")
        
        # Save results if requested
        if save_results:
            save_battery_results(battery_id, results, analysis_results)
        
        # Plot results if requested
        if plot_results:
            plot_battery_analysis(battery_id, results, analysis_results)
        
        return {
            'battery_id': battery_id,
            'measurement_results': results,
            'analysis_results': analysis_results
        }
        
    except Exception as e:
        print(f"Error during battery monitoring: {e}")
        return None
    finally:
        # Clean up
        analyzer.disconnect()


def save_battery_results(battery_id, measurement_results, analysis_results):
    """
    Save battery measurement and analysis results to files.
    
    Parameters
    ----------
    battery_id : str
        Identifier for the battery being tested.
    measurement_results : dict
        Measurement results from the analyzer.
    analysis_results : dict
        Analysis results from the AI analyzer.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'battery')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.fromtimestamp(measurement_results['timestamp']).strftime('%Y%m%d_%H%M%S')
    
    # Save measurement results
    measurement_file = os.path.join(output_dir, f"{battery_id}_{timestamp}_measurement.json")
    with open(measurement_file, 'w') as f:
        # Prepare data for JSON serialization (convert numpy arrays to lists)
        serializable_results = {}
        for key, value in measurement_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, f"{battery_id}_{timestamp}_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def plot_battery_analysis(battery_id, measurement_results, analysis_results):
    """
    Plot battery measurement and analysis results.
    
    Parameters
    ----------
    battery_id : str
        Identifier for the battery being tested.
    measurement_results : dict
        Measurement results from the analyzer.
    analysis_results : dict
        Analysis results from the AI analyzer.
    """
    # Create output directory for plots if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'battery', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.fromtimestamp(measurement_results['timestamp']).strftime('%Y%m%d_%H%M%S')
    
    # Extract data
    electrical_data = measurement_results.get('electrical', {})
    thermal_data = measurement_results.get('thermal', {})
    
    # Create a multi-panel figure
    plt.figure(figsize=(12, 10))
    
    # Plot electrical impedance (Nyquist plot)
    plt.subplot(2, 2, 1)
    plt.plot(electrical_data.get('real', []), -np.array(electrical_data.get('imag', [])), 'o-', color='blue')
    plt.xlabel('Real Impedance (Ω)')
    plt.ylabel('-Imaginary Impedance (Ω)')
    plt.title('Electrical Impedance: Nyquist Plot')
    plt.grid(True)
    
    # Plot electrical impedance magnitude vs frequency (Bode plot)
    plt.subplot(2, 2, 2)
    plt.loglog(electrical_data.get('frequency', []), electrical_data.get('magnitude', []), 'o-', color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance Magnitude (Ω)')
    plt.title('Electrical Impedance: Bode Plot')
    plt.grid(True, which="both", ls="-")
    
    # Plot thermal impedance magnitude vs frequency
    plt.subplot(2, 2, 3)
    plt.loglog(thermal_data.get('frequency', []), thermal_data.get('magnitude', []), 'o-', color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Thermal Impedance (K/W)')
    plt.title('Thermal Impedance Magnitude')
    plt.grid(True, which="both", ls="-")
    
    # Plot temperature response amplitude vs frequency
    plt.subplot(2, 2, 4)
    plt.loglog(thermal_data.get('frequency', []), thermal_data.get('temperature_amplitude', []), 'o-', color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Temperature Amplitude (K)')
    plt.title('Thermal Response Amplitude')
    plt.grid(True, which="both", ls="-")
    
    # Add overall title
    plt.suptitle(f'Battery {battery_id} Analysis - {timestamp}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plot_file = os.path.join(output_dir, f"{battery_id}_{timestamp}_plots.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plots saved to {plot_file}")
    
    # Display figure if in interactive mode
    plt.show()


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Battery Health Monitoring Example')
    parser.add_argument('--battery-id', type=str, default='18650_001',
                        help='Identifier for the battery being tested')
    parser.add_argument('--no-save', action='store_false', dest='save_results',
                        help='Do not save results to a file')
    parser.add_argument('--no-plot', action='store_false', dest='plot_results',
                        help='Do not plot results')
    parser.add_argument('--device-id', type=str, default=None,
                        help='Device ID for hardware connection')
    
    args = parser.parse_args()
    
    # Run battery monitoring
    run_battery_monitoring(
        battery_id=args.battery_id,
        save_results=args.save_results,
        plot_results=args.plot_results,
        device_id=args.device_id
    )


if __name__ == "__main__":
    main()
