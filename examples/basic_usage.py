#!/usr/bin/env python
"""
Basic usage example for the Integrated Electrical-Thermal Impedance Analyzer.

This example demonstrates how to initialize the analyzer, configure it,
perform measurements, analyze the results, and visualize the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from impedance_analyzer import IntegratedImpedanceAnalyzer

def main():
    """Main function demonstrating basic usage of the analyzer."""
    
    print("Integrated Electrical-Thermal Impedance Analyzer - Basic Example")
    print("===============================================================")
    
    # Initialize the analyzer
    # Note: For real hardware, provide the device_id parameter
    analyzer = IntegratedImpedanceAnalyzer(
        thermal_control=True,
        pcm_type="pcm1"
    )
    
    print(f"\nAnalyzer initialized with PCM type: {analyzer.pcm_type}")
    
    # Configure measurement parameters
    analyzer.configure(
        electrical_freq_range=(0.1, 100000),  # Hz
        thermal_freq_range=(0.01, 1),         # Hz
        voltage_amplitude=10e-3,              # V
        thermal_pulse_power=100e-3            # W
    )
    
    print("\nMeasurement parameters configured:")
    print(f"  Electrical frequency range: {analyzer.electrical_freq_range[0]} to {analyzer.electrical_freq_range[1]} Hz")
    print(f"  Thermal frequency range: {analyzer.thermal_freq_range[0]} to {analyzer.thermal_freq_range[1]} Hz")
    print(f"  Voltage amplitude: {analyzer.voltage_amplitude * 1000} mV")
    print(f"  Thermal pulse power: {analyzer.thermal_pulse_power * 1000} mW")
    
    # Perform measurement
    print("\nPerforming measurement...")
    start_time = time.time()
    
    results = analyzer.measure(
        target_temperature=25.0,
        wait_for_stability=True,
        stability_threshold=0.1
    )
    
    elapsed_time = time.time() - start_time
    print(f"Measurement completed in {elapsed_time:.2f} seconds")
    
    # Display basic results
    print("\nBasic measurement results:")
    print(f"  Timestamp: {results['timestamp']}")
    print(f"  Measurement time: {results['measurement_time']:.2f} seconds")
    print(f"  Number of electrical frequency points: {len(results['electrical']['frequency'])}")
    print(f"  Number of thermal frequency points: {len(results['thermal']['frequency'])}")
    
    # Analyze the results
    print("\nAnalyzing measurement data...")
    analysis = analyzer.analyze(results)
    
    # Display analysis results
    print("\nAnalysis results:")
    
    print("\nElectrical parameters:")
    for param, data in analysis['electrical_parameters'].items():
        print(f"  {param}: {data['value']:.4f} {data['unit']}")
    
    print("\nThermal parameters:")
    for param, data in analysis['thermal_parameters'].items():
        print(f"  {param}: {data['value']:.4f} {data['unit']}")
    
    print("\nCombined analysis:")
    for param, data in analysis['combined_analysis'].items():
        print(f"  {param}: {data['value']:.4f} {data['unit']}")
    
    # Visualize the data
    print("\nVisualizing impedance spectra...")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot electrical impedance
    freq_e = results['electrical']['frequency']
    z_real = results['electrical']['real']
    z_imag = results['electrical']['imag']
    
    ax1.loglog(freq_e, np.sqrt(z_real**2 + z_imag**2), 'b-', linewidth=2)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ω)')
    ax1.set_title('Electrical Impedance Magnitude')
    ax1.grid(True, which="both", ls="--")
    
    # Plot thermal impedance
    freq_t = results['thermal']['frequency']
    z_real_t = results['thermal']['real']
    z_imag_t = results['thermal']['imag']
    
    ax2.loglog(freq_t, np.sqrt(z_real_t**2 + z_imag_t**2), 'r-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('|Z_th| (K/W)')
    ax2.set_title('Thermal Impedance Magnitude')
    ax2.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('impedance_spectra.png', dpi=300)
    print("Figure saved as 'impedance_spectra.png'")
    
    # Show the figure (comment this out for non-interactive environments)
    plt.show()
    
    print("\nExample completed successfully")

if __name__ == "__main__":
    main()
