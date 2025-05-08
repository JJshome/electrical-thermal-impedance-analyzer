#!/usr/bin/env python3
"""
Thermal Impedance Spectroscopy Example

This example demonstrates thermal impedance spectroscopy for a Li-ion battery,
implementing the heat-pulse response analysis method described in:
Barsoukov, E., Jang, J. H., & Lee, H. (2002). Thermal impedance spectroscopy for 
Li-ion batteries using heat-pulse response analysis. Journal of Power Sources, 109(2), 313-320.

The implementation is based on patented technology by Ucaretron Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft
import time

class ThermalImpedanceAnalyzer:
    """
    Class for thermal impedance spectroscopy using heat-pulse response analysis
    """
    
    def __init__(self):
        """Initialize the thermal impedance analyzer"""
        self.freq_range = None
        self.thermal_data = None
        self.impedance_spectrum = None
        self.model_params = None
    
    def configure(self, freq_range=(0.01, 1), freq_points=20, pulse_power=0.5, 
                  measurement_time=1000, sampling_rate=10):
        """
        Configure the thermal impedance analyzer
        
        Parameters:
        -----------
        freq_range : tuple
            Min and max frequency for the thermal impedance spectrum (Hz)
        freq_points : int
            Number of frequency points in the spectrum
        pulse_power : float
            Power of the heat pulse (W)
        measurement_time : float
            Total measurement time (s)
        sampling_rate : float
            Sampling rate for temperature measurement (Hz)
        """
        self.freq_range = freq_range
        self.freq_points = freq_points
        self.pulse_power = pulse_power
        self.measurement_time = measurement_time
        self.sampling_rate = sampling_rate
        
        # Generate logarithmically spaced frequency points
        self.frequencies = np.logspace(
            np.log10(freq_range[0]), 
            np.log10(freq_range[1]), 
            freq_points
        )
        
        # Initialize data structures
        self.time_points = np.arange(0, measurement_time, 1/sampling_rate)
        self.thermal_data = {
            'time': self.time_points,
            'temperature': np.zeros_like(self.time_points),
            'heat_pulse': np.zeros_like(self.time_points)
        }
        
        print(f"Configured thermal impedance analyzer:")
        print(f"  - Frequency range: {freq_range[0]:.3f} Hz to {freq_range[1]:.3f} Hz")
        print(f"  - Frequency points: {freq_points}")
        print(f"  - Heat pulse power: {pulse_power} W")
        print(f"  - Measurement time: {measurement_time} s")
        print(f"  - Sampling rate: {sampling_rate} Hz")
    
    def apply_heat_pulse(self, duration=40):
        """
        Simulate application of a heat pulse to the battery
        
        Parameters:
        -----------
        duration : float
            Duration of the heat pulse (s)
        
        Returns:
        --------
        heat_pulse : ndarray
            Heat pulse profile over time
        """
        # Create a rectangular heat pulse
        heat_pulse = np.zeros_like(self.time_points)
        pulse_samples = int(duration * self.sampling_rate)
        heat_pulse[:pulse_samples] = self.pulse_power
        
        return heat_pulse
    
    def simulate_temperature_response(self, heat_pulse, thermal_params):
        """
        Simulate the temperature response to a heat pulse
        
        Parameters:
        -----------
        heat_pulse : ndarray
            Heat pulse profile over time
        thermal_params : dict
            Thermal parameters of the battery
            - 'R_th': Thermal resistance (K/W)
            - 'C_th': Thermal capacitance (J/K)
            - 'tau': Thermal time constant (s)
        
        Returns:
        --------
        temperature : ndarray
            Temperature response over time
        """
        # Extract thermal parameters
        R_th = thermal_params['R_th']  # Thermal resistance (K/W)
        C_th = thermal_params['C_th']  # Thermal capacitance (J/K)
        tau = thermal_params['tau']    # Thermal time constant (s)
        
        # Calculate temperature response using convolution with exponential decay
        dt = 1 / self.sampling_rate
        t = self.time_points
        impulse_response = (R_th / tau) * np.exp(-t / tau)
        temperature = np.convolve(heat_pulse, impulse_response)[:len(t)] * dt
        
        # Add some realistic noise
        noise_level = 0.01  # 0.01°C noise
        temperature += np.random.normal(0, noise_level, size=len(temperature))
        
        return temperature
    
    def measure(self, battery_params):
        """
        Perform a thermal impedance measurement
        
        Parameters:
        -----------
        battery_params : dict
            Thermal parameters of the battery
            - 'R_th': Thermal resistance (K/W)
            - 'C_th': Thermal capacitance (J/K)
            - 'tau': Thermal time constant (s)
        
        Returns:
        --------
        thermal_data : dict
            Dictionary containing measurement data
        """
        print("Starting thermal impedance measurement...")
        
        # Apply heat pulse
        heat_pulse = self.apply_heat_pulse()
        
        # Simulate temperature response
        temperature = self.simulate_temperature_response(heat_pulse, battery_params)
        
        # Store the data
        self.thermal_data = {
            'time': self.time_points,
            'temperature': temperature,
            'heat_pulse': heat_pulse
        }
        
        print("Measurement completed.")
        return self.thermal_data
    
    def calculate_impedance_spectrum(self):
        """
        Calculate thermal impedance spectrum from the measured data
        using the heat-pulse response analysis method
        
        Returns:
        --------
        impedance_spectrum : dict
            Dictionary containing impedance spectrum data
        """
        print("Calculating thermal impedance spectrum...")
        
        # Check if measurement data exists
        if self.thermal_data is None:
            raise ValueError("No measurement data available. Run measure() first.")
        
        # Extract data
        time = self.thermal_data['time']
        temperature = self.thermal_data['temperature']
        heat_pulse = self.thermal_data['heat_pulse']
        
        # Initialize arrays for impedance spectrum
        freq = self.frequencies
        z_real = np.zeros_like(freq)
        z_imag = np.zeros_like(freq)
        
        # Calculate impedance at each frequency using FFT-based method
        # This is a simplified implementation of the Barsoukov et al. method
        for i, f in enumerate(freq):
            # Calculate complex impedance at frequency f
            omega = 2 * np.pi * f
            dt = 1 / self.sampling_rate
            
            # Apply windowing to reduce spectral leakage
            window = np.hanning(len(time))
            temp_windowed = (temperature - temperature[0]) * window
            heat_windowed = heat_pulse * window
            
            # Calculate FFT
            temp_fft = fft(temp_windowed)
            heat_fft = fft(heat_windowed)
            
            # Find nearest frequency bin
            freq_bin = int(f * len(time) * dt)
            if freq_bin >= len(temp_fft):
                freq_bin = len(temp_fft) - 1
            
            # Calculate complex impedance
            if abs(heat_fft[freq_bin]) > 1e-10:  # Avoid division by zero
                z_complex = temp_fft[freq_bin] / heat_fft[freq_bin]
                z_real[i] = np.real(z_complex)
                z_imag[i] = np.imag(z_complex)
            else:
                z_real[i] = 0
                z_imag[i] = 0
        
        # Store the impedance spectrum
        self.impedance_spectrum = {
            'frequency': freq,
            'real': z_real,
            'imag': z_imag,
            'magnitude': np.sqrt(z_real**2 + z_imag**2),
            'phase': np.arctan2(z_imag, z_real)
        }
        
        print("Impedance spectrum calculation completed.")
        return self.impedance_spectrum
    
    def fit_thermal_model(self, model_type='foster'):
        """
        Fit thermal model to the impedance spectrum
        
        Parameters:
        -----------
        model_type : str
            Type of thermal model to fit
            - 'foster': Foster RC network model
            - 'cauer': Cauer RC network model
            - 'simple': Simple RC model
        
        Returns:
        --------
        model_params : dict
            Fitted model parameters
        """
        print(f"Fitting {model_type} thermal model...")
        
        # Check if impedance spectrum exists
        if self.impedance_spectrum is None:
            raise ValueError("No impedance spectrum available. Run calculate_impedance_spectrum() first.")
        
        # Extract data
        freq = self.impedance_spectrum['frequency']
        z_real = self.impedance_spectrum['real']
        z_imag = self.impedance_spectrum['imag']
        
        # Define model functions
        def simple_rc_model(omega, R, C):
            """Simple RC thermal model"""
            z = R / (1 + 1j * omega * R * C)
            return np.hstack([np.real(z), np.imag(z)])
        
        def foster_model(omega, R1, C1, R2, C2):
            """Foster RC network thermal model with two RC pairs"""
            z1 = R1 / (1 + 1j * omega * R1 * C1)
            z2 = R2 / (1 + 1j * omega * R2 * C2)
            z = z1 + z2
            return np.hstack([np.real(z), np.imag(z)])
        
        # Prepare data for fitting
        omega = 2 * np.pi * freq
        z_data = np.hstack([z_real, z_imag])
        
        # Fit the appropriate model
        if model_type == 'simple':
            # Initial parameter guess [R, C]
            p0 = [np.max(z_real), 1 / (2 * np.pi * freq[np.argmax(z_imag)] * np.max(z_real))]
            popt, pcov = curve_fit(
                lambda w, R, C: simple_rc_model(w, R, C), 
                omega, z_data, p0=p0
            )
            self.model_params = {
                'type': 'simple',
                'R': popt[0],
                'C': popt[1],
                'tau': popt[0] * popt[1]
            }
        
        elif model_type == 'foster':
            # Initial parameter guess [R1, C1, R2, C2]
            p0 = [
                np.max(z_real) * 0.7, 
                1 / (2 * np.pi * freq[np.argmax(z_imag)] * np.max(z_real)),
                np.max(z_real) * 0.3,
                1 / (2 * np.pi * freq[0] * np.max(z_real) * 0.3)
            ]
            popt, pcov = curve_fit(
                lambda w, R1, C1, R2, C2: foster_model(w, R1, C1, R2, C2), 
                omega, z_data, p0=p0
            )
            self.model_params = {
                'type': 'foster',
                'R1': popt[0],
                'C1': popt[1],
                'tau1': popt[0] * popt[1],
                'R2': popt[2],
                'C2': popt[3],
                'tau2': popt[2] * popt[3],
                'R_total': popt[0] + popt[2]
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print("Model fitting completed.")
        return self.model_params
    
    def plot_temperature_response(self):
        """Plot the temperature response to the heat pulse"""
        if self.thermal_data is None:
            raise ValueError("No measurement data available. Run measure() first.")
        
        plt.figure(figsize=(10, 6))
        
        # Plot heat pulse
        plt.subplot(2, 1, 1)
        plt.plot(self.thermal_data['time'], self.thermal_data['heat_pulse'], 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Power (W)')
        plt.title('Heat Pulse')
        plt.grid(True)
        
        # Plot temperature response
        plt.subplot(2, 1, 2)
        plt.plot(self.thermal_data['time'], self.thermal_data['temperature'], 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Response')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_impedance_spectrum(self):
        """Plot the thermal impedance spectrum"""
        if self.impedance_spectrum is None:
            raise ValueError("No impedance spectrum available. Run calculate_impedance_spectrum() first.")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot real vs imaginary part (Nyquist plot)
        plt.subplot(2, 2, 1)
        plt.plot(self.impedance_spectrum['real'], -self.impedance_spectrum['imag'], 'bo-')
        plt.xlabel('Re(Z) (K/W)')
        plt.ylabel('-Im(Z) (K/W)')
        plt.title('Thermal Impedance Nyquist Plot')
        plt.grid(True)
        
        # Plot magnitude vs frequency
        plt.subplot(2, 2, 2)
        plt.loglog(self.impedance_spectrum['frequency'], self.impedance_spectrum['magnitude'], 'ro-')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('|Z| (K/W)')
        plt.title('Thermal Impedance Magnitude')
        plt.grid(True)
        
        # Plot real part vs frequency
        plt.subplot(2, 2, 3)
        plt.semilogx(self.impedance_spectrum['frequency'], self.impedance_spectrum['real'], 'go-')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Re(Z) (K/W)')
        plt.title('Thermal Impedance Real Part')
        plt.grid(True)
        
        # Plot imaginary part vs frequency
        plt.subplot(2, 2, 4)
        plt.semilogx(self.impedance_spectrum['frequency'], -self.impedance_spectrum['imag'], 'mo-')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('-Im(Z) (K/W)')
        plt.title('Thermal Impedance Imaginary Part')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_battery_characteristics(self):
        """Print the battery thermal characteristics based on the fitted model"""
        if self.model_params is None:
            raise ValueError("No model parameters available. Run fit_thermal_model() first.")
        
        print("\n===== Battery Thermal Characteristics =====")
        
        if self.model_params['type'] == 'simple':
            print(f"Model type: Simple RC model")
            print(f"Thermal resistance: {self.model_params['R']:.3f} K/W")
            print(f"Thermal capacitance: {self.model_params['C']:.3f} J/K")
            print(f"Thermal time constant: {self.model_params['tau']:.3f} s")
        
        elif self.model_params['type'] == 'foster':
            print(f"Model type: Foster RC network model")
            print(f"Total thermal resistance: {self.model_params['R_total']:.3f} K/W")
            print(f"First RC pair:")
            print(f"  - Thermal resistance: {self.model_params['R1']:.3f} K/W")
            print(f"  - Thermal capacitance: {self.model_params['C1']:.3f} J/K")
            print(f"  - Thermal time constant: {self.model_params['tau1']:.3f} s")
            print(f"Second RC pair:")
            print(f"  - Thermal resistance: {self.model_params['R2']:.3f} K/W")
            print(f"  - Thermal capacitance: {self.model_params['C2']:.3f} J/K")
            print(f"  - Thermal time constant: {self.model_params['tau2']:.3f} s")
        
        print("===========================================")


def main():
    """Main function demonstrating thermal impedance spectroscopy for a battery"""
    print("Demonstrating Thermal Impedance Spectroscopy for Li-ion Battery")
    print("Based on the heat-pulse response analysis method by Barsoukov et al. (2002)")
    print("Implementing patented technology by Ucaretron Inc.")
    print("----------------------------------------------------------------")
    
    # Initialize the thermal impedance analyzer
    analyzer = ThermalImpedanceAnalyzer()
    
    # Configure the analyzer
    analyzer.configure(
        freq_range=(0.005, 1),
        freq_points=30,
        pulse_power=1.0,  # 1W heat pulse
        measurement_time=500,  # 500s measurement
        sampling_rate=10  # 10Hz sampling
    )
    
    # Define battery thermal parameters (for simulation)
    # These parameters represent a typical 18650 Li-ion cell
    battery_params = {
        'R_th': 2.5,    # Thermal resistance (K/W)
        'C_th': 60.0,   # Thermal capacitance (J/K)
        'tau': 150.0    # Thermal time constant (s)
    }
    
    # Perform the measurement
    thermal_data = analyzer.measure(battery_params)
    
    # Plot the temperature response
    analyzer.plot_temperature_response()
    
    # Calculate impedance spectrum
    impedance_spectrum = analyzer.calculate_impedance_spectrum()
    
    # Plot the impedance spectrum
    analyzer.plot_impedance_spectrum()
    
    # Fit thermal model
    model_params = analyzer.fit_thermal_model(model_type='foster')
    
    # Print battery characteristics
    analyzer.print_battery_characteristics()
    
    print("\nThermal impedance analysis complete.")


if __name__ == "__main__":
    main()
