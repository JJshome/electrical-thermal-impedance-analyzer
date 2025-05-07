#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-invasive Continuous Glucose Monitoring Application Example

This example demonstrates how to use the integrated electrical-thermal impedance
analyzer system for non-invasive glucose monitoring applications. The system combines
electrical impedance spectroscopy (EIS) and thermal impedance spectroscopy (TIS)
to estimate blood glucose levels without the need for blood sampling.

The implementation is based on the research and technology developed by Ucaretron Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from impedance_analyzer import IntegratedImpedanceAnalyzer

# Constants
MEASUREMENT_INTERVAL = 5  # minutes
MONITORING_DURATION = 24  # hours
EIS_FREQUENCIES = np.logspace(1, 6, 20)  # 10Hz to 1MHz, 20 points
TIS_FREQUENCIES = np.logspace(-2, 0, 10)  # 0.01Hz to 1Hz, 10 points
VOLTAGE_AMPLITUDE = 10e-3  # V
THERMAL_PULSE_POWER = 100e-3  # W

# Simulated meal times for demonstration
MEAL_TIMES = [
    datetime.now() + timedelta(hours=1),    # Breakfast
    datetime.now() + timedelta(hours=5),    # Lunch
    datetime.now() + timedelta(hours=10),   # Dinner
    datetime.now() + timedelta(hours=15),   # Snack
]

# Simulated physical activity times
ACTIVITY_TIMES = [
    (datetime.now() + timedelta(hours=3), datetime.now() + timedelta(hours=3.5)),     # Morning exercise
    (datetime.now() + timedelta(hours=13), datetime.now() + timedelta(hours=13.75)),  # Evening exercise
]

class GlucoseMonitoring:
    """Continuous Glucose Monitoring using Integrated Impedance Analysis"""
    
    def __init__(self):
        """Initialize the glucose monitoring system"""
        # Create the integrated impedance analyzer
        self.analyzer = IntegratedImpedanceAnalyzer()
        
        # Configure measurement parameters
        self.analyzer.configure(
            electrical_freq_range=(EIS_FREQUENCIES[0], EIS_FREQUENCIES[-1]),
            thermal_freq_range=(TIS_FREQUENCIES[0], TIS_FREQUENCIES[-1]),
            voltage_amplitude=VOLTAGE_AMPLITUDE,
            thermal_pulse_power=THERMAL_PULSE_POWER,
        )
        
        # Additional parameters specific to glucose monitoring
        self.analyzer.set_advanced_parameters(
            integration_time=2.0,  # seconds
            averages=5,            # number of measurements to average
            pcm_control=True,      # enable Phase Change Material temperature control
            target_temperature=33.0 # °C (skin temperature)
        )
        
        # Load calibration data if available
        try:
            self.calibration_data = pd.read_csv('glucose_calibration.csv')
            self.is_calibrated = True
            print("Calibration data loaded successfully")
        except FileNotFoundError:
            self.is_calibrated = False
            print("No calibration data found. Using default model parameters.")
        
        # Initialize ML model for glucose prediction
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize the machine learning model for glucose estimation"""
        # In a real implementation, this would load a pre-trained model
        # For demonstration, we're using a simplistic model
        self.scaler = StandardScaler()
        
        # Simulated model parameters (would be loaded from a file in real implementation)
        # These weights represent the importance of each feature in glucose prediction
        self.model_weights = {
            'eis_real_low': 0.25,      # Real part of EIS at low frequencies
            'eis_imag_low': 0.15,      # Imaginary part of EIS at low frequencies
            'eis_real_high': 0.05,     # Real part of EIS at high frequencies
            'eis_imag_high': 0.05,     # Imaginary part of EIS at high frequencies
            'tis_real_low': 0.20,      # Real part of TIS at low frequencies
            'tis_imag_low': 0.15,      # Imaginary part of TIS at low frequencies
            'tis_real_high': 0.10,     # Real part of TIS at high frequencies
            'tis_imag_high': 0.05,     # Imaginary part of TIS at high frequencies
        }
    
    def calibrate(self, reference_glucose_values):
        """
        Calibrate the system using reference glucose measurements
        
        Args:
            reference_glucose_values: List of tuples (time, glucose_mg_dl)
        """
        print("Starting calibration process...")
        
        # Collect reference data
        reference_data = []
        for time, glucose_value in reference_glucose_values:
            print(f"Calibration measurement at {time}, reference: {glucose_value} mg/dL")
            
            # Take measurement at calibration time
            measurement = self.analyzer.measure()
            
            # Extract features
            features = self._extract_features(measurement)
            
            # Store calibration data
            reference_data.append({
                'timestamp': time,
                'reference_glucose': glucose_value,
                **features
            })
        
        # Create calibration dataset
        self.calibration_data = pd.DataFrame(reference_data)
        
        # Update model parameters based on calibration data
        X = self.calibration_data.drop(['timestamp', 'reference_glucose'], axis=1)
        y = self.calibration_data['reference_glucose']
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Save calibration data
        self.calibration_data.to_csv('glucose_calibration.csv', index=False)
        
        self.is_calibrated = True
        print("Calibration completed successfully")
    
    def _extract_features(self, measurement):
        """
        Extract relevant features from the impedance measurements
        
        Args:
            measurement: Raw impedance measurement data
            
        Returns:
            Dictionary of extracted features
        """
        # Extract electrical impedance data
        eis_data = measurement['electrical_impedance']
        eis_freq = eis_data['frequency']
        eis_real = eis_data['real']
        eis_imag = eis_data['imaginary']
        
        # Extract thermal impedance data
        tis_data = measurement['thermal_impedance']
        tis_freq = tis_data['frequency']
        tis_real = tis_data['real']
        tis_imag = tis_data['imaginary']
        
        # Low and high frequency indices
        eis_low_idx = np.where(eis_freq < 1000)[0]  # Below 1kHz
        eis_high_idx = np.where(eis_freq >= 1000)[0]  # Above 1kHz
        tis_low_idx = np.where(tis_freq < 0.1)[0]    # Below 0.1Hz
        tis_high_idx = np.where(tis_freq >= 0.1)[0]  # Above 0.1Hz
        
        # Calculate mean values for each frequency band
        features = {
            'eis_real_low': np.mean(eis_real[eis_low_idx]),
            'eis_imag_low': np.mean(eis_imag[eis_low_idx]),
            'eis_real_high': np.mean(eis_real[eis_high_idx]),
            'eis_imag_high': np.mean(eis_imag[eis_high_idx]),
            'tis_real_low': np.mean(tis_real[tis_low_idx]),
            'tis_imag_low': np.mean(tis_imag[tis_low_idx]),
            'tis_real_high': np.mean(tis_real[tis_high_idx]),
            'tis_imag_high': np.mean(tis_imag[tis_high_idx]),
        }
        
        return features
    
    def estimate_glucose(self, measurement=None):
        """
        Estimate glucose level from impedance measurements
        
        Args:
            measurement: Optional measurement data (if None, a new measurement is taken)
            
        Returns:
            Estimated glucose level in mg/dL
        """
        if not self.is_calibrated:
            print("Warning: System not calibrated. Results may be inaccurate.")
        
        # Take a new measurement if none provided
        if measurement is None:
            measurement = self.analyzer.measure()
        
        # Extract features
        features = self._extract_features(measurement)
        
        # Prepare feature vector
        X = pd.DataFrame([features])
        
        # Normalize features
        X_scaled = self.scaler.transform(X)
        
        # Simplified glucose estimation for demonstration
        # In a real implementation, this would use a proper ML model
        glucose_estimate = 0
        for i, feature_name in enumerate(features.keys()):
            glucose_estimate += X_scaled[0, i] * self.model_weights[feature_name]
        
        # Scale to realistic glucose range (70-180 mg/dL is typical range)
        glucose_estimate = 70 + (glucose_estimate + 1) * 55
        
        # Apply boundary conditions
        glucose_estimate = max(40, min(400, glucose_estimate))
        
        return glucose_estimate
    
    def start_continuous_monitoring(self, duration_hours=MONITORING_DURATION, 
                                  interval_minutes=MEASUREMENT_INTERVAL):
        """
        Start continuous glucose monitoring for the specified duration
        
        Args:
            duration_hours: Duration of monitoring in hours
            interval_minutes: Measurement interval in minutes
        """
        print(f"Starting continuous glucose monitoring for {duration_hours} hours...")
        print(f"Measurement interval: {interval_minutes} minutes")
        
        # Calculate number of measurements
        num_measurements = int(duration_hours * 60 / interval_minutes)
        
        # Initialize results storage
        timestamps = []
        glucose_values = []
        
        # Function to determine if time is near a meal or activity
        def is_near_meal(current_time):
            return any(abs((meal_time - current_time).total_seconds()) < 60*60 for meal_time in MEAL_TIMES)
            
        def is_during_activity(current_time):
            return any(start <= current_time <= end for start, end in ACTIVITY_TIMES)
        
        # Simulate monitoring
        start_time = datetime.now()
        for i in range(num_measurements):
            # Calculate current timestamp
            current_time = start_time + timedelta(minutes=i*interval_minutes)
            timestamps.append(current_time)
            
            # Perform measurement
            print(f"Taking measurement at {current_time.strftime('%H:%M:%S')}...", end="")
            measurement = self.analyzer.measure()
            
            # Estimate glucose level
            glucose = self.estimate_glucose(measurement)
            
            # Add realistic variations based on simulated meals and activities
            if is_near_meal(current_time):
                # Simulate post-meal glucose rise
                meal_effect = 40 * np.exp(-((current_time - min(MEAL_TIMES, key=lambda x: abs(x - current_time))).total_seconds() / 3600))
                glucose += meal_effect
                print(f" Post-meal effect: +{meal_effect:.1f} mg/dL", end="")
                
            if is_during_activity(current_time):
                # Simulate exercise-induced glucose decrease
                glucose -= 20
                print(" Exercise effect: -20 mg/dL", end="")
                
            glucose_values.append(glucose)
            print(f" Glucose: {glucose:.1f} mg/dL")
            
            # In a real implementation, we would actually wait between measurements
            # For simulation, we're just processing the data points sequentially
        
        # Plot results
        self._plot_results(timestamps, glucose_values)
        
        return timestamps, glucose_values
    
    def _plot_results(self, timestamps, glucose_values):
        """
        Plot glucose monitoring results
        
        Args:
            timestamps: List of measurement timestamps
            glucose_values: List of glucose level estimates
        """
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, glucose_values, 'b-', linewidth=2)
        
        # Add horizontal lines for normal range
        plt.axhline(y=70, color='g', linestyle='--', alpha=0.7)
        plt.axhline(y=180, color='r', linestyle='--', alpha=0.7)
        
        # Shade the normal range
        plt.axhspan(70, 180, alpha=0.2, color='green')
        
        # Add markers for meals and activities
        for meal_time in MEAL_TIMES:
            plt.axvline(x=meal_time, color='orange', linestyle='-', alpha=0.5)
            plt.text(meal_time, max(glucose_values)+10, "Meal", rotation=90)
            
        for start, end in ACTIVITY_TIMES:
            plt.axvspan(start, end, alpha=0.2, color='blue')
            plt.text(start + (end - start)/2, min(glucose_values)-10, "Exercise", 
                    horizontalalignment='center')
        
        plt.title('Continuous Glucose Monitoring')
        plt.xlabel('Time')
        plt.ylabel('Glucose (mg/dL)')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for ranges
        plt.text(timestamps[0], 190, "Hyperglycemia", color='red')
        plt.text(timestamps[0], 125, "Normal Range", color='green')
        plt.text(timestamps[0], 60, "Hypoglycemia", color='red')
        
        plt.tight_layout()
        plt.savefig('glucose_monitoring_results.png')
        plt.show()


def main():
    """Main function to demonstrate glucose monitoring functionality"""
    # Create glucose monitoring system
    glucose_system = GlucoseMonitoring()
    
    # Reference glucose values for calibration (time, value in mg/dL)
    reference_values = [
        (datetime.now(), 95),                                # Fasting glucose
        (datetime.now() + timedelta(hours=2), 140),          # Post-breakfast
        (datetime.now() + timedelta(hours=6), 120),          # Post-lunch
        (datetime.now() + timedelta(hours=12), 110),         # Evening
    ]
    
    # Calibrate the system
    glucose_system.calibrate(reference_values)
    
    # Start continuous monitoring (default 24 hours with 5-minute intervals)
    timestamps, glucose_values = glucose_system.start_continuous_monitoring()
    
    # Print summary statistics
    mean_glucose = np.mean(glucose_values)
    std_glucose = np.std(glucose_values)
    min_glucose = np.min(glucose_values)
    max_glucose = np.max(glucose_values)
    
    print("\nGlucose Monitoring Summary:")
    print(f"Mean Glucose: {mean_glucose:.1f} mg/dL")
    print(f"Standard Deviation: {std_glucose:.1f} mg/dL")
    print(f"Range: {min_glucose:.1f} - {max_glucose:.1f} mg/dL")
    
    # Calculate time in range (70-180 mg/dL is the typical target range)
    time_in_range = sum(70 <= g <= 180 for g in glucose_values) / len(glucose_values) * 100
    print(f"Time in Range (70-180 mg/dL): {time_in_range:.1f}%")


if __name__ == "__main__":
    main()
