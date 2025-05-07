#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hydration Status Monitoring Application Example

This example demonstrates how to use the integrated electrical-thermal impedance
analyzer system for continuous hydration status monitoring. The system combines
electrical impedance spectroscopy (EIS) and thermal impedance spectroscopy (TIS)
to assess body fluid balance and hydration levels non-invasively.

The implementation is based on the research and technology developed by Ucaretron Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import pandas as pd
from impedance_analyzer import IntegratedImpedanceAnalyzer

# Constants
MEASUREMENT_INTERVAL = 15  # minutes
MONITORING_DURATION = 24   # hours
EIS_FREQUENCIES = np.logspace(2, 6, 20)  # 100Hz to 1MHz, 20 points
TIS_FREQUENCIES = np.logspace(-2, 0, 10)  # 0.01Hz to 1Hz, 10 points
VOLTAGE_AMPLITUDE = 10e-3  # V
THERMAL_PULSE_POWER = 100e-3  # W

# Hydration status reference levels (in percentage)
HYDRATION_LEVELS = {
    'dehydrated': (0, 60),
    'mild_dehydration': (60, 70),
    'normal': (70, 90),
    'overhydrated': (90, 100),
}

# Simulated water intake events for demonstration
WATER_INTAKE_EVENTS = [
    (datetime.now() + timedelta(hours=1), 250),     # 250 ml morning
    (datetime.now() + timedelta(hours=4), 350),     # 350 ml mid-morning
    (datetime.now() + timedelta(hours=7), 500),     # 500 ml lunch
    (datetime.now() + timedelta(hours=10), 350),    # 350 ml afternoon
    (datetime.now() + timedelta(hours=13), 300),    # 300 ml dinner
    (datetime.now() + timedelta(hours=16), 200),    # 200 ml evening
]

# Simulated activity events that affect hydration
ACTIVITY_EVENTS = [
    (datetime.now() + timedelta(hours=2), datetime.now() + timedelta(hours=3), "Exercise", 500),   # 500ml loss
    (datetime.now() + timedelta(hours=9), datetime.now() + timedelta(hours=9.5), "Hot Environment", 200),  # 200ml loss
    (datetime.now() + timedelta(hours=14), datetime.now() + timedelta(hours=15), "Exercise", 600),  # 600ml loss
]


class HydrationMonitoring:
    """Hydration Status Monitoring using Integrated Impedance Analysis"""
    
    def __init__(self):
        """Initialize the hydration monitoring system"""
        # Create the integrated impedance analyzer
        self.analyzer = IntegratedImpedanceAnalyzer()
        
        # Configure measurement parameters
        self.analyzer.configure(
            electrical_freq_range=(EIS_FREQUENCIES[0], EIS_FREQUENCIES[-1]),
            thermal_freq_range=(TIS_FREQUENCIES[0], TIS_FREQUENCIES[-1]),
            voltage_amplitude=VOLTAGE_AMPLITUDE,
            thermal_pulse_power=THERMAL_PULSE_POWER,
        )
        
        # Additional parameters specific to hydration monitoring
        self.analyzer.set_advanced_parameters(
            integration_time=3.0,        # seconds
            averages=3,                  # number of measurements to average
            pcm_control=True,            # enable Phase Change Material temperature control
            target_temperature=32.0,     # °C (skin temperature)
            electrode_config="tetrapolar"  # 4-electrode configuration for better depth penetration
        )
        
        # Load calibration data if available
        try:
            self.calibration_data = pd.read_csv('hydration_calibration.csv')
            self.is_calibrated = True
            print("Calibration data loaded successfully")
            
            # Extract model parameters from calibration data
            self.model_params = {
                'intracellular_factor': float(self.calibration_data['intracellular_factor'].iloc[0]),
                'extracellular_factor': float(self.calibration_data['extracellular_factor'].iloc[0]),
                'reference_weight': float(self.calibration_data['reference_weight'].iloc[0]),
                'height': float(self.calibration_data['height'].iloc[0]),
                'age': float(self.calibration_data['age'].iloc[0]),
                'sex': self.calibration_data['sex'].iloc[0],
            }
        except FileNotFoundError:
            self.is_calibrated = False
            # Default model parameters
            self.model_params = {
                'intracellular_factor': 0.65,
                'extracellular_factor': 0.35,
                'reference_weight': 70.0,  # kg
                'height': 170.0,           # cm
                'age': 35,                 # years
                'sex': 'male',
            }
            print("No calibration data found. Using default model parameters.")
        
        # Calculate reference total body water based on Watson formula
        if self.model_params['sex'] == 'male':
            self.reference_tbw = 2.447 - 0.09516 * self.model_params['age'] + 0.1074 * self.model_params['height'] + 0.3362 * self.model_params['reference_weight']
        else:  # female
            self.reference_tbw = -2.097 + 0.1069 * self.model_params['height'] + 0.2466 * self.model_params['reference_weight']
            
        print(f"Reference total body water: {self.reference_tbw:.2f} L")
        
        # Set initial hydration status
        self.current_hydration = 0.75 * 100  # 75% as starting point
        self.current_tbw = self.reference_tbw * (self.current_hydration / 100)
        
        # Initialize fluid balance tracking
        self.fluid_balance = 0.0  # net fluid balance in ml
        self.hourly_changes = []  # to store hourly fluid changes
        
    def calibrate(self, subject_data, reference_measurements=None):
        """
        Calibrate the system using subject data and optional reference measurements
        
        Args:
            subject_data: Dictionary with keys 'weight', 'height', 'age', 'sex'
            reference_measurements: Optional list of reference hydration measurements
        """
        print("Starting calibration process...")
        
        # Store subject data
        self.model_params = {
            'intracellular_factor': 0.65,  # Default starting value
            'extracellular_factor': 0.35,  # Default starting value
            'reference_weight': subject_data['weight'],
            'height': subject_data['height'],
            'age': subject_data['age'],
            'sex': subject_data['sex'],
        }
        
        # Recalculate reference total body water
        if self.model_params['sex'] == 'male':
            self.reference_tbw = 2.447 - 0.09516 * self.model_params['age'] + 0.1074 * self.model_params['height'] + 0.3362 * self.model_params['reference_weight']
        else:  # female
            self.reference_tbw = -2.097 + 0.1069 * self.model_params['height'] + 0.2466 * self.model_params['reference_weight']
            
        print(f"Reference total body water: {self.reference_tbw:.2f} L")
        
        # If reference measurements are provided, use them to adjust the model
        if reference_measurements:
            print("Calibrating with reference measurements...")
            
            for method, value in reference_measurements:
                print(f"Reference measurement using {method}: {value}")
                
                # Take impedance measurement
                measurement = self.analyzer.measure()
                
                # Extract features for calibration
                eis_data = measurement['electrical_impedance']
                tis_data = measurement['thermal_impedance']
                
                # Adjust factors based on reference measurements
                # In a real implementation, this would use a more sophisticated algorithm
                # For now, we're using a simplified approach
                if method == 'bioimpedance':
                    # Bioimpedance provides total body water estimate
                    tbw_ratio = value / self.reference_tbw
                    self.model_params['intracellular_factor'] *= tbw_ratio
                    self.model_params['extracellular_factor'] *= tbw_ratio
                    
                elif method == 'deuterium_dilution':
                    # Gold standard for total body water
                    self.reference_tbw = value
                    # Adjust the factors to match this reference
                    intracellular_estimate = self._estimate_icw(eis_data, tis_data)
                    extracellular_estimate = self._estimate_ecw(eis_data, tis_data)
                    total_estimate = intracellular_estimate + extracellular_estimate
                    
                    # Correction factors
                    icw_correction = (value * 0.6) / intracellular_estimate
                    ecw_correction = (value * 0.4) / extracellular_estimate
                    
                    self.model_params['intracellular_factor'] *= icw_correction
                    self.model_params['extracellular_factor'] *= ecw_correction
        
        # Save calibration data
        calibration_df = pd.DataFrame([self.model_params])
        calibration_df.to_csv('hydration_calibration.csv', index=False)
        
        self.is_calibrated = True
        print("Calibration completed successfully")
    
    def _estimate_icw(self, eis_data, tis_data):
        """
        Estimate intracellular water volume based on impedance data
        
        Args:
            eis_data: Electrical impedance data
            tis_data: Thermal impedance data
            
        Returns:
            Estimated intracellular water volume in liters
        """
        # Extract high frequency impedance data (>50kHz) which correlates with intracellular water
        high_freq_mask = eis_data['frequency'] > 50000
        high_freq_z = np.mean(np.sqrt(eis_data['real'][high_freq_mask]**2 + eis_data['imaginary'][high_freq_mask]**2))
        
        # Calculate intracellular water estimate using the model parameters
        # The formula is a simplified model for demonstration
        icw = self.model_params['intracellular_factor'] * (self.model_params['height']**2 / high_freq_z)
        
        # Apply thermal impedance correction (simplified model)
        thermal_correction = 1.0 + 0.1 * np.mean(tis_data['real']) / np.max(tis_data['real'])
        
        return icw * thermal_correction
    
    def _estimate_ecw(self, eis_data, tis_data):
        """
        Estimate extracellular water volume based on impedance data
        
        Args:
            eis_data: Electrical impedance data
            tis_data: Thermal impedance data
            
        Returns:
            Estimated extracellular water volume in liters
        """
        # Extract low frequency impedance data (<10kHz) which correlates with extracellular water
        low_freq_mask = eis_data['frequency'] < 10000
        low_freq_z = np.mean(np.sqrt(eis_data['real'][low_freq_mask]**2 + eis_data['imaginary'][low_freq_mask]**2))
        high_freq_mask = eis_data['frequency'] > 50000
        high_freq_z = np.mean(np.sqrt(eis_data['real'][high_freq_mask]**2 + eis_data['imaginary'][high_freq_mask]**2))
        
        # Calculate impedance of extracellular path
        z_ecw = (low_freq_z * high_freq_z) / (low_freq_z - high_freq_z)
        
        # Calculate extracellular water estimate using the model parameters
        ecw = self.model_params['extracellular_factor'] * (self.model_params['height']**2 / z_ecw)
        
        # Apply thermal impedance correction (simplified model)
        thermal_correction = 1.0 + 0.05 * np.mean(tis_data['imaginary']) / np.max(tis_data['imaginary'])
        
        return ecw * thermal_correction
    
    def assess_hydration(self, measurement=None):
        """
        Assess hydration status from impedance measurements
        
        Args:
            measurement: Optional measurement data (if None, a new measurement is taken)
            
        Returns:
            Dictionary with hydration assessment results
        """
        if not self.is_calibrated:
            print("Warning: System not calibrated. Results may be inaccurate.")
        
        # Take a new measurement if none provided
        if measurement is None:
            measurement = self.analyzer.measure()
        
        # Extract impedance data
        eis_data = measurement['electrical_impedance']
        tis_data = measurement['thermal_impedance']
        
        # Estimate intracellular and extracellular water
        icw = self._estimate_icw(eis_data, tis_data)
        ecw = self._estimate_ecw(eis_data, tis_data)
        tbw = icw + ecw
        
        # Calculate hydration status as percentage of reference TBW
        hydration_percent = (tbw / self.reference_tbw) * 100
        
        # Store current hydration values
        self.current_hydration = hydration_percent
        self.current_tbw = tbw
        
        # Determine hydration status category
        status = 'unknown'
        for level, (min_val, max_val) in HYDRATION_LEVELS.items():
            if min_val <= hydration_percent < max_val:
                status = level
                break
        
        # Calculate ECW/ICW ratio (indicator of fluid distribution)
        ecw_icw_ratio = ecw / icw if icw > 0 else 0
        
        # Create results dictionary
        results = {
            'timestamp': datetime.now(),
            'total_body_water': tbw,
            'intracellular_water': icw,
            'extracellular_water': ecw,
            'hydration_percent': hydration_percent,
            'hydration_status': status,
            'ecw_icw_ratio': ecw_icw_ratio,
            'reference_tbw': self.reference_tbw,
        }
        
        return results
    
    def track_fluid_intake(self, volume_ml, timestamp=None):
        """
        Track fluid intake
        
        Args:
            volume_ml: Volume of fluid intake in ml
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update fluid balance
        self.fluid_balance += volume_ml
        
        # Add to hourly changes
        self.hourly_changes.append({
            'timestamp': timestamp,
            'change': volume_ml,
            'type': 'intake',
            'balance': self.fluid_balance
        })
        
        print(f"Recorded fluid intake: +{volume_ml} ml at {timestamp.strftime('%H:%M:%S')}")
        print(f"Current fluid balance: {self.fluid_balance} ml")
    
    def track_fluid_loss(self, volume_ml, type_of_loss="Other", timestamp=None):
        """
        Track fluid loss
        
        Args:
            volume_ml: Volume of fluid loss in ml
            type_of_loss: Type of fluid loss (e.g., "Sweat", "Urine")
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update fluid balance
        self.fluid_balance -= volume_ml
        
        # Add to hourly changes
        self.hourly_changes.append({
            'timestamp': timestamp,
            'change': -volume_ml,
            'type': type_of_loss,
            'balance': self.fluid_balance
        })
        
        print(f"Recorded fluid loss ({type_of_loss}): -{volume_ml} ml at {timestamp.strftime('%H:%M:%S')}")
        print(f"Current fluid balance: {self.fluid_balance} ml")
    
    def start_continuous_monitoring(self, duration_hours=MONITORING_DURATION, 
                                  interval_minutes=MEASUREMENT_INTERVAL):
        """
        Start continuous hydration monitoring for the specified duration
        
        Args:
            duration_hours: Duration of monitoring in hours
            interval_minutes: Measurement interval in minutes
        """
        print(f"Starting continuous hydration monitoring for {duration_hours} hours...")
        print(f"Measurement interval: {interval_minutes} minutes")
        
        # Calculate number of measurements
        num_measurements = int(duration_hours * 60 / interval_minutes)
        
        # Initialize results storage
        timestamps = []
        hydration_values = []
        tbw_values = []
        fluid_balance_values = []
        
        # Simulate monitoring
        start_time = datetime.now()
        
        # Reset fluid balance for this simulation
        self.fluid_balance = 0.0
        self.hourly_changes = []
        
        for i in range(num_measurements):
            # Calculate current timestamp
            current_time = start_time + timedelta(minutes=i*interval_minutes)
            timestamps.append(current_time)
            
            # Check for water intake events
            for intake_time, volume in WATER_INTAKE_EVENTS:
                if (abs((current_time - intake_time).total_seconds()) < interval_minutes * 30 and 
                    intake_time not in [entry['timestamp'] for entry in self.hourly_changes if entry['type'] == 'intake']):
                    self.track_fluid_intake(volume, intake_time)
            
            # Check for activity events
            for activity_start, activity_end, activity_type, loss_volume in ACTIVITY_EVENTS:
                if (activity_start <= current_time <= activity_end and 
                    activity_start not in [entry['timestamp'] for entry in self.hourly_changes if entry['type'] == activity_type]):
                    self.track_fluid_loss(loss_volume, activity_type, activity_start)
            
            # Perform measurement and assessment
            print(f"Taking measurement at {current_time.strftime('%H:%M:%S')}...", end="")
            measurement = self.analyzer.measure()
            results = self.assess_hydration(measurement)
            
            hydration_values.append(results['hydration_percent'])
            tbw_values.append(results['total_body_water'])
            fluid_balance_values.append(self.fluid_balance)
            
            print(f" Hydration: {results['hydration_percent']:.1f}%, "
                  f"Status: {results['hydration_status']}, "
                  f"TBW: {results['total_body_water']:.2f} L")
            
            # In a real implementation, we would actually wait between measurements
            # For simulation, we're just processing the data points sequentially
        
        # Plot results
        self._plot_results(timestamps, hydration_values, tbw_values, fluid_balance_values)
        
        # Create 24-hour hydration chart
        self._create_hydration_chart(hydration_values[-1], self.hourly_changes)
        
        return timestamps, hydration_values, tbw_values, fluid_balance_values
    
    def _plot_results(self, timestamps, hydration_values, tbw_values, fluid_balance_values):
        """
        Plot hydration monitoring results
        
        Args:
            timestamps: List of measurement timestamps
            hydration_values: List of hydration percentage values
            tbw_values: List of total body water values
            fluid_balance_values: List of fluid balance values
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot hydration percentage
        ax1.plot(timestamps, hydration_values, 'b-', linewidth=2)
        ax1.set_ylabel('Hydration (%)')
        ax1.set_title('Hydration Status Monitoring')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal lines and shading for hydration levels
        colors = {'dehydrated': 'red', 'mild_dehydration': 'orange', 'normal': 'green', 'overhydrated': 'blue'}
        for level, (min_val, max_val) in HYDRATION_LEVELS.items():
            ax1.axhspan(min_val, max_val, alpha=0.2, color=colors[level])
            ax1.axhline(y=min_val, color=colors[level], linestyle='--', alpha=0.7)
        
        # Add labels for hydration zones
        for level, (min_val, max_val) in HYDRATION_LEVELS.items():
            mid_val = (min_val + max_val) / 2
            ax1.text(timestamps[0], mid_val, level.replace('_', ' ').title(), 
                    fontsize=9, ha='left', va='center')
        
        # Plot total body water
        ax2.plot(timestamps, tbw_values, 'g-', linewidth=2)
        ax2.set_ylabel('Total Body Water (L)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.reference_tbw, color='r', linestyle='--', alpha=0.7, label='Reference TBW')
        ax2.legend()
        
        # Plot fluid balance
        ax3.plot(timestamps, fluid_balance_values, 'm-', linewidth=2)
        ax3.set_ylabel('Fluid Balance (ml)')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Add markers for water intake and activities
        for intake_time, volume in WATER_INTAKE_EVENTS:
            ax3.scatter(intake_time, volume, color='blue', marker='^', s=100)
            ax3.text(intake_time, volume + 50, f"+{volume}ml", ha='center', fontsize=8)
            
        for activity_start, _, activity_type, loss_volume in ACTIVITY_EVENTS:
            ax3.scatter(activity_start, -loss_volume, color='red', marker='v', s=100)
            ax3.text(activity_start, -loss_volume - 50, f"-{loss_volume}ml\n{activity_type}", ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('hydration_monitoring_results.png')
        plt.show()
    
    def _create_hydration_chart(self, current_hydration, hourly_changes):
        """
        Create a circular chart showing 24-hour hydration status
        
        Args:
            current_hydration: Current hydration percentage
            hourly_changes: List of hourly fluid changes
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        # Set up the polar plot as a 24-hour clock
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)
        ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
        ax.set_xticklabels([f"{h:d}:00" for h in range(24)])
        
        # Create color gradient for hydration levels
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        norm = mcolors.Normalize(vmin=50, vmax=90)  # 50-90% hydration range
        
        # Create background ring showing hydration status zones
        background_radius = 0.9
        for level, (min_val, max_val) in HYDRATION_LEVELS.items():
            color = {'dehydrated': 'red', 'mild_dehydration': 'orange', 
                     'normal': 'green', 'overhydrated': 'blue'}[level]
            wedge = Wedge((0, 0), background_radius, 0, 360, width=0.2, 
                           facecolor=color, alpha=0.2, edgecolor='gray', linewidth=0.5)
            ax.add_patch(wedge)
        
        # Calculate hourly fluid balance
        hourly_data = []
        if hourly_changes:
            # Group changes by hour
            changes_df = pd.DataFrame(hourly_changes)
            changes_df['hour'] = changes_df['timestamp'].apply(lambda x: x.hour)
            hourly_sums = changes_df.groupby('hour')['change'].sum()
            
            # Create hourly data
            for hour in range(24):
                if hour in hourly_sums.index:
                    hourly_data.append(hourly_sums[hour])
                else:
                    hourly_data.append(0)
        else:
            hourly_data = [0] * 24
        
        # Plot hourly changes as bars
        theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
        width = 2*np.pi / 24
        max_change = max(max(abs(min(hourly_data)), abs(max(hourly_data))), 100)
        bars = ax.bar(theta, [abs(h) / max_change * 0.6 for h in hourly_data], width=width, alpha=0.6)
        
        # Color bars based on intake (blue) vs loss (red)
        for i, bar in enumerate(bars):
            if hourly_data[i] > 0:
                bar.set_facecolor('blue')
                bar.set_edgecolor('darkblue')
            else:
                bar.set_facecolor('red')
                bar.set_edgecolor('darkred')
        
        # Add central circle showing current hydration status
        hydration_color = cmap(norm(min(max(current_hydration, 50), 90)))
        central_circle = Circle((0, 0), 0.3, facecolor=hydration_color, edgecolor='white', linewidth=2)
        ax.add_patch(central_circle)
        
        # Add text with current hydration
        plt.text(0, 0, f"{current_hydration:.1f}%", ha='center', va='center', fontsize=24, 
                 fontweight='bold', color='white')
        
        # Add current TBW text
        plt.text(0, -0.1, f"TBW: {self.current_tbw:.2f}L", ha='center', va='center', 
                 fontsize=12, color='white')
        
        # Add labels for the time points with fluid changes
        for entry in hourly_changes:
            hour = entry['timestamp'].hour
            theta_pos = 2*np.pi * hour/24
            r = 1.0  # Slightly outside the outer ring
            
            if entry['change'] > 0:
                label = f"+{entry['change']}ml"
                color = 'blue'
            else:
                label = f"{entry['change']}ml"
                color = 'red'
                
            ax.text(theta_pos, r, label, ha='center', va='center', fontsize=8, color=color,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Set chart title
        ax.set_title("24-Hour Hydration Status", y=1.1, fontsize=16)
        
        # Add legend for hydration zones
        legend_elements = []
        for level, (min_val, max_val) in HYDRATION_LEVELS.items():
            color = {'dehydrated': 'red', 'mild_dehydration': 'orange', 
                     'normal': 'green', 'overhydrated': 'blue'}[level]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=10, alpha=0.3,
                                            label=f"{level.replace('_', ' ').title()} ({min_val}-{max_val}%)"))
        
        # Add legend for bars
        legend_elements.append(plt.Line2D([0], [0], color='blue', lw=10, alpha=0.6, label='Fluid Intake'))
        legend_elements.append(plt.Line2D([0], [0], color='red', lw=10, alpha=0.6, label='Fluid Loss'))
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=3)
        
        # Remove radial grid and labels
        ax.set_yticklabels([])
        ax.grid(False)
        
        # Save and show
        plt.tight_layout()
        plt.savefig('hydration_chart.png')
        plt.show()


def main():
    """Main function to demonstrate hydration monitoring functionality"""
    # Create hydration monitoring system
    hydration_system = HydrationMonitoring()
    
    # Subject data for calibration
    subject_data = {
        'weight': 75.0,  # kg
        'height': 175.0, # cm
        'age': 35,       # years
        'sex': 'male',   # 'male' or 'female'
    }
    
    # Optional reference measurements for calibration
    reference_measurements = [
        ('bioimpedance', 42.5),  # 42.5L measured with bioimpedance analyzer
        ('deuterium_dilution', 43.1),  # 43.1L measured with deuterium dilution method
    ]
    
    # Calibrate the system
    hydration_system.calibrate(subject_data, reference_measurements)
    
    # Perform a single hydration assessment
    results = hydration_system.assess_hydration()
    print("\nHydration Assessment Results:")
    print(f"Total Body Water: {results['total_body_water']:.2f} L")
    print(f"Intracellular Water: {results['intracellular_water']:.2f} L")
    print(f"Extracellular Water: {results['extracellular_water']:.2f} L")
    print(f"Hydration Percentage: {results['hydration_percent']:.1f}%")
    print(f"Hydration Status: {results['hydration_status']}")
    print(f"ECW/ICW Ratio: {results['ecw_icw_ratio']:.2f}")
    
    # Start continuous monitoring (default 24 hours with 15-minute intervals)
    timestamps, hydration_values, tbw_values, fluid_balance = hydration_system.start_continuous_monitoring()
    
    # Print summary statistics
    mean_hydration = np.mean(hydration_values)
    std_hydration = np.std(hydration_values)
    min_hydration = np.min(hydration_values)
    max_hydration = np.max(hydration_values)
    
    print("\nHydration Monitoring Summary:")
    print(f"Mean Hydration: {mean_hydration:.1f}%")
    print(f"Standard Deviation: {std_hydration:.1f}%")
    print(f"Range: {min_hydration:.1f}% - {max_hydration:.1f}%")
    
    # Calculate time in normal hydration range
    time_in_range = sum(HYDRATION_LEVELS['normal'][0] <= h <= HYDRATION_LEVELS['normal'][1] 
                       for h in hydration_values) / len(hydration_values) * 100
    print(f"Time in Normal Range: {time_in_range:.1f}%")
    
    # Calculate final fluid balance
    final_balance = fluid_balance[-1]
    print(f"Net Fluid Balance: {final_balance:+.0f} ml")

    # Provide recommendations based on results
    if mean_hydration < HYDRATION_LEVELS['normal'][0]:
        print("\nRecommendation: Increase fluid intake. You appear to be dehydrated.")
    elif mean_hydration > HYDRATION_LEVELS['normal'][1]:
        print("\nRecommendation: Monitor fluid intake. You appear to be overhydrated.")
    else:
        print("\nRecommendation: Maintain current hydration practices. Your hydration level is within normal range.")


if __name__ == "__main__":
    main()
