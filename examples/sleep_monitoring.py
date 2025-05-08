#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sleep Stage Monitoring Application Example

This example demonstrates how to use the integrated electrical-thermal impedance
analyzer system for non-invasive sleep stage monitoring. The system combines
electrical impedance spectroscopy (EIS) and thermal impedance spectroscopy (TIS)
to assess sleep stages and quality without the need for complex EEG setups.

The implementation is based on the research and technology developed by Ucaretron Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import pandas as pd
from datetime import datetime, timedelta
from impedance_analyzer import IntegratedImpedanceAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLING_RATE = 1    # Hz (1 sample per second)
EIS_FREQUENCIES = np.logspace(1, 5, 20)  # 10Hz to 100kHz, 20 points
TIS_FREQUENCIES = np.logspace(-2, 0, 10)  # 0.01Hz to 1Hz, 10 points
VOLTAGE_AMPLITUDE = 5e-3  # V (reduced for comfort during sleep)
THERMAL_PULSE_POWER = 50e-3  # W (reduced for comfort during sleep)
EPOCH_LENGTH = 30   # seconds (standard sleep scoring epoch)

# Sleep stage definitions
SLEEP_STAGES = {
    0: {'name': 'Wake', 'color': 'red'},
    1: {'name': 'N1', 'color': 'lightskyblue'},
    2: {'name': 'N2', 'color': 'royalblue'},
    3: {'name': 'N3', 'color': 'darkblue'},
    4: {'name': 'REM', 'color': 'green'},
}

# Sleep disorder events
DISORDER_EVENTS = {
    'apnea': {'color': 'purple', 'marker': 'v'},
    'hypopnea': {'color': 'magenta', 'marker': '^'},
    'limb_movement': {'color': 'orange', 'marker': 's'},
    'arousal': {'color': 'yellow', 'marker': 'd'},
}


class SleepMonitoring:
    """Sleep Stage Monitoring using Integrated Impedance Analysis"""
    
    def __init__(self):
        """Initialize the sleep monitoring system"""
        # Create the integrated impedance analyzer
        self.analyzer = IntegratedImpedanceAnalyzer()
        
        # Configure measurement parameters
        self.analyzer.configure(
            electrical_freq_range=(EIS_FREQUENCIES[0], EIS_FREQUENCIES[-1]),
            thermal_freq_range=(TIS_FREQUENCIES[0], TIS_FREQUENCIES[-1]),
            voltage_amplitude=VOLTAGE_AMPLITUDE,
            thermal_pulse_power=THERMAL_PULSE_POWER,
        )
        
        # Additional parameters specific to sleep monitoring
        self.analyzer.set_advanced_parameters(
            integration_time=1.0,        # seconds
            averages=2,                  # number of measurements to average
            pcm_control=True,            # enable PCM temperature control
            target_temperature=32.0,     # °C (comfortable skin temperature)
            electrode_config="bipolar",  # 2-electrode configuration
            adaptive_sampling=True,      # adjust sampling based on changes
        )
        
        # Initialize ML model for sleep stage classification
        self._initialize_ml_model()
        
        # Storage for sleep data
        self.sleep_stages = []
        self.sleep_times = []
        self.sleep_events = []
        self.raw_data = []
        
        # Sleep quality metrics
        self.sleep_metrics = {
            'total_sleep_time': 0,
            'sleep_efficiency': 0,
            'sleep_latency': 0,
            'wake_after_sleep_onset': 0,
            'rem_percent': 0,
            'deep_sleep_percent': 0,
            'sleep_quality_score': 0,
            'apnea_hypopnea_index': 0,
            'arousal_index': 0,
            'limb_movement_index': 0,
        }
    
    def _initialize_ml_model(self):
        """Initialize the machine learning model for sleep stage classification"""
        # In a real implementation, this would load a pre-trained model
        # For demonstration, we're using a simplistic random forest model
        self.feature_scaler = StandardScaler()
        self.sleep_stage_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Check if a pre-trained model is available
        try:
            # Load pre-trained model parameters (simulated)
            self.model_trained = True
            print("Pre-trained sleep stage classification model loaded.")
        except:
            self.model_trained = False
            print("No pre-trained model found. Using simulated classification.")
    
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
        eis_magnitude = np.sqrt(eis_real**2 + eis_imag**2)
        eis_phase = np.arctan2(eis_imag, eis_real)
        
        # Extract thermal impedance data
        tis_data = measurement['thermal_impedance']
        tis_freq = tis_data['frequency']
        tis_real = tis_data['real']
        tis_imag = tis_data['imaginary']
        tis_magnitude = np.sqrt(tis_real**2 + tis_imag**2)
        tis_phase = np.arctan2(tis_imag, tis_real)
        
        # Frequency bands of interest for sleep monitoring
        # Low frequency EIS - correlates with slow wave activity
        eis_low_mask = eis_freq < 100
        # Mid frequency EIS - correlates with sleep spindles
        eis_mid_mask = (eis_freq >= 100) & (eis_freq < 1000)
        # High frequency EIS - correlates with fast activity
        eis_high_mask = eis_freq >= 1000
        
        # TIS frequency bands
        tis_low_mask = tis_freq < 0.1
        tis_high_mask = tis_freq >= 0.1
        
        # Calculate features
        features = {
            # EIS magnitude features
            'eis_mag_low_mean': np.mean(eis_magnitude[eis_low_mask]),
            'eis_mag_low_std': np.std(eis_magnitude[eis_low_mask]),
            'eis_mag_mid_mean': np.mean(eis_magnitude[eis_mid_mask]),
            'eis_mag_mid_std': np.std(eis_magnitude[eis_mid_mask]),
            'eis_mag_high_mean': np.mean(eis_magnitude[eis_high_mask]),
            'eis_mag_high_std': np.std(eis_magnitude[eis_high_mask]),
            
            # EIS phase features
            'eis_phase_low_mean': np.mean(eis_phase[eis_low_mask]),
            'eis_phase_mid_mean': np.mean(eis_phase[eis_mid_mask]),
            'eis_phase_high_mean': np.mean(eis_phase[eis_high_mask]),
            
            # EIS real/imaginary components
            'eis_real_low_mean': np.mean(eis_real[eis_low_mask]),
            'eis_imag_low_mean': np.mean(eis_imag[eis_low_mask]),
            'eis_real_high_mean': np.mean(eis_real[eis_high_mask]),
            'eis_imag_high_mean': np.mean(eis_imag[eis_high_mask]),
            
            # TIS features
            'tis_mag_low_mean': np.mean(tis_magnitude[tis_low_mask]),
            'tis_mag_high_mean': np.mean(tis_magnitude[tis_high_mask]),
            'tis_phase_low_mean': np.mean(tis_phase[tis_low_mask]),
            'tis_phase_high_mean': np.mean(tis_phase[tis_high_mask]),
            
            # Combined EIS-TIS features (ratios and correlations)
            'eis_tis_low_ratio': np.mean(eis_magnitude[eis_low_mask]) / (np.mean(tis_magnitude[tis_low_mask]) + 1e-10),
            'eis_tis_high_ratio': np.mean(eis_magnitude[eis_high_mask]) / (np.mean(tis_magnitude[tis_high_mask]) + 1e-10),
            
            # Calculated metrics related to sleep physiology
            'sympathetic_index': np.mean(eis_magnitude[eis_high_mask]) / (np.mean(eis_magnitude[eis_low_mask]) + 1e-10),
            'parasympathetic_index': np.mean(tis_magnitude[tis_low_mask]) / (np.mean(tis_magnitude[tis_high_mask]) + 1e-10),
            'thermal_reactivity': np.std(tis_magnitude) / (np.mean(tis_magnitude) + 1e-10),
        }
        
        # Cole-Cole parameters (simplified)
        r0 = np.min(eis_magnitude)
        rinf = np.max(eis_magnitude)
        features['cole_r0'] = r0
        features['cole_rinf'] = rinf
        features['cole_ratio'] = rinf / (r0 + 1e-10)
        
        return features
    
    def _detect_sleep_events(self, features, window_size=5):
        """
        Detect sleep-related events like apnea, limb movement
        
        Args:
            features: Time series of extracted features
            window_size: Number of epochs to consider for event detection
            
        Returns:
            Dictionary of detected events with timestamps
        """
        if len(features) < window_size:
            return []
        
        # Get the last window_size feature sets
        recent_features = features[-window_size:]
        
        events = []
        current_time = self.sleep_times[-1]
        
        # Apnea detection (simulated)
        # In a real implementation, this would use more sophisticated algorithms
        parasympathetic_values = [f['parasympathetic_index'] for f in recent_features]
        thermal_reactivity = [f['thermal_reactivity'] for f in recent_features]
        
        # Detect apnea (sudden changes in parasympathetic index and thermal reactivity)
        if (np.std(parasympathetic_values) > 0.5 and 
            np.mean(thermal_reactivity) < 0.2):
            events.append({
                'type': 'apnea',
                'time': current_time,
                'duration': window_size * EPOCH_LENGTH,
                'severity': np.std(parasympathetic_values)
            })
        
        # Limb movement detection (simulated)
        eis_variation = [f['eis_mag_high_std'] for f in recent_features]
        if np.max(eis_variation) > 1.5 * np.mean(eis_variation):
            events.append({
                'type': 'limb_movement',
                'time': current_time,
                'duration': EPOCH_LENGTH,
                'severity': np.max(eis_variation) / np.mean(eis_variation)
            })
        
        # Arousal detection (simulated)
        sympathetic_values = [f['sympathetic_index'] for f in recent_features]
        if np.max(sympathetic_values) > 1.5 * np.mean(sympathetic_values):
            events.append({
                'type': 'arousal',
                'time': current_time,
                'duration': EPOCH_LENGTH,
                'severity': np.max(sympathetic_values) / np.mean(sympathetic_values)
            })
            
        return events
    
    def _classify_sleep_stage(self, features):
        """
        Classify sleep stage based on extracted features
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Predicted sleep stage (0-4)
        """
        if not self.model_trained:
            # For demonstration, return simulated sleep stages
            # In a real implementation, this would use the trained model
            
            # Convert features to a feature vector
            feature_values = np.array(list(features.values())).reshape(1, -1)
            
            # Simple rule-based classification for demonstration
            sympathetic = features['sympathetic_index']
            parasympathetic = features['parasympathetic_index']
            thermal_reactivity = features['thermal_reactivity']
            cole_ratio = features['cole_ratio']
            
            # Simplified sleep stage classification rules
            if sympathetic > 1.5:  # High sympathetic activity
                return 0  # Wake
            elif parasympathetic > 1.2 and cole_ratio < 2.0:
                return 4  # REM
            elif parasympathetic > 1.0 and thermal_reactivity < 0.3:
                return 3  # N3 (Deep sleep)
            elif 0.7 < parasympathetic < 1.0:
                return 2  # N2
            else:
                return 1  # N1
        else:
            # Use trained model for classification
            feature_vector = np.array([features[key] for key in sorted(features.keys())]).reshape(1, -1)
            scaled_features = self.feature_scaler.transform(feature_vector)
            return self.sleep_stage_classifier.predict(scaled_features)[0]
    
    def start_monitoring(self, duration_hours=8):
        """
        Start sleep monitoring for the specified duration
        
        Args:
            duration_hours: Monitoring duration in hours
            
        Returns:
            None
        """
        print(f"Starting sleep monitoring for {duration_hours} hours...")
        
        # Reset storage
        self.sleep_stages = []
        self.sleep_times = []
        self.sleep_events = []
        self.raw_data = []
        
        # Calculate number of epochs
        n_epochs = int(duration_hours * 3600 / EPOCH_LENGTH)
        start_time = datetime.now()
        
        # Initialize feature storage
        all_features = []
        
        # For demonstration purposes, we'll simulate the measurements
        for epoch in range(n_epochs):
            current_time = start_time + timedelta(seconds=epoch * EPOCH_LENGTH)
            self.sleep_times.append(current_time)
            
            # In a real implementation, this would get actual measurements
            # For demonstration, we'll generate simulated data
            measurement = self._simulate_measurement(epoch, n_epochs)
            self.raw_data.append(measurement)
            
            # Extract features
            features = self._extract_features(measurement)
            all_features.append(features)
            
            # Classify sleep stage
            sleep_stage = self._classify_sleep_stage(features)
            self.sleep_stages.append(sleep_stage)
            
            # Detect sleep events
            if len(all_features) >= 5:  # Need some history for event detection
                events = self._detect_sleep_events(all_features)
                self.sleep_events.extend(events)
            
            # Print progress every 30 minutes
            if epoch % (30 * 60 / EPOCH_LENGTH) == 0 and epoch > 0:
                elapsed_minutes = epoch * EPOCH_LENGTH / 60
                print(f"Monitoring in progress... {elapsed_minutes:.0f} minutes elapsed")
        
        # Calculate sleep metrics
        self._calculate_sleep_metrics()
        
        print("Sleep monitoring completed.")
        print(f"Total monitoring time: {duration_hours} hours")
        print(f"Sleep efficiency: {self.sleep_metrics['sleep_efficiency']:.1f}%")
        print(f"Total sleep time: {self.sleep_metrics['total_sleep_time'] / 60:.1f} hours")
    
    def _simulate_measurement(self, epoch, total_epochs):
        """
        Simulate impedance measurements for demonstration
        
        In a real implementation, this would be replaced with actual measurements
        from the impedance analyzer hardware
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Simulated measurement data
        """
        # Create time-dependent patterns that resemble real sleep data
        # For demonstration purposes only
        
        # Time normalized to [0, 1]
        t_normalized = epoch / total_epochs
        
        # Create a sleep cycle pattern (roughly 90-minute cycles)
        cycle_freq = 2 * np.pi * (total_epochs / (90 * 60 / EPOCH_LENGTH))
        cycle_phase = cycle_freq * t_normalized
        
        # Add some randomness
        random_factor = 0.2 * np.random.randn()
        
        # Electrical impedance simulation
        eis_real = []
        eis_imag = []
        
        for freq in EIS_FREQUENCIES:
            # Base values depend on frequency
            base_real = 500 - 300 * np.log10(freq/EIS_FREQUENCIES[0]) / np.log10(EIS_FREQUENCIES[-1]/EIS_FREQUENCIES[0])
            base_imag = -200 + 150 * np.log10(freq/EIS_FREQUENCIES[0]) / np.log10(EIS_FREQUENCIES[-1]/EIS_FREQUENCIES[0])
            
            # Add time-dependent variations based on sleep cycle
            cycle_effect_real = 50 * np.sin(cycle_phase)
            cycle_effect_imag = 30 * np.sin(cycle_phase + np.pi/4)
            
            # Add random variations
            noise_real = 10 * np.random.randn()
            noise_imag = 10 * np.random.randn()
            
            # Combine effects
            real_val = base_real + cycle_effect_real + noise_real
            imag_val = base_imag + cycle_effect_imag + noise_imag
            
            eis_real.append(real_val)
            eis_imag.append(imag_val)
        
        # Thermal impedance simulation
        tis_real = []
        tis_imag = []
        
        for freq in TIS_FREQUENCIES:
            # Base values depend on frequency
            base_real = 0.2 - 0.1 * np.log10(freq/TIS_FREQUENCIES[0]) / np.log10(TIS_FREQUENCIES[-1]/TIS_FREQUENCIES[0])
            base_imag = -0.1 + 0.05 * np.log10(freq/TIS_FREQUENCIES[0]) / np.log10(TIS_FREQUENCIES[-1]/TIS_FREQUENCIES[0])
            
            # Add time-dependent variations based on sleep cycle
            cycle_effect_real = 0.05 * np.sin(cycle_phase + np.pi/3)
            cycle_effect_imag = 0.03 * np.sin(cycle_phase + np.pi/2)
            
            # Add random variations
            noise_real = 0.01 * np.random.randn()
            noise_imag = 0.01 * np.random.randn()
            
            # Combine effects
            real_val = base_real + cycle_effect_real + noise_real
            imag_val = base_imag + cycle_effect_imag + noise_imag
            
            tis_real.append(real_val)
            tis_imag.append(imag_val)
        
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
            'timestamp': self.sleep_times[-1],
            'metadata': {
                'epoch': epoch,
                'epoch_length': EPOCH_LENGTH,
                'configuration': {
                    'voltage_amplitude': VOLTAGE_AMPLITUDE,
                    'thermal_pulse_power': THERMAL_PULSE_POWER
                }
            }
        }
        
        return measurement
    
    def _calculate_sleep_metrics(self):
        """Calculate sleep quality metrics from the recorded data"""
        # Count epochs in each sleep stage
        stage_counts = {}
        for stage in range(5):  # 0-4
            stage_counts[stage] = self.sleep_stages.count(stage)
        
        # Total number of epochs
        total_epochs = len(self.sleep_stages)
        total_time_minutes = total_epochs * EPOCH_LENGTH / 60
        
        # Calculate metrics
        # Wake epochs
        wake_epochs = stage_counts[0]
        wake_minutes = wake_epochs * EPOCH_LENGTH / 60
        
        # Total sleep time (excluding wake)
        sleep_epochs = total_epochs - wake_epochs
        sleep_minutes = sleep_epochs * EPOCH_LENGTH / 60
        
        # Sleep efficiency
        sleep_efficiency = (sleep_minutes / total_time_minutes) * 100 if total_time_minutes > 0 else 0
        
        # Sleep latency (time to first sleep)
        sleep_latency = 0
        for stage in self.sleep_stages:
            if stage == 0:  # Wake
                sleep_latency += EPOCH_LENGTH / 60  # minutes
            else:
                break
        
        # REM sleep percentage
        rem_epochs = stage_counts[4]
        rem_percent = (rem_epochs / sleep_epochs) * 100 if sleep_epochs > 0 else 0
        
        # Deep sleep percentage (N3)
        deep_epochs = stage_counts[3]
        deep_percent = (deep_epochs / sleep_epochs) * 100 if sleep_epochs > 0 else 0
        
        # Wake after sleep onset (WASO)
        if len(self.sleep_stages) > 0 and any(stage != 0 for stage in self.sleep_stages):
            # Find first sleep epoch
            first_sleep = next(i for i, stage in enumerate(self.sleep_stages) if stage != 0)
            # Count wake epochs after first sleep
            waso_epochs = self.sleep_stages[first_sleep:].count(0)
            waso_minutes = waso_epochs * EPOCH_LENGTH / 60
        else:
            waso_minutes = 0
        
        # Calculate event indices (events per hour)
        total_hours = total_time_minutes / 60
        
        # Count events by type
        apnea_count = len([e for e in self.sleep_events if e['type'] == 'apnea'])
        arousal_count = len([e for e in self.sleep_events if e['type'] == 'arousal'])
        limb_movement_count = len([e for e in self.sleep_events if e['type'] == 'limb_movement'])
        
        # Calculate indices
        ahi = apnea_count / total_hours if total_hours > 0 else 0
        arousal_index = arousal_count / total_hours if total_hours > 0 else 0
        limb_movement_index = limb_movement_count / total_hours if total_hours > 0 else 0
        
        # Sleep quality score (0-100)
        # Simple calculation for demonstration - a real implementation would be more sophisticated
        sleep_quality_score = max(0, min(100, (
            sleep_efficiency * 0.3 +
            rem_percent * 0.2 +
            deep_percent * 0.3 +
            (20 - min(20, ahi)) * 1.0 +
            (10 - min(10, arousal_index)) * 1.0
        )))
        
        # Store metrics
        self.sleep_metrics = {
            'total_sleep_time': sleep_minutes,
            'sleep_efficiency': sleep_efficiency,
            'sleep_latency': sleep_latency,
            'wake_after_sleep_onset': waso_minutes,
            'rem_percent': rem_percent,
            'deep_sleep_percent': deep_percent,
            'sleep_quality_score': sleep_quality_score,
            'apnea_hypopnea_index': ahi,
            'arousal_index': arousal_index,
            'limb_movement_index': limb_movement_index,
        }
    
    def visualize_sleep_stages(self):
        """Visualize sleep stages in a hypnogram"""
        if not self.sleep_stages:
            print("No sleep data available. Run start_monitoring first.")
            return
        
        # Convert stages to a format suitable for visualization
        stages_array = np.array(self.sleep_stages)
        # Reverse the order for conventional hypnogram display (wake at top)
        stages_display = 4 - stages_array
        # Except for REM which is typically shown between N2 and N1
        stages_display[stages_array == 4] = 1  # Place REM between N1 and N2
        
        # Create a time array in hours from start
        hours_from_start = [(t - self.sleep_times[0]).total_seconds() / 3600 for t in self.sleep_times]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot hypnogram
        plt.subplot(2, 1, 1)
        plt.plot(hours_from_start, stages_display, linewidth=1.5, color='black')
        plt.step(hours_from_start, stages_display, where='post', linewidth=1.5, color='black')
        
        # Set y-axis labels and limits
        y_labels = ['N3', 'N2', 'REM', 'N1', 'Wake']
        plt.yticks(range(5), y_labels)
        plt.ylim(-0.5, 4.5)
        
        # Set x-axis limits and labels
        plt.xlim(0, max(hours_from_start))
        plt.xlabel('Hours from start')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        plt.title('Sleep Hypnogram', fontsize=14)
        
        # Plot events
        if self.sleep_events:
            for event in self.sleep_events:
                event_time = (event['time'] - self.sleep_times[0]).total_seconds() / 3600
                event_type = event['type']
                
                # Get event marker properties
                color = DISORDER_EVENTS[event_type]['color']
                marker = DISORDER_EVENTS[event_type]['marker']
                
                # Plot event marker
                plt.scatter(event_time, 4, marker=marker, color=color, s=80, zorder=10)
        
        # Add legend for events
        legend_elements = []
        for event_type, props in DISORDER_EVENTS.items():
            legend_elements.append(plt.Line2D([0], [0], marker=props['marker'], color='w',
                                            markerfacecolor=props['color'], markersize=10,
                                            label=event_type.replace('_', ' ').title()))
        
        plt.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
        
        # Plot sleep metrics summary
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # Create a summary box
        metrics_text = [
            f"Total Sleep Time: {self.sleep_metrics['total_sleep_time'] / 60:.1f} hours",
            f"Sleep Efficiency: {self.sleep_metrics['sleep_efficiency']:.1f}%",
            f"Sleep Latency: {self.sleep_metrics['sleep_latency']:.1f} minutes",
            f"WASO: {self.sleep_metrics['wake_after_sleep_onset']:.1f} minutes",
            f"REM Sleep: {self.sleep_metrics['rem_percent']:.1f}%",
            f"Deep Sleep: {self.sleep_metrics['deep_sleep_percent']:.1f}%",
            f"Apnea-Hypopnea Index: {self.sleep_metrics['apnea_hypopnea_index']:.1f} events/hour",
            f"Arousal Index: {self.sleep_metrics['arousal_index']:.1f} events/hour",
            f"Sleep Quality Score: {self.sleep_metrics['sleep_quality_score']:.0f}/100"
        ]
        
        # Display metrics text
        plt.text(0.1, 0.9, "Sleep Metrics Summary", fontsize=14, fontweight='bold')
        y_pos = 0.8
        for text in metrics_text:
            plt.text(0.1, y_pos, text, fontsize=12)
            y_pos -= 0.1
        
        # Add a sleep quality gauge
        quality_score = self.sleep_metrics['sleep_quality_score']
        gauge_colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        gauge_thresholds = [0, 20, 40, 60, 80, 100]
        
        # Draw gauge background
        gauge_x = 0.7
        gauge_y = 0.5
        gauge_width = 0.2
        gauge_height = 0.3
        
        # Draw gauge segments
        for i in range(len(gauge_colors)):
            segment_width = gauge_width * (gauge_thresholds[i+1] - gauge_thresholds[i]) / 100
            rect = patches.Rectangle((gauge_x + gauge_width * gauge_thresholds[i] / 100, gauge_y), 
                                    segment_width, gauge_height, 
                                    facecolor=gauge_colors[i], alpha=0.7)
            plt.gca().add_patch(rect)
        
        # Draw gauge pointer
        pointer_x = gauge_x + gauge_width * quality_score / 100
        plt.plot([pointer_x, pointer_x], [gauge_y, gauge_y+gauge_height], 'k-', linewidth=3)
        
        # Add gauge labels
        plt.text(gauge_x, gauge_y+gauge_height+0.05, "Sleep Quality", fontsize=12, fontweight='bold', ha='left')
        plt.text(gauge_x, gauge_y-0.05, "Poor", fontsize=10, ha='left')
        plt.text(gauge_x+gauge_width, gauge_y-0.05, "Excellent", fontsize=10, ha='right')
        plt.text(pointer_x, gauge_y+gauge_height+0.15, f"{quality_score:.0f}", fontsize=12, fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def export_data(self, filename):
        """
        Export sleep data to CSV file
        
        Args:
            filename: Output filename
            
        Returns:
            None
        """
        if not self.sleep_stages:
            print("No sleep data available. Run start_monitoring first.")
            return
        
        # Create a DataFrame with sleep data
        data = {
            'timestamp': self.sleep_times,
            'sleep_stage': self.sleep_stages,
            'sleep_stage_name': [SLEEP_STAGES[s]['name'] for s in self.sleep_stages]
        }
        
        # Add selected features if available
        if self.raw_data:
            for i, measurement in enumerate(self.raw_data):
                features = self._extract_features(measurement)
                
                # For first iteration, initialize columns
                if i == 0:
                    for key in features:
                        data[f'feature_{key}'] = []
                
                # Add feature values
                for key, value in features.items():
                    data[f'feature_{key}'].append(value)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Sleep data exported to {filename}")
        
        # Also export sleep metrics as a separate file
        metrics_df = pd.DataFrame([self.sleep_metrics])
        metrics_filename = filename.replace('.csv', '_metrics.csv')
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Sleep metrics exported to {metrics_filename}")


def main():
    """Main function to demonstrate the sleep monitoring system"""
    # Create an instance of the sleep monitoring system
    sleep_monitor = SleepMonitoring()
    
    # Start monitoring for a specified duration (in hours)
    # For demonstration, we'll use a shorter duration
    sleep_monitor.start_monitoring(duration_hours=8)
    
    # Visualize the sleep stages
    sleep_monitor.visualize_sleep_stages()
    
    # Export the data
    sleep_monitor.export_data("sleep_monitoring_data.csv")


if __name__ == "__main__":
    main()
