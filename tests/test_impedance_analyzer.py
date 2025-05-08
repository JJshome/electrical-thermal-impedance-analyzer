#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the IntegratedImpedanceAnalyzer class
"""

import unittest
import numpy as np
from impedance_analyzer import IntegratedImpedanceAnalyzer

class TestImpedanceAnalyzer(unittest.TestCase):
    """Test cases for IntegratedImpedanceAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = IntegratedImpedanceAnalyzer()
    
    def tearDown(self):
        """Tear down test fixtures"""
        pass
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertTrue(self.analyzer.status['connected'])
        self.assertFalse(self.analyzer.status['calibrated'])
    
    def test_configuration(self):
        """Test analyzer configuration"""
        # Initial configuration
        self.assertEqual(self.analyzer.config['electrical_freq_range'], (0.1, 100000))
        self.assertEqual(self.analyzer.config['thermal_freq_range'], (0.01, 1))
        
        # Update configuration
        self.analyzer.configure(
            electrical_freq_range=(1, 10000),
            thermal_freq_range=(0.1, 0.5),
            voltage_amplitude=5e-3,
            thermal_pulse_power=50e-3
        )
        
        # Check updated configuration
        self.assertEqual(self.analyzer.config['electrical_freq_range'], (1, 10000))
        self.assertEqual(self.analyzer.config['thermal_freq_range'], (0.1, 0.5))
        self.assertEqual(self.analyzer.config['voltage_amplitude'], 5e-3)
        self.assertEqual(self.analyzer.config['thermal_pulse_power'], 50e-3)
    
    def test_advanced_parameters(self):
        """Test advanced parameter configuration"""
        # Initial parameters
        self.assertEqual(self.analyzer.advanced_params['integration_time'], 1.0)
        self.assertEqual(self.analyzer.advanced_params['electrode_config'], "4-wire")
        
        # Update parameters
        self.analyzer.set_advanced_parameters(
            integration_time=0.5,
            electrode_config="bipolar"
        )
        
        # Check updated parameters
        self.assertEqual(self.analyzer.advanced_params['integration_time'], 0.5)
        self.assertEqual(self.analyzer.advanced_params['electrode_config'], "bipolar")
        
        # Get parameters
        params = self.analyzer.get_advanced_parameters()
        self.assertEqual(params['integration_time'], 0.5)
        self.assertEqual(params['electrode_config'], "bipolar")
    
    def test_calibration(self):
        """Test system calibration"""
        # Initially not calibrated
        self.assertFalse(self.analyzer.status['calibrated'])
        
        # Perform calibration
        result = self.analyzer.calibrate()
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(self.analyzer.status['calibrated'])
        self.assertIsNotNone(self.analyzer.calibration_data)
    
    def test_measurement(self):
        """Test impedance measurement"""
        # Calibrate first (for better results)
        self.analyzer.calibrate()
        
        # Perform measurement
        result = self.analyzer.measure()
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertIn('electrical_impedance', result)
        self.assertIn('thermal_impedance', result)
        
        # Check electrical impedance data
        eis_data = result['electrical_impedance']
        self.assertIn('frequency', eis_data)
        self.assertIn('real', eis_data)
        self.assertIn('imaginary', eis_data)
        
        # Check dimensions
        self.assertEqual(len(eis_data['frequency']), len(eis_data['real']))
        self.assertEqual(len(eis_data['frequency']), len(eis_data['imaginary']))
        
        # Check thermal impedance data
        tis_data = result['thermal_impedance']
        self.assertIn('frequency', tis_data)
        self.assertIn('real', tis_data)
        self.assertIn('imaginary', tis_data)
        
        # Check dimensions
        self.assertEqual(len(tis_data['frequency']), len(tis_data['real']))
        self.assertEqual(len(tis_data['frequency']), len(tis_data['imaginary']))
    
    def test_analysis(self):
        """Test impedance analysis"""
        # Perform measurement
        measurement = self.analyzer.measure()
        
        # Analyze the data
        analysis = self.analyzer.analyze(measurement)
        
        # Check analysis results
        self.assertIsNotNone(analysis)
        self.assertIn('electrical_parameters', analysis)
        self.assertIn('thermal_parameters', analysis)
        self.assertIn('cross_domain_analysis', analysis)
        
        # Check electrical parameters
        elec_params = analysis['electrical_parameters']
        self.assertIn('resistance', elec_params)
        self.assertIn('capacitance', elec_params)
        
        # Check thermal parameters
        thermal_params = analysis['thermal_parameters']
        self.assertIn('thermal_resistance', thermal_params)
        self.assertIn('thermal_capacitance', thermal_params)
        
        # Check cross-domain analysis
        cross_domain = analysis['cross_domain_analysis']
        self.assertIn('electro_thermal_correlation', cross_domain)
    
    def test_self_test(self):
        """Test the system self-test functionality"""
        result = self.analyzer.self_test()
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertIn('test_results', result)
        self.assertIn('all_passed', result)
        
        # Check individual test results
        test_results = result['test_results']
        self.assertIn('power_supply', test_results)
        self.assertIn('communication', test_results)
        self.assertIn('signal_generation', test_results)
        self.assertIn('signal_acquisition', test_results)
        self.assertIn('temperature_control', test_results)
        self.assertIn('calibration', test_results)
    
    def test_simulated_capacitor_impedance(self):
        """Test the capacitor impedance simulation"""
        capacitance = 1e-6  # 1 μF
        num_points = 10
        
        # Get simulated impedance
        impedance = self.analyzer._simulate_capacitor_impedance(capacitance, num_points)
        
        # Check result
        self.assertEqual(len(impedance), num_points)
        
        # Check mathematical correctness (for a few points)
        frequencies = np.logspace(1, 5, num_points)  # 10Hz to 100kHz
        for i in range(num_points):
            expected_magnitude = 1 / (2 * np.pi * frequencies[i] * capacitance)
            actual_magnitude = abs(impedance[i])
            # Allow for some floating point error
            self.assertAlmostEqual(actual_magnitude, expected_magnitude, delta=1e-10)
    
    def test_export_data(self):
        """Test data export functionality (to memory)"""
        # Perform measurement
        measurement = self.analyzer.measure()
        
        # Test CSV export (simulated by checking format)
        csv_export = self.analyzer._make_json_serializable(measurement)
        
        # Check result structure
        self.assertIsNotNone(csv_export)
        self.assertIn('electrical_impedance', csv_export)
        self.assertIn('thermal_impedance', csv_export)


if __name__ == "__main__":
    unittest.main()
