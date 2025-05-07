"""
Electrical models module for the Integrated Electrical-Thermal Impedance Analyzer

This module implements various electrical equivalent circuit models for analyzing
electrical impedance data, including Randles circuit, RC circuits, and CPE models.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple

class BaseElectricalModel:
    """Base class for electrical circuit models."""
    
    def __init__(self):
        """Initialize base electrical model."""
        self.params = {}
        self.fitted = False
    
    def impedance(self, freq):
        """Calculate impedance at given frequencies."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def fit(self, freq, imp_data):
        """Fit model parameters to measured impedance data."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot_nyquist(self, freq=None, imp_data=None, fig=None, ax=None):
        """Plot Nyquist plot (imaginary vs real part of impedance)."""
        if not self.fitted and imp_data is None:
            raise ValueError("Model must be fitted or measured data must be provided")
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate model impedance if needed
        if self.fitted and freq is not None:
            model_imp = self.impedance(freq)
            ax.plot(np.real(model_imp), -np.imag(model_imp), 'r-', label='Model')
        
        # Plot measured data if provided
        if imp_data is not None:
            ax.plot(np.real(imp_data), -np.imag(imp_data), 'bo', label='Measured')
        
        ax.set_xlabel('Real Z (Ω)')
        ax.set_ylabel('-Imaginary Z (Ω)')
        ax.grid(True)
        ax.axis('equal')  # Equal aspect ratio
        ax.legend()
        
        fig.tight_layout()
        return fig, ax
    
    def plot_bode(self, freq, imp_data=None, fig=None, ax=None):
        """Plot Bode plot (magnitude and phase vs frequency)."""
        if not self.fitted and imp_data is None:
            raise ValueError("Model must be fitted or measured data must be provided")
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Calculate model impedance
        if self.fitted:
            model_imp = self.impedance(freq)
            ax[0].loglog(freq, np.abs(model_imp), 'r-', label='Model')
            ax[1].semilogx(freq, np.angle(model_imp, deg=True), 'r-', label='Model')
        
        # Plot measured data if provided
        if imp_data is not None:
            ax[0].loglog(freq, np.abs(imp_data), 'bo', label='Measured')
            ax[1].semilogx(freq, np.angle(imp_data, deg=True), 'bo', label='Measured')
        
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('|Z| (Ω)')
        ax[0].grid(True, which="both", ls="--")
        ax[0].legend()
        
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Phase (deg)')
        ax[1].grid(True, which="both", ls="--")
        ax[1].legend()
        
        fig.tight_layout()
        return fig, ax


class ConstantPhaseElement:
    """
    Constant Phase Element (CPE) implementation.
    
    The CPE is an equivalent circuit component that models the behavior of
    a non-ideal capacitor. Its impedance is given by:
    Z_CPE = 1 / (Q * (j * omega)^alpha)
    
    where:
    - Q is a constant with units S*s^alpha (Siemens * second^alpha)
    - alpha is between 0 and 1
        - alpha = 1 corresponds to an ideal capacitor
        - alpha = 0 corresponds to a pure resistor
        - alpha = 0.5 corresponds to a Warburg element
    """
    
    def __init__(self, Q=1e-6, alpha=0.8):
        """
        Initialize a Constant Phase Element.
        
        Parameters
        ----------
        Q : float
            CPE coefficient (S*s^alpha)
        alpha : float
            CPE exponent (0 <= alpha <= 1)
        """
        self.Q = Q
        self.alpha = alpha
    
    def impedance(self, freq):
        """
        Calculate CPE impedance.
        
        Parameters
        ----------
        freq : array_like
            Frequencies in Hz
        
        Returns
        -------
        complex_array
            CPE impedance at each frequency
        """
        omega = 2 * np.pi * np.asarray(freq)
        return 1 / (self.Q * (1j * omega) ** self.alpha)


class RCCircuit(BaseElectricalModel):
    """
    RC Circuit model.
    
    This model represents a simple series/parallel combination of
    resistors and capacitors.
    """
    
    def __init__(self, n_elements=2, topology='parallel'):
        """
        Initialize RC Circuit model.
        
        Parameters
        ----------
        n_elements : int
            Number of RC elements
        topology : str
            'series' or 'parallel'
        """
        super().__init__()
        self.n_elements = n_elements
        self.topology = topology
        self.params = {
            'R': np.zeros(n_elements),  # Resistances
            'C': np.zeros(n_elements)   # Capacitances
        }
        
        if self.topology not in ['series', 'parallel']:
            raise ValueError("Topology must be 'series' or 'parallel'")
    
    def impedance(self, freq):
        """
        Calculate impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        Z = np.zeros(len(omega), dtype=complex)
        
        if self.topology == 'parallel':
            # Parallel RC elements in series
            for i in range(self.n_elements):
                R = self.params['R'][i]
                C = self.params['C'][i]
                Z_RC = R / (1 + 1j * omega * R * C)
                Z += Z_RC
        else:  # 'series'
            # Series RC elements
            for i in range(self.n_elements):
                R = self.params['R'][i]
                C = self.params['C'][i]
                Z_R = R
                Z_C = 1 / (1j * omega * C)
                Z += Z_R + Z_C
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        omega = 2 * np.pi * np.asarray(freq)
        
        # Define the error function to minimize
        def error_func(params):
            # Extract parameters
            R_values = params[:self.n_elements]
            C_values = params[self.n_elements:]
            
            # Ensure positive values
            R_values = np.abs(R_values)
            C_values = np.abs(C_values)
            
            # Calculate model impedance
            Z_model = np.zeros(len(omega), dtype=complex)
            
            if self.topology == 'parallel':
                for i in range(self.n_elements):
                    R = R_values[i]
                    C = C_values[i]
                    Z_RC = R / (1 + 1j * omega * R * C)
                    Z_model += Z_RC
            else:  # 'series'
                for i in range(self.n_elements):
                    R = R_values[i]
                    C = C_values[i]
                    Z_R = R
                    Z_C = 1 / (1j * omega * C)
                    Z_model += Z_R + Z_C
            
            # Calculate error
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + phase_error**2)
        
        # Initial parameter guess
        if self.topology == 'parallel':
            # For parallel topology, estimate from Nyquist plot features
            Z_real = np.real(imp_data)
            Z_imag = np.imag(imp_data)
            
            # R estimate from DC resistance
            R_total = np.max(Z_real)
            R_guess = np.ones(self.n_elements) * R_total / self.n_elements
            
            # C estimate from time constants
            # Identify frequencies with max imaginary component
            if self.n_elements == 1:
                max_imag_idx = np.argmax(-Z_imag)
                f_characteristic = freq[max_imag_idx]
                tau = 1 / (2 * np.pi * f_characteristic)
                C_guess = np.array([tau / R_guess[0]])
            else:
                # For multiple elements, distribute time constants
                f_min = np.min(freq)
                f_max = np.max(freq)
                f_characteristic = np.logspace(np.log10(f_min), np.log10(f_max), self.n_elements)
                tau = 1 / (2 * np.pi * f_characteristic)
                C_guess = tau / R_guess
        else:  # 'series'
            # For series topology, simple initial estimates
            Z_real = np.real(imp_data)
            Z_imag = np.imag(imp_data)
            
            # R estimate from high frequency real impedance
            R_guess = np.ones(self.n_elements) * np.min(Z_real) / self.n_elements
            
            # C estimate from imaginary part
            f_mid = np.sqrt(np.min(freq) * np.max(freq))
            mid_idx = np.argmin(np.abs(freq - f_mid))
            C_total = 1 / (2 * np.pi * f_mid * np.abs(Z_imag[mid_idx]))
            C_guess = np.ones(self.n_elements) * C_total / self.n_elements
        
        initial_params = np.concatenate((R_guess, C_guess))
        
        # Parameter bounds
        bounds = [(1e-6, None) for _ in range(self.n_elements)]  # R > 0
        bounds += [(1e-12, None) for _ in range(self.n_elements)]  # C > 0
        
        # Fit the model
        result = optimize.minimize(error_func, initial_params, bounds=bounds)
        
        if not result.success:
            print("Warning: Fitting did not converge.")
        
        # Extract optimized parameters
        R_values = np.abs(result.x[:self.n_elements])
        C_values = np.abs(result.x[self.n_elements:])
        
        # Store parameters
        self.params = {
            'R': R_values,
            'C': C_values
        }
        self.fitted = True
        
        return self.params


class RLCCircuit(BaseElectricalModel):
    """
    RLC Circuit model.
    
    This model extends the RC circuit model to include inductance (L),
    which is important for high-frequency applications.
    """
    
    def __init__(self, topology='series'):
        """
        Initialize RLC Circuit model.
        
        Parameters
        ----------
        topology : str
            'series' or 'parallel'
        """
        super().__init__()
        self.topology = topology
        self.params = {
            'R': 0,  # Resistance
            'L': 0,  # Inductance
            'C': 0   # Capacitance
        }
        
        if self.topology not in ['series', 'parallel']:
            raise ValueError("Topology must be 'series' or 'parallel'")
    
    def impedance(self, freq):
        """
        Calculate impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        
        R = self.params['R']
        L = self.params['L']
        C = self.params['C']
        
        if self.topology == 'series':
            # Series RLC
            Z_R = R
            Z_L = 1j * omega * L
            Z_C = 1 / (1j * omega * C)
            Z = Z_R + Z_L + Z_C
        else:  # 'parallel'
            # Parallel RLC
            Y_R = 1 / R if R != 0 else 0
            Y_L = 1 / (1j * omega * L) if L != 0 else 0
            Y_C = 1j * omega * C
            Y = Y_R + Y_L + Y_C
            Z = 1 / Y
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        omega = 2 * np.pi * np.asarray(freq)
        
        # Define the error function to minimize
        def error_func(params):
            # Extract parameters
            R, L, C = np.abs(params)  # Ensure positive values
            
            # Calculate model impedance
            if self.topology == 'series':
                # Series RLC
                Z_R = R
                Z_L = 1j * omega * L
                Z_C = 1 / (1j * omega * C)
                Z_model = Z_R + Z_L + Z_C
            else:  # 'parallel'
                # Parallel RLC
                Y_R = 1 / R if R != 0 else 0
                Y_L = 1 / (1j * omega * L) if L != 0 else 0
                Y_C = 1j * omega * C
                Y = Y_R + Y_L + Y_C
                Z_model = 1 / Y
            
            # Calculate error
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + phase_error**2)
        
        # Initial parameter guess
        Z_real = np.real(imp_data)
        Z_imag = np.imag(imp_data)
        
        if self.topology == 'series':
            # For series RLC
            R_guess = np.mean(Z_real)
            
            # Find resonance frequency (where impedance is minimum or phase crosses zero)
            Z_mag = np.abs(imp_data)
            res_idx = np.argmin(Z_mag)
            f_res = freq[res_idx]
            
            # Estimate L and C from resonance frequency
            # At resonance, omega * L = 1 / (omega * C)
            # so omega^2 * L * C = 1
            # and L * C = 1 / omega^2
            
            omega_res = 2 * np.pi * f_res
            L_C_product = 1 / (omega_res ** 2)
            
            # Estimate L and C individually from impedance magnitude at resonance
            L_guess = np.sqrt(L_C_product * R_guess)
            C_guess = L_C_product / L_guess
        else:  # 'parallel'
            # For parallel RLC
            R_guess = np.max(Z_real)
            
            # Find resonance frequency (where impedance is maximum)
            Z_mag = np.abs(imp_data)
            res_idx = np.argmax(Z_mag)
            f_res = freq[res_idx]
            
            # Estimate L and C from resonance frequency
            omega_res = 2 * np.pi * f_res
            L_C_product = 1 / (omega_res ** 2)
            
            # Estimate L and C individually
            C_guess = 1 / (omega_res * R_guess)
            L_guess = L_C_product / C_guess
        
        initial_params = [R_guess, L_guess, C_guess]
        
        # Parameter bounds
        bounds = [(1e-6, None), (1e-12, None), (1e-12, None)]  # R, L, C > 0
        
        # Fit the model
        result = optimize.minimize(error_func, initial_params, bounds=bounds)
        
        if not result.success:
            print("Warning: Fitting did not converge.")
        
        # Extract optimized parameters
        R, L, C = np.abs(result.x)
        
        # Store parameters
        self.params = {
            'R': R,
            'L': L,
            'C': C
        }
        self.fitted = True
        
        return self.params
    
    def resonance_frequency(self):
        """Calculate the resonance frequency of the RLC circuit."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        L = self.params['L']
        C = self.params['C']
        
        if L <= 0 or C <= 0:
            raise ValueError("L and C must be positive for resonance frequency calculation")
        
        f_res = 1 / (2 * np.pi * np.sqrt(L * C))
        return f_res
    
    def quality_factor(self):
        """Calculate the quality factor (Q) of the RLC circuit."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        R = self.params['R']
        L = self.params['L']
        C = self.params['C']
        
        if self.topology == 'series':
            # Q for series RLC
            Q = (1/R) * np.sqrt(L/C)
        else:  # 'parallel'
            # Q for parallel RLC
            Q = R * np.sqrt(C/L)
        
        return Q


class ColeColeModel(BaseElectricalModel):
    """
    Cole-Cole model for impedance analysis.
    
    This model is widely used for analyzing dielectric and biological systems.
    The impedance is given by:
    
    Z = R_inf + (R_0 - R_inf) / (1 + (j*omega*tau)^alpha)
    
    where:
    - R_0 is the low frequency (DC) resistance
    - R_inf is the high frequency resistance
    - tau is the characteristic time constant
    - alpha is the dispersion parameter (0 < alpha <= 1)
    """
    
    def __init__(self):
        """Initialize Cole-Cole model."""
        super().__init__()
        self.params = {
            'R_0': 0,     # Low frequency resistance
            'R_inf': 0,   # High frequency resistance
            'tau': 0,     # Time constant
            'alpha': 0.8  # Dispersion parameter
        }
    
    def impedance(self, freq):
        """
        Calculate impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        
        R_0 = self.params['R_0']
        R_inf = self.params['R_inf']
        tau = self.params['tau']
        alpha = self.params['alpha']
        
        Z = R_inf + (R_0 - R_inf) / (1 + (1j * omega * tau) ** alpha)
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        omega = 2 * np.pi * np.asarray(freq)
        
        # Define the error function to minimize
        def error_func(params):
            # Extract parameters
            R_0, R_inf, tau, alpha = params
            
            # Ensure physical constraints
            R_0 = abs(R_0)
            R_inf = abs(R_inf)
            tau = abs(tau)
            alpha = min(max(abs(alpha), 0.01), 1.0)  # 0 < alpha <= 1
            
            # Calculate model impedance
            Z_model = R_inf + (R_0 - R_inf) / (1 + (1j * omega * tau) ** alpha)
            
            # Calculate error
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + 0.1*phase_error**2)
        
        # Initial parameter guess
        Z_real = np.real(imp_data)
        
        # R_inf from high frequency data
        R_inf_guess = np.min(Z_real)
        
        # R_0 from low frequency data
        R_0_guess = np.max(Z_real)
        
        # tau from frequency at which imaginary part is maximum
        Z_imag = -np.imag(imp_data)  # Use negative imaginary for easier peak finding
        max_imag_idx = np.argmax(Z_imag)
        f_characteristic = freq[max_imag_idx]
        tau_guess = 1 / (2 * np.pi * f_characteristic)
        
        # alpha - typical value for many systems
        alpha_guess = 0.8
        
        initial_params = [R_0_guess, R_inf_guess, tau_guess, alpha_guess]
        
        # Parameter bounds
        bounds = [
            (1e-6, None),     # R_0 > 0
            (1e-6, None),     # R_inf > 0
            (1e-12, None),    # tau > 0
            (0.01, 1.0)       # 0 < alpha <= 1
        ]
        
        # Fit the model
        result = optimize.minimize(error_func, initial_params, bounds=bounds)
        
        if not result.success:
            print("Warning: Fitting did not converge.")
        
        # Extract optimized parameters
        R_0, R_inf, tau, alpha = result.x
        
        # Ensure physical constraints
        R_0 = abs(R_0)
        R_inf = abs(R_inf)
        tau = abs(tau)
        alpha = min(max(abs(alpha), 0.01), 1.0)  # 0 < alpha <= 1
        
        # Store parameters
        self.params = {
            'R_0': R_0,
            'R_inf': R_inf,
            'tau': tau,
            'alpha': alpha
        }
        self.fitted = True
        
        return self.params


class RandlesCircuit(BaseElectricalModel):
    """
    Randles circuit model.
    
    This model is commonly used for electrochemical systems and consists of:
    - R_s: Solution resistance
    - R_ct: Charge transfer resistance
    - C_dl: Double layer capacitance (or CPE)
    - Z_w: Warburg impedance (for diffusion)
    """
    
    def __init__(self, use_cpe=True, include_warburg=True):
        """
        Initialize Randles circuit model.
        
        Parameters
        ----------
        use_cpe : bool
            Use CPE instead of ideal capacitor
        include_warburg : bool
            Include Warburg impedance for diffusion
        """
        super().__init__()
        self.use_cpe = use_cpe
        self.include_warburg = include_warburg
        
        if use_cpe:
            self.params = {
                'R_s': 0,     # Solution resistance
                'R_ct': 0,    # Charge transfer resistance
                'Q': 0,       # CPE coefficient
                'alpha': 0.8  # CPE exponent
            }
        else:
            self.params = {
                'R_s': 0,     # Solution resistance
                'R_ct': 0,    # Charge transfer resistance
                'C_dl': 0     # Double layer capacitance
            }
        
        if include_warburg:
            self.params['sigma'] = 0  # Warburg coefficient
    
    def impedance(self, freq):
        """
        Calculate impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        
        # Solution resistance
        Z_rs = self.params['R_s']
        
        # Double layer capacitance or CPE
        if self.use_cpe:
            Q = self.params['Q']
            alpha = self.params['alpha']
            Z_dl = 1 / (Q * (1j * omega) ** alpha)
        else:
            C_dl = self.params['C_dl']
            Z_dl = 1 / (1j * omega * C_dl)
        
        # Warburg impedance for diffusion
        if self.include_warburg:
            sigma = self.params['sigma']
            Z_w = sigma * (1 - 1j) / np.sqrt(omega)
        else:
            Z_w = 0
        
        # Charge transfer resistance
        Z_ct = self.params['R_ct']
        
        # Combine impedances according to Randles circuit
        Z_faradaic = Z_ct + Z_w
        Z_parallel = 1 / (1/Z_faradaic + 1/Z_dl)
        Z = Z_rs + Z_parallel
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        omega = 2 * np.pi * np.asarray(freq)
        
        # Define the error function to minimize
        def error_func(params):
            # Extract parameters based on model configuration
            if self.use_cpe and self.include_warburg:
                R_s, R_ct, Q, alpha, sigma = params
                self.params = {
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'Q': abs(Q),
                    'alpha': min(max(abs(alpha), 0.01), 1.0),
                    'sigma': abs(sigma)
                }
            elif self.use_cpe and not self.include_warburg:
                R_s, R_ct, Q, alpha = params
                self.params = {
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'Q': abs(Q),
                    'alpha': min(max(abs(alpha), 0.01), 1.0)
                }
            elif not self.use_cpe and self.include_warburg:
                R_s, R_ct, C_dl, sigma = params
                self.params = {
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'C_dl': abs(C_dl),
                    'sigma': abs(sigma)
                }
            else:  # not self.use_cpe and not self.include_warburg
                R_s, R_ct, C_dl = params
                self.params = {
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'C_dl': abs(C_dl)
                }
            
            # Calculate model impedance
            Z_model = self.impedance(freq)
            
            # Calculate error
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + 0.1*phase_error**2)
        
        # Initial parameter guess
        Z_real = np.real(imp_data)
        Z_imag = -np.imag(imp_data)  # Negative imaginary for easier analysis
        
        # R_s from high frequency intercept
        R_s_guess = np.min(Z_real)
        
        # R_ct from diameter of semicircle
        # (Difference between low and high frequency intercepts)
        R_ct_guess = np.max(Z_real) - R_s_guess
        
        # Time constant from frequency at max -Im(Z)
        max_imag_idx = np.argmax(Z_imag)
        f_characteristic = freq[max_imag_idx]
        tau_guess = 1 / (2 * np.pi * f_characteristic)
        
        if self.use_cpe:
            # CPE parameters
            alpha_guess = 0.8
            Q_guess = 1 / (R_ct_guess * (2 * np.pi * f_characteristic) ** alpha_guess)
            cap_param = [Q_guess, alpha_guess]
        else:
            # Ideal capacitance
            C_dl_guess = tau_guess / R_ct_guess
            cap_param = [C_dl_guess]
        
        if self.include_warburg:
            # Warburg coefficient from low frequency behavior
            # At low frequencies, Warburg dominates and Im(Z) ~ Re(Z) ~ sigma/sqrt(omega)
            low_freq_idx = np.argmin(freq)
            sigma_guess = abs(Z_imag[low_freq_idx]) * np.sqrt(omega[low_freq_idx])
            warb_param = [sigma_guess]
        else:
            warb_param = []
        
        # Combine parameters
        initial_params = [R_s_guess, R_ct_guess] + cap_param + warb_param
        
        # Parameter bounds
        bounds = [(1e-6, None), (1e-6, None)]  # R_s, R_ct > 0
        
        if self.use_cpe:
            bounds += [(1e-12, None), (0.01, 1.0)]  # Q > 0, 0 < alpha <= 1
        else:
            bounds += [(1e-12, None)]  # C_dl > 0
        
        if self.include_warburg:
            bounds += [(1e-6, None)]  # sigma > 0
        
        # Fit the model
        result = optimize.minimize(error_func, initial_params, bounds=bounds)
        
        if not result.success:
            print("Warning: Fitting did not converge.")
        
        # Parameters are already updated in error_func
        self.fitted = True
        
        return self.params


class BatteryModel(BaseElectricalModel):
    """
    Advanced battery model combining electrical and thermal characteristics.
    
    This model is specifically designed for battery analysis as described in the
    patent, incorporating both electrical impedance and thermal effects.
    """
    
    def __init__(self, use_cpe=True, include_warburg=True, include_thermal=True):
        """
        Initialize battery model.
        
        Parameters
        ----------
        use_cpe : bool
            Use CPE instead of ideal capacitor
        include_warburg : bool
            Include Warburg impedance for diffusion
        include_thermal : bool
            Include thermal coupling effects
        """
        super().__init__()
        self.use_cpe = use_cpe
        self.include_warburg = include_warburg
        self.include_thermal = include_thermal
        
        # Basic Randles circuit parameters
        self.params = {
            'R_s': 0,     # Series resistance
            'R_ct': 0,    # Charge transfer resistance
            'R_sei': 0,   # SEI layer resistance
        }
        
        if use_cpe:
            self.params.update({
                'Q_dl': 0,        # Double layer CPE coefficient
                'alpha_dl': 0.8,  # Double layer CPE exponent
                'Q_sei': 0,       # SEI layer CPE coefficient
                'alpha_sei': 0.8  # SEI layer CPE exponent
            })
        else:
            self.params.update({
                'C_dl': 0,  # Double layer capacitance
                'C_sei': 0  # SEI layer capacitance
            })
        
        if include_warburg:
            self.params['sigma'] = 0  # Warburg coefficient
        
        if include_thermal:
            self.params.update({
                'R_th': 0,     # Thermal resistance
                'C_th': 0,     # Thermal capacitance
                'alpha_th': 0  # Thermal coupling coefficient
            })
    
    def impedance(self, freq, temp=25):
        """
        Calculate impedance at given frequencies and temperature.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        temp : float, optional
            Temperature in °C
        
        Returns
        -------
        complex_array
            Complex impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        
        # Temperature adjustment factor (Arrhenius-like)
        T_ref = 25  # Reference temperature in °C
        T_k = temp + 273.15  # Convert to Kelvin
        T_ref_k = T_ref + 273.15
        
        # Activation energy (typical values for Li-ion processes)
        E_a_r = 0.4  # eV for resistive processes
        E_a_c = 0.2  # eV for capacitive processes
        
        # Boltzmann constant
        k_B = 8.617e-5  # eV/K
        
        # Temperature adjustment factors
        f_r = np.exp((E_a_r / k_B) * (1/T_ref_k - 1/T_k))  # For resistances
        f_c = np.exp((E_a_c / k_B) * (1/T_ref_k - 1/T_k))  # For capacitances
        
        # Adjust parameters for temperature
        R_s = self.params['R_s'] * f_r
        R_ct = self.params['R_ct'] * f_r
        R_sei = self.params['R_sei'] * f_r
        
        # SEI layer impedance
        if self.use_cpe:
            Q_sei = self.params['Q_sei'] * f_c
            alpha_sei = self.params['alpha_sei']
            Z_sei = R_sei / (1 + R_sei * Q_sei * (1j * omega) ** alpha_sei)
        else:
            C_sei = self.params['C_sei'] * f_c
            Z_sei = R_sei / (1 + 1j * omega * R_sei * C_sei)
        
        # Charge transfer and double layer impedance
        if self.use_cpe:
            Q_dl = self.params['Q_dl'] * f_c
            alpha_dl = self.params['alpha_dl']
            Z_dl = 1 / (Q_dl * (1j * omega) ** alpha_dl)
        else:
            C_dl = self.params['C_dl'] * f_c
            Z_dl = 1 / (1j * omega * C_dl)
        
        # Warburg impedance for diffusion
        if self.include_warburg:
            sigma = self.params['sigma'] * f_r
            Z_w = sigma * (1 - 1j) / np.sqrt(omega)
        else:
            Z_w = 0
        
        # Combine impedances
        Z_faradaic = R_ct + Z_w
        Z_parallel = 1 / (1/Z_faradaic + 1/Z_dl)
        Z = R_s + Z_sei + Z_parallel
        
        # Add thermal effects if included
        if self.include_thermal:
            R_th = self.params['R_th']
            C_th = self.params['C_th']
            alpha_th = self.params['alpha_th']
            
            # Thermal impedance (simple RC)
            Z_th = R_th / (1 + 1j * omega * R_th * C_th)
            
            # Couple electrical and thermal impedances
            Z = Z + alpha_th * Z_th * np.abs(Z) ** 2
        
        return Z
    
    def fit(self, freq, imp_data, temp=25):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex impedance data
        temp : float, optional
            Temperature in °C
        
        Returns
        -------
        dict
            Fitted parameters
        """
        # This fitting is complex due to many parameters
        # Here we implement a two-step approach:
        # 1. Fit electrical parameters first
        # 2. Then fit thermal parameters if needed
        
        # Step 1: Fit electrical parameters
        
        # Define the electrical error function
        def electrical_error_func(params):
            if self.use_cpe and self.include_warburg:
                R_s, R_ct, R_sei, Q_dl, alpha_dl, Q_sei, alpha_sei, sigma = params
                self.params.update({
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'R_sei': abs(R_sei),
                    'Q_dl': abs(Q_dl),
                    'alpha_dl': min(max(abs(alpha_dl), 0.01), 1.0),
                    'Q_sei': abs(Q_sei),
                    'alpha_sei': min(max(abs(alpha_sei), 0.01), 1.0),
                    'sigma': abs(sigma)
                })
            elif self.use_cpe and not self.include_warburg:
                R_s, R_ct, R_sei, Q_dl, alpha_dl, Q_sei, alpha_sei = params
                self.params.update({
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'R_sei': abs(R_sei),
                    'Q_dl': abs(Q_dl),
                    'alpha_dl': min(max(abs(alpha_dl), 0.01), 1.0),
                    'Q_sei': abs(Q_sei),
                    'alpha_sei': min(max(abs(alpha_sei), 0.01), 1.0)
                })
            elif not self.use_cpe and self.include_warburg:
                R_s, R_ct, R_sei, C_dl, C_sei, sigma = params
                self.params.update({
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'R_sei': abs(R_sei),
                    'C_dl': abs(C_dl),
                    'C_sei': abs(C_sei),
                    'sigma': abs(sigma)
                })
            else:  # not self.use_cpe and not self.include_warburg
                R_s, R_ct, R_sei, C_dl, C_sei = params
                self.params.update({
                    'R_s': abs(R_s),
                    'R_ct': abs(R_ct),
                    'R_sei': abs(R_sei),
                    'C_dl': abs(C_dl),
                    'C_sei': abs(C_sei)
                })
            
            # Temporarily disable thermal effects
            include_thermal_orig = self.include_thermal
            self.include_thermal = False
            
            # Calculate model impedance
            Z_model = self.impedance(freq, temp)
            
            # Restore thermal setting
            self.include_thermal = include_thermal_orig
            
            # Calculate error
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + 0.1*phase_error**2)
        
        # Initial parameter guess for electrical components
        Z_real = np.real(imp_data)
        Z_imag = -np.imag(imp_data)
        
        # High-frequency intercept for R_s
        R_s_guess = np.min(Z_real)
        
        # SEI and charge transfer resistances (distribute remaining resistance)
        R_total = np.max(Z_real) - R_s_guess
        R_sei_guess = R_total * 0.3  # 30% of total
        R_ct_guess = R_total * 0.7   # 70% of total
        
        if self.use_cpe:
            # CPE parameters
            alpha_dl_guess = 0.8
            alpha_sei_guess = 0.7
            
            # Find characteristic frequencies
            # For SEI - typically higher frequency
            f_sei_guess = 1000  # Hz
            Q_sei_guess = 1 / (R_sei_guess * (2 * np.pi * f_sei_guess) ** alpha_sei_guess)
            
            # For double layer - typically lower frequency
            f_dl_guess = 10  # Hz
            Q_dl_guess = 1 / (R_ct_guess * (2 * np.pi * f_dl_guess) ** alpha_dl_guess)
            
            cap_param = [Q_dl_guess, alpha_dl_guess, Q_sei_guess, alpha_sei_guess]
        else:
            # Ideal capacitances
            C_dl_guess = 1e-3  # F
            C_sei_guess = 1e-6  # F
            cap_param = [C_dl_guess, C_sei_guess]
        
        if self.include_warburg:
            # Warburg coefficient from low frequency data
            low_freq_idx = np.argmin(freq)
            omega_low = 2 * np.pi * freq[low_freq_idx]
            sigma_guess = abs(Z_imag[low_freq_idx]) * np.sqrt(omega_low)
            warb_param = [sigma_guess]
        else:
            warb_param = []
        
        # Combine parameters
        initial_params = [R_s_guess, R_ct_guess, R_sei_guess] + cap_param + warb_param
        
        # Parameter bounds
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]  # R_s, R_ct, R_sei > 0
        
        if self.use_cpe:
            bounds += [(1e-12, None), (0.01, 1.0), (1e-12, None), (0.01, 1.0)]  # Q_dl, alpha_dl, Q_sei, alpha_sei
        else:
            bounds += [(1e-12, None), (1e-12, None)]  # C_dl, C_sei > 0
        
        if self.include_warburg:
            bounds += [(1e-6, None)]  # sigma > 0
        
        # Fit electrical parameters
        result = optimize.minimize(electrical_error_func, initial_params, bounds=bounds)
        
        if not result.success:
            print("Warning: Electrical parameter fitting did not converge.")
        
        # Step 2: Fit thermal parameters if needed
        if self.include_thermal:
            # Define the thermal error function
            def thermal_error_func(params):
                R_th, C_th, alpha_th = params
                self.params.update({
                    'R_th': abs(R_th),
                    'C_th': abs(C_th),
                    'alpha_th': abs(alpha_th)
                })
                
                # Calculate model impedance with thermal effects
                Z_model = self.impedance(freq, temp)
                
                # Calculate error
                mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
                phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
                
                # Combined error
                return np.sum(mag_error**2 + 0.1*phase_error**2)
            
            # Initial thermal parameter guess
            R_th_guess = 10.0  # K/W
            C_th_guess = 100.0  # J/K
            alpha_th_guess = 0.01  # Coupling coefficient
            
            thermal_params = [R_th_guess, C_th_guess, alpha_th_guess]
            
            # Parameter bounds
            thermal_bounds = [
                (0.1, 100.0),    # R_th: 0.1-100 K/W
                (10.0, 1000.0),  # C_th: 10-1000 J/K
                (0.0, 0.1)       # alpha_th: 0-0.1
            ]
            
            # Fit thermal parameters
            result_thermal = optimize.minimize(thermal_error_func, thermal_params, bounds=thermal_bounds)
            
            if not result_thermal.success:
                print("Warning: Thermal parameter fitting did not converge.")
        
        self.fitted = True
        
        return self.params"""