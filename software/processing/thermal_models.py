"""
Thermal models module for the Integrated Electrical-Thermal Impedance Analyzer

This module implements various thermal equivalent circuit models for analyzing
thermal impedance data, including Foster and Cauer networks, and distributed
parameter models.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class BaseThermalModel:
    """Base class for thermal models."""
    
    def __init__(self):
        """Initialize base thermal model."""
        self.params = {}
        self.fitted = False
    
    def impedance(self, freq):
        """Calculate thermal impedance at given frequencies."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def fit(self, freq, imp_data):
        """Fit model parameters to measured impedance data."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def step_response(self, time):
        """Calculate thermal step response."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot_impedance(self, freq, measured_imp=None, fig=None, ax=None):
        """Plot the thermal impedance spectrum."""
        if not self.fitted and measured_imp is None:
            raise ValueError("Model must be fitted or measured data must be provided")
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Calculate model impedance
        if self.fitted:
            model_imp = self.impedance(freq)
            ax[0].loglog(freq, np.abs(model_imp), 'r-', label='Model')
            ax[1].semilogx(freq, np.angle(model_imp, deg=True), 'r-', label='Model')
        
        # Plot measured data if provided
        if measured_imp is not None:
            ax[0].loglog(freq, np.abs(measured_imp), 'bo', label='Measured')
            ax[1].semilogx(freq, np.angle(measured_imp, deg=True), 'bo', label='Measured')
        
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('|Z| (K/W)')
        ax[0].grid(True, which="both", ls="--")
        ax[0].legend()
        
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Phase (deg)')
        ax[1].grid(True, which="both", ls="--")
        ax[1].legend()
        
        fig.tight_layout()
        return fig, ax
    
    def plot_step_response(self, time, fig=None, ax=None):
        """Plot the thermal step response."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        response = self.step_response(time)
        ax.plot(time, response, 'b-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, which="both", ls="--")
        
        fig.tight_layout()
        return fig, ax


class FosterNetwork(BaseThermalModel):
    """
    Foster network thermal model.
    
    This model represents the thermal system as a parallel connection of 
    R-C pairs (thermal resistances and capacitances).
    """
    
    def __init__(self, n_elements=3):
        """
        Initialize Foster network model.
        
        Parameters
        ----------
        n_elements : int
            Number of R-C pairs in the network
        """
        super().__init__()
        self.n_elements = n_elements
        self.params = {
            'R': np.zeros(n_elements),  # Thermal resistances
            'C': np.zeros(n_elements)   # Thermal capacitances
        }
    
    def impedance(self, freq):
        """
        Calculate thermal impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex thermal impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        Z = np.zeros(len(omega), dtype=complex)
        
        for i in range(self.n_elements):
            R = self.params['R'][i]
            C = self.params['C'][i]
            tau = R * C
            
            # Each R-C pair contributes to the total impedance
            Z += R / (1 + 1j * omega * tau)
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex thermal impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        omega = 2 * np.pi * np.asarray(freq)
        
        # Define the error function to minimize
        def error_func(params):
            R_values = params[:self.n_elements]
            tau_values = params[self.n_elements:]
            
            # Calculate model impedance
            Z_model = np.zeros(len(omega), dtype=complex)
            for i in range(self.n_elements):
                Z_model += R_values[i] / (1 + 1j * omega * tau_values[i])
            
            # Calculate error (magnitude and phase components)
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + phase_error**2)
        
        # Initial parameter guess
        # For a good initial guess, we estimate time constants spanning the frequency range
        tau_min = 1 / (2 * np.pi * np.max(freq))
        tau_max = 1 / (2 * np.pi * np.min(freq))
        tau_guess = np.logspace(np.log10(tau_min), np.log10(tau_max), self.n_elements)
        
        # Equal distribution of the total resistance
        R_total_guess = np.real(imp_data[np.argmin(freq)])
        R_guess = np.ones(self.n_elements) * R_total_guess / self.n_elements
        
        initial_params = np.concatenate((R_guess, tau_guess))
        
        # Parameter bounds
        bounds = [(1e-6, None) for _ in range(2 * self.n_elements)]  # All params > 0
        
        # Fit the model
        result = optimize.minimize(error_func, initial_params, bounds=bounds)
        
        if not result.success:
            print("Warning: Fitting did not converge.")
        
        # Extract optimized parameters
        R_values = result.x[:self.n_elements]
        tau_values = result.x[self.n_elements:]
        C_values = tau_values / R_values
        
        # Store parameters
        self.params = {
            'R': R_values,
            'C': C_values,
            'tau': tau_values
        }
        self.fitted = True
        
        return self.params
    
    def step_response(self, time):
        """
        Calculate thermal step response.
        
        Parameters
        ----------
        time : array_like
            Time array in seconds
        
        Returns
        -------
        array_like
            Temperature response to a step power input
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        response = np.zeros_like(time)
        
        for i in range(self.n_elements):
            R = self.params['R'][i]
            tau = self.params['tau'][i]
            
            # Each R-C pair contributes to the step response
            response += R * (1 - np.exp(-time / tau))
        
        return response


class CauerNetwork(BaseThermalModel):
    """
    Cauer network thermal model.
    
    This model represents the thermal system as a ladder network of
    thermal resistances and capacitances, which better corresponds to
    the physical structure of the system.
    """
    
    def __init__(self, n_elements=3):
        """
        Initialize Cauer network model.
        
        Parameters
        ----------
        n_elements : int
            Number of R-C sections in the network
        """
        super().__init__()
        self.n_elements = n_elements
        self.params = {
            'R': np.zeros(n_elements),  # Thermal resistances
            'C': np.zeros(n_elements)   # Thermal capacitances
        }
    
    def impedance(self, freq):
        """
        Calculate thermal impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex thermal impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        Z = np.zeros(len(omega), dtype=complex)
        
        # Calculate impedance using the continued fraction expansion
        # Start from the deepest element and work backwards
        for i in range(self.n_elements-1, -1, -1):
            R = self.params['R'][i]
            C = self.params['C'][i]
            
            if i == self.n_elements-1:
                # Last element
                Z = R + 1/(1j * omega * C)
            else:
                # Chain with previous elements
                Z = R + 1 / (1j * omega * C + 1/Z)
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex thermal impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        omega = 2 * np.pi * np.asarray(freq)
        
        # Define the error function to minimize
        def error_func(params):
            # Extract R and C values
            params = np.abs(params)  # Ensure positive values
            R_values = params[:self.n_elements]
            C_values = params[self.n_elements:]
            
            # Calculate model impedance using continued fraction
            Z_model = np.zeros(len(omega), dtype=complex)
            
            for i in range(self.n_elements-1, -1, -1):
                R = R_values[i]
                C = C_values[i]
                
                if i == self.n_elements-1:
                    # Last element
                    Z_model = R + 1/(1j * omega * C)
                else:
                    # Chain with previous elements
                    Z_model = R + 1 / (1j * omega * C + 1/Z_model)
            
            # Calculate error (magnitude and phase components)
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + phase_error**2)
        
        # Initial parameter guess
        # For Cauer network, we need more care in choosing initial values
        R_total_guess = np.real(imp_data[np.argmin(freq)])
        R_guess = np.ones(self.n_elements) * R_total_guess / self.n_elements
        
        # Estimate capacitances from frequency range
        f_max = np.max(freq)
        f_min = np.min(freq)
        f_range = np.logspace(np.log10(f_min), np.log10(f_max), self.n_elements)
        C_guess = 1 / (2 * np.pi * f_range * R_guess)
        
        initial_params = np.concatenate((R_guess, C_guess))
        
        # Fit the model
        result = optimize.minimize(error_func, initial_params, method='L-BFGS-B')
        
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
    
    def step_response(self, time):
        """
        Calculate thermal step response.
        
        For Cauer networks, the step response is calculated using numerical
        integration of the state-space model.
        
        Parameters
        ----------
        time : array_like
            Time array in seconds
        
        Returns
        -------
        array_like
            Temperature response to a step power input
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Set up state-space model
        n = self.n_elements
        A = np.zeros((n, n))
        B = np.zeros(n)
        C = np.zeros(n)
        
        for i in range(n):
            R = self.params['R'][i]
            C = self.params['C'][i]
            
            if i == 0:
                A[i, i] = -1 / (R * C)
                if n > 1:
                    A[i, i+1] = 1 / (R * C)
                B[i] = 1 / C
            elif i == n-1:
                A[i, i] = -1 / (R * C)
                A[i, i-1] = 1 / (R * C)
            else:
                A[i, i] = -2 / (R * C)
                A[i, i-1] = 1 / (R * C)
                A[i, i+1] = 1 / (R * C)
        
        # Output matrix - temperature at the input node
        C[0] = 1
        
        # Solve state-space model for step response
        from scipy import signal
        sys = signal.StateSpace(A, B, C, 0)
        _, response = signal.step(sys, T=time)
        
        return response


class DistributedParameterModel(BaseThermalModel):
    """
    Distributed parameter thermal model.
    
    This model represents the thermal system using a distributed parameter
    approach, which can better model complex thermal systems with non-uniform
    properties.
    """
    
    def __init__(self, n_layers=3):
        """
        Initialize distributed parameter model.
        
        Parameters
        ----------
        n_layers : int
            Number of layers in the distributed model
        """
        super().__init__()
        self.n_layers = n_layers
        self.params = {
            'k': np.zeros(n_layers),    # Thermal conductivity
            'rho': np.zeros(n_layers),  # Density
            'cp': np.zeros(n_layers),   # Specific heat
            'd': np.zeros(n_layers)     # Layer thickness
        }
    
    def impedance(self, freq):
        """
        Calculate thermal impedance at given frequencies.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex thermal impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        omega = 2 * np.pi * np.asarray(freq)
        Z = np.zeros(len(omega), dtype=complex)
        
        # Calculate impedance using the transmission line theory
        for i in range(len(omega)):
            # Start with the last layer (farthest from the heat source)
            Z_i = 0
            
            for j in range(self.n_layers-1, -1, -1):
                k = self.params['k'][j]    # Thermal conductivity
                rho = self.params['rho'][j] # Density
                cp = self.params['cp'][j]   # Specific heat
                d = self.params['d'][j]     # Layer thickness
                
                # Thermal diffusivity
                alpha = k / (rho * cp)
                
                # Complex propagation constant
                gamma = np.sqrt(1j * omega[i] / alpha)
                
                # Characteristic impedance
                Z0 = 1 / (k * gamma)
                
                # Input impedance (transmission line equation)
                Z_i = Z0 * (Z_i + Z0 * np.tanh(gamma * d)) / (Z0 + Z_i * np.tanh(gamma * d))
            
            Z[i] = Z_i
        
        return Z
    
    def fit(self, freq, imp_data):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex thermal impedance data
        
        Returns
        -------
        dict
            Fitted parameters
        """
        # For distributed parameter models, fitting is complex
        # Here we implement a simplified approach using global optimization
        
        # Define the error function to minimize
        def error_func(params):
            # Extract and reshape parameters
            params = np.abs(params)  # Ensure positive values
            k_values = params[:self.n_layers]
            rho_cp_values = params[self.n_layers:2*self.n_layers]  # Combined rho*cp
            d_values = params[2*self.n_layers:]
            
            # Fixed parameters
            rho_values = np.ones(self.n_layers) * 2700  # Aluminum density
            cp_values = rho_cp_values / rho_values
            
            # Set parameters temporarily
            temp_params = {
                'k': k_values,
                'rho': rho_values,
                'cp': cp_values,
                'd': d_values
            }
            
            old_params = self.params
            self.params = temp_params
            
            # Calculate model impedance
            Z_model = self.impedance(freq)
            
            # Restore original parameters
            self.params = old_params
            
            # Calculate error (magnitude and phase components)
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + 0.1*phase_error**2)
        
        # Initial parameter guess
        # Typical values for common materials
        k_guess = np.linspace(200, 100, self.n_layers)  # W/(m*K), conductivity decreasing with distance
        rho_guess = np.ones(self.n_layers) * 2700       # kg/m³, typical for aluminum
        cp_guess = np.ones(self.n_layers) * 900         # J/(kg*K), typical for aluminum
        
        # Total thickness estimation from DC thermal resistance
        R_th = np.real(imp_data[np.argmin(freq)])
        A = 1e-4  # Assuming heat transfer area of 1 cm²
        total_thickness = R_th * np.mean(k_guess) * A
        d_guess = np.ones(self.n_layers) * total_thickness / self.n_layers
        
        # Combine rho and cp for optimization
        rho_cp_guess = rho_guess * cp_guess
        
        initial_params = np.concatenate((k_guess, rho_cp_guess, d_guess))
        
        # Use global optimization
        bounds = [(10, 500) for _ in range(self.n_layers)]         # k: 10-500 W/(m*K)
        bounds += [(1e6, 5e6) for _ in range(self.n_layers)]       # rho*cp: 1e6-5e6 J/(m³*K)
        bounds += [(1e-6, 1e-2) for _ in range(self.n_layers)]     # d: 1µm-1cm
        
        result = optimize.dual_annealing(error_func, bounds, maxiter=1000)
        
        if not result.success:
            print("Warning: Fitting did not converge.")
        
        # Extract optimized parameters
        params = np.abs(result.x)
        k_values = params[:self.n_layers]
        rho_cp_values = params[self.n_layers:2*self.n_layers]
        d_values = params[2*self.n_layers:]
        
        # Fixed density
        rho_values = np.ones(self.n_layers) * 2700
        cp_values = rho_cp_values / rho_values
        
        # Store parameters
        self.params = {
            'k': k_values,
            'rho': rho_values,
            'cp': cp_values,
            'd': d_values
        }
        
        self.fitted = True
        
        return self.params
    
    def step_response(self, time):
        """
        Calculate thermal step response.
        
        For distributed parameter models, this is calculated using
        numerical solution of the heat equation.
        
        Parameters
        ----------
        time : array_like
            Time array in seconds
        
        Returns
        -------
        array_like
            Temperature response to a step power input
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # For distributed parameter models, step response calculation
        # is complex and would require solving PDEs
        # Here we use an approximation method using the frequency domain
        
        # Create a wide frequency range
        f_min = 1 / (10 * np.max(time))
        f_max = 1 / (0.1 * np.min(time))
        freq = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
        
        # Calculate impedance in frequency domain
        Z = self.impedance(freq)
        
        # Convert to time domain using numerical inverse Laplace transform
        # This is an approximation using the method of Gaver-Stehfest
        response = np.zeros_like(time)
        
        for i, t in enumerate(time):
            s = 1j * 2 * np.pi * freq
            integrand = Z * np.exp(s * t) / (1j * 2 * np.pi)
            response[i] = np.trapz(integrand.real, freq)
        
        return response


class PCMThermalModel(BaseThermalModel):
    """
    Phase Change Material (PCM) thermal model.
    
    This model extends the distributed parameter model to include
    phase change effects, which are crucial for accurate modeling of
    PCM-based thermal management systems as described in the patent.
    """
    
    def __init__(self, n_layers=3):
        """
        Initialize PCM thermal model.
        
        Parameters
        ----------
        n_layers : int
            Number of layers in the distributed model
        """
        super().__init__()
        self.n_layers = n_layers
        self.params = {
            'k': np.zeros(n_layers),     # Thermal conductivity
            'rho': np.zeros(n_layers),   # Density
            'cp': np.zeros(n_layers),    # Specific heat
            'd': np.zeros(n_layers),     # Layer thickness
            'T_phase': 0,                # Phase change temperature
            'L_phase': 0,                # Latent heat of phase change
            'phase_range': 0             # Temperature range of phase change
        }
    
    def impedance(self, freq):
        """
        Calculate thermal impedance at given frequencies.
        
        For PCM models, the impedance calculation is more complex due to
        the nonlinear behavior of the phase change.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        
        Returns
        -------
        complex_array
            Complex thermal impedance at each frequency
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # PCM models require special handling
        # Here we implement a linearized model around the operating point
        
        omega = 2 * np.pi * np.asarray(freq)
        Z = np.zeros(len(omega), dtype=complex)
        
        # Calculate basic distributed parameter model impedance
        base_model = DistributedParameterModel(self.n_layers)
        base_model.params = {
            'k': self.params['k'],
            'rho': self.params['rho'],
            'cp': self.params['cp'],
            'd': self.params['d']
        }
        base_model.fitted = True
        
        Z_base = base_model.impedance(freq)
        
        # Add PCM effect - simplified model
        # Phase change creates an effective heat capacity increase
        T_phase = self.params['T_phase']
        L_phase = self.params['L_phase']
        phase_range = self.params['phase_range']
        
        # Effective heat capacity modification factor
        # This is a simplification - in reality, the effect depends on
        # the operating point temperature relative to phase change temperature
        pcm_factor = np.ones(len(omega), dtype=complex)
        
        # PCM effect is strongest at low frequencies
        for i, f in enumerate(freq):
            # Create frequency-dependent modification factor
            # PCM effect diminishes at high frequencies
            pcm_effect = L_phase / (phase_range * self.params['cp'][0])
            factor = 1 + pcm_effect / (1 + 1j * omega[i] * (phase_range**2) / (self.params['k'][0] / (self.params['rho'][0] * self.params['cp'][0])))
            pcm_factor[i] = factor
        
        # Apply PCM effect to base impedance
        Z = Z_base * pcm_factor
        
        return Z
    
    def fit(self, freq, imp_data, T_phase=45, L_phase=200000, phase_range=2):
        """
        Fit model parameters to measured impedance data.
        
        Parameters
        ----------
        freq : array_like
            Frequency array in Hz
        imp_data : array_like
            Complex thermal impedance data
        T_phase : float
            Phase change temperature (°C)
        L_phase : float
            Latent heat of phase change (J/kg)
        phase_range : float
            Temperature range over which phase change occurs (°C)
        
        Returns
        -------
        dict
            Fitted parameters
        """
        # First, fit the basic distributed parameter model
        base_model = DistributedParameterModel(self.n_layers)
        base_params = base_model.fit(freq, imp_data)
        
        # Store base parameters
        self.params.update(base_params)
        
        # Add PCM-specific parameters
        self.params['T_phase'] = T_phase
        self.params['L_phase'] = L_phase
        self.params['phase_range'] = phase_range
        
        # Additional refinement for PCM-specific parameters
        # This is a simplified approach - in a real implementation,
        # you would need a more sophisticated optimization
        
        def error_func(pcm_params):
            L_phase, phase_range = pcm_params
            
            # Update PCM parameters
            self.params['L_phase'] = L_phase
            self.params['phase_range'] = phase_range
            
            # Calculate model impedance
            Z_model = self.impedance(freq)
            
            # Calculate error
            mag_error = np.abs(np.abs(Z_model) - np.abs(imp_data))
            phase_error = np.abs(np.angle(Z_model) - np.angle(imp_data))
            
            # Combined error
            return np.sum(mag_error**2 + 0.1*phase_error**2)
        
        # Initial PCM parameter guess
        initial_pcm_params = [L_phase, phase_range]
        
        # PCM parameter bounds
        bounds = [(50000, 500000), (0.5, 10)]  # L_phase, phase_range
        
        # Fit PCM parameters
        result = optimize.minimize(error_func, initial_pcm_params, bounds=bounds)
        
        if not result.success:
            print("Warning: PCM parameter fitting did not converge.")
        
        # Update PCM parameters
        self.params['L_phase'] = result.x[0]
        self.params['phase_range'] = result.x[1]
        
        self.fitted = True
        
        return self.params
    
    def step_response(self, time):
        """
        Calculate thermal step response.
        
        For PCM models, this requires special handling due to the
        nonlinear behavior of the phase change.
        
        Parameters
        ----------
        time : array_like
            Time array in seconds
        
        Returns
        -------
        array_like
            Temperature response to a step power input
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # PCM step response is complex due to nonlinear behavior
        # Here we use a simplified approach
        
        # Calculate base distributed parameter model response
        base_model = DistributedParameterModel(self.n_layers)
        base_model.params = {
            'k': self.params['k'],
            'rho': self.params['rho'],
            'cp': self.params['cp'],
            'd': self.params['d']
        }
        base_model.fitted = True
        
        response_base = base_model.step_response(time)
        
        # Modify response to account for PCM effect
        T_phase = self.params['T_phase']
        L_phase = self.params['L_phase']
        phase_range = self.params['phase_range']
        
        response = np.copy(response_base)
        
        # Find where temperature crosses phase change region
        for i in range(1, len(time)):
            t = time[i]
            if response[i-1] < T_phase - phase_range/2 and response[i] > T_phase - phase_range/2:
                # Calculate extra energy needed for phase change
                extra_energy = L_phase * self.params['rho'][0] * np.prod(self.params['d'])
                
                # Estimate power input
                power_input = response[i] / (base_model.params['R'][0] * t)
                
                # Calculate time delay due to phase change
                delay = extra_energy / power_input
                
                # Apply delay to subsequent response
                time_shift = np.clip(time - delay, 0, None)
                response[i:] = np.interp(time_shift[i:], time, response_base)
                break
        
        return response"""