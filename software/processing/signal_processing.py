"""
Signal processing module for the Integrated Electrical-Thermal Impedance Analyzer

This module provides core signal processing functionality for analyzing electrical
and thermal impedance data, including filtering, feature extraction, and data transformation.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
from scipy import signal, optimize
from scipy.fft import fft, ifft
import pywt  # PyWavelets for wavelet analysis

def process_impedance_data(frequency, impedance, phase=None, mode='electrical'):
    """
    Process raw impedance data to prepare for analysis.
    
    Parameters
    ----------
    frequency : array_like
        Array of frequency points in Hz
    impedance : array_like
        Array of impedance magnitudes in ohms (electrical) or K/W (thermal)
    phase : array_like, optional
        Array of phase angles in degrees/radians
    mode : str, optional
        'electrical' or 'thermal' to specify the type of impedance
    
    Returns
    -------
    dict
        Processed impedance data including real part, imaginary part, etc.
    """
    if phase is None:
        # Assume impedance is a complex array
        Z = impedance
    else:
        # Convert magnitude and phase to complex
        if np.max(np.abs(phase)) > np.pi*2:
            # Convert from degrees to radians if necessary
            phase_rad = np.deg2rad(phase)
        else:
            phase_rad = phase
        
        # Calculate complex impedance
        Z = impedance * np.exp(1j * phase_rad)
    
    # Extract components
    Z_real = np.real(Z)
    Z_imag = np.imag(Z)
    Z_mag = np.abs(Z)
    Z_phase = np.angle(Z)
    
    # Create processed data dictionary
    processed_data = {
        'frequency': frequency,
        'impedance': Z,
        'real': Z_real,
        'imaginary': Z_imag,
        'magnitude': Z_mag,
        'phase': Z_phase,
        'mode': mode
    }
    
    return processed_data

def extract_cole_cole_parameters(freq, z):
    """
    Extract Cole-Cole model parameters from impedance data.
    
    Parameters
    ----------
    freq : array_like
        Frequency array
    z : array_like
        Complex impedance array
    
    Returns
    -------
    dict
        Cole-Cole model parameters (R0, R_inf, tau, alpha)
    """
    # Helper function for fitting
    def cole_cole(params, f):
        R0, R_inf, tau, alpha = params
        omega = 2 * np.pi * f
        return R_inf + (R0 - R_inf) / (1 + (1j * omega * tau) ** alpha)
    
    # Error function to minimize
    def error_func(params):
        return np.sum(np.abs(cole_cole(params, freq) - z) ** 2)
    
    # Initial parameter guess
    R_inf_guess = np.real(z[np.argmax(freq)])
    R0_guess = np.real(z[np.argmin(freq)])
    tau_guess = 1 / (2 * np.pi * np.sqrt(np.min(freq) * np.max(freq)))
    alpha_guess = 0.8
    
    initial_params = [R0_guess, R_inf_guess, tau_guess, alpha_guess]
    
    # Fit the model
    result = optimize.minimize(error_func, initial_params, 
                              bounds=((0, None), (0, None), (0, None), (0, 1)))
    
    # Extract optimized parameters
    R0, R_inf, tau, alpha = result.x
    
    return {
        'R0': R0,  # Low frequency resistance
        'R_inf': R_inf,  # High frequency resistance
        'tau': tau,  # Time constant
        'alpha': alpha,  # Distribution parameter
    }

def smooth_data(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    
    Parameters
    ----------
    x : array_like
        Input signal
    window_len : int, optional
        Size of the smoothing window
    window : str, optional
        Type of window ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')
    
    Returns
    -------
    array_like
        Smoothed signal
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len < 3:
        return x
    
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    
    y = np.convolve(w / w.sum(), s, mode='valid')
    
    # Adjust the output to have the same length as the input
    y = y[(window_len//2-1):-(window_len//2)]
    
    return y

def wavelet_denoise(data, wavelet='db4', level=1, threshold_factor=1.0):
    """
    Denoise data using wavelet transform.
    
    Parameters
    ----------
    data : array_like
        Input signal
    wavelet : str, optional
        Wavelet type
    level : int, optional
        Decomposition level
    threshold_factor : float, optional
        Threshold scaling factor
    
    Returns
    -------
    array_like
        Denoised signal
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Threshold calculation
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(data)))
    
    # Apply threshold
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    # Reconstruction
    return pywt.waverec(coeffs, wavelet)

def calculate_drt(frequency, impedance, reg_param=0.001, n_tau=50, tau_min=1e-6, tau_max=1e3):
    """
    Calculate Distribution of Relaxation Times (DRT) from impedance data.
    
    Parameters
    ----------
    frequency : array_like
        Frequency array in Hz
    impedance : array_like
        Complex impedance array
    reg_param : float, optional
        Regularization parameter for Tikhonov regularization
    n_tau : int, optional
        Number of time constants in the DRT
    tau_min : float, optional
        Minimum time constant
    tau_max : float, optional
        Maximum time constant
    
    Returns
    -------
    dict
        tau and gamma arrays representing the DRT
    """
    # Extract real part of impedance (subtract high-freq limit)
    Z_real = np.real(impedance)
    R_inf = np.min(Z_real)
    Z_real_adjusted = Z_real - R_inf
    
    # Create time constant array
    log_tau = np.linspace(np.log10(tau_min), np.log10(tau_max), n_tau)
    tau = 10**log_tau
    
    # Create kernel matrix
    A = np.zeros((len(frequency), n_tau))
    
    for i, f in enumerate(frequency):
        omega = 2 * np.pi * f
        for j, t in enumerate(tau):
            A[i, j] = 1 / (1 + (omega * t) ** 2)
    
    # Create Tikhonov regularization matrix
    L = np.eye(n_tau)  # Identity for zero-order regularization
    
    # Solve regularized least squares problem
    ATA = A.T @ A
    ATb = A.T @ Z_real_adjusted
    gamma = np.linalg.solve(ATA + reg_param * (L.T @ L), ATb)
    
    return {
        'tau': tau,
        'gamma': gamma
    }

def frequency_response_analysis(frequency, impedance, fs=None):
    """
    Analyze frequency response characteristics of impedance data.
    
    Parameters
    ----------
    frequency : array_like
        Frequency array in Hz
    impedance : array_like
        Complex impedance array
    fs : float, optional
        Sampling frequency (if applicable)
    
    Returns
    -------
    dict
        Frequency response characteristics
    """
    # Extract magnitude and phase
    magnitude = np.abs(impedance)
    phase = np.angle(impedance, deg=True)
    
    # Calculate the slope of the magnitude response (in dB per decade)
    mag_db = 20 * np.log10(magnitude)
    freq_log = np.log10(frequency)
    
    # Calculate slope using linear fit in log-log space
    try:
        slope, _ = np.polyfit(freq_log, mag_db, 1)
    except:
        slope = np.nan
    
    # Identify characteristic frequencies
    # - Peaks/valleys in impedance magnitude
    # - Zero crossings or extrema in phase
    
    peaks_idx = signal.find_peaks(magnitude)[0]
    valleys_idx = signal.find_peaks(-magnitude)[0]
    
    # Characteristic frequencies
    characteristic_freq = {
        'peak_frequencies': frequency[peaks_idx] if len(peaks_idx) > 0 else [],
        'valley_frequencies': frequency[valleys_idx] if len(valleys_idx) > 0 else [],
        'slope_db_decade': slope
    }
    
    # For EIS data, calculate cutoff frequencies
    if len(valleys_idx) > 0:
        characteristic_freq['cutoff_frequency'] = frequency[valleys_idx[0]]
    
    return characteristic_freq

def synchronize_electrical_thermal_data(elec_freq, elec_imp, thermal_freq, thermal_imp):
    """
    Synchronize electrical and thermal impedance data for integrated analysis.
    
    Parameters
    ----------
    elec_freq : array_like
        Electrical frequency array
    elec_imp : array_like
        Electrical impedance array
    thermal_freq : array_like
        Thermal frequency array
    thermal_imp : array_like
        Thermal impedance array
    
    Returns
    -------
    dict
        Synchronized data for integrated analysis
    """
    # Create common frequency grid for interpolation if needed
    if not np.array_equal(elec_freq, thermal_freq):
        # Determine frequency range for interpolation
        min_freq = max(np.min(elec_freq), np.min(thermal_freq))
        max_freq = min(np.max(elec_freq), np.max(thermal_freq))
        
        # Create log-spaced frequency grid
        common_freq = np.logspace(np.log10(min_freq), np.log10(max_freq), 
                                 max(len(elec_freq), len(thermal_freq)))
        
        # Interpolate electrical data (real and imaginary parts separately)
        elec_real_interp = np.interp(common_freq, elec_freq, np.real(elec_imp))
        elec_imag_interp = np.interp(common_freq, elec_freq, np.imag(elec_imp))
        elec_imp_interp = elec_real_interp + 1j * elec_imag_interp
        
        # Interpolate thermal data
        thermal_real_interp = np.interp(common_freq, thermal_freq, np.real(thermal_imp))
        thermal_imag_interp = np.interp(common_freq, thermal_freq, np.imag(thermal_imp))
        thermal_imp_interp = thermal_real_interp + 1j * thermal_imag_interp
    else:
        # No interpolation needed
        common_freq = elec_freq
        elec_imp_interp = elec_imp
        thermal_imp_interp = thermal_imp
    
    # Calculate cross-correlation between electrical and thermal
    cross_correlation = np.correlate(np.abs(elec_imp_interp), np.abs(thermal_imp_interp), mode='full')
    
    return {
        'frequency': common_freq,
        'electrical_impedance': elec_imp_interp,
        'thermal_impedance': thermal_imp_interp,
        'cross_correlation': cross_correlation,
        'electro_thermal_ratio': np.abs(elec_imp_interp) / np.abs(thermal_imp_interp)
    }

def extract_features(freq, impedance, mode='electrical'):
    """
    Extract key features from impedance data for machine learning models.
    
    Parameters
    ----------
    freq : array_like
        Frequency array
    impedance : array_like
        Complex impedance array
    mode : str, optional
        'electrical' or 'thermal'
    
    Returns
    -------
    dict
        Dictionary of extracted features
    """
    features = {}
    
    # Basic statistical features
    magnitude = np.abs(impedance)
    phase = np.angle(impedance)
    
    features['mean_magnitude'] = np.mean(magnitude)
    features['std_magnitude'] = np.std(magnitude)
    features['mean_phase'] = np.mean(phase)
    features['std_phase'] = np.std(phase)
    
    # Frequency-specific features
    # Low frequency impedance
    low_freq_idx = np.argmin(freq)
    features['low_freq_impedance'] = magnitude[low_freq_idx]
    features['low_freq_phase'] = phase[low_freq_idx]
    
    # High frequency impedance
    high_freq_idx = np.argmax(freq)
    features['high_freq_impedance'] = magnitude[high_freq_idx]
    features['high_freq_phase'] = phase[high_freq_idx]
    
    # Extract Cole-Cole parameters
    try:
        cole_cole_params = extract_cole_cole_parameters(freq, impedance)
        features.update(cole_cole_params)
    except:
        # In case the fitting fails
        pass
    
    # Additional mode-specific features
    if mode == 'electrical':
        # Calculate approximated equivalent circuit values
        try:
            # Simple RC circuit approximation for electrical
            R = np.real(impedance[low_freq_idx])
            f_cutoff = freq[np.argmin(np.abs(magnitude - 0.707 * magnitude[low_freq_idx]))]
            C = 1 / (2 * np.pi * f_cutoff * R)
            
            features['estimated_R'] = R
            features['estimated_C'] = C
        except:
            pass
    
    elif mode == 'thermal':
        # Thermal-specific features
        try:
            # Thermal time constant approximation
            thermal_tau = 1 / (2 * np.pi * freq[np.argmax(np.imag(impedance))])
            features['thermal_time_constant'] = thermal_tau
            
            # Thermal resistance approximation
            R_thermal = np.real(impedance[low_freq_idx])
            features['thermal_resistance'] = R_thermal
        except:
            pass
    
    return features
"""