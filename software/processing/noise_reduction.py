"""
Noise reduction module for the Integrated Electrical-Thermal Impedance Analyzer

This module implements various noise filtering and reduction techniques for
cleaning electrical and thermal impedance data, including adaptive filtering,
wavelet denoising, and Kalman filtering.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
from scipy import signal
import pywt
from enum import Enum
import matplotlib.pyplot as plt


class SignalType(Enum):
    """Enumeration for different signal types."""
    ELECTRICAL = 1
    THERMAL = 2
    COMBINED = 3


def median_filter(data, kernel_size=5):
    """
    Apply median filter to remove impulse noise.
    
    Parameters
    ----------
    data : array_like
        Input signal (can be complex)
    kernel_size : int, optional
        Size of the median filter kernel (window)
    
    Returns
    -------
    array_like
        Filtered signal
    """
    # Handle complex data
    if np.iscomplexobj(data):
        real_part = signal.medfilt(np.real(data), kernel_size)
        imag_part = signal.medfilt(np.imag(data), kernel_size)
        return real_part + 1j * imag_part
    else:
        return signal.medfilt(data, kernel_size)


def savitzky_golay_filter(data, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter for smoothing.
    
    Parameters
    ----------
    data : array_like
        Input signal (can be complex)
    window_length : int, optional
        Length of the filter window
    polyorder : int, optional
        Order of the polynomial used to fit the samples
    
    Returns
    -------
    array_like
        Filtered signal
    """
    # Check window parameters
    if window_length % 2 == 0:
        window_length += 1  # Ensure odd window length
    
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    # Handle complex data
    if np.iscomplexobj(data):
        real_part = signal.savgol_filter(np.real(data), window_length, polyorder)
        imag_part = signal.savgol_filter(np.imag(data), window_length, polyorder)
        return real_part + 1j * imag_part
    else:
        return signal.savgol_filter(data, window_length, polyorder)


def wavelet_denoise(data, wavelet='db4', level=None, threshold_mode='soft',
                   threshold_method='universal', threshold_factor=1.0):
    """
    Wavelet-based denoising for impedance data.
    
    Parameters
    ----------
    data : array_like
        Input signal (can be complex)
    wavelet : str, optional
        Wavelet type (e.g., 'db4', 'sym8')
    level : int, optional
        Decomposition level. If None, computed based on data length.
    threshold_mode : str, optional
        'soft' or 'hard' thresholding
    threshold_method : str, optional
        'universal' or 'bayes' threshold selection method
    threshold_factor : float, optional
        Factor to scale the threshold (higher values = more denoising)
    
    Returns
    -------
    array_like
        Denoised signal
    """
    # Handle complex data
    if np.iscomplexobj(data):
        real_part = wavelet_denoise_real(np.real(data), wavelet, level, 
                                        threshold_mode, threshold_method, 
                                        threshold_factor)
        imag_part = wavelet_denoise_real(np.imag(data), wavelet, level,
                                        threshold_mode, threshold_method,
                                        threshold_factor)
        return real_part + 1j * imag_part
    else:
        return wavelet_denoise_real(data, wavelet, level, threshold_mode,
                                  threshold_method, threshold_factor)


def wavelet_denoise_real(data, wavelet='db4', level=None, threshold_mode='soft',
                       threshold_method='universal', threshold_factor=1.0):
    """
    Wavelet-based denoising for real-valued data.
    
    Parameters
    ----------
    data : array_like
        Input signal (real)
    wavelet : str, optional
        Wavelet type (e.g., 'db4', 'sym8')
    level : int, optional
        Decomposition level. If None, computed based on data length.
    threshold_mode : str, optional
        'soft' or 'hard' thresholding
    threshold_method : str, optional
        'universal' or 'bayes' threshold selection method
    threshold_factor : float, optional
        Factor to scale the threshold (higher values = more denoising)
    
    Returns
    -------
    array_like
        Denoised signal
    """
    # Determine decomposition level if not provided
    if level is None:
        level = min(int(np.log2(len(data))), pywt.dwt_max_level(len(data), wavelet))
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Calculate threshold
    if threshold_method == 'universal':
        # Universal threshold (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust noise estimate
        N = len(data)
        threshold = sigma * np.sqrt(2 * np.log(N)) * threshold_factor
    elif threshold_method == 'bayes':
        # BayesShrink (level-dependent threshold)
        thresholds = []
        for i in range(1, len(coeffs)):  # Skip approximation coefficients
            sigma = np.median(np.abs(coeffs[i])) / 0.6745  # Robust noise estimate
            threshold = sigma * np.sqrt(2 * np.log(len(coeffs[i]))) * threshold_factor
            thresholds.append(threshold)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Apply thresholding
    if threshold_method == 'universal':
        # Apply same threshold to all detail coefficients
        for i in range(1, len(coeffs)):  # Skip approximation coefficients
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)
    else:  # 'bayes'
        # Apply level-dependent thresholds
        for i in range(1, len(coeffs)):  # Skip approximation coefficients
            coeffs[i] = pywt.threshold(coeffs[i], thresholds[i-1], mode=threshold_mode)
    
    # Reconstruct signal
    return pywt.waverec(coeffs, wavelet)


def kalman_filter(data, process_variance=1e-5, measurement_variance=1e-2, 
                 initial_state=None, is_complex=False):
    """
    Apply Kalman filter for optimal state estimation.
    
    Parameters
    ----------
    data : array_like
        Input signal
    process_variance : float, optional
        Process noise variance
    measurement_variance : float, optional
        Measurement noise variance
    initial_state : float, optional
        Initial state estimate. If None, use first measurement.
    is_complex : bool, optional
        Whether the data is complex
    
    Returns
    -------
    array_like
        Filtered signal
    """
    if is_complex or np.iscomplexobj(data):
        real_part = kalman_filter_real(np.real(data), process_variance, 
                                     measurement_variance, 
                                     initial_state)
        imag_part = kalman_filter_real(np.imag(data), process_variance, 
                                     measurement_variance,
                                     initial_state)
        return real_part + 1j * imag_part
    else:
        return kalman_filter_real(data, process_variance, measurement_variance, 
                               initial_state)


def kalman_filter_real(data, process_variance=1e-5, measurement_variance=1e-2, 
                     initial_state=None):
    """
    Apply Kalman filter for real-valued data.
    
    Parameters
    ----------
    data : array_like
        Input signal (real)
    process_variance : float, optional
        Process noise variance
    measurement_variance : float, optional
        Measurement noise variance
    initial_state : float, optional
        Initial state estimate. If None, use first measurement.
    
    Returns
    -------
    array_like
        Filtered signal
    """
    # Initialize state
    if initial_state is None:
        initial_state = data[0]
    
    n = len(data)
    filtered_data = np.zeros(n)
    
    # Initial estimates
    x_hat = initial_state  # State estimate
    p = 1.0  # Error estimate
    
    # Kalman filter iteration
    for i in range(n):
        # Prediction
        x_hat_minus = x_hat
        p_minus = p + process_variance
        
        # Update
        K = p_minus / (p_minus + measurement_variance)  # Kalman gain
        x_hat = x_hat_minus + K * (data[i] - x_hat_minus)
        p = (1 - K) * p_minus
        
        # Store filtered value
        filtered_data[i] = x_hat
    
    return filtered_data


def adaptive_filter(data, reference, step_size=0.01, filter_length=10, is_complex=False):
    """
    Apply adaptive filter using LMS algorithm.
    
    Parameters
    ----------
    data : array_like
        Primary input signal (can be complex)
    reference : array_like
        Reference signal for adaptive filtering
    step_size : float, optional
        Step size (learning rate) for the LMS algorithm
    filter_length : int, optional
        Length of the adaptive filter
    is_complex : bool, optional
        Whether the data is complex
    
    Returns
    -------
    array_like
        Filtered signal
    """
    if is_complex or np.iscomplexobj(data):
        real_part = adaptive_filter_real(np.real(data), np.real(reference), 
                                       step_size, filter_length)
        imag_part = adaptive_filter_real(np.imag(data), np.imag(reference), 
                                       step_size, filter_length)
        return real_part + 1j * imag_part
    else:
        return adaptive_filter_real(data, reference, step_size, filter_length)


def adaptive_filter_real(data, reference, step_size=0.01, filter_length=10):
    """
    Apply adaptive filter (LMS) for real-valued data.
    
    Parameters
    ----------
    data : array_like
        Primary input signal (real)
    reference : array_like
        Reference signal for adaptive filtering
    step_size : float, optional
        Step size (learning rate) for the LMS algorithm
    filter_length : int, optional
        Length of the adaptive filter
    
    Returns
    -------
    array_like
        Filtered signal
    """
    n = len(data)
    filtered_data = np.zeros(n)
    
    # Initialize filter weights
    w = np.zeros(filter_length)
    
    # Initialize reference signal buffer
    x_buffer = np.zeros(filter_length)
    
    # LMS algorithm
    for i in range(n):
        # Update reference signal buffer
        x_buffer = np.roll(x_buffer, 1)
        x_buffer[0] = reference[i]
        
        # Calculate filter output
        y = np.dot(w, x_buffer)
        
        # Calculate error
        e = data[i] - y
        
        # Update weights
        w = w + step_size * e * x_buffer
        
        # Store filtered value
        filtered_data[i] = e
    
    return filtered_data


def multi_band_filter(data, sampling_rate, freq_bands, filter_type='bandpass', 
                     order=4, is_complex=False):
    """
    Apply multi-band filtering to extract signals in specific frequency bands.
    
    Parameters
    ----------
    data : array_like
        Input signal (can be complex)
    sampling_rate : float
        Sampling rate in Hz
    freq_bands : list of tuples
        List of frequency bands (low, high) in Hz
    filter_type : str, optional
        'bandpass' or 'bandstop'
    order : int, optional
        Filter order
    is_complex : bool, optional
        Whether the data is complex
    
    Returns
    -------
    list of array_like
        Filtered signals for each frequency band
    """
    if is_complex or np.iscomplexobj(data):
        real_results = multi_band_filter_real(np.real(data), sampling_rate, 
                                           freq_bands, filter_type, order)
        imag_results = multi_band_filter_real(np.imag(data), sampling_rate, 
                                           freq_bands, filter_type, order)
        
        # Combine real and imaginary parts
        results = []
        for real_part, imag_part in zip(real_results, imag_results):
            results.append(real_part + 1j * imag_part)
        
        return results
    else:
        return multi_band_filter_real(data, sampling_rate, freq_bands, 
                                   filter_type, order)


def multi_band_filter_real(data, sampling_rate, freq_bands, filter_type='bandpass', 
                         order=4):
    """
    Apply multi-band filtering for real-valued data.
    
    Parameters
    ----------
    data : array_like
        Input signal (real)
    sampling_rate : float
        Sampling rate in Hz
    freq_bands : list of tuples
        List of frequency bands (low, high) in Hz
    filter_type : str, optional
        'bandpass' or 'bandstop'
    order : int, optional
        Filter order
    
    Returns
    -------
    list of array_like
        Filtered signals for each frequency band
    """
    nyquist = 0.5 * sampling_rate
    filtered_signals = []
    
    for low, high in freq_bands:
        # Normalize frequencies to Nyquist
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Ensure frequencies are within valid range
        low_norm = max(0, min(1, low_norm))
        high_norm = max(0, min(1, high_norm))
        
        # Design filter
        if filter_type == 'bandpass':
            b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        elif filter_type == 'bandstop':
            b, a = signal.butter(order, [low_norm, high_norm], btype='bandstop')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply filter
        filtered = signal.filtfilt(b, a, data)
        filtered_signals.append(filtered)
    
    return filtered_signals


def signal_decomposition(data, wavelet='db4', level=None):
    """
    Decompose signal into approximation and detail coefficients using
    wavelet transform.
    
    Parameters
    ----------
    data : array_like
        Input signal (can be complex)
    wavelet : str, optional
        Wavelet type (e.g., 'db4', 'sym8')
    level : int, optional
        Decomposition level. If None, computed based on data length.
    
    Returns
    -------
    dict
        Dictionary containing approximation and detail coefficients
    """
    # Handle complex data
    if np.iscomplexobj(data):
        real_decomp = signal_decomposition_real(np.real(data), wavelet, level)
        imag_decomp = signal_decomposition_real(np.imag(data), wavelet, level)
        
        # Combine real and imaginary parts
        decomp = {'approximation': real_decomp['approximation'] + 1j * imag_decomp['approximation']}
        for k in range(len(real_decomp) - 1):
            decomp[f'detail_{k+1}'] = real_decomp[f'detail_{k+1}'] + 1j * imag_decomp[f'detail_{k+1}']
        
        return decomp
    else:
        return signal_decomposition_real(data, wavelet, level)


def signal_decomposition_real(data, wavelet='db4', level=None):
    """
    Decompose real signal using wavelet transform.
    
    Parameters
    ----------
    data : array_like
        Input signal (real)
    wavelet : str, optional
        Wavelet type (e.g., 'db4', 'sym8')
    level : int, optional
        Decomposition level. If None, computed based on data length.
    
    Returns
    -------
    dict
        Dictionary containing approximation and detail coefficients
    """
    # Determine decomposition level if not provided
    if level is None:
        level = min(int(np.log2(len(data))), pywt.dwt_max_level(len(data), wavelet))
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Store results
    decomp = {'approximation': pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(data))}
    
    for i in range(1, len(coeffs)):
        decomp[f'detail_{i}'] = pywt.upcoef('d', coeffs[i], wavelet, level=level-i+1, take=len(data))
    
    return decomp


def empirical_mode_decomposition(data, max_imfs=10, sift_iters=10):
    """
    Perform Empirical Mode Decomposition (EMD) to decompose signal
    into Intrinsic Mode Functions (IMFs).
    
    Parameters
    ----------
    data : array_like
        Input signal (real)
    max_imfs : int, optional
        Maximum number of IMFs to extract
    sift_iters : int, optional
        Number of sifting iterations
    
    Returns
    -------
    list of array_like
        List of IMFs
    """
    # EMD requires real signals
    if np.iscomplexobj(data):
        raise ValueError("EMD is only defined for real signals. "
                        "Please process real and imaginary parts separately.")
    
    # Check if EMD package is available
    try:
        from PyEMD import EMD
    except ImportError:
        raise ImportError("PyEMD package is required for EMD. "
                         "Install it with: pip install EMD-signal")
    
    # Initialize EMD
    emd = EMD()
    emd.MAX_ITERATION = sift_iters
    
    # Perform decomposition
    imfs = emd(data, max_imfs)
    
    return imfs


class DenoiseMethod(Enum):
    """Enumeration for different denoising methods."""
    WAVELET = 1
    KALMAN = 2
    SAVITZKY_GOLAY = 3
    MEDIAN = 4
    ADAPTIVE = 5


def auto_denoise(data, method=DenoiseMethod.WAVELET, signal_type=SignalType.ELECTRICAL,
                reference=None, sampling_rate=None, **kwargs):
    """
    Automatically denoise impedance data using the specified method.
    
    Parameters
    ----------
    data : array_like
        Input signal (can be complex)
    method : DenoiseMethod, optional
        Denoising method to use
    signal_type : SignalType, optional
        Type of signal (electrical, thermal, combined)
    reference : array_like, optional
        Reference signal for adaptive filtering
    sampling_rate : float, optional
        Sampling rate in Hz, required for some methods
    **kwargs
        Additional parameters for the specific denoising method
    
    Returns
    -------
    array_like
        Denoised signal
    """
    # Use optimal parameters based on signal type
    if signal_type == SignalType.ELECTRICAL:
        # Parameters optimized for electrical impedance data
        if method == DenoiseMethod.WAVELET:
            wavelet = kwargs.get('wavelet', 'sym8')
            level = kwargs.get('level', None)
            threshold_mode = kwargs.get('threshold_mode', 'soft')
            threshold_factor = kwargs.get('threshold_factor', 1.0)
            return wavelet_denoise(data, wavelet, level, threshold_mode, 
                                 'universal', threshold_factor)
        
        elif method == DenoiseMethod.KALMAN:
            process_var = kwargs.get('process_variance', 1e-6)
            measurement_var = kwargs.get('measurement_variance', 1e-2)
            return kalman_filter(data, process_var, measurement_var)
        
        elif method == DenoiseMethod.SAVITZKY_GOLAY:
            window = kwargs.get('window_length', 15)
            polyorder = kwargs.get('polyorder', 3)
            return savitzky_golay_filter(data, window, polyorder)
        
        elif method == DenoiseMethod.MEDIAN:
            kernel_size = kwargs.get('kernel_size', 5)
            return median_filter(data, kernel_size)
        
        elif method == DenoiseMethod.ADAPTIVE:
            if reference is None:
                raise ValueError("Reference signal required for adaptive filtering")
            step_size = kwargs.get('step_size', 0.01)
            filter_length = kwargs.get('filter_length', 20)
            return adaptive_filter(data, reference, step_size, filter_length)
    
    elif signal_type == SignalType.THERMAL:
        # Parameters optimized for thermal impedance data
        if method == DenoiseMethod.WAVELET:
            wavelet = kwargs.get('wavelet', 'db6')
            level = kwargs.get('level', None)
            threshold_mode = kwargs.get('threshold_mode', 'soft')
            threshold_factor = kwargs.get('threshold_factor', 1.5)
            return wavelet_denoise(data, wavelet, level, threshold_mode, 
                                 'universal', threshold_factor)
        
        elif method == DenoiseMethod.KALMAN:
            process_var = kwargs.get('process_variance', 1e-7)
            measurement_var = kwargs.get('measurement_variance', 1e-3)
            return kalman_filter(data, process_var, measurement_var)
        
        elif method == DenoiseMethod.SAVITZKY_GOLAY:
            window = kwargs.get('window_length', 21)
            polyorder = kwargs.get('polyorder', 2)
            return savitzky_golay_filter(data, window, polyorder)
        
        elif method == DenoiseMethod.MEDIAN:
            kernel_size = kwargs.get('kernel_size', 7)
            return median_filter(data, kernel_size)
        
        elif method == DenoiseMethod.ADAPTIVE:
            if reference is None:
                raise ValueError("Reference signal required for adaptive filtering")
            step_size = kwargs.get('step_size', 0.005)
            filter_length = kwargs.get('filter_length', 30)
            return adaptive_filter(data, reference, step_size, filter_length)
    
    elif signal_type == SignalType.COMBINED:
        # For combined signals, use more conservative parameters
        if method == DenoiseMethod.WAVELET:
            wavelet = kwargs.get('wavelet', 'coif3')
            level = kwargs.get('level', None)
            threshold_mode = kwargs.get('threshold_mode', 'soft')
            threshold_factor = kwargs.get('threshold_factor', 1.2)
            return wavelet_denoise(data, wavelet, level, threshold_mode, 
                                 'universal', threshold_factor)
        
        elif method == DenoiseMethod.KALMAN:
            process_var = kwargs.get('process_variance', 1e-5)
            measurement_var = kwargs.get('measurement_variance', 5e-3)
            return kalman_filter(data, process_var, measurement_var)
        
        elif method == DenoiseMethod.SAVITZKY_GOLAY:
            window = kwargs.get('window_length', 17)
            polyorder = kwargs.get('polyorder', 3)
            return savitzky_golay_filter(data, window, polyorder)
        
        elif method == DenoiseMethod.MEDIAN:
            kernel_size = kwargs.get('kernel_size', 5)
            return median_filter(data, kernel_size)
        
        elif method == DenoiseMethod.ADAPTIVE:
            if reference is None:
                raise ValueError("Reference signal required for adaptive filtering")
            step_size = kwargs.get('step_size', 0.007)
            filter_length = kwargs.get('filter_length', 25)
            return adaptive_filter(data, reference, step_size, filter_length)
    
    raise ValueError(f"Unknown method or signal type: {method}, {signal_type}")


def evaluate_denoising(original, noisy, denoised):
    """
    Evaluate denoising performance.
    
    Parameters
    ----------
    original : array_like
        Original clean signal
    noisy : array_like
        Noisy signal
    denoised : array_like
        Denoised signal
    
    Returns
    -------
    dict
        Performance metrics
    """
    # Handle complex signals
    if np.iscomplexobj(original) or np.iscomplexobj(noisy) or np.iscomplexobj(denoised):
        # Convert to magnitude
        original_mag = np.abs(original)
        noisy_mag = np.abs(noisy)
        denoised_mag = np.abs(denoised)
        
        # Evaluate on magnitude
        metrics_mag = evaluate_denoising_real(original_mag, noisy_mag, denoised_mag)
        
        # Calculate phase metrics
        original_phase = np.angle(original)
        noisy_phase = np.angle(noisy)
        denoised_phase = np.angle(denoised)
        
        metrics_phase = evaluate_denoising_real(original_phase, noisy_phase, denoised_phase)
        
        # Combine metrics
        metrics = {
            'magnitude': metrics_mag,
            'phase': metrics_phase
        }
        
        # Overall metrics
        metrics['overall_snr_improvement'] = (metrics_mag['snr_improvement'] + 
                                           metrics_phase['snr_improvement']) / 2
        metrics['overall_mse_reduction'] = (metrics_mag['mse_reduction'] + 
                                         metrics_phase['mse_reduction']) / 2
        
        return metrics
    else:
        return evaluate_denoising_real(original, noisy, denoised)


def evaluate_denoising_real(original, noisy, denoised):
    """
    Evaluate denoising performance for real-valued signals.
    
    Parameters
    ----------
    original : array_like
        Original clean signal
    noisy : array_like
        Noisy signal
    denoised : array_like
        Denoised signal
    
    Returns
    -------
    dict
        Performance metrics
    """
    # Mean Squared Error (MSE)
    mse_noisy = np.mean(np.abs(original - noisy) ** 2)
    mse_denoised = np.mean(np.abs(original - denoised) ** 2)
    mse_reduction = 100 * (1 - mse_denoised / mse_noisy) if mse_noisy > 0 else 0
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(np.abs(original) ** 2)
    noise_power_before = np.mean(np.abs(original - noisy) ** 2)
    noise_power_after = np.mean(np.abs(original - denoised) ** 2)
    
    snr_before = 10 * np.log10(signal_power / noise_power_before) if noise_power_before > 0 else float('inf')
    snr_after = 10 * np.log10(signal_power / noise_power_after) if noise_power_after > 0 else float('inf')
    snr_improvement = snr_after - snr_before
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_value = np.max(np.abs(original))
    psnr_before = 20 * np.log10(max_value / np.sqrt(noise_power_before)) if noise_power_before > 0 else float('inf')
    psnr_after = 20 * np.log10(max_value / np.sqrt(noise_power_after)) if noise_power_after > 0 else float('inf')
    psnr_improvement = psnr_after - psnr_before
    
    # Correlation coefficient
    corr_before = np.corrcoef(original, noisy)[0, 1]
    corr_after = np.corrcoef(original, denoised)[0, 1]
    corr_improvement = corr_after - corr_before
    
    # Return metrics
    return {
        'mse_before': mse_noisy,
        'mse_after': mse_denoised,
        'mse_reduction': mse_reduction,
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_improvement': snr_improvement,
        'psnr_before': psnr_before,
        'psnr_after': psnr_after,
        'psnr_improvement': psnr_improvement,
        'correlation_before': corr_before,
        'correlation_after': corr_after,
        'correlation_improvement': corr_improvement
    }


def plot_denoising_results(original, noisy, denoised, title=None, is_complex=False):
    """
    Plot original, noisy, and denoised signals for comparison.
    
    Parameters
    ----------
    original : array_like
        Original clean signal
    noisy : array_like
        Noisy signal
    denoised : array_like
        Denoised signal
    title : str, optional
        Plot title
    is_complex : bool, optional
        Whether the signals are complex
    
    Returns
    -------
    tuple
        Figure and axes objects
    """
    if is_complex or np.iscomplexobj(original) or np.iscomplexobj(noisy) or np.iscomplexobj(denoised):
        # Complex signal plots (magnitude and phase)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        axs[0].plot(np.abs(original), 'b-', label='Original', linewidth=1.5)
        axs[0].plot(np.abs(noisy), 'r-', label='Noisy', alpha=0.6)
        axs[0].plot(np.abs(denoised), 'g-', label='Denoised', linewidth=1.5)
        axs[0].set_ylabel('Magnitude')
        axs[0].set_title(f'Magnitude {title}' if title else 'Magnitude')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Phase plot
        axs[1].plot(np.angle(original, deg=True), 'b-', label='Original', linewidth=1.5)
        axs[1].plot(np.angle(noisy, deg=True), 'r-', label='Noisy', alpha=0.6)
        axs[1].plot(np.angle(denoised, deg=True), 'g-', label='Denoised', linewidth=1.5)
        axs[1].set_xlabel('Sample')
        axs[1].set_ylabel('Phase (degrees)')
        axs[1].set_title('Phase')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
    else:
        # Real signal plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(original, 'b-', label='Original', linewidth=1.5)
        ax.plot(noisy, 'r-', label='Noisy', alpha=0.6)
        ax.plot(denoised, 'g-', label='Denoised', linewidth=1.5)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_title(title if title else 'Denoising Results')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        axs = [ax]
    
    # Add metrics to the plot
    metrics = evaluate_denoising(original, noisy, denoised)
    
    if is_complex or np.iscomplexobj(original):
        # Complex signal metrics
        text = (f"Magnitude: SNR Improvement: {metrics['magnitude']['snr_improvement']:.2f} dB, "
              f"MSE Reduction: {metrics['magnitude']['mse_reduction']:.2f}%\n"
              f"Phase: SNR Improvement: {metrics['phase']['snr_improvement']:.2f} dB, "
              f"MSE Reduction: {metrics['phase']['mse_reduction']:.2f}%")
    else:
        # Real signal metrics
        text = (f"SNR Improvement: {metrics['snr_improvement']:.2f} dB\n"
              f"MSE Reduction: {metrics['mse_reduction']:.2f}%\n"
              f"Correlation Improvement: {metrics['correlation_improvement']:.4f}")
    
    fig.text(0.5, 0.01, text, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", 
                                                               facecolor='white', alpha=0.7))
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig, axs"""