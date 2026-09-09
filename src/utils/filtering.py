import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch

def bandpass_filter(data, fs, low_freq=10, high_freq=900, order=4, plot= False):
    """
        Apply bandpass filter to sEMG data
        
        Parameters:
        -----------
        data : ndarray, shape (n_channels, n_samples)
            Input EMG data
        fs : float
            Sampling frequency (Hz)
        low_freq : float, default=10
            Low cutoff frequency (Hz)
        high_freq : float, default=900  
            High cutoff frequency (Hz)
        order : int, default=4
            Filter order
        plot : bool, default=False
            Whether to plot spectral components before and after filtering
            
        Returns:
        --------
        filtered_data : ndarray, shape (n_channels, n_samples)
            Bandpass filtered data as np.float64
        """
    # Normalize frequencies by Nyquist frequency
    nyquist = fs / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low_norm, high_norm], btype='band')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = filtfilt(b, a, data[ch, :])
    
    # Plot spectral components if requested
    if plot:
        plot_spectrum_comparison(data, filtered_data, fs, low_freq, high_freq)
    
    return filtered_data

def plot_spectrum_comparison(original_data, filtered_data, fs, low_freq, high_freq):
    """
    Plot power spectral density before and after filtering
    """
    n_channels = original_data.shape[0]
    n_plot_channels = min(4, n_channels)  # Plot up to 4 channels
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Spectral Analysis: Before vs After Bandpass Filtering', fontsize=14)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for i in range(n_plot_channels):
        # Compute power spectral density
        f_orig, psd_orig = welch(original_data[i, :], fs, nperseg=min(1024, len(original_data[i, :])//4))
        f_filt, psd_filt = welch(filtered_data[i, :], fs, nperseg=min(1024, len(filtered_data[i, :])//4))
        
        # Plot
        axes[i].semilogy(f_orig, psd_orig, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[i].semilogy(f_filt, psd_filt, 'r-', alpha=0.8, label='Filtered', linewidth=1.5)
        
        # Add vertical lines for filter cutoffs
        axes[i].axvline(low_freq, color='g', linestyle='--', alpha=0.7, label=f'Low: {low_freq} Hz')
        axes[i].axvline(high_freq, color='orange', linestyle='--', alpha=0.7, label=f'High: {high_freq} Hz')
        
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('PSD (V²/Hz)')
        axes[i].set_title(f'Channel {i+1}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
        axes[i].set_xlim(0, min(2000, fs/2))  # Show up to 2000 Hz or Nyquist
    
    # Hide unused subplots
    for i in range(n_plot_channels, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
