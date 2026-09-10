# Ripple Analysis Functions

# Loading/Import related packages
import sys
import os

#Usual suspects
import pandas as pd
import numpy as np
import json
import pickle as pkl
import matplotlib.pyplot as plt

#Extras for plotting
from matplotlib.patches import Patch

#Extra for typing
from collections import defaultdict

import glob 

###
def get_task_data(s_dir, n_reps, movement_names, file_type, sampling_freq, win=None):
    """
    Load EMG data from multiple pickle files, 
    then combine repitions per movement type, with the option of specifying a window within each repition to keep.

    Parameters:
    - s_dir: str, directory where the pickle files are located.
    - n_reps: int, number of repetitions per movement.
    - movement_names: list of str, names of movements sorted in the order of collection in task.
    - file_type: str, the type of files to load (e.g., 'move' for movement files, or 'rest' for rest files).
    - sampling_freq: float, sampling frequency of data collection in Hz.
    - win: float, optional, window size in seconds to extract from the middle of the movement
      duration. If None, the entire rep duration is used.

    Returns:
    - iso_dict: dict containing each movement as keys, containing:
        - emg_reps: list of dataframes, each containing the EMG data for a single repetition
        - combined_emg_reps: ndarray, concatenated (wide) EMG data for all repetitions
        - durations: list of floats, duration of each repetition
    """
    task_files = sorted(glob.glob(os.path.join(s_dir, f'*{file_type}*')))

    half_win = win / 2 if win is not None else None
    iso_dict = {}

    for movement_num, movement_type in enumerate(movement_names):
        start = movement_num * n_reps
        end = start + n_reps
        movement_files = task_files[start:end]

        movement_data = [pd.read_pickle(file_name) for file_name in movement_files]
        durations = [rep_data['duration_s'] for rep_data in movement_data]

        rep_emg = []
        for rep_data, dur in zip(movement_data, durations):
            rep_df = pd.DataFrame(rep_data['data'])

            if win is None:
                rep_data = rep_df  # no windowing, take entire rep duration
            else:
                iso_start = int(((dur / 2) - half_win) * sampling_freq)
                iso_end = int(((dur / 2) + half_win) * sampling_freq)
                rep_data = rep_df.iloc[:, iso_start:iso_end]

            rep_emg.append(rep_data)

        iso_dict[movement_type] = {
            'emg_reps': rep_emg,
            'combined_emg_reps': pd.concat(rep_emg, ignore_index=True, axis=1).to_numpy(),
            'durations': durations
        }

    return iso_dict

#### Preprocessing
from scipy.interpolate import interp1d

def remove_spikes(emg, threshold_std=5):
    """
    Remove large spikes via simple thresholding and linear interpolation.
    
    Parameters
    ----------
    emg : ndarray, shape (n_channels, n_samples)
    threshold_std : float
        Number of standard deviations for threshold
    
    Returns
    -------
    emg_clean : ndarray
    spike_mask : ndarray, bool
    """
    emg_clean = emg.copy()
    spike_mask = np.zeros_like(emg, dtype=bool)
    
    for ch in range(emg.shape[0]):
        signal = emg[ch]
        threshold = threshold_std * np.std(signal)
        
        spikes = np.abs(signal) > threshold
        spike_mask[ch] = spikes
        
        if np.any(spikes):
            clean_idx = np.where(~spikes)[0]
            spike_idx = np.where(spikes)[0]
            
            if len(clean_idx) > 1:
                interp = interp1d(clean_idx, signal[clean_idx], 
                                  kind='linear', bounds_error=False, 
                                  fill_value='extrapolate')
                emg_clean[ch, spike_idx] = interp(spike_idx)
    
    return emg_clean, spike_mask


def remove_spikes_dict(data_dict, threshold_std=3):
    """
    Remove large spikes from EMG data in dictionary. 
    Can remove spikes from each repetition separately or from the combined EMG repetitions.

    Paramters:
    - data_dict: dict, containing each movement as keys, and 'emg_reps', 'combined_emg_reps', 'durations' as values
    - threshold_std: float, number of standard deviations for spike thresholding as required by remove_spikes function

    Returns:
    - clean_dict: dict, containing each movement as keys, and clean emg data and spike masks as values.
    """

    clean_dict = {}

    for movement, data in data_dict.items():

        # include if you want to clean each repitition separately:
        # emg_reps_clean = []
        # spike_mask_reps = []

       
        # for rep_df in data['emg_reps']:
        #     rep_clean, rep_mask = remove_spikes(rep_df.to_numpy(), threshold_std)
        #     emg_reps_clean.append(pd.DataFrame(rep_clean, index=rep_df.index, columns=rep_df.columns))
        #     spike_mask_reps.append(rep_mask)
        
        combined_clean, combined_mask = remove_spikes(data['combined_emg_reps'], threshold_std)


        clean_dict[movement] = {
            **data,  # keep original emg_reps, combined_emg_reps, durations
            # 'emg_reps_clean': emg_reps_clean, # uncomment for cleaning each repitition separately
            # 'spike_mask_reps': spike_mask_reps,
            'combined_emg_reps_clean': combined_clean,
            'spike_mask_combined': combined_mask,
        }

    return clean_dict

#### Signal Quality
from scipy.signal import welch

def calc_PSD(sig, fsamp=2048, nperseg=2048, noverlap=2048/2):
    '''
    Compute the power spectral density for each channel using Welch's method.

    Args:
        sig (ndarray): Multi-channel signal (Channels x Samples)
        fsamp (float): Sampling rate in Hz
        nperseg (int): Number of data points per segment
        overlap (int): Number of overlapping samples
    '''

    f, _ = welch(sig[0,:], fs=fsamp, nperseg=nperseg, noverlap=noverlap)

    P = np.zeros((sig.shape[0], f.shape[0]))

    for i in range(sig.shape[0]):
        _ , P[i,:] = welch(sig[i,:], fs=fsamp, nperseg=nperseg, noverlap=noverlap)

    return P, f

def emg_inspect(move_dict, noise_dict, samp_freq, emg_name_key='combined_emg_reps'):
    """
    Collect variables to inspect EMG data quality, including power spectral density (PSD), power, and signal-to-noise ratio (SNR).

    Parameters:
    - move_dict: dict, containing each movement as keys, with combined EMG repetitions as values
    - noise_dict: dict, containing noise data as keys, with combined EMG repetitions as values
    - samp_freq: int, sampling frequency
    - emg_name_key: str, key to access the combined EMG repetitions in the dictionaries (
        default is 'combined_emg_reps', can be changed to 'combined_emg_reps_clean' if using cleaned data

    Returns:
    - inspect_dict: dict, containing each movement and noise as keys, with PSD, power, and SNR values for inspection
    """
    inspect_dict = {}

    for noise_key in noise_dict.keys():
        noise = noise_dict[noise_key][emg_name_key]
        P_n, f_n = calc_PSD(noise, fsamp=samp_freq, nperseg=samp_freq, noverlap=samp_freq/2)
        p_noise = np.mean(noise**2)
        rms_noise = p_noise**0.5
        P_mean = np.mean(P_n, axis=0)
        P_tot = np.sum(P_mean)
        
        inspect_dict[noise_key] = {
            'psd': (P_n, f_n),
            'power': p_noise,
            'P_tot': P_tot,
            'rms_noise': rms_noise
        }

    for sig_key in move_dict.keys():
        sig = move_dict[sig_key][emg_name_key]
        P_s, f_s = calc_PSD(sig, fsamp=samp_freq, nperseg=samp_freq, noverlap=samp_freq/2)
        p_sig = np.mean(sig**2)
        

        inspect_dict[sig_key] = {
            'psd': (P_s, f_s),
            'power': p_sig,
            'snr': {}
        }

        for noise_key in noise_dict.keys():
            p_noise = inspect_dict[noise_key]['power']
            snr = 10 * np.log10(p_sig / p_noise)
            inspect_dict[sig_key]['snr'][noise_key] = snr

    return inspect_dict

# generate_psd_fig requires noise and signal data to be combined in emg_data
def generate_psd_fig(emg_data, fn, fs, pn, ps, samp_freq, channels=np.arange(13, 25), zoom_xlim=(24.9, 25.1), save_path=None):
    """
    Plot EMG signals (full + zoomed) together with their PSD.

    Parameters
    ----------
    emg_data : dict
        Dictionary containing the cleaned EMG data.
        Expected structure:
        emg_data['fist']['combined_emg_reps_clean']

    emg_inspect : dict
        Dictionary containing PSD information.
        Expected structure:
        emg_inspect['rest']['psd'] = [Pn, fn]
        emg_inspect['fist']['psd'] = [Ps, fs]

    samp_freq : float
        Sampling frequency of the EMG data (Hz).

    channels : array-like, optional
        Channel indices to plot.

    zoom_xlim : tuple, optional
        Time range for the zoomed signal plot.

    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    # Time (x axis) vector
    t = np.linspace(0, (emg_data.shape[1] - 1) / samp_freq, emg_data.shape[1])

    # -------------------------
    # Create figure
    # -------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # -------------------------
    # Full EMG signal
    # -------------------------
    for i, ch_idx in enumerate(channels):
        trace = emg_data[ch_idx, :]
        ax[0].plot(t, trace + i, lw=0.5
        )

    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude + offset (mV)")
    ax[0].set_title("Raw EMG signals")

    # -------------------------
    # Zoomed EMG signal
    # -------------------------
    for i, ch_idx in enumerate(channels):
        trace = emg_data[ch_idx, :]
        ax[1].plot(t,trace + i, lw=0.5
        )

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude + offset (mV)")
    ax[1].set_xlim(*zoom_xlim)
    ax[1].set_xticks(list(zoom_xlim))
    ax[1].set_title("Raw EMG signals (zoom)")

    # -------------------------
    # PSD
    # -------------------------

    # Threshold lines
    ax[2].semilogy(fn,np.ones(len(fn)) * 5.21e-8, color="darkgreen", linestyle="dashed",lw=1)

    ax[2].semilogy(fn,np.ones(len(fn)) * 20.83e-8, color="green", linestyle="dashed", lw=1)

    ax[2].semilogy(fn,np.ones(len(fn)) * 46.87e-8, color="orange", linestyle="dashed",lw=1)

    ax[2].semilogy(fn,np.ones(len(fn)) * 83.33e-8, color="red", linestyle="dashed", lw=1)

    # Individual PSDs
    ax[2].semilogy(fn,pn.T, lw=0.1, color=[0.7, 0.7, 1])

    ax[2].semilogy( fs, ps.T, lw=0.1, color=[1, 0.7, 0.7])

    # Median PSD
    ax[2].semilogy(fn,np.percentile(pn, 50, axis=0),lw=1,color="blue",label="noise" )

    ax[2].semilogy(fs,np.percentile(ps, 50, axis=0),lw=1,color="red", label="signal" )

    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("PSD (mV$^2$/Hz)")
    ax[2].set_title("PSD of the raw data")
    ax[2].legend()

    # Save / display
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path,dpi=300,bbox_inches="tight")

    plt.show()

    return fig, ax




def plot_emg_with_psd(sig, noise, pn, fn, ps, fs, fsamp, channels=np.arange(13, 25), save_path=None):

    # Time vectors for signal and noise
    t_sig = np.linspace(0,(sig.shape[1] - 1) / fsamp,sig.shape[1])

    t_noise = np.linspace(0,(noise.shape[1] - 1) / fsamp,noise.shape[1])

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # --------------------------------------------------
    # Signal
    # --------------------------------------------------

    for i, ch_idx in enumerate(channels):
        trace = sig[ch_idx, :]
        ax[0].plot(t_sig,trace + i,lw=0.5)

    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude + offset (mV)")
    ax[0].set_title("Fist EMG signals")
    ax[0].margins(y=0.05)


    # --------------------------------------------------
    # Noise
    # --------------------------------------------------

    for i, ch_idx in enumerate(channels):
        trace = noise[ch_idx, :]
        ax[1].plot(t_noise, trace + i, lw=0.5)

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude + offset (mV)")
    ax[1].set_title("Rest EMG signals")
    ax[1].margins(y=0.05)

    # --------------------------------------------------
    # PSD
    # --------------------------------------------------

    ax[2].semilogy(fn,np.ones(len(fn)) * 5.21e-8,color="darkgreen",linestyle="dashed", lw=1)

    ax[2].semilogy(fn,np.ones(len(fn)) * 20.83e-8,color="green",linestyle="dashed",lw=1)

    ax[2].semilogy(fn, np.ones(len(fn)) * 46.87e-8,color="orange",linestyle="dashed", lw=1)

    ax[2].semilogy(fn,np.ones(len(fn)) * 83.33e-8,color="red",linestyle="dashed",lw=1)

    # Individual PSDs
    ax[2].semilogy(fn,pn.T,lw=0.1,color=[0.7, 0.7, 1])

    ax[2].semilogy(fs, ps.T,lw=0.1,color=[1, 0.7, 0.7])

    # Median PSDs
    ax[2].semilogy(fn,np.percentile(pn, 50, axis=0),lw=1,color="blue",label="noise")

    ax[2].semilogy(fs, np.percentile(ps, 50, axis=0),lw=1,color="red",label="signal")

    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("PSD (mV$^2$/Hz)")
    #ax[2].set_ylim(1e-9, 1e-2)
    ax[2].legend()
    ax[2].set_title("PSD of the raw data")

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path,dpi=300,bbox_inches="tight")

    plt.show()
    
    return fig, ax