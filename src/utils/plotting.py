import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as mpl_patches


import pandas as pd


import torch
from itertools import groupby


import matplotlib.pyplot as plt
import numpy as np



from matplotlib import rcParams
import seaborn as sns


def plot_muapts(muapts_train, muapts_bin_train, n_mu: int, FS: int = 2048 ):


    t = np.arange(muapts_train.shape[0]) / FS

    f, axes = plt.subplots(
        nrows=n_mu, sharex="all", figsize=(16, 8), layout="constrained"
    )
    f.suptitle(f"MUAPTs extracted from training set (subject {0})")
    f.supxlabel("Time [s]")
    f.supylabel("Amplitude [a.u.]")

    idx = 0
    for i in range(n_mu):
        spike_loc = np.flatnonzero(muapts_bin_train[f"MU{i}"])
        axes[idx].plot(t, muapts_train[f"MU{i}"])
        axes[idx].plot(spike_loc / FS, muapts_train[f"MU{i}"].iloc[spike_loc], "x")
        print(i)
        idx += 1

    plt.show()


    ################################################################# Functions originally from adaptive_bss_semg, can be found in the package in src ################################################################


def plot_signal(
    s: np.ndarray | pd.DataFrame | pd.Series | torch.Tensor,
    fs: float = 1.0,
    labels: np.ndarray | pd.Series | None = None,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Helper function to plot a signal with multiple channels, each in a different subplot.

    Parameters
    ----------
    s : ndarray or DataFrame or Series or Tensor
        Signal to plot:
        - if it's a NumPy array or PyTorch Tensor, the shape must be (n_channels, n_samples);
        - if it's a DataFrame or Series, the index and column(s) must represent
          the samples and the channel(s), respectively.
    fs : float, default=1.0
        Sampling frequency of the signal (relevant if s is a NumPy array).
    labels : ndarray or Series or None, default=None
        NumPy array or Series containing a label for each sample.
    title : str or None, default=None
        Title of the plot.
    x_label : str, default="Time [s]"
        Label for X axis.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Convert signal to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    elif isinstance(s, pd.Series):
        s_df = s.to_frame()
    else:
        s_arr = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        if len(s_arr.shape) == 1:
            s_arr = s_arr.reshape(1, -1)
        s_df = pd.DataFrame(s_arr.T, index=np.arange(s_arr.shape[1]) / fs)

    # Create figure with subplots and shared X axis
    n_cols = 1
    n_rows = s_df.shape[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )
    axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes
    # Set title and label of X and Y axes
    if title is not None:
        fig.suptitle(title, fontsize="xx-large")
    fig.supxlabel(x_label)
    fig.supylabel(y_label)

    # Plot signal
    if labels is not None:
        # Get label intervals
        labels_intervals = []
        labels_tmp = [
            list(group)
            for _, group in groupby(
                labels.reset_index().to_numpy().tolist(), key=lambda t: t[1]
            )
        ]
        for cur_label in labels_tmp:
            cur_label_start, cur_label_name = cur_label[0]
            cur_label_stop = cur_label[-1][0]
            labels_intervals.append((cur_label_name, cur_label_start, cur_label_stop))
        # Get set of unique labels
        label_set = set(map(lambda t: t[0], labels_intervals))
        # Create dictionary label -> color
        cmap = cm.get_cmap("plasma", len(label_set))
        color_dict = {lab: cmap(i) for i, lab in enumerate(label_set)}
        for i, ch_i in enumerate(s_df):
            for label, idx_from, idx_to in labels_intervals:
                axes[i].plot(
                    s_df[ch_i].loc[idx_from:idx_to],
                    color=color_dict[label],
                )
        # Add legend
        fig.legend(
            handles=[
                mpl_patches.Patch(color=c, label=lab) for lab, c in color_dict.items()
            ],
            loc="center right",
        )
    else:
        for i, ch_i in enumerate(s_df):
            axes[i].plot(s_df[ch_i])

    # Show or save plot
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_waveforms(
    wfs: np.ndarray,
    fs: float = 1.0,
    n_cols: int = 10,
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Function to plot MUAP waveforms.

    Parameters
    ----------
    wfs : ndarray
        MUAP waveforms with shape (n_mu, n_channels, waveform_len).
    fs : float, default=1.0
        Sampling frequency of the signal.
    n_cols : int, default=10
        Number of columns for subplots.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    n_ch = wfs.shape[1]
    assert (
        n_ch % n_cols == 0
    ), "The number of channels must be divisible for the number of columns."
    n_rows = n_ch // n_cols
    t = np.arange(wfs.shape[2]) * 1000 / fs

    f, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex="all",
        sharey="all",
        figsize=fig_size,
        layout="constrained",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            axes[i, j].set_title(f"Ch{idx}")
            axes[i, j].plot(t, wfs[:, idx].T)
            axes[i, j].axvline(t[wfs.shape[2] // 2], color="k", linestyle="--")

    f.suptitle("MUAP waveforms")
    f.supxlabel("Time [ms]")
    f.supylabel(y_label)

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


"""
Publication-ready plotting utilities for EMG analysis.
Follows Nature/Science style guidelines for clear, professional figures.
"""


def setup_publication_style():
    """
    Configure matplotlib with publication-ready settings.
    Based on Nature and Science journal style guidelines.
    """
    # Set the style parameters
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 8
    rcParams['axes.labelsize'] = 9
    rcParams['axes.titlesize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 8
    rcParams['figure.titlesize'] = 11
    
    # Line widths
    rcParams['axes.linewidth'] = 0.8
    rcParams['lines.linewidth'] = 1.2
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    
    # Remove top and right spines (Nature style)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    
    # Grid style
    rcParams['grid.linewidth'] = 0.5
    rcParams['grid.alpha'] = 0.3
    
    # Use colorblind-friendly palette
    rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("colorblind"))


def get_color_palette(n_colors=10):
    """
    Get a colorblind-friendly color palette.
    
    Parameters:
    -----------
    n_colors : int
        Number of colors needed
        
    Returns:
    --------
    list : List of RGB color tuples
    """
    if n_colors <= 10:
        return sns.color_palette("colorblind", n_colors)
    else:
        # For more colors, use a combination of palettes
        base = sns.color_palette("colorblind", 10)
        extra = sns.color_palette("husl", n_colors - 10)
        return base + extra
    


    