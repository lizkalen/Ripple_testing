import pandas as pd
import logging
import numpy as np
import os
import glob
import re
# -----------------------------



# 1) Loading utilities
# -----------------------------
def load_channel_map(channels_tsv: str) -> pd.DataFrame:
    """
    Load the cyberglove channel map TSV with columns:
    name, type, units, sampling_frequency, tracked_point, component (optional)
    """
    df = pd.read_csv(channels_tsv, sep="\t")
    # normalize column names for safety
    df.columns = [c.strip() for c in df.columns]
    return df

# ---- 2) Load *unnamed* timeseries and align to map order ----
def load_and_align_timeseries_with_map(
    data_path: str,
    ch_map: pd.DataFrame,
    time_col_index: int = 0,
    force_no_header: bool = True,   # set True if your CSV truly has no header row
    sep_candidates=(",", "\t", ";", r"\s+"),
):
    """
    Loads the motion file assuming column 0 is time and remaining columns are
    channels in the same order as ch_map['name'].

    If lengths differ, the shorter length is used, and a warning is printed.
    """
    # 1) Load the file (robust delimiter, optional header)
    df = None
    if force_no_header:
        for sep in sep_candidates:
            try:
                df = pd.read_csv(data_path, sep=sep, header=None, engine="python")
                if df.shape[1] >= 2:
                    break
            except Exception:
                pass
    else:
        # Try with a header first, else fall back to no header
        for sep in sep_candidates:
            try:
                tmp = pd.read_csv(data_path, sep=sep, engine="python")
                # If first row looks numeric across all columns, it's probably not a header
                if pd.to_numeric(tmp.iloc[0], errors="coerce").notna().all():
                    # reload as no header
                    df = pd.read_csv(data_path, sep=sep, header=None, engine="python")
                else:
                    df = tmp
                if df.shape[1] >= 2:
                    break
            except Exception:
                pass

    if df is None:
        raise ValueError("Could not read timeseries file with common delimiters.")

    # 2) Build target column names: time + ordered channel names from the map
    map_names = ch_map["name"].astype(str).tolist()
    n_file_channels = df.shape[1] - 1  # minus time col
    n_map_channels = len(map_names)

    n_used = min(n_file_channels, n_map_channels)
    if n_file_channels != n_map_channels:
        print(f"[WARN] File has {n_file_channels} channels; map has {n_map_channels}. Using first {n_used}.")

    target_cols = ["time"] + map_names[:n_used]

    # 3) Trim/align the dataframe to exactly 1 + n_used columns
    # Keep `time` at index time_col_index, then the next n_used columns as channels
    if time_col_index != 0:
        # Move time column to col 0 if it’s not already there
        cols = df.columns.tolist()
        cols[0], cols[time_col_index] = cols[time_col_index], cols[0]
        df = df[cols]

    # Slice to time + first n_used channel columns
    used_df = df.iloc[:, : (1 + n_used)].copy()
    used_df.columns = target_cols

    return used_df

# ---- 3) Convenience: get time + signals split ----
def split_time_signals(aligned_df: pd.DataFrame):
    t = aligned_df["time"].to_numpy()
    signals = aligned_df.drop(columns=["time"])
    return t, signals


def load_timeseries(data_path: str) -> pd.DataFrame:
    """
    Load the time-series. Auto-detect delimiter (comma/semicolon/tab).
    """
    # Try comma, tab, semicolon
    for sep in [",", "\t", ";", r"\s+"]:
        try:
            df = pd.read_csv(data_path, sep=sep, engine="python")
            # Heuristic: at least 3 columns to be useful
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    raise ValueError("Could not read timeseries file with common delimiters.")

def get_time_vector(df: pd.DataFrame, fs: float, time_col_candidates=("time", "Time", "t", "sec", "seconds")) -> np.ndarray:
    """
    Return a time vector. Prefer a present time column; else build from fs.
    """
    for c in time_col_candidates:
        if c in df.columns:
            return df[c].to_numpy()
    n = len(df)
    return np.arange(n) / float(fs if fs and fs > 0 else 100.0)




def load_sessantaquattro_csv(folder_path: str) -> dict:
    """
    Load CSV data from files containing 'sessantaquattro' in the name 
    and extract task identifiers.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing the CSV files.
        
    Returns
    -------
    dict
        Dictionary with task identifiers (two letters after 'sub_NN_task-') 
        as keys and loaded DataFrame objects as values.
    """
    
    # Find all CSV files with 'sessantaquattro' in the name
    pattern = os.path.join(folder_path, "*sessantaquattro*.csv")
    # Additional logging for debugging
    logging.info(f"Searching for sessantaquattro CSV files in {folder_path}")
    # Print the files found for debugging
    print(f"Found files: {glob.glob(folder_path)}")
    
    csv_files = glob.glob(pattern)
    
    
    if not csv_files:
        logging.warning(f"No CSV files with 'sessantaquattro' found in {folder_path}")
        return {}
    
    # Dictionary to store results
    data_dict = {}
    
    # Process each file
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Extract task identifier (two letters after 'sub_NN_task-')
        match = re.search(r'sub_\d+_task-(\w{2})', filename)
        if match:
            task_id = match.group(1)
        else:
            logging.warning(f"Could not extract task ID from {filename}, using full filename")
            task_id = filename
        
        # Load CSV data
        df = pd.read_csv(file_path)
        
        # Store in dictionary
        data_dict[task_id] = df
        logging.info(f"Loaded data for task '{task_id}' from {filename}")
    
    return data_dict






def load_kyn_csv(folder_path: str) -> dict:
    """
    Load CSV data from files containing 'cyberglove' in the name 
    and extract task identifiers.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing the CSV files.
        
    Returns
    -------
    dict
        Dictionary with task identifiers (two letters after 'sub_NN_task-') 
        as keys and loaded DataFrame objects as values.
    """
    
    # Find all CSV files with 'cyberglove' in the name
    pattern = os.path.join(folder_path, "*cyberglove*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        logging.warning(f"No CSV files with 'cyberglove' found in {folder_path}")
        return {}
    
    # Dictionary to store results
    data_dict = {}
    
    # Process each file
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Extract task identifier (two letters after 'sub_NN_task-')
        match = re.search(r'sub_\d+_task-(\w{2})', filename)
        if match:
            task_id = match.group(1)
        else:
            logging.warning(f"Could not extract task ID from {filename}, using full filename")
            task_id = filename
        
        # Load CSV data
        df = pd.read_csv(file_path)
        
        # Store in dictionary
        data_dict[task_id] = df
        logging.info(f"Loaded data for task '{task_id}' from {filename}")
    
    return data_dict




