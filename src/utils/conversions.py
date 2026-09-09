import numpy as np
import pandas as pd

def convert_sparse_to_binary_format(firing_df, fs=2048, duration=20):
    """
    Convert sparse firing format to dense binary format.
    
    Parameters:
    firing_df: DataFrame with columns ['source_idx', 'discharge_samples', 'discharge_seconds', 'squared_amplitude'], as obtained from
    the Formento 2021 decomposition output (see simulation notebook for examples)
    fs: sampling frequency (Hz)
    duration: signal duration (seconds)
    
    Returns:
    binary_df: DataFrame in format similar to muapts_bin_train
    """
    
    # Create time index (same as target format)
    n_samples = int(fs * duration)
    time_index = np.arange(n_samples) / fs  # Time in seconds
    
    # Get unique motor units
    unique_mus = sorted(firing_df['source_idx'].unique())
    n_mus = len(unique_mus)
    print(f"Found {n_mus} motor units: {unique_mus}")
    
    # Create column names like 'MU0', 'MU1', etc.
    mu_columns = [f'MU{i}' for i in range(n_mus)]
    
    # Initialize binary DataFrame with zeros
    binary_df = pd.DataFrame(0, index=time_index, columns=mu_columns)
    
    # Fill in the firings
    for _, row in firing_df.iterrows():
        mu_idx = int(row['source_idx'])  # Original MU index
        firing_time = row['discharge_seconds']
        
        # Find closest time index
        closest_idx = np.argmin(np.abs(time_index - firing_time))
        
        # Set firing to 1
        if mu_idx < n_mus:  # Safety check
            col_name = f'MU{mu_idx}'
            if col_name in binary_df.columns:
                binary_df.iloc[closest_idx][col_name] = 1
    
    print(f"Converted to binary format: {binary_df.shape}")
    print(f"Columns: {binary_df.columns.tolist()}")
    print(f"Total firings: {binary_df.sum().sum()}")
    
    return binary_df

# Usage example:

# muapts_bin_train_converted = convert_sparse_to_binary_format(pd.DataFrame(firing))


def convert_hugh_to_binary_format(hugh_data_edit, fs=2048, duration=20):
    """
    Convert HUGH format data to binary firing format.
    
    Parameters:
    hugh_data_edit: dict with 'MUPulses' and 'MUIDs' keys
    fs: sampling frequency (Hz)  
    duration: signal duration (seconds)
    
    Returns:
    binary_df: DataFrame with binary firing patterns
    """
    
    # Create time index
    n_samples = int(fs * duration)
    time_index = np.arange(n_samples) / fs  # Time in seconds
    
    # Get motor unit information
    mu_pulses = hugh_data_edit["MUPulses"]  # Shape should be (1, n_mus) or similar
    mu_ids = hugh_data_edit["MUIDs"]
    
    # Determine number of motor units
    if len(mu_pulses.shape) == 2:
        n_mus = mu_pulses.shape[1]  # Assuming shape is (1, n_mus)
        mu_data_row = 0  # Use first row
    else:
        n_mus = len(mu_pulses)
        mu_data_row = None
    
    print(f"Found {n_mus} motor units")
    print(f"MU names: {[str(mu_ids[i][0]) for i in range(min(n_mus, len(mu_ids)))]}")
    
    # Create column names
    mu_columns = [f'MU{i}' for i in range(n_mus)]
    
    # Initialize binary DataFrame with zeros
    binary_df = pd.DataFrame(0, index=time_index, columns=mu_columns)
    
    # Fill in the firings for each motor unit
    for mu_idx in range(n_mus):
        try:
            # Get firing samples for this MU
            if mu_data_row is not None:
                firing_samples = mu_pulses[mu_data_row, mu_idx]
            else:
                firing_samples = mu_pulses[mu_idx]
            
            # Convert to array if needed
            if hasattr(firing_samples, '__len__') and len(firing_samples) > 0:
                firing_samples = np.array(firing_samples).flatten()
                
                # Convert samples to time indices
                firing_times = firing_samples / fs
                
                # Set binary values
                col_name = f'MU{mu_idx}'
                for firing_time in firing_times:
                    # Find closest time index
                    if firing_time < duration:  # Make sure within signal duration
                        closest_idx = np.argmin(np.abs(time_index - firing_time))
                        binary_df.iloc[closest_idx][col_name] = 1
                
                print(f"MU{mu_idx}: {len(firing_samples)} firings")
            else:
                print(f"MU{mu_idx}: No firings found")
                
        except Exception as e:
            print(f"Error processing MU{mu_idx}: {e}")
            continue
    
    print(f"\nConverted to binary format: {binary_df.shape}")
    print(f"Total firings: {binary_df.sum().sum()}")
    
    return binary_df

