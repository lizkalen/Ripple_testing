import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle






import sys
import os
sys.path.append(os.path.abspath('..'))

from src.formento.decomposition import EmgDecomposition
from src.formento.parameters import EmgDecompositionParams

import src.utils.conversions as conv
import src.utils.validation as val


def analyze_multiple_decompositions(bin_trains, tolerance=0.001, min_match_percentage=0.3, n_iterations = 5):
    """
    Analyze multiple decomposition runs to find consistent motor units
    
    Parameters:

    bin_trains: list of DataFrames from convert_sparse_to_binary_format
    each dataframe correspond to the binary firings of one decomposition run
    tolerance: float, time tolerance for matching firings: meaning two firings within this time are considered a match
    min_match_percentage: float, minimum percentage of matching firings to consider two MUs as matching MUs
    n_iterations: int, number of decomposition runs


    Returns:

    all_run_timings: list of lists of arrays, each sublist corresponds to a run and contains arrays of firing times for each MU
    consistency_matrix: 2D array, matrix where entry (i,j) is the number of matching MUs between run i and run j
    match_details: array of dicts, Each dictionary in your list represents a single motor unit that was detected consistently across multiple decomposition runs.
                For example: {'group_id': 0, 'units': [(4, 9), (3, 1), (2, 3), (0, 2), (1, 6)], 'n_runs': 5, 'percentage': 100.0}
                means that motor unit group 0 was found in runs (i) 0,1,2,3,4 with respective MU indices (j) 2,6,3,1,9 and it appeared in all 5 runs (100%).
    """
    n_runs = len(bin_trains)
    
    # Extract firing times from each run
    all_run_timings = []
    
    for run_idx, muapts_bin in enumerate(bin_trains):
        DC_timings = []
        for col in muapts_bin.columns:
            # Find where this MU fired (non-zero values)
            firing_mask = muapts_bin[col] > 0
            # Get the time indices where firings occurred
            firing_times = muapts_bin.index[firing_mask].values
            DC_timings.append(firing_times)
        
        all_run_timings.append(DC_timings)
        print(f"Run {run_idx}: {len(DC_timings)} motor units detected")
    
    # Compare all pairs of runs
    consistency_matrix = np.zeros((n_runs, n_runs))
    match_details = {}
    
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            matches = val.find_matching_motor_units(
                all_run_timings[i], 
                all_run_timings[j], 
                tolerance=tolerance, 
                min_match_percentage=min_match_percentage
            )
            consistency_matrix[i,j] = len(matches)
            consistency_matrix[j,i] = len(matches)
            match_details[(i,j)] = matches
            
            print(f"\nRun {i} vs Run {j}: {len(matches)} matches")
            for match in matches:
                print(f"  Run{i} MU{match[0]} ↔ Run{j} MU{match[1]} ({match[2]:.1%})")
    
    return all_run_timings, consistency_matrix, match_details

def find_consensus_motor_units(match_details, all_timings, min_runs=3):
    """
    Find motor units that appear in at least min_runs
    """
    n_runs = len(all_timings)
    
    # Build groups of matching units
    unit_groups = []
    
    for (run1, run2), matches in match_details.items():
        for mu1_idx, mu2_idx, match_pct in matches:
            # Find if this match belongs to an existing group
            added_to_group = False
            
            for group in unit_groups:
                # Check if either unit is already in this group
                if any((r == run1 and mu == mu1_idx) or (r == run2 and mu == mu2_idx) 
                       for r, mu in group):
                    group.add((run1, mu1_idx))
                    group.add((run2, mu2_idx))
                    added_to_group = True
                    break
            
            if not added_to_group:
                # Create new group
                unit_groups.append({(run1, mu1_idx), (run2, mu2_idx)})
    
    # Merge overlapping groups
    merged = True
    while merged:
        merged = False
        for i in range(len(unit_groups)):
            for j in range(i+1, len(unit_groups)):
                if unit_groups[i] & unit_groups[j]:  # If groups share any units
                    unit_groups[i] |= unit_groups[j]
                    unit_groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    # Find groups present in enough runs
    consensus_units = []
    for group_idx, group in enumerate(unit_groups):
        runs_present = len(set(run for run, _ in group))
        if runs_present >= min_runs:
            consensus_units.append({
                'group_id': group_idx,
                'units': list(group),
                'n_runs': runs_present,
                'percentage': runs_present / n_runs * 100
            })
    
    return consensus_units

def plot_consensus_units(consensus_units, all_timings, time_window=(0, 30), n_iterations=5):
    """
    Plot firing patterns of consensus motor units across runs
    """
    n_consensus = len(consensus_units)
    if n_consensus == 0:
        print("No consensus units to plot")
        return
    
    fig, axes = plt.subplots(n_consensus, 1, figsize=(15, 3*n_consensus))
    if n_consensus == 1:
        axes = [axes]
    
    for idx, unit_info in enumerate(consensus_units):
        ax = axes[idx]
        
        # Plot firings from each run for this consensus unit
        colors = plt.cm.tab10(np.linspace(0, 1, n_iterations))
        
        for run_idx, (run, mu) in enumerate(unit_info['units']):
            firing_times = all_timings[run][mu]
            times_in_window = firing_times[(firing_times >= time_window[0]) & 
                                          (firing_times <= time_window[1])]
            
            ax.scatter(times_in_window, [run]*len(times_in_window), 
                      color=colors[run], s=20, alpha=0.7,
                      label=f'Run {run}, MU {mu}')
        
        ax.set_title(f"Consensus MU {unit_info['group_id']} "
                    f"(appears in {unit_info['n_runs']}/5 runs)")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Run')
        ax.set_xlim(time_window)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_consensus_dataframe(consensus, bin_trains):

    # TBD
    # Create consensus dataframe by picking first instance from each group, for now just picks the first MU from each group


    consensus_df = pd.concat([
        bin_trains[unit['units'][0][0]][f"MU{unit['units'][0][1]}"].rename(f"Consensus_MU{unit['group_id']}")
        for unit in consensus
    ], axis=1)


    return consensus_df


################################################################################################Wrapper Function####################################################################################################################



def consensus_analysis(emg_signal: np.array, fs: int, tolerance=0.001, min_match_percentage=0.3, min_runs=11, n_iterations=20, plot=True, from_pkl=False, pkl_path='decomposition_results.pkl', 
                       cuda = True):

    """

    Wrapper function to analyze multiple decompositions and find consensus motor units
    
    Parameters:

    emg_signal: np.array, raw EMG signal to decompose  of shape n_channels x n_samples
    fs: int, sampling frequency of the EMG signal
    tolerance: float, time tolerance for matching firings: meaning two firings within this time are considered a match
    min_match_percentage: float, minimum percentage of matching firings to consider two MUs as matching MUs
    min_runs: int, minimum number of runs a MU must appear in to be considered consensus
    n_iterations: int, number of decomposition runs
    plot: bool, whether to plot the consistency matrix and consensus units
    from_pkl: bool, whether to load previous results from pickle file or run new decompositions
    pkl_path: str, path to pickle file for loading/saving results
    cuda: bool, whether to use CUDA for decomposition if available

    Returns:
    decompositions: list of EmgDecomposition objects from each run (None if loaded from pickle, as they cannot be saved at the moment)
    consensus_df: DataFrame of consensus motor unit firings in binary format
    consensus: list of dicts, each dict contains info about a consensus motor unit
    all_timings: list of lists of arrays, firing times from all runs
    match_details: array of dicts, details of matching motor units between runs
    consistency_matrix: 2D array, matrix of number of matching MUs between runs


    """


    if from_pkl:

        print("INFO: Loading previous decomposition results from pickle...")


        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)

        decompositions = None
        firings = results['firings']
        bin_trains = results['bin_trains']
        consensus = results['consensus']
        consensus_df = results['consensus_df']
        n_iterations = results['n_iterations']




    elif not from_pkl:

        print("INFO: Starting new decomposition runs...")


        decompositions = []
        firings = []
        bin_trains = []



        for i in range(n_iterations):

            print(f"Decomposition iteration {i+1}/{n_iterations}")
            decomp = EmgDecomposition(params=EmgDecompositionParams(sampling_rate= fs), use_cuda=cuda)
            decompositions.append(decomp)

            firing = decomp.decompose(emg_signal)
            firings.append(firing)

            print(pd.DataFrame(firing))

            muapts_bin_train = conv.convert_sparse_to_binary_format(pd.DataFrame(firing), duration=30, fs=fs)

            bin_trains.append(muapts_bin_train)


            all_timings, consistency_matrix, match_details = analyze_multiple_decompositions(
            bin_trains, tolerance, min_match_percentage, n_iterations
        )
        if plot:

            # Visualize consistency matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(consistency_matrix, cmap='Blues')
            plt.colorbar(label='Number of matching MUs')
            plt.xlabel('Run number')
            plt.ylabel('Run number')
            plt.title('Motor Unit Consistency Across 5 Decomposition Runs')
            for i in range(n_iterations):
                for j in range(n_iterations):
                    plt.text(j, i, f'{int(consistency_matrix[i,j])}', 
                            ha='center', va='center', 
                    color='white' if consistency_matrix[i,j] > consistency_matrix.max()/2 else 'black')
            plt.show()

        consensus= find_consensus_motor_units(match_details, all_timings, min_runs)
        
        print(f"\nConsensus Motor Units (appearing in ≥{min_runs} runs):")
        print(f"Found {len(consensus)} consistent motor units\n")


        if plot:
            plot_consensus_units(consensus, all_timings, n_iterations=n_iterations)

        consensus_df = create_consensus_dataframe(consensus, bin_trains)

        # Save everything
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                # 'decompositions': decompositions,
                'firings': firings,
                'bin_trains': bin_trains,
                'consensus': consensus,
                'consensus_df': consensus_df,
                'n_iterations': n_iterations
            }, f)
            
    else:
        raise ValueError("What did you put for from_pkl?")

    return decompositions, consensus_df, consensus, all_timings, match_details, consistency_matrix
