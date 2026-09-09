import numpy as np
import matplotlib.pyplot as plt


def calculate_roa(gt_spikes, dc_spikes, tolerance=0.001):
    """
    OPTIMIZED: Calculate Rate-of-Agreement (RoA) between two spike trains using vectorized operations.

    RoA = (C / (C + O + I)) · 100%
    where:
    - C: number of spikes identified in both (common/matched)
    - O: number of spikes only in GT (GT only)
    - I: number of spikes only in decomposition (DC only)

    Parameters:
    gt_spikes: numpy array of ground truth spike times
    dc_spikes: numpy array of decomposed spike times
    tolerance: time tolerance for matching spikes (in seconds)

    Returns:
    roa: Rate-of-Agreement percentage (0-100)
    matched_counts: dict with 'C', 'O', 'I' counts for debugging
    """
    if len(gt_spikes) == 0 and len(dc_spikes) == 0:
        return 100.0, {'C': 0, 'O': 0, 'I': 0}

    if len(gt_spikes) == 0:
        return 0.0, {'C': 0, 'O': 0, 'I': len(dc_spikes)}

    if len(dc_spikes) == 0:
        return 0.0, {'C': 0, 'O': len(gt_spikes), 'I': 0}

    # Convert to numpy arrays
    gt_spikes = np.array(gt_spikes)
    dc_spikes = np.array(dc_spikes)

    # OPTIMIZED: Vectorized matching using broadcasting
    # Create distance matrix: shape (len(gt_spikes), len(dc_spikes))
    distances = np.abs(gt_spikes[:, np.newaxis] - dc_spikes[np.newaxis, :])

    # Find minimum distance for each GT spike to any DC spike
    min_distances_per_gt = np.min(distances, axis=1)
    min_indices_per_gt = np.argmin(distances, axis=1)

    # Find matches within tolerance
    gt_matched = np.zeros(len(gt_spikes), dtype=bool)
    dc_matched = np.zeros(len(dc_spikes), dtype=bool)

    # Process GT spikes in order, ensuring one-to-one matching
    for i in range(len(gt_spikes)):
        if min_distances_per_gt[i] <= tolerance:
            dc_idx = min_indices_per_gt[i]
            if not dc_matched[dc_idx]:
                gt_matched[i] = True
                dc_matched[dc_idx] = True

    # Calculate counts
    C = int(np.sum(gt_matched))  # Common spikes (matched)
    O = int(np.sum(~gt_matched))  # GT only spikes
    I = int(np.sum(~dc_matched))  # DC only spikes

    # Calculate RoA
    total = C + O + I
    roa = (C / total * 100) if total > 0 else 0.0

    return roa, {'C': C, 'O': O, 'I': I}


def find_matching_motor_units(GT_timings, DC_timings, tolerance=0.001, min_match_percentage=0.3):
    """
    Compare GT and DC timings to find matching motor units.
    First screens using bidirectional matching, then calculates RoA for qualifying pairs.
    
    Parameters:
    GT_timings: list of arrays with ground truth firing times
    DC_timings: list of lists with decomposed firing times
    tolerance: time tolerance for matching spikes (in seconds)
    min_match_percentage: minimum percentage of matches required for initial screening
    
    Returns:
    matches: list of tuples (GT_idx, DC_idx, roa_percentage, match_percentage_ij, match_percentage_ji)
    """
    matches = []
    
    for gt_idx, gt_spikes in enumerate(GT_timings):
        for dc_idx, dc_spikes in enumerate(DC_timings):
            # Convert DC timings to numpy array for easier computation
            dc_spikes_array = np.array(dc_spikes)
            
            if len(gt_spikes) == 0 or len(dc_spikes) == 0:
                continue
            
            # Initial screening: Count matches in both directions (original approach)
            matched_count_ij = 0
            for gt_spike in gt_spikes:
                time_diffs = np.abs(dc_spikes_array - gt_spike)
                min_diff = np.min(time_diffs)
                if min_diff <= tolerance:
                    matched_count_ij += 1

            matched_count_ji = 0
            for dc_spike in dc_spikes:
                time_diffs = np.abs(gt_spikes - dc_spike)
                min_diff = np.min(time_diffs)
                if min_diff <= tolerance:
                    matched_count_ji += 1
            
            match_percentage_ij = matched_count_ij / len(gt_spikes)
            match_percentage_ji = matched_count_ji / len(dc_spikes)

            # Check if both directions meet the threshold (initial screening)
            if match_percentage_ij >= min_match_percentage and match_percentage_ji >= min_match_percentage:
                # Only now calculate RoA for qualifying pairs
                roa, counts = calculate_roa(gt_spikes, dc_spikes, tolerance)
                
                print(f"GT MU {gt_idx} and DC MU {dc_idx} match: "
                      f"{match_percentage_ij:.1%} (GT->DC), {match_percentage_ji:.1%} (DC->GT) "
                      f"-> RoA: {roa:.1f}% (C:{counts['C']}, O:{counts['O']}, I:{counts['I']})")
                
                matches.append((gt_idx, dc_idx, roa, match_percentage_ij, match_percentage_ji))
    
    return matches


def validate_decomposition_results(ground_truth, muapts_bin_train, fs=2048, GT_format="Dataframe"):
    """
    Main validation function using RoA metric.
    """
    # Convert ground truth firing times from samples to seconds
    GT_timings = []
    if GT_format == "Dataframe":
        if ground_truth is not None:
            print("DEBUG: ground_truth info:", ground_truth.info())
            for col in ground_truth.columns:
                firing_mask = ground_truth[col] > 0
                if firing_mask.sum() == 0:
                    print(f"Warning: No firings detected for {col}")
                    firing_times = []
                else:
                    firing_times = ground_truth.index[firing_mask].values
                GT_timings.append(firing_times)
    else:
        try:    
            for mu in range(ground_truth['n_mus']):
                firing_samples = ground_truth['firing_times'][mu]
                firing_seconds = firing_samples / fs
                GT_timings.append(firing_seconds)
        except:
            raise ValueError("Error processing ground truth firing times")
    
    # Extract decomposed firing times from pandas DataFrame
    DC_timings = []
    if muapts_bin_train is not None:
        print("DEBUG: muapts_bin_train info:", muapts_bin_train.info())
        for col in muapts_bin_train.columns:
            firing_mask = muapts_bin_train[col] > 0
            if firing_mask.sum() == 0:
                print(f"Warning: No firings detected for {col}")
                firing_times = []
            else:
                firing_times = muapts_bin_train.index[firing_mask].values
            DC_timings.append(firing_times)
    
    # Find matching motor units
    matches = find_matching_motor_units(GT_timings, DC_timings)
    
    # Print results
    print("\n" + "="*50)
    print("DECOMPOSITION VALIDATION RESULTS")
    print("="*50)
    print(f"Ground truth motor units: {len(GT_timings)}")
    print(f"Decomposed motor units: {len(DC_timings)}")
    
    print("\nMatching Motor Units (GT_idx, DC_idx, RoA%, GT->DC%, DC->GT%):")
    for match in matches:
        print(f"GT MU {match[0]} matches DC MU {match[1]} with {match[2]:.1f}% RoA "
              f"({match[3]:.1%} GT->DC, {match[4]:.1%} DC->GT)")
    
    print(f"\nTotal matches found: {len(matches)}")
    
    # Calculate overall performance
    correctly_identified = len(matches)
    missed_mus = len(GT_timings) - correctly_identified
    false_positives = len(DC_timings) - correctly_identified

    print(f"Correctly identified: {correctly_identified}/{len(GT_timings)}")
    print(f"Missed MUs: {missed_mus}")
    print(f"False positives: {false_positives}")

    if len(GT_timings) > 0:
        sensitivity = correctly_identified / len(GT_timings)
        print(f"Sensitivity (recall): {sensitivity:.2%}")
    
    return matches, GT_timings, DC_timings


def check_time_shifted_matches(GT_timings, DC_timings, max_shift=0.01, shift_step=0.0005, min_match_percentage=0.3):
    """
    Check if DC firing patterns match GT patterns when time-shifted.
    Now (13/10/25) Uses vectorized operations and early termination for speed.

    Parameters:
    GT_timings: list of GT firing time arrays
    DC_timings: list of DC firing time arrays
    max_shift: maximum time shift to test (seconds)
    shift_step: step size for shift search (seconds)
    min_match_percentage: minimum percentage of matches required for initial screening

    Returns:
    best_matches: list of (gt_idx, dc_idx, best_shift, best_roa, match_ij, match_ji)
    dc_best_matches: dict mapping dc_idx to best match info
    match_matrix: n_dc x n_gt structured array with all match information
    """
    print("\n" + "="*60)
    print("CHECKING TIME-SHIFTED MATCHES (OPTIMIZED)")
    print("="*60)
    print(f"Testing shifts from -{max_shift*1000:.1f}ms to +{max_shift*1000:.1f}ms")
    print(f"Step size: {shift_step*1000:.1f}ms")

    # Generate shift values to test
    shifts = np.arange(-max_shift, max_shift + shift_step, shift_step)
    tolerance = 0.001

    best_matches = []
    dc_best_matches = {}  # dc_idx -> (gt_idx, shift, roa, match_ij, match_ji, counts)

    # Initialize match matrix: n_dc x n_gt with detailed match information
    n_dc = len(DC_timings)
    n_gt = len(GT_timings)

    # Create structured array to store all match information
    match_matrix_dtype = [
        ('roa', 'f4'),                    # Rate of Agreement
        ('best_shift', 'f4'),             # Best time shift in seconds
        ('match_ij', 'f4'),               # GT->DC match percentage
        ('match_ji', 'f4'),               # DC->GT match percentage
        ('C', 'i4'),                      # Common spikes
        ('O', 'i4'),                      # GT only spikes
        ('I', 'i4'),                      # DC only spikes
        ('meets_threshold', 'bool')       # Whether it meets min_match_percentage
    ]
    match_matrix = np.zeros((n_dc, n_gt), dtype=match_matrix_dtype)

    total_comparisons = len(GT_timings) * len(DC_timings)
    comparison_count = 0

    for gt_idx, gt_times in enumerate(GT_timings):
        if len(gt_times) == 0:
            continue

        # Convert to numpy array once
        gt_array = np.array(gt_times)

        for dc_idx, dc_times in enumerate(DC_timings):
            comparison_count += 1
            if comparison_count % 10 == 0 or comparison_count == total_comparisons:
                print(f"Progress: {comparison_count}/{total_comparisons} comparisons ({comparison_count/total_comparisons*100:.1f}%)\033[K", end='\r')

            if len(dc_times) == 0:
                continue

            # Convert to numpy array once
            dc_array = np.array(dc_times)

            best_shift = 0
            best_match_ij = 0
            best_match_ji = 0
            best_roa = 0
            best_counts = None
            absolute_best_shift = 0
            absolute_best_match_ij = 0
            absolute_best_match_ji = 0
            absolute_best_roa = 0
            absolute_best_counts = None

            # Track the best raw match score to ensure we always have a fallback
            best_raw_score = 0
            best_raw_shift = 0
            best_raw_match_ij = 0
            best_raw_match_ji = 0

            # Test each time shift
            for shift in shifts:
                shifted_dc_array = dc_array + shift

                # OPTIMIZED: Vectorized distance calculation for GT->DC matches
                # Create distance matrix: shape (len(gt_times), len(dc_times))
                distances_ij = np.abs(gt_array[:, np.newaxis] - shifted_dc_array[np.newaxis, :])
                min_distances_ij = np.min(distances_ij, axis=1)
                matched_count_ij = np.sum(min_distances_ij <= tolerance)

                # OPTIMIZED: Vectorized distance calculation for DC->GT matches
                distances_ji = np.abs(shifted_dc_array[:, np.newaxis] - gt_array[np.newaxis, :])
                min_distances_ji = np.min(distances_ji, axis=1)
                matched_count_ji = np.sum(min_distances_ji <= tolerance)

                match_percentage_ij = matched_count_ij / len(gt_times)
                match_percentage_ji = matched_count_ji / len(shifted_dc_array)

                # Track best raw match for fallback
                raw_score = match_percentage_ij + match_percentage_ji
                if raw_score > best_raw_score:
                    best_raw_score = raw_score
                    best_raw_shift = shift
                    best_raw_match_ij = match_percentage_ij
                    best_raw_match_ji = match_percentage_ji

                # Early termination: skip RoA calculation if match percentages are too low
                if match_percentage_ij < min_match_percentage * 0.5 and match_percentage_ji < min_match_percentage * 0.5:
                    continue

                # Calculate RoA for promising matches
                roa, counts = calculate_roa(gt_times, shifted_dc_array, tolerance=tolerance)

                # Update absolute best match for this GT-DC pair (no threshold)
                if roa > absolute_best_roa:
                    absolute_best_roa = roa
                    absolute_best_shift = shift
                    absolute_best_match_ij = match_percentage_ij
                    absolute_best_match_ji = match_percentage_ji
                    absolute_best_counts = counts

                # Check if both directions meet threshold for qualified matches
                if match_percentage_ij >= min_match_percentage and match_percentage_ji >= min_match_percentage:
                    # Update best match for this GT-DC pair (with threshold)
                    if roa > best_roa:
                        best_roa = roa
                        best_shift = shift
                        best_match_ij = match_percentage_ij
                        best_match_ji = match_percentage_ji
                        best_counts = counts

            # If no RoA was calculated due to early termination, calculate for the best raw match
            if absolute_best_counts is None:
                shifted_dc_array = dc_array + best_raw_shift
                absolute_best_roa, absolute_best_counts = calculate_roa(gt_times, shifted_dc_array, tolerance=tolerance)
                absolute_best_shift = best_raw_shift
                absolute_best_match_ij = best_raw_match_ij
                absolute_best_match_ji = best_raw_match_ji

            # Store match information in matrix (dc_idx x gt_idx)
            match_matrix[dc_idx, gt_idx]['roa'] = absolute_best_roa
            match_matrix[dc_idx, gt_idx]['best_shift'] = absolute_best_shift
            match_matrix[dc_idx, gt_idx]['match_ij'] = absolute_best_match_ij
            match_matrix[dc_idx, gt_idx]['match_ji'] = absolute_best_match_ji
            match_matrix[dc_idx, gt_idx]['C'] = absolute_best_counts['C']
            match_matrix[dc_idx, gt_idx]['O'] = absolute_best_counts['O']
            match_matrix[dc_idx, gt_idx]['I'] = absolute_best_counts['I']
            match_matrix[dc_idx, gt_idx]['meets_threshold'] = (
                absolute_best_match_ij >= min_match_percentage and
                absolute_best_match_ji >= min_match_percentage
            )

            # Track absolute best match for each DC motor unit
            if dc_idx not in dc_best_matches or absolute_best_roa > dc_best_matches[dc_idx][2]:
                dc_best_matches[dc_idx] = (gt_idx, absolute_best_shift, absolute_best_roa,
                                          absolute_best_match_ij, absolute_best_match_ji,
                                          absolute_best_counts)

            # Report if good match found
            if best_roa > 0:  # Any qualifying match found
                print(f"\nGT MU {gt_idx} <-> DC MU {dc_idx}: {best_match_ij:.1%} (GT->DC), {best_match_ji:.1%} (DC->GT) "
                      f"at {best_shift*1000:+.1f}ms shift -> RoA: {best_roa:.1f}% "
                      f"(C:{best_counts['C']}, O:{best_counts['O']}, I:{best_counts['I']})")
                best_matches.append((gt_idx, dc_idx, best_shift, best_roa, best_match_ij, best_match_ji))

    print(f"\n{len(best_matches)} time-shifted matches found above {min_match_percentage*100}% bidirectional threshold")

    # Print best matches for all DC motor units (even those below threshold)
    print("\n" + "="*60)
    print("BEST MATCH FOR EACH DC MOTOR UNIT (regardless of threshold)")
    print("="*60)
    for dc_idx in sorted(dc_best_matches.keys()):
        gt_idx, shift, roa, match_ij, match_ji, counts = dc_best_matches[dc_idx]
        threshold_met = roa > 0 and (dc_idx, gt_idx) in [(m[1], m[0]) for m in best_matches]
        status = "[MATCHED]" if threshold_met else "[BELOW THRESHOLD]"
        print(f"{status} DC MU {dc_idx} -> GT MU {gt_idx}: "
              f"{match_ij:.1%} (GT->DC), {match_ji:.1%} (DC->GT) "
              f"at {shift*1000:+.1f}ms shift -> RoA: {roa:.1f}% "
              f"(C:{counts['C']}, O:{counts['O']}, I:{counts['I']})")

    return best_matches, dc_best_matches, match_matrix


def plot_shifted_comparison(GT_timings, DC_timings, best_matches, time_window=(0, 30)):
    """
    Plot GT vs DC firings with best time shifts applied.
    """
    if len(best_matches) == 0:
        print("No matches to plot")
        return
        
    fig, axes = plt.subplots(len(best_matches), 1, figsize=(15, 3*len(best_matches)))
    if len(best_matches) == 1:
        axes = [axes]
    
    for i, (gt_idx, dc_idx, best_shift, best_roa, match_ij, match_ji) in enumerate(best_matches):
        ax = axes[i]
        
        # Get data
        gt_times = GT_timings[gt_idx]
        dc_times = np.array(DC_timings[dc_idx]) + best_shift  # Apply best shift
        
        # Filter to time window
        gt_window = gt_times[(gt_times >= time_window[0]) & (gt_times <= time_window[1])]
        dc_window = dc_times[(dc_times >= time_window[0]) & (dc_times <= time_window[1])]
        
        # Plot
        ax.scatter(gt_window, [1]*len(gt_window), alpha=0.8, s=50, 
                  label=f'GT MU {gt_idx}', color='blue')
        ax.scatter(dc_window, [0.5]*len(dc_window), alpha=0.8, s=50, 
                  label=f'DC MU {dc_idx} (shifted {best_shift*1000:+.1f}ms)', 
                  color='red', marker='x')
        
        ax.set_title(f'GT MU {gt_idx} vs DC MU {dc_idx}: {best_roa:.1f}% RoA, '
                    f'{best_shift*1000:+.1f}ms shift '
                    f'({match_ij:.1%} GT->DC, {match_ji:.1%} DC->GT)')
        ax.set_xlim(time_window)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def debug_firing_times(GT_timings, DC_timings, max_time_display=5.0):
    """
    Debug function to examine firing times in detail.
    """
    print("\n" + "="*60)
    print("DEBUGGING FIRING TIMES")
    print("="*60)
    
    print(f"\nGROUND TRUTH ({len(GT_timings)} MUs):")
    for i, gt_times in enumerate(GT_timings):
        early_times = gt_times[gt_times < max_time_display]
        print(f"  GT MU {i}: {len(gt_times)} total firings")
        print(f"    First few times: {early_times[:10]}")
        if len(gt_times) > 0:
            print(f"    Time range: {gt_times.min():.3f} - {gt_times.max():.3f} sec")
            print(f"    Mean interval: {np.mean(np.diff(gt_times)):.3f} sec")
    
    print(f"\nDECOMPOSED ({len(DC_timings)} MUs):")
    for i, dc_times in enumerate(DC_timings):
        early_times = dc_times[dc_times < max_time_display] if len(dc_times) > 0 else []
        print(f"  DC MU {i}: {len(dc_times)} total firings")
        if len(dc_times) > 0:
            print(f"    First few times: {early_times[:10]}")
            print(f"    Time range: {dc_times.min():.3f} - {dc_times.max():.3f} sec")
            print(f"    Mean interval: {np.mean(np.diff(dc_times)):.3f} sec")
        else:
            print(f"    No firings detected!")


def plot_firing_comparison(GT_timings, DC_timings, time_window=(0, 10)):
    """
    Plot ground truth vs decomposed firing times for visual comparison.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot Ground Truth
    ax1.set_title("Ground Truth Firing Times", fontsize=14)
    for i, gt_times in enumerate(GT_timings):
        times_in_window = gt_times[(gt_times >= time_window[0]) & (gt_times <= time_window[1])]
        ax1.scatter(times_in_window, [i+1]*len(times_in_window), 
                   alpha=0.8, s=30, label=f'GT MU {i}')
    ax1.set_ylabel("Motor Unit")
    ax1.set_xlim(time_window)
    ax1.set_ylim(0.5, len(GT_timings) + 0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Decomposed
    ax2.set_title("Decomposed Firing Times", fontsize=14)
    for i, dc_times in enumerate(DC_timings):
        if len(dc_times) > 0:
            times_in_window = dc_times[(dc_times >= time_window[0]) & (dc_times <= time_window[1])]
            ax2.scatter(times_in_window, [i+1]*len(times_in_window), 
                       alpha=0.8, s=30, label=f'DC MU {i}', marker='x')
    ax2.set_ylabel("Motor Unit")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_xlim(time_window)
    if len(DC_timings) > 0:
        ax2.set_ylim(0.5, len(DC_timings) + 0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_different_tolerances(GT_timings, DC_timings):
    """
    Test matching with different tolerance values using two-step approach:
    bidirectional screening then RoA calculation.
    """
    tolerances = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]  # seconds
    
    print("\n" + "="*50)
    print("TESTING DIFFERENT TOLERANCES")
    print("="*50)
    
    for tol in tolerances:
        matches = find_matching_motor_units(GT_timings, DC_timings, 
                                          tolerance=tol, min_match_percentage=0.3)
        
        print(f"Tolerance {tol*1000:4.1f}ms: {len(matches)} matches found")
        for match in matches:
            print(f"  GT MU {match[0]} ↔ DC MU {match[1]} "
                  f"(RoA: {match[2]:.1f}%, GT->DC: {match[3]:.1%}, DC->GT: {match[4]:.1%})")


def comprehensive_debug(ground_truth, muapts_train, muapts_bin_train, fs=2048, 
                       GT_format="Dataframe", time_window=(0, 30)):
    """
    Complete debugging suite for decomposition validation using RoA metric.
    """
    # Get the timings
    matches, GT_timings, DC_timings = validate_decomposition_results(
        ground_truth=ground_truth, muapts_bin_train=muapts_bin_train, 
        fs=fs, GT_format=GT_format
    )
    
    # Debug firing times
    debug_firing_times(GT_timings, DC_timings)

    # Check for time-shifted matches
    shifted_matches, dc_best_matches, match_matrix = check_time_shifted_matches(GT_timings, DC_timings)

    # Plot shifted comparisons
    if len(shifted_matches) > 0:
        plot_shifted_comparison(GT_timings, DC_timings, shifted_matches, time_window=time_window)

    # Test different tolerances
    test_different_tolerances(GT_timings, DC_timings)

    # Visual comparison
    plot_firing_comparison(GT_timings, DC_timings, time_window=time_window)

    return matches, GT_timings, DC_timings, shifted_matches, match_matrix