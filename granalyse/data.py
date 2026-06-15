"""Data preparation helpers for Granalyse.

Ported from data_helpers.py (NSGF/Grendel). Functions that belong to
later phases (fitting, normalisation, numax retrieval from catalogues)
are not included here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Known Kepler SC systematics (µHz).  Values are {freq: [label, width_desc]}
# or {freq: [label, width_desc, lo_bound, hi_bound]} for explicitly bounded peaks.
SC_sys_peaks = {
    86.8: ['RWTH', 'Broad'],
    196.5: ['none', 'Broad'],
    242.5: ['none', 'Q7 Very Broad'],
    290.0: ['U1', 'Broad'],
    340.0: ['U2', 'Broad'],
    344.9: ['none', 'Very Broad', 341.0, 349.1],
    360.0: ['U3', 'Narrow'],
    362.8: ['none', 'Q7 Very Broad', 360.9, 364.1],
    365.8: ['none', 'Broad'],
    370.4: ['U4', 'Narrow'],
    393.0: ['Harmonic', 'Broad'],
    421.0: ['U5-U8', 'Narrow'],
    566.4: ['1/LC', 'Narrow'],
    589.6: ['Harmonic', 'Broad'],
    733.6: ['Harmonic', 'Broad'],
    1132.8: ['2/LC', 'Narrow'],
    1680.1: ['none', 'Narrow'],
    1682.4: ['Harmonic', 'Narrow'],
    1699.2: ['3/LC', 'Narrow'],
    1803.4: ['None', 'Q7'],
    1825.2: ['none', 'unknown'],
    1828.4: ['Harmonic', 'unknown'],
    2088.5: ['Harmonic', 'Q6.3'],
    2092.6: ['none', 'unknown'],
    2265.6: ['4/LC', 'Narrow'],
    2832.0: ['5/LC', 'Narrow'],
    3398.4: ['6/LC', 'Narrow'],
    3964.8: ['7/LC', 'Narrow'],
    4428.4: ['none', 'unknown'],
    4451.4: ['shifted 1/LC', 'unknown'],
    4531.4: ['8/LC', 'Narrow'],
    4995.8: ['none', 'Q7'],
    5016.2: ['shifted 1/LC', 'unknown'],
    5097.5: ['9/LC', 'Narrow'],
    5564.4: ['none', 'Q6+Q7'],
    5584.5: ['shifted 1/LC', 'unknown'],
    5663.9: ['10/LC', 'Narrow'],
    6230.3: ['11/LC', 'Narrow'],
    6796.7: ['12/LC', 'Narrow'],
    7024.0: ['U5', 'Narrow'],
    7363.1: ['13/LC', 'Narrow'],
    7444.0: ['U6', 'Narrow'],
    7865.0: ['U7', 'Narrow'],
    7929.5: ['14/LC', 'Narrow'],
    8286.0: ['U8', 'Narrow'],
    8495.9: ['15/LC', 'Narrow'],
}

# Solar reference values for the scaling relation
logg_sun = 4.44    # dex
teff_sun = 5777    # K
numax_sun = 3090   # µHz
dnu_sun = 135.1    # µHz


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def read_data(fit_class):
    """Load PDS from file and return (freq, power) arrays.

    The file path is taken from ``fit_class.PDSfile``.
    """
    with open(fit_class.PDSfile, 'r') as f:
        PDS = np.genfromtxt(f)
    freq = PDS[:, 0]   # µHz
    power = PDS[:, 1]  # ppm^2/µHz
    return freq, power


def mask_systematics(freq, power, numax, sys_dict=SC_sys_peaks,
                     # Detection (adaptive local mean)
                     threshold=7,
                     rel_span_frac=0.02, min_span_muHz=5.0, max_span_muHz=30.0,
                     # Matching
                     sys_match_tol_muHz=5.0,
                     # Seed expansion (coarse)
                     expand_factor=0.5,
                     max_replace_width_muHz=5.0,
                     # Refinement pass (sideband sigma method)
                     refine=True,
                     sideband_width_muHz=8.0,
                     refine_detect_k=2.5,
                     refine_quiet_bins=8,
                     gap_tolerance=3,
                     min_core_bins=5,
                     smooth_power=3,
                     # Replacement
                     edge_buffer_bins=20,
                     replace_method='interp',
                     # Output
                     verbose=False,
                     dump_info=False,
                     plot=False):
    """Detect and clean known systematic peaks.

    Pipeline:
      1) Adaptive local mean (frequency-span based) -> mark bins where
         power > threshold*local_mean.
      2) Group contiguous bins -> clusters.
      3) Match cluster centers to catalog systematics within sys_match_tol_muHz.
      4) (Seed) Expand cluster limited by max_replace_width_muHz using
         expand_factor*threshold.
      5) (Refine, optional) Recompute precise contaminated width using sidebands:
            - sidebands just outside ±max_replace_width_muHz/2 give baseline &
              scatter (MAD->sigma)
            - expand from catalog center until refine_quiet_bins consecutive bins
              fall below baseline + refine_detect_k*sigma (bridging small gaps
              ≤ gap_tolerance)
            - enforce width cap & min_core_bins.
      6) Replacement: linear interpolation (or median) between robust side medians
         (edge buffers) over final contaminated span.

    Returns
      (power_clean, mask_info) normally, or
      (peaks, peak_freqs, local_mean, win_lengths, power_clean, mask_info)
      when dump_info=True.
    """
    freq = np.asarray(freq)
    power = np.asarray(power)
    df = np.median(np.diff(freq))
    n = power.size

    # ---------- Adaptive local mean ----------
    span_muHz = np.clip(rel_span_frac * freq, min_span_muHz, max_span_muHz)
    win_lengths = np.maximum(3, (span_muHz / df).astype(int))
    win_lengths += (win_lengths % 2 == 0)
    csum = np.concatenate(([0.0], np.cumsum(power)))
    half = win_lengths // 2
    local_mean = np.empty_like(power, dtype=float)
    for i in range(n):
        L = max(0, i - half[i])
        R = min(n, i + half[i] + 1)
        local_mean[i] = (csum[R] - csum[L]) / (R - L)

    # ---------- Seed detection ----------
    peaks = np.where(power > threshold * local_mean)[0]
    peak_freqs = freq[peaks]
    if verbose or plot:
        print(f"[mask_systematics] Seed: {len(peak_freqs)} bins > {threshold}×local_mean "
              f"(median window {int(np.median(win_lengths))} bins "
              f"≈ {np.median(win_lengths)*df:.2f} µHz)")

    # ---------- Group contiguous ----------
    def group_contiguous(idx):
        if len(idx) == 0:
            return []
        groups = []
        s = prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
            else:
                groups.append((s, prev))
                s = prev = k
        groups.append((s, prev))
        return groups

    seed_groups = group_contiguous(peaks)

    sys_freqs = np.array(sorted(sys_dict.keys()))
    power_clean = power.copy()
    mask_info = {}
    expand_thresh_mult = expand_factor * threshold

    # Precompute smoothed series for refine boundary detection
    if refine and smooth_power > 1:
        from scipy.ndimage import median_filter
        det_series = median_filter(power, size=smooth_power)
    else:
        det_series = power

    def mad_sigma(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(x)
        return med, sigma

    for g_start, g_end in seed_groups:
        # ----- Match to nearest systematic -----
        g_center_f = 0.5 * (freq[g_start] + freq[g_end])
        i_sys = np.searchsorted(sys_freqs, g_center_f)
        cand = []
        if i_sys < len(sys_freqs):
            cand.append(i_sys)
        if i_sys > 0:
            cand.append(i_sys - 1)
        matched_sys = None
        min_sep = None
        for ci in cand:
            sep = abs(sys_freqs[ci] - g_center_f)
            if sep <= sys_match_tol_muHz and (min_sep is None or sep < min_sep):
                matched_sys = sys_freqs[ci]
                min_sep = sep
        if matched_sys is None:
            continue
        # Safety: skip systematics within ±15% of numax
        if numax is not None and (numax * 0.85 <= matched_sys <= numax * 1.15):
            if verbose:
                print(f"[mask_systematics] Skip {matched_sys:.2f} µHz: "
                      f"within ±15% of νmax={numax:.2f}")
            continue

        # ----- Seed expansion (coarse) -----
        left, right = g_start, g_end
        while left > 0:
            if freq[g_end] - freq[left - 1] > max_replace_width_muHz:
                break
            if power[left - 1] > expand_thresh_mult * local_mean[left - 1]:
                left -= 1
            else:
                break
        while right < n - 1:
            if freq[right + 1] - freq[g_start] > max_replace_width_muHz:
                break
            if power[right + 1] > expand_thresh_mult * local_mean[right + 1]:
                right += 1
            else:
                break

        # ----- Refine exact contaminated span -----
        if refine:
            c_idx = np.argmin(np.abs(freq - matched_sys))
            c_freq = freq[c_idx]
            half_cap = 0.5 * max_replace_width_muHz
            f_cap_lo = c_freq - half_cap
            f_cap_hi = c_freq + half_cap
            i_cap_lo = np.searchsorted(freq, f_cap_lo, 'left')
            i_cap_hi = np.searchsorted(freq, f_cap_hi, 'right') - 1
            i_cap_lo = max(0, i_cap_lo)
            i_cap_hi = min(n - 1, i_cap_hi)

            # Sidebands
            sb_lo1 = max(freq[0], f_cap_lo - sideband_width_muHz)
            sb_lo2 = f_cap_lo
            sb_hi1 = f_cap_hi
            sb_hi2 = min(freq[-1], f_cap_hi + sideband_width_muHz)
            idx_sb_left = np.where((freq >= sb_lo1) & (freq < sb_lo2))[0]
            idx_sb_right = np.where((freq > sb_hi1) & (freq <= sb_hi2))[0]
            if idx_sb_left.size >= 3 and idx_sb_right.size >= 3:
                side_vals = np.concatenate([power[idx_sb_left], power[idx_sb_right]])
                base, sigma = mad_sigma(side_vals)
                thresh_ref = base + refine_detect_k * sigma
                # expand left
                L = c_idx
                quiet = 0
                i = c_idx
                while i > i_cap_lo:
                    i -= 1
                    if det_series[i] <= thresh_ref:
                        quiet += 1
                        if quiet >= refine_quiet_bins:
                            L = i + refine_quiet_bins
                            break
                    else:
                        quiet = 0
                        L = i
                # expand right
                R = c_idx
                quiet = 0
                i = c_idx
                while i < i_cap_hi:
                    i += 1
                    if det_series[i] <= thresh_ref:
                        quiet += 1
                        if quiet >= refine_quiet_bins:
                            R = i - refine_quiet_bins
                            break
                    else:
                        quiet = 0
                        R = i
                # Bridge small gaps
                if R > L:
                    mask_det = det_series[L:R + 1] > thresh_ref
                    gap_start = None
                    for k, v in enumerate(mask_det):
                        if not v and gap_start is None:
                            gap_start = k
                        if v and gap_start is not None:
                            if (k - gap_start) <= gap_tolerance:
                                mask_det[gap_start:k] = True
                            gap_start = None
                    if gap_start is not None:
                        if (len(mask_det) - gap_start) <= gap_tolerance:
                            mask_det[gap_start:] = True
                    if mask_det.any():
                        rel = np.where(mask_det)[0]
                        L = L + rel[0]
                        R = L + (rel[-1] - rel[0])
                # Enforce cap
                while freq[R] - freq[L] > max_replace_width_muHz and R > L:
                    if (det_series[L] - base) < (det_series[R] - base):
                        L += 1
                    else:
                        R -= 1
                if R - L + 1 >= min_core_bins:
                    left, right = L, R

        # ----- Context (edge medians) -----
        left_lo = max(0, left - edge_buffer_bins)
        left_hi = left
        right_lo = right + 1
        right_hi = min(n, right + 1 + edge_buffer_bins)
        if left_lo >= left_hi or right_lo >= right_hi:
            continue
        left_val = np.median(power_clean[left_lo:left_hi])
        right_val = np.median(power_clean[right_lo:right_hi])

        # ----- Replacement -----
        if replace_method == 'interp':
            repl = np.linspace(left_val, right_val, right - left + 1)
        else:
            repl = np.full(right - left + 1, 0.5 * (left_val + right_val))
        power_clean[left:right + 1] = repl

        mask_info[matched_sys] = {
            'descriptor': sys_dict[matched_sys][1],
            'freq_range': (freq[left], freq[right]),
            'width_muHz': freq[right] - freq[left],
            'nbins': right - left + 1,
            'sys_sep_muHz': float(min_sep),
            'refined': refine,
        }
        if verbose:
            print(f"Cleaned {matched_sys:.2f} µHz: "
                  f"width {mask_info[matched_sys]['width_muHz']:.2f} µHz "
                  f"({mask_info[matched_sys]['nbins']} bins)")

    # ---------- Plot ----------
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(freq, power, label='Original', color='C0', alpha=0.35)
        plt.plot(freq, local_mean, label='Local mean', color='C1', lw=0.8)
        plt.plot(freq, power_clean, label='Cleaned', color='k', lw=0.7)
        for info in mask_info.values():
            f1, f2 = info['freq_range']
            plt.axvspan(f1, f2, color='tab:red', alpha=0.15)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (µHz)')
        plt.ylabel('Power Density')
        plt.title('Systematic Cleaning (merged refine)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if dump_info:
        return peaks, peak_freqs, local_mean, win_lengths, power_clean, mask_info
    else:
        return power_clean, mask_info


def bin_to_res_fixed_count(freq, power, version, resolution_muHz=0.5, verbose=True):
    """Bin PDS so every output bin contains exactly s_int points.

    s_int is chosen to give achieved resolution closest to the requested
    resolution, while never going below 0.3 µHz (for Peakbogging mode).

    Returns
    -------
    binned_freq : ndarray
    binned_power : ndarray
    s_int : int (points per bin; dof = 2*s_int)
    achieved_res_muHz : float
    n_dropped : int
    """
    freq = np.asarray(freq)
    power = np.asarray(power)

    native_spacing = np.median(np.diff(freq))

    if native_spacing >= resolution_muHz:
        print(
            f"Requested resolution ({resolution_muHz} µHz) is finer than "
            f"the data spacing ({native_spacing:.3f} µHz). "
            "Cannot bin to a higher resolution than native."
        )
        return freq, power, 1, native_spacing, 0

    s_float = resolution_muHz / native_spacing
    s_round = int(np.round(s_float))
    if s_round < 1:
        s_round = 1

    if version == 'Peakbogging':
        min_s = int(np.ceil(0.3 / native_spacing))
    else:
        min_s = 1
    if min_s < 1:
        min_s = 1

    candidates = [max(min_s, s_round - 1), max(min_s, s_round), max(min_s, s_round + 1)]
    candidates = sorted(set(c for c in candidates if c >= 1))

    achieved_options = [c * native_spacing for c in candidates]
    idx_best = int(np.argmin([abs(resolution_muHz - a) for a in achieved_options]))
    s_int = candidates[idx_best]
    achieved_res = achieved_options[idx_best]

    n_points = freq.size
    n_full_bins = n_points // s_int
    n_dropped = n_points - (n_full_bins * s_int)

    n_used = n_full_bins * s_int
    freq_used = freq[:n_used]
    power_used = power[:n_used]

    freq_blocks = freq_used.reshape(n_full_bins, s_int)
    power_blocks = power_used.reshape(n_full_bins, s_int)

    binned_freq = np.mean(freq_blocks, axis=1)
    binned_power = np.mean(power_blocks, axis=1)

    if verbose:
        print(f"Native_spacing = {native_spacing:.6g} µHz")
        print(f"Requested_resolution = {resolution_muHz:.6g} µHz (s_float = {s_float:.3f})")
        print(f"Chosen s_int = {s_int} -> Achieved_resolution = {achieved_res:.6g} µHz")
        if n_dropped:
            print(f"Dropped {n_dropped} points at the high-frequency tail")
        if s_int == 1:
            print("Note: s_int == 1 -> no effective binning")

    return binned_freq, binned_power, s_int, achieved_res, n_dropped


def bin_data_simplistic(freq, power, bin_size):
    """s-point independent binning (mean of s consecutive points).

    Parameters
    ----------
    freq : ndarray
    power : ndarray
    bin_size : int

    Returns
    -------
    binned_freq : ndarray
    binned_power : ndarray
    """
    freq = np.array(freq)
    power = np.array(power)
    num_points = len(freq)
    num_bins = num_points // bin_size

    freq = freq[:num_bins * bin_size]
    power = power[:num_bins * bin_size]

    binned_freq = freq.reshape((num_bins, bin_size)).mean(axis=1)
    binned_power = power.reshape((num_bins, bin_size)).mean(axis=1)

    return binned_freq, binned_power


def process_binning(freq, power, fit, bin_data, bin_res, bin_visuals):
    """Handle binning of stellar power spectra.

    Parameters
    ----------
    freq : ndarray
        Frequency array in µHz.
    power : ndarray
        Power array.
    fit : FitCase (or SingleFit-compatible object)
        State object updated in-place with binned data and metadata.
    bin_data : bool
        Whether to bin to a fixed resolution.
    bin_res : float
        Target binning resolution (µHz) for fixed-count binning.
    bin_visuals : int
        Points-per-bin for visual binning when main binning is skipped.

    Returns
    -------
    freq : ndarray
    power : ndarray
    bin_info : dict
    """
    bin_info = {}

    if bin_data:
        fit.bin_data = True
        freq_raw, power_raw = freq.copy(), power.copy()
        version = fit.pulsmod
        freq, power, bin_size, outres, ndrop = bin_to_res_fixed_count(
            freq_raw, power_raw, version, bin_res, verbose=False
        )
        print(f'Binned data to {outres:.2f} µHz resolution -- bin size: {bin_size}')
        bin_info.update({'bin_size': bin_size, 'outres': outres, 'ndrop': ndrop})

        if bin_size < 2:
            # fallback: visual-only binning
            fit.bin_data = False
            fit.bin_size = bin_visuals
            freq_binned, power_binned = bin_data_simplistic(freq, power, fit.bin_size)
            fit.freq = freq
            fit.power = power
            fit.freq_binned = freq_binned
            fit.power_binned = power_binned
            fit.freq_raw = freq_raw
            fit.power_raw = power_raw
            print(f'Bin size was 1 -- Adding binned data with {bin_visuals} points per bin '
                  f'for visualization purposes')
            fit.dof = 2  # Appourchaux (2003a)
        else:
            fit.freq = freq
            fit.power = power
            fit.freq_raw = freq_raw
            fit.power_raw = power_raw
            fit.bin_size = bin_size
            fit.dof = 2 * bin_size  # Appourchaux (2003a)

    else:
        # No main binning; just visualization binning
        fit.bin_data = False
        fit.bin_size = bin_visuals

        freq_binned, power_binned = bin_data_simplistic(freq, power, fit.bin_size)
        fit.freq = freq
        fit.power = power
        fit.freq_binned = freq_binned
        fit.power_binned = power_binned

        fit.dof = 2  # Appourchaux (2003a)
        bin_info.update({'bin_size': fit.bin_size})

    return freq, power, bin_info


def cut_low_frequencies(freq, power, fit, remove_low_freq):
    """Apply a low-frequency cutoff and update the fit object.

    Parameters
    ----------
    freq : ndarray
    power : ndarray
    fit : FitCase-compatible object
    remove_low_freq : float or None
        Cutoff in µHz.  None means no cutoff.

    Returns
    -------
    freq : ndarray
    power : ndarray
    cut_info : dict
    """
    cut_info = {"low_freq_cut": None}

    if remove_low_freq is None:
        fit.freq = freq
        fit.power = power
        return freq, power, cut_info

    low_freq_cut = float(remove_low_freq)
    mask = freq >= low_freq_cut

    fit.freq_removed = freq[~mask]
    fit.power_removed = power[~mask]

    freq = freq[mask]
    power = power[mask]

    fit.freq = freq
    fit.power = power

    print(f"Applied low-frequency cutoff at {low_freq_cut:.3f} µHz")
    cut_info["low_freq_cut"] = low_freq_cut

    return freq, power, cut_info


def calc_eta(fit_class):
    """Apodisation correction for Nyquist roll-off.

    Parameters
    ----------
    fit_class : object with .freq and .nyquist attributes.

    Returns
    -------
    eta : ndarray
    """
    x = (np.pi / 2) * fit_class.freq / fit_class.nyquist
    eta = np.sinc(x / np.pi)
    return eta


def calc_numax(logg, teff):
    """Estimate numax from the scaling relation.

    .. math::

        \\nu_{\\max} = \\frac{10^{\\log g}}{10^{\\log g_\\odot}}
                       \\left(\\frac{T_{\\rm eff}}{T_{\\rm eff,\\odot}}\\right)^{-0.5}
                       \\nu_{\\max,\\odot}

    Solar reference values used: logg_sun=4.44, teff_sun=5777 K,
    numax_sun=3090 µHz (identical to the original data_helpers.py).

    Note: the original ``calc_numax(fit_class)`` read logg/teff from a
    STAGGER model table on disk and required ``fit_class.data_format ==
    'STAGGER'``.  That table-lookup path is not reproduced here; call this
    pure function directly with the stellar parameters instead.

    Parameters
    ----------
    logg : float
        Surface gravity in dex.
    teff : float
        Effective temperature in K.

    Returns
    -------
    numax : float
        Estimated numax in µHz.
    """
    numax = float(
        (10 ** logg / 10 ** logg_sun) * (teff / teff_sun) ** (-0.5) * numax_sun
    )
    return numax
