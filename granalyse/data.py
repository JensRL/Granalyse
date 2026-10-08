"""Data preparation helpers for Granalyse."""

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
    """Load PDS from file and return (freq, power) arrays."""
    with open(fit_class.PDSfile, 'r') as f:
        PDS = np.genfromtxt(f)
    freq = PDS[:, 0]   # µHz
    power = PDS[:, 1]  # ppm^2/µHz
    return freq, power


# Half-width [muHz] of a catalogue entry's exclusion window by width class.
# Entries with explicit bounds use those; unclassified ones count as broad.
SYS_HALF_WIDTH = {"narrow": 0.5, "broad": 2.5, "very broad": 4.0}


def _sys_window(f0, meta):
    """(lo, hi) exclusion window [muHz] of an SC_sys_peaks entry."""
    if len(meta) >= 4:
        return float(meta[2]), float(meta[3])
    desc = str(meta[1]).lower()
    if "very broad" in desc:
        hw = SYS_HALF_WIDTH["very broad"]
    elif "narrow" in desc:
        hw = SYS_HALF_WIDTH["narrow"]
    else:
        hw = SYS_HALF_WIDTH["broad"]
    return f0 - hw, f0 + hw


def _variance_inflation(x, n, max_lag):
    """Var(sum of n consecutive x) / (n Var(x)) from the sample autocorrelation of
    x up to max_lag; >= 1. Absorbs the weak correlation of neighbouring bins."""
    x = np.asarray(x, float) - np.mean(x)
    var = np.mean(x * x)
    if x.size < 3 * max_lag or var <= 0:
        return 1.0
    kappa = 1.0
    for k in range(1, min(max_lag, n - 1) + 1):
        rho = np.mean(x[:-k] * x[k:]) / var
        kappa += 2.0 * (1.0 - k / n) * rho
    return max(1.0, kappa)


def detect_systematics(freq, power, numax, sys_dict=SC_sys_peaks, fap=0.01,
                       test_half_widths=(0.1, 0.5, 2.5), sideband_gap=1.0,
                       sideband_width=15.0, min_sideband_bins=50, max_lag=10,
                       protect_fwhm=1.5, verbose=False):
    """Flag known instrumental peaks that are clearly present, for exclusion.

    For each catalogue entry, the mean power in windows of several widths
    centred on it is compared with the local background B, the mean power of
    the neighbouring sidebands (5% of the frequency wide, at most
    sideband_width; log-linearly interpolated across the window).
    For independent native bins of chi2_2 noise the ratio R = mean(P/B) follows
    F(2 n_win, 2 n_side) when no systematic is present (cf. Appourchaux 2004),
    which accounts for the noise of both the window and the background
    estimate. The weak correlation of neighbouring bins is absorbed by
    variance-inflation factors kappa measured in the sidebands, reducing the
    degrees of freedom to 2n/kappa. An entry is flagged when any of its tests
    falls below fap / (number of tests), i.e. a false-alarm probability <= fap
    per spectrum (Bonferroni), and its whole catalogue window is then excluded.

    Entries whose window lies within protect_fwhm x FWHM of numax (envelope FWHM
    of Mosser et al. 2012) are not tested, so oscillation modes are never taken
    for systematics. Expects the native (unbinned, non-oversampled) spectrum.

    Returns (mask, info): mask is True for native bins to exclude from the
    likelihood (bins are never replaced); info has one dict per catalogue entry.
    """
    from scipy.stats import f as f_dist

    freq = np.asarray(freq, float)
    power = np.asarray(power, float)
    mask = np.zeros(freq.size, bool)

    entries = [(f0, sys_dict[f0], *_sys_window(f0, sys_dict[f0])) for f0 in sorted(sys_dict)]
    in_cat = np.zeros(freq.size, bool)
    for _, _, lo, hi in entries:
        in_cat |= (freq >= lo) & (freq <= hi)
    if numax is not None:
        fwhm = 0.66 * numax ** 0.88
        zone = (numax - protect_fwhm * fwhm, numax + protect_fwhm * fwhm)
    else:
        zone = None

    info, tests = [], []
    for f0, meta, lo, hi in entries:
        rec = {"freq": f0, "label": meta[0], "descriptor": meta[1], "lo": lo, "hi": hi,
               "status": "", "p_value": np.nan, "excess": np.nan,
               "test_halfwidth": np.nan, "kappa": np.nan, "n_masked": 0}
        info.append(rec)
        if hi < freq[0] or lo > freq[-1]:
            rec["status"] = "outside data"
            continue
        if zone is not None and hi >= zone[0] and lo <= zone[1]:
            rec["status"] = "protected"
            continue

        # Local background from the sidebands (other catalogue windows left out)
        sides = []
        # Sidebands span 5% of the frequency (at most sideband_width) so a curved
        # background (granulation knee, activity) is close to log-linear across them
        sbw = min(sideband_width, 0.05 * f0)
        for a, b in ((lo - sideband_gap - sbw, lo - sideband_gap),
                     (hi + sideband_gap, hi + sideband_gap + sbw)):
            sel = (freq >= a) & (freq < b) & ~in_cat
            if sel.sum() >= 10:
                sides.append(sel)
        if sum(s.sum() for s in sides) < min_sideband_bins:
            rec["status"] = "untestable"
            continue
        f_side = np.array([np.mean(freq[s]) for s in sides])
        lb_side = np.log([np.mean(power[s]) for s in sides])
        n_side = np.array([s.sum() for s in sides], float)
        # Correlation of neighbouring bins, from the sidebands' power ratios
        ratios = [power[s] / np.mean(power[s]) for s in sides]

        def interp_weights(f):
            """Weights of the sidebands in the log-linear background at f."""
            if len(sides) == 1:
                return np.ones((f.size, 1))
            t = np.clip((f - f_side[0]) / (f_side[1] - f_side[0]), 0.0, 1.0)
            return np.column_stack([1.0 - t, t])

        # Nested test windows centred on the catalogue frequency, up to the full window
        windows = [(max(lo, f0 - h), min(hi, f0 + h)) for h in test_half_widths
                   if f0 - h > lo or f0 + h < hi] + [(lo, hi)]
        for a, b in windows:
            sel = (freq >= a) & (freq <= b)
            n = int(sel.sum())
            if n == 0:
                continue
            wts = interp_weights(freq[sel])
            R = float(np.mean(power[sel] / np.exp(wts @ lb_side)))
            # Degrees of freedom: window (n bins) and the background estimate,
            # whose relative variance is sum_j (w_j^2 kappa_j / n_j)
            kappa_w = float(np.dot(n_side, [_variance_inflation(r, n, max_lag) for r in ratios])
                            / n_side.sum())
            w_bar = wts.mean(axis=0)
            kappa_s = np.array([_variance_inflation(r, r.size, max_lag) for r in ratios])
            n_bg = 1.0 / np.sum(w_bar ** 2 * kappa_s / n_side)
            p = float(f_dist.sf(R, 2.0 * n / kappa_w, 2.0 * n_bg))
            tests.append((rec, p, R, 0.5 * (b - a), kappa_w))

    # Bonferroni over all tests of this spectrum, then exclude whole catalogue windows
    p_crit = fap / max(len(tests), 1)
    for rec in info:
        mine = [t for t in tests if t[0] is rec]
        if not mine:
            continue
        _, p, excess, half, kappa = min(mine, key=lambda t: t[1])
        rec.update(p_value=p, excess=excess, test_halfwidth=round(half, 3), kappa=kappa)
        if p < p_crit:
            sel = (freq >= rec["lo"]) & (freq <= rec["hi"])
            mask |= sel
            rec["status"] = "masked"
            rec["n_masked"] = int(sel.sum())
        else:
            rec["status"] = "not significant"

    if verbose:
        hits = [f"{r['freq']:.1f}" for r in info if r["status"] == "masked"]
        print(f"Systematics: {len(hits)} masked ({', '.join(hits) or 'none'}), "
              f"{mask.sum()} native bins excluded")
    return mask, info


def bin_to_res_fixed_count(freq, power, version, resolution_muHz=0.5, verbose=True,
                            mask=None):
    """Bin PDS so every output bin contains exactly s_int points. s_int is chosen to give 
    achieved resolution closest to the requested resolution. Native bins flagged in
    ``mask`` are excluded: a binned point containing any of them is dropped.

    Returns
    -------
    binned_freq : ndarray
    binned_power : ndarray
    s_int : int (points per bin; chi2 dof = 2*s_int, Gamma shape = s_int)
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
        keep = ~mask if mask is not None else slice(None)
        return freq[keep], power[keep], 1, native_spacing, 0

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
    if mask is not None:
        keep = ~np.asarray(mask)[:n_used].reshape(n_full_bins, s_int).any(axis=1)
        binned_freq, binned_power = binned_freq[keep], binned_power[keep]

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


def process_binning(freq, power, fit, bin_data, bin_res, bin_visuals, mask=None):
    """Handle binning of stellar power spectra.

    Parameters
    ----------
    freq : ndarray
        Frequency array in µHz.
    power : ndarray
        Power array.
    fit : FitCase-compatible object
        State object updated in-place with binned data and metadata.
    bin_data : bool
        Whether to bin to a fixed resolution.
    bin_res : float
        Target binning resolution (µHz) for fixed-count binning.
    bin_visuals : int
        Points-per-bin for visual binning when main binning is skipped.
    mask : bool ndarray or None
        Native bins to exclude (detected systematics); never filled in.

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
            freq_raw, power_raw, version, bin_res, verbose=False, mask=mask
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
            fit.dof = 2 * bin_size  # chi2 dof; likelihood uses Gamma shape dof/2 (Appourchaux 2003a)

    else:
        # No main binning; just visualization binning
        fit.bin_data = False
        if mask is not None:
            freq, power = freq[~mask], power[~mask]
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
    """Apodisation correction using Nyquist frequency.

    Parameters
    ----------
    fit_class : object with .freq and .nyquist attributes.

    Returns
    -------
    eta : ndarray
    """
    x = (np.pi / 2) * fit_class.freq / fit_class.nyquist
    eta = np.sinc(x / np.pi) #IMPORTANT: np.sinc definition is normalized as sin(pi*x)/(pi*x)
    return eta


def calc_numax(logg, teff):
    """
    Estimate numax using the asteroseismic scaling relation.

    Solar reference values used: logg_sun=4.44, teff_sun=5777 K,
    numax_sun=3090 µHz.

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
