"""
Plotting and presentation functions for fitted granulation models.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
from matplotlib import rc_context
import corner
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from .models import get_model
from .scaling import gran_a, gran_b, gran_d
from .data import bin_data_simplistic

plt.rcParams["agg.path.chunksize"] = 20000

_CORNER_RASTER_ZORDER = 10
_CORNER_RASTER_DPI = 150

# PDS figure y-range constraints
_PDS_COMPONENT_MARGIN_DEX = 0.5
_PDS_DATA_MAX_FRACTION = 0.75


def _save_corner(fig, path):
    """Save a corner figure with its dense contour/scatter layers rasterised
    and all text/axes kept as vectors."""
    for ax in fig.get_axes():
        ax.set_rasterization_zorder(_CORNER_RASTER_ZORDER)
    fig.savefig(path, dpi=_CORNER_RASTER_DPI)


def _peakbogging_likelihood_plotting(P, dof, mod_back, beta, term1):
    """Per-frequency background and peak log-likelihoods (for visualisation)."""
    term2 = (dof - 1) * jnp.log(P) - dof * jnp.log(mod_back)
    term3 = -dof * P / mod_back
    Lback = term1 + term2 + term3
    term2peak = (dof - 1) * jnp.log(P) - dof * jnp.log(mod_back * beta)
    term3peak = -dof * P / (mod_back * beta)
    Lpeak = term1 + term2peak + term3peak
    return jnp.sum(Lback), jnp.sum(Lpeak), Lback, Lpeak

# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _bin_sigmoid(sigmoid, bin_size):
    """Bin a sigmoid array by the same method used for freq/power."""
    sigmoid = np.array(sigmoid)
    num_bins = len(sigmoid) // bin_size
    sigmoid = sigmoid[: num_bins * bin_size]
    return sigmoid.reshape((num_bins, bin_size)).mean(axis=1)


def _one_component(nu, amplitude, charfreq, exponent, eta):
    """Single Harvey or activity component for plotting."""
    if exponent == 2:
        return eta**2 * ((2 / np.pi) ** 2 * (amplitude**2 / charfreq) / (1 + (nu / charfreq) ** exponent))
    return eta**2 * (amplitude**2 / charfreq) / (1 + (nu / charfreq) ** exponent)


def _downsample_freq(freq, model_arr, target_length=1000):
    """Downsample freq + model arrays to at most target_length points."""
    if len(freq) <= target_length:
        return freq, model_arr
    indices = np.linspace(0, len(freq) - 1, target_length, dtype=int)
    return freq[indices], model_arr[indices]


# ---------------------------------------------------------------------------
# Build the parameter dict that model.func_real / func_back expects.
# ---------------------------------------------------------------------------

def _has_key(source, key):
    """Check if key is present in source regardless of dict vs DataFrame."""
    if isinstance(source, pd.DataFrame):
        return key in source.columns
    return key in source


def _resolve_corr_params(fit, model, source, i=None, numax_for_scaling=None):
    """
    Return (a, b, d) from either posterior samples or prior_samples.
    Returns numpy scalars safe to pass to jax model functions.
    """
    def _get(key):
        val = source[key]
        if i is None:
            return float(val[0])
        return float(val[i])

    # numax used for scaling-relation reconstruction
    if numax_for_scaling is not None:
        numax_sc = float(numax_for_scaling)
    else:
        numax_sc = _get("numax_fit")

    corr = model.corr_computed

    if "a" in corr and _has_key(source, "sigma_a"):
        a = float(gran_a(numax_sc, _get("sigma_a")))
    else:
        a = _get("a")

    if "b" in corr and _has_key(source, "sigma_b"):
        b = float(gran_b(numax_sc, _get("sigma_b")))
    else:
        b = _get("b")

    if "d" in model.params:
        if "d" in corr and _has_key(source, "sigma_d"):
            d = float(gran_d(numax_sc, _get("sigma_d")))
        else:
            d = _get("d")
    else:
        d = None

    return a, b, d


def _call_model_func(func, nu, eta, param_dict):
    """Call a model jax function and return a plain numpy array."""
    return np.asarray(func(nu=jnp.asarray(nu), eta=jnp.asarray(eta), **{
        k: jnp.asarray(v) if hasattr(v, "__len__") else v
        for k, v in param_dict.items()
    }))


# ---------------------------------------------------------------------------
# prior_corner  
# ---------------------------------------------------------------------------

def _prior_corner(fit):
    prior_samples = fit.prior_samples
    PlotData_priors = np.vstack([prior_samples[p] for p in fit.fitparams]).T
    labels = [fit.paramlabels[p] for p in fit.fitparams]
    cornerfig = plt.figure(figsize=(20, 18))
    corner.corner(
        PlotData_priors,
        labels=labels,
        labelpad=0.3,
        label_kwargs={"fontsize": 15},
        show_titles=True,
        title_quantiles=[0.16, 0.50, 0.84],
        title_kwargs={"fontsize": 8.5},
        fig=cornerfig,
        quiet=True,
    )
    cornerfig.savefig(fit.priorCornerfigname, dpi=500)
    plt.close(cornerfig)
    return True


# ---------------------------------------------------------------------------
# prior_predictive_PDS
# ---------------------------------------------------------------------------

def prior_predictive_PDS(fit, model=None):
    """Plot prior-predictive model traces over the raw/binned PDS."""
    if model is None:
        model = get_model(fit.model)

    if fit.prior_samples is None:
        from .priors import draw_prior_samples
        fit.prior_samples = draw_prior_samples(fit)
    prior_samples = fit.prior_samples
    freq = fit.freq
    power = fit.power
    freq_binned, power_binned = bin_data_simplistic(freq, power, fit.bin_size)

    # How many prior samples to draw
    if len(freq) > 1000:
        N = 100 if len(freq) > 1e6 else 1000
        N = min(N, len(prior_samples["W"]))
        zorder_samples = 30
        alpha_samples = 0.5
    else:
        N = len(prior_samples["W"])
        zorder_samples = 1
        alpha_samples = 0.05

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(freq, power, "grey", linewidth=0.5, alpha=0.6, label="Fitted PDS", zorder=10)

    if fit.likelihood == "mixture":
        sigmoid = fit.sigmoid
        ax.plot(freq, sigmoid, "r", linewidth=1, zorder=100, label="Sigmoid weighting")
        ax.hlines(fit.bagmean, np.min(freq), np.max(freq), colors="r",
                  linestyles="dashed", linewidth=1, zorder=100, label="Bag mean")
        sigmoid_binned = _bin_sigmoid(sigmoid, 10)
        alpha_values = np.clip(sigmoid_binned, 0.1, 1.0)
        segments = [
            [(freq_binned[i], power_binned[i]), (freq_binned[i + 1], power_binned[i + 1])]
            for i in range(len(freq_binned) - 1)
        ]
        colors = [(0, 0, 0, alpha_values[i]) for i in range(len(freq_binned) - 1)]
        lc = LineCollection(segments, colors=colors, linewidths=0.8, zorder=90,
                            label="Binned PDS with sigmoid weighting")
        ax.add_collection(lc)
        ax.autoscale()
    else:
        ax.plot(freq_binned, power_binned, "k", linewidth=1, label="Binned PDS", zorder=10)

    # Plot N prior-predictive model traces
    nu_jax = jnp.asarray(freq)
    eta_jax = jnp.asarray(fit.eta)

    for i in range(N):
        # Resolve physical granulation params.
        a, b, d = _resolve_corr_params(fit, model, prior_samples, i,
                                       numax_for_scaling=fit.numax if fit.corr_inference else None)

        # Collect remaining fitparams from prior_samples at index i
        # Build the full parameter dict the model expects
        p = {}
        for param in model.params:
            if param == "a":
                p["a"] = a
            elif param == "b":
                p["b"] = b
            elif param == "d":
                p["d"] = d
            else:
                p[param] = float(prior_samples[param][i])

        # Oscillation / activity params
        if fit.pulsmod == "Gaussian":
            numax_key = "numax_fit"
            numax_val = (float(prior_samples[numax_key][i])
                         if fit.data_format == "STAR"
                         else fit.numax)
            p["Posc"] = float(prior_samples["Posc"][i])
            p["sigma"] = float(prior_samples["sigma"][i])
        else:
            # Peakbogging: no oscillation envelope
            numax_val = 0.0
            p["Posc"] = 0.0
            p["sigma"] = 1.0

        p["numax_fit"] = numax_val
        p["W"] = float(prior_samples["W"][i])
        p["a2"] = float(prior_samples["a2"][i])
        p["b2"] = float(prior_samples["b2"][i])

        mod = _call_model_func(model.func_real, nu_jax, eta_jax, p)

        freq_ds, mod_ds = _downsample_freq(freq, mod)
        ax.plot(freq_ds, mod_ds, "b", linewidth=0.2, alpha=alpha_samples,
                zorder=zorder_samples, label="Model samples")

    ax.vlines(fit.numax, 0, np.max(power), colors="k", linestyles="dashed",
              linewidth=1,
              label=r"$\nu_\mathrm{max}$" + f"= {float(fit.numax):.2f} " + r"$\mu$Hz")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Frequency [$\mu$Hz]")
    ax.set_ylabel(r"Power Density [ppm$^2/\mu$Hz]")
    ax.set_title(f"Prior predictive PDS for model {fit.model} for {fit.name}")
    ax.set_xlim(0.5*np.min(freq), np.max(freq))
    lower_ylim = np.mean(power[-100:]) - 0.8 * np.mean(power[-100:])
    ax.set_ylim(lower_ylim, np.max(power) + 0.3 * np.max(power))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=8)
    plt.tight_layout()
    fig.savefig(fit.priorpredictivePDSfigname, dpi=400)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# present_fit  (PDS + posterior corner plots)
# ---------------------------------------------------------------------------

def present_fit(fit, model=None):
    """Full posterior presentation: PDS fit figure + corner plots."""
    if model is None:
        model = get_model(fit.model)

    if fit.prior_samples is None:
        from .priors import draw_prior_samples
        fit.prior_samples = draw_prior_samples(fit)

    # ======================== PDS figure ========================
    if fit.pulsmod == "Gaussian":
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

    freq = fit.freq
    power = fit.power

    # ---- Plot the data ----
    if fit.bin_data:
        freq_raw = fit.freq_raw
        power_raw = fit.power_raw
        #Avoid thin grey lines when plotting
        split = int(np.searchsorted(freq_raw, np.min(freq)))
        if fit.freq_removed is not None and split > 1:
            ax1.plot(freq_raw[:split + 1], power_raw[:split + 1],
                     color="grey", alpha=0.4, linewidth=0.5, label="Raw PDS",
                     rasterized=True)
            ax1.plot(freq_raw[split:], power_raw[split:],
                     color="grey", alpha=0.8, linewidth=0.5, rasterized=True)
        else:
            ax1.plot(freq_raw, power_raw, color="grey", alpha=0.8,
                     linewidth=0.5, label="Raw PDS", rasterized=True)
        if fit.likelihood not in {"mixture"}:
            ax1.plot(freq, power, color="k", linewidth=1,
                     label="Fitted Binned PDS", rasterized=True)
    else:
        freq_binned = fit.freq_binned
        power_binned = fit.power_binned
        ax1.plot(freq, power, color="grey", alpha=0.8, linewidth=0.5,
                 label="Fitted Raw PDS", rasterized=True)
        if fit.likelihood not in {"mixture"}:
            if fit.freq_removed is not None:
                keep = freq_binned >= np.min(freq)
                freq_b, power_b = freq_binned[keep], power_binned[keep]
            else:
                freq_b, power_b = freq_binned, power_binned
            if freq_b.size:
                ax1.plot(freq_b, power_b, color="k", alpha=1, linewidth=1,
                         label="Binned PDS", rasterized=True)

    if fit.likelihood in {"mixture"}:
        sigmoid = fit.sigmoid
        # Ensure binned arrays are available 
        if "freq_binned" not in dir() or freq_binned is None:
            freq_binned = fit.freq_binned if hasattr(fit, "freq_binned") else None
            power_binned = fit.power_binned if hasattr(fit, "power_binned") else None
        if freq_binned is not None and power_binned is not None:
            sigmoid_binned = _bin_sigmoid(sigmoid, 10)
            alpha_values = np.clip(sigmoid_binned, 0.1, 1.0)
            segments = [
                [(freq_binned[i], power_binned[i]), (freq_binned[i + 1], power_binned[i + 1])]
                for i in range(len(freq_binned) - 1)
            ]
            colors = [(0, 0, 0, alpha_values[i]) for i in range(len(freq_binned) - 1)]
            lc = LineCollection(segments, colors=colors, linewidths=1, zorder=90,
                                label="Binned PDS with sigmoid weighting")
            ax1.add_collection(lc)
            ax1.autoscale()
        else:
            print("WARNING: mixture likelihood overlay skipped — binned arrays not available")

    if fit.freq_removed is not None and fit.power_removed is not None:
        ax1.plot(fit.freq_removed, fit.power_removed,
                 color="grey", alpha=0.4, linewidth=0.4, rasterized=True)

    # ---- numax vline ----
    if fit.data_format == "STAR":
        numax = fit.fitresults["numax_fit"][0]
        numax_plus = fit.fitresults["numax_fit"][1]
        numax_minus = fit.fitresults["numax_fit"][2]
        numax_label = (
            r"$\nu_\mathrm{max} = "
            f"{numax:.2f}"
            r"^{+" + f"{numax_plus:.2f}" + r"}_{-" + f"{numax_minus:.2f}" + r"}\,\mu\mathrm{Hz}$"
        )
    else:
        numax = fit.numax
        numax_label = r"$\nu_\mathrm{max}$" + f"= {float(numax):.2f}" + r"$\mu$Hz"

    ax1.vlines(numax, 0, 0.8 * np.max(power), colors="k",
               linestyles="dashed", linewidth=1, label=numax_label)

    # ---- Resolve median posterior parameters ----
    a, b, d = _resolve_corr_params(fit, model, fit.fitresults, i=None)

    # Collect non-granulation params from fitresults
    median_p = {}
    for param in model.params:
        if param == "a":
            median_p["a"] = a
        elif param == "b":
            median_p["b"] = b
        elif param == "d":
            if d is not None:
                median_p["d"] = d
        else:
            median_p[param] = float(fit.fitresults[param][0])

    if fit.pulsmod == "Gaussian":
        median_p["Posc"] = float(fit.fitresults["Posc"][0])
        median_p["sigma"] = float(fit.fitresults["sigma"][0])
    else:
        median_p["Posc"] = 0.0
        median_p["sigma"] = 1.0

    median_p["numax_fit"] = numax
    median_p["W"] = float(fit.fitresults["W"][0])
    median_p["a2"] = float(fit.fitresults["a2"][0])
    median_p["b2"] = float(fit.fitresults["b2"][0])

    W = median_p["W"]
    a2 = median_p["a2"]
    b2 = median_p["b2"]

    # noise label
    noiselabel = "Noise" if fit.data_format == "STAR" else "Offset"

    # ---- Draw N posterior sample traces ----
    N = 50
    nu_jax = jnp.asarray(freq)
    eta_jax = jnp.asarray(fit.eta)

    fd = fit.fitdata  # DataFrame with posterior samples

    def _rc(key):
        return np.random.choice(fd[key].values, N)

    numaxs_N = _rc("numax_fit")
    if fit.corr_inference:
        if "sigma_a" in fd.columns:
            sigma_as_N = _rc("sigma_a")
            a_samples = np.array([gran_a(nm, sa) for nm, sa in zip(numaxs_N, sigma_as_N)])
        else:
            a_samples = _rc("a")
        sigma_bs_N = _rc("sigma_b")
        b_samples = np.array([gran_b(nm, sb) for nm, sb in zip(numaxs_N, sigma_bs_N)])
    else:
        a_samples = _rc("a")
        b_samples = _rc("b")

    # Slope l is always present
    l_samples = _rc("l")
    W_samples = _rc("W")

    if fit.pulsmod == "Gaussian":
        Posc_samples = _rc("Posc")
        sigma_samples = _rc("sigma")

    a2_samples = _rc("a2")
    b2_samples = _rc("b2")

    # Second char-freq d / slope k (absent in model D)
    has_d = "d" in model.params
    if has_d:
        if fit.corr_inference:
            sigma_ds_N = _rc("sigma_d")
            d_samples = np.array([gran_d(nm, sd) for nm, sd in zip(numaxs_N, sigma_ds_N)])
        else:
            d_samples = _rc("d")
        k_samples = _rc("k")

    # Model-specific extra params (c for H/T; e,f,m for T)
    extra_samples = {}
    for param in model.params:
        if param not in ("a", "b", "d", "l", "k"):
            extra_samples[param] = _rc(param)

    if fit.data_format in {"STAGGER", "STAR"}:
        # Median model
        ax1.plot(freq,
                 _call_model_func(model.func_real, nu_jax, eta_jax, median_p),
                 "r", linewidth=1, label="Model fit", zorder=10, rasterized=True)

        # Sample traces
        for i in range(N):
            p_i = {}
            for param in model.params:
                if param == "a":
                    p_i["a"] = a_samples[i]
                elif param == "b":
                    p_i["b"] = b_samples[i]
                elif param == "d":
                    p_i["d"] = d_samples[i]
                elif param == "l":
                    p_i["l"] = l_samples[i]
                elif param == "k":
                    p_i["k"] = k_samples[i]
                else:
                    p_i[param] = extra_samples[param][i]

            p_i["W"] = W_samples[i]
            p_i["a2"] = a2_samples[i]
            p_i["b2"] = b2_samples[i]
            if fit.pulsmod == "Gaussian":
                p_i["Posc"] = Posc_samples[i]
                p_i["sigma"] = sigma_samples[i]
                p_i["numax_fit"] = numax
            else:
                p_i["Posc"] = 0.0
                p_i["sigma"] = 1.0
                p_i["numax_fit"] = 0.0

            ax1.plot(freq,
                     _call_model_func(model.func_real, nu_jax, eta_jax, p_i),
                     "r", linewidth=0.5, alpha=0.2, rasterized=True)

    # ---- Granulation components (delegated to the model object) ----
    component_levels = model.draw_pds_components(ax1, freq, fit.eta,
                                                 median_p, np.max(power))

    # No-oscillation model line (STAR + Gaussian only)
    if fit.data_format == "STAR" and fit.pulsmod == "Gaussian":
        back_p = {k: v for k, v in median_p.items()}
        back_p["Posc"] = 0.0
        ax1.plot(freq,
                 _call_model_func(model.func_back, nu_jax, eta_jax, back_p),
                 "b", linestyle="dashed", linewidth=1,
                 label="Model w/o Oscillations", zorder=8, rasterized=True)

    # White noise
    ax1.hlines(W, np.min(freq), np.max(freq), colors="#FF800E",
               linestyles="dashed", linewidth=1, label=noiselabel)

    # Activity
    ax1.plot(freq, _one_component(freq, a2, b2, 2, fit.eta),
             "g", linestyle="dashdot", linewidth=1, label="Activity")

    # ---- Peakbogging inset ----
    if fit.pulsmod == "Peakbogging":
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 1)
        H_val = fit.fitresults["H"][0] if "H" in fit.fitresults else None
        sigma_pb = fit.sigwidth
        if H_val is not None:
            # Gaussian oscillation-peak mixture coefficient
            mod_peak = np.asarray(
                jax.jit(lambda nu: H_val * jnp.exp(-0.5 * ((nu - numax) / sigma_pb) ** 2))(
                    jnp.asarray(freq)
                )
            )
            ax2.plot(freq, mod_peak, "b", linewidth=1, label="Peakbog Excess",
                     zorder=8, rasterized=True)
        ax2.set_ylabel(r"$H$ (Peakbogging detection probability)", color="b")

        # Background model for likelihood ratio
        back_params = {k: v for k, v in median_p.items()
                       if k not in ("Posc", "sigma", "numax_fit")}
        mod_back = _call_model_func(model.func_back, nu_jax, eta_jax, back_params)

        if H_val is not None:
            # Likelihood ratio per frequency
            dof = int(fit.dof)
            term1 = (dof - 1) * jnp.log(dof) - jax.scipy.special.gammaln(dof - 1)
            beta_val = float(fit.fitresults["beta"][0])
            _, _, Lback_freq, Lpeak_freq = _peakbogging_likelihood_plotting(
                jnp.asarray(power), dof, jnp.asarray(mod_back), beta_val, term1
            )
            Q = jnp.asarray(mod_peak)
            loglike_ratio = jnp.log(Q) + Lpeak_freq - Lback_freq

            with rc_context(rc={"xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 8}):
                axins = inset_axes(
                    ax1, width="30%", height="30%", loc="lower left",
                    bbox_to_anchor=(0.1, 0.08, 1, 1),
                    bbox_transform=ax1.transAxes, borderpad=0,
                )
                axins.plot(freq, np.asarray(loglike_ratio), color="darkred",
                           zorder=10, rasterized=True)
                axins.set_xlabel(r"$\nu$ [$\mu$Hz]", fontsize=8)
                axins.set_ylabel(r"$\log(L_{fore} - L_{back})$", fontsize=8)
                axins.set_xlim(0.6 * fit.numax, 1.4 * fit.numax)
                region_mask = (freq > 0.6 * fit.numax) & (freq < 1.4 * fit.numax)
                lr_region = np.asarray(loglike_ratio)[region_mask]
                finite_mask = np.isfinite(lr_region)
                lr_clean = lr_region[finite_mask]
                if lr_clean.size > 0:
                    y_min, y_max = np.min(lr_clean), np.max(lr_clean)
                    y_pad = 0.2 * (y_max - y_min)
                    axins.set_ylim(y_min - 0.5 * y_pad, y_max + y_pad)
                else:
                    axins.set_ylim(-30, 300)
                axins.tick_params(axis="both", labelsize=8, pad=2)
                axins.xaxis.labelpad = 1
                axins.yaxis.labelpad = 1
                axins.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                axins.patch.set_facecolor("white")
                axins.patch.set_alpha(1)
                axins.patch.set_edgecolor("black")
                axins.patch.set_linewidth(0.5)
                xticks = [0.7 * fit.numax, fit.numax, 1.3 * fit.numax]
                labels_dict = {
                    0.7 * fit.numax: r"$0.7\nu_{\mathrm{max}}$",
                    fit.numax: r"$\nu_{\mathrm{max}}$",
                    1.3 * fit.numax: r"$1.3\nu_{\mathrm{max}}$",
                }

                def tick_formatter(x, _):
                    for tv, lbl in labels_dict.items():
                        if np.isclose(x, tv, rtol=1e-4):
                            return lbl
                    return ""

                axins.set_xticks(xticks)
                axins.xaxis.set_major_formatter(FuncFormatter(tick_formatter))
                axins.xaxis.set_minor_locator(plt.NullLocator())

            # Fill where peak dominates (loglike_ratio > 2)
            dominant = np.asarray(loglike_ratio) > 2
            ax1.fill_between(
                freq, y1=mod_back, y2=power,
                where=dominant, facecolor="cyan", alpha=1,
                edgecolor="cyan", linewidth=0.08, step="mid", zorder=50,
            )

    # ---- Axis formatting ----
    legend = ax1.legend(loc="upper right", fontsize=8, facecolor="white",
                        framealpha=1, edgecolor="black", ncol=2,
                        bbox_to_anchor=(0.65, 0.99))
    legend.set_zorder(100)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Frequency [$\mu$Hz]")
    ax1.set_ylabel(r"Power Density [ppm$^2/\mu$Hz]")
    ax1.set_title(f"Model {fit.model} for {fit.name}", fontsize=16)
    ax1.set_xlim(0.5*np.min(freq), np.max(freq))

    if fit.bin_data:
        lower_ylim = np.min(power) - 0.3 * np.min(power)
        data_max = np.max(power)
    else:
        lower_ylim = np.min(fit.power_binned) - 0.3 * np.min(fit.power_binned)
        data_max = np.max(fit.power_binned)

    # Ensure components are visible
    for comp_amp in component_levels:
        lower_ylim = min(lower_ylim, comp_amp / 10**_PDS_COMPONENT_MARGIN_DEX)

    if lower_ylim < 1e-3 and fit.data_format == "STAR":
        lower_ylim = 1e-3

    log_lo = np.log10(max(lower_ylim, 1e-10))
    log_peak = np.log10(max(data_max, lower_ylim * 10))
    upper_ylim = 10 ** (log_lo + (log_peak - log_lo) / _PDS_DATA_MAX_FRACTION)

    ax1.set_ylim(lower_ylim, upper_ylim)

    plt.draw()
    fig.tight_layout()
    fig.savefig(fit.PDSfigname, dpi=500)
    plt.close(fig)

    # ======================== Posterior corner plot ========================
    try:
        cornerfig = plt.figure(figsize=(20, 18), dpi=300)
        PlotData = np.vstack([fit.fitdata[p] for p in fit.fitparams]).T
        labels = [v for v in fit.paramlabels.values()]
        quantiles = [0.16, 0.84]
        corner.corner(
            PlotData,
            labels=labels,
            labelpad=0.3,
            label_kwargs={"fontsize": 15},
            show_titles=True,
            title_quantiles=[0.16, 0.50, 0.84],
            title_kwargs={"fontsize": 8.5},
            quantiles=quantiles,
            fig=cornerfig,
            quiet=True,
        )
        cornerfig.subplots_adjust(left=0.06, bottom=0.06)
        cornerfig.suptitle(
            f"Posterior corner plot of model {fit.model} for {fit.name}", fontsize=40
        )
        _save_corner(cornerfig, fit.Cornerfigname)
        plt.close(cornerfig)
    except ValueError as e:
        print(f"WARNING: Failed to create posterior corner plot for {fit.name}")
        print(f"  Error: {e}")
        if "cornerfig" in dir():
            plt.close(cornerfig)

    # ======================== Prior / Posterior corner ========================
    try:
        corner2 = plt.figure(figsize=(20, 18), dpi=300)
        prior_samples = fit.prior_samples
        PlotData_priors = np.vstack([prior_samples[p] for p in fit.fitparams]).T
        labels = [v for v in fit.paramlabels.values()]

        prior_medians = [np.median(prior_samples[k]) for k in fit.fitparams]
        prior_84 = []
        prior_16 = []
        for param in fit.fitparams:
            arr = pd.Series(prior_samples[param])
            sort = sorted(arr)
            idx = int(len(sort) * 0.16)
            plus = sort[len(sort) - idx] - np.median(arr)
            minus = np.median(arr) - sort[idx]
            prior_84.append(plus)
            prior_16.append(minus)

        quantiles = [0.16, 0.84]
        cornerfig = corner.corner(
            PlotData_priors,
            weights=np.ones(len(prior_samples["W"])) / len(prior_samples["W"]),
            labels=labels,
            color="blue",
            fig=corner2,
            quiet=True,
        )

        try:
            cornerfig = corner.corner(
                PlotData,
                weights=np.ones(len(fit.fitdata["W"])) / len(fit.fitdata["W"]),
                labels=labels,
                labelpad=0.3,
                label_kwargs={"fontsize": 15},
                show_titles=True,
                title_quantiles=[0.16, 0.50, 0.84],
                title_kwargs={"fontsize": 8.5},
                quantiles=quantiles,
                fig=corner2,
                quiet=True,
            )
            text = [
                f"{labels[i]} = {prior_medians[i]:.2f} +{prior_84[i]:.2f} -{prior_16[i]:.2f}"
                for i in range(len(fit.fitparams))
            ]
            x_text, y_text = 0.8, 0.85
            for i, t in enumerate(text):
                corner2.text(x_text, y_text - i * 0.03, t,
                             transform=corner2.transFigure, fontsize=15)
            corner2.text(x_text, y_text + 0.05, "Prior centroids:",
                         transform=corner2.transFigure, fontsize=20)
            corner2.subplots_adjust(left=0.06, bottom=0.06)
            corner2.suptitle(
                f"Priors and posteriors of model {fit.model} for {fit.name}", fontsize=40
            )
        except ValueError as e:
            print(f"WARNING: Failed to overlay posteriors on prior/posterior corner plot")
            print(f"  Error: {e}")
            print(f"  Plotting priors only...")
            text = [
                f"{labels[i]} = {prior_medians[i]:.2f} +{prior_84[i]:.2f} -{prior_16[i]:.2f}"
                for i in range(len(fit.fitparams))
            ]
            x_text, y_text = 0.8, 0.85
            for i, t in enumerate(text):
                corner2.text(x_text, y_text - i * 0.03, t,
                             transform=corner2.transFigure, fontsize=15)
            corner2.text(x_text, y_text + 0.05, "Prior centroids:",
                         transform=corner2.transFigure, fontsize=20)
            corner2.subplots_adjust(left=0.06, bottom=0.06)
            corner2.suptitle(
                f"Prior distributions for model {fit.model} for {fit.name} (posteriors failed)",
                fontsize=40, color="red",
            )

        _save_corner(corner2, fit.PriorPostCornerfigname)
        plt.close(corner2)

    except ValueError as e:
        print(f"WARNING: Failed to create prior/posterior corner plot for {fit.name}")
        print(f"  Error: {e}")
        if "corner2" in dir():
            plt.close(corner2)

    # ======================== Correlated-params corner ========================
    if fit.corr_inference:
        try:
            corrfig = plt.figure(figsize=(12, 10), dpi=300)
            numax_samples = fit.fitdata["numax_fit"].values
            # Reconstruct the physical granulation samples from scatter parameters
            gran_funcs = {"a": gran_a, "b": gran_b, "d": gran_d}
            units = {"a": "ppm", "b": r"$\mu$Hz", "d": r"$\mu$Hz"}
            cols = [numax_samples]
            labels = [r"$\nu_{max}$ [$\mu$Hz]"]
            for p in ("a", "b", "d"):
                if f"sigma_{p}" in fit.fitdata.columns:
                    cols.append(gran_funcs[p](numax_samples,
                                              fit.fitdata[f"sigma_{p}"].values))
                elif p in fit.fitdata.columns:
                    cols.append(fit.fitdata[p].values)
                else:
                    continue
                labels.append(f"${p}$ [{units[p]}]")

            corner.corner(
                np.vstack(cols).T,
                labels=labels,
                weights=np.ones(len(numax_samples)) / len(numax_samples),
                label_kwargs={"fontsize": 15},
                show_titles=True,
                title_quantiles=[0.16, 0.50, 0.84],
                title_kwargs={"fontsize": 8.5},
                quantiles=[0.16, 0.84],
                fig=corrfig,
                quiet=True,
            )
            corrfig.subplots_adjust(top=0.92)
            corrfig.suptitle(
                f"Correlated inference corner plot for model {fit.model} for {fit.name}",
                fontsize=20,
            )
            # Filename: Cornerfigname with _corrparams suffix
            plotname = fit.Cornerfigname.replace(".pdf", "_corrparams.pdf")
            _save_corner(corrfig, plotname)
            plt.close(corrfig)
        except ValueError as e:
            print(f"WARNING: Failed to create correlated inference corner plot for {fit.name}")
            print(f"  Error: {e}")
            if "corrfig" in dir():
                plt.close(corrfig)

    return True


# ---------------------------------------------------------------------------
# Radius inference diagnostics (two-panel posterior figure)
# ---------------------------------------------------------------------------

_RADIUS_COLORS = {
    "J": np.array([230, 159, 0]) / 255,
    "H": np.array([0, 128, 0]) / 255,
    "T": np.array([0, 114, 178]) / 255,
}


def radius_posterior_plot(radius_result, outpath):
    """Two-panel radius diagnostic:
    left  - per-model radius posteriors with their BMA weights,
    right - the BMA-combined posterior with median and 16/84 percentiles.
    """
    from scipy import stats

    R_combined = radius_result["R_combined"]
    model_results = radius_result["model_results"]

    all_s = np.concatenate(
        [model_results[m]["samples"] for m in ["H", "J", "T"] if m in model_results]
        + [radius_result["combined_samples"]]
    )
    lo = np.percentile(all_s, 0.5)
    hi = np.percentile(all_s, 99.5)
    span = max(R_combined - lo, hi - R_combined) * 1.05
    x_min, x_max = R_combined - span, R_combined + span
    x_g = np.linspace(x_min, x_max, 600)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for m in model_results:
        s = model_results[m]["samples"]
        R_m = model_results[m]["R_median"]
        w = radius_result["weights"].get(m, np.nan)
        color = _RADIUS_COLORS.get(m, "tab:gray")
        ax_l.hist(s, bins=75, density=True, color=color,
                  alpha=0.50, edgecolor="none")
        kde = stats.gaussian_kde(s)
        ax_l.plot(x_g, kde(x_g), color=color, lw=1.5,
                  label=rf"$R_{{{m}}} = {R_m:.2f}\,R_{{\odot}}\; (w={w:.2f})$")
        ax_l.axvline(R_m, color=color, linestyle="--", lw=1.2, alpha=0.8)

    ax_l.set_xlabel(r"Radius [$R_\odot$]")
    ax_l.set_ylabel("Density")
    ax_l.set_title("Per-model posteriors")
    ax_l.legend(fontsize=10, frameon=False, loc="upper left")

    s_c = radius_result["combined_samples"]
    R_16 = radius_result["R_16_combined"]
    R_84 = radius_result["R_84_combined"]
    ax_r.hist(s_c, bins=75, density=True, color="black", alpha=0.50, edgecolor="none")
    kde_c = stats.gaussian_kde(s_c)
    ax_r.plot(x_g, kde_c(x_g), color="black", lw=1.6)
    ax_r.axvline(R_combined, color="black", linestyle="--", lw=1.8)
    ax_r.axvline(R_16, color="black", linestyle="-.", lw=1.2, alpha=0.8)
    ax_r.axvline(R_84, color="black", linestyle="-.", lw=1.2, alpha=0.8)
    R_p = R_84 - R_combined
    R_m_ = R_combined - R_16
    ax_r.text(0.03, 0.97,
              rf"$R_{{\mathrm{{gran}}}} = {R_combined:.2f}_{{-{R_m_:.2f}}}^{{+{R_p:.2f}}}\,R_\odot$",
              transform=ax_r.transAxes, ha="left", va="top", fontsize=12)
    ax_r.set_xlabel(r"Radius [$R_\odot$]")
    ax_r.set_title("BMA combined posterior")

    for ax in (ax_l, ax_r):
        ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    fig.suptitle(f"Radius inference for {radius_result['star_name']}", y=1.02)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True
