"""Core data containers: user-facing Star input and the per-fit FitCase state.

FitCase replaces the SingleFit class that was duplicated across the original
NSGF main scripts. Attribute names are kept identical to SingleFit so ported
functions remain directly comparable to the originals.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Star:
    """User-facing description of one star to analyse.

    Either `numax` is given directly, or `logg` and `teff` are given and
    numax is estimated from the scaling relation (see data.calc_numax).
    `teff` and `feh` (with uncertainties) are additionally required for
    radius inference; `mission` selects the bolometric correction.
    """

    name: str
    pds_file: str
    numax: float | None = None
    logg: float | None = None
    teff: float | None = None
    teff_err: float | None = None
    feh: float | None = None
    feh_err: float | None = None
    mission: str = "Kepler"  # 'Kepler' or 'TESS'
    nyquist: float | None = None  # muHz; defaults per mission if None

    def __post_init__(self):
        if self.numax is None and (self.logg is None or self.teff is None):
            raise ValueError(
                f"Star '{self.name}': provide either numax directly, "
                "or both logg and teff for the scaling-relation estimate."
            )


@dataclass
class FitCase:
    """State of a single fit: one star x one background model x one mode.

    Attribute names mirror the original SingleFit class.
    """

    # General information
    name: str
    model: str
    data_format: str = "STAR"
    seed: int | None = None
    noisy: bool | None = None
    inits: dict | None = None
    fitparams: list | None = None
    logPars: list | None = None
    paramlabels: list | None = None

    # Data
    bin_data: bool | None = None
    bin_size: int | None = None
    freq: np.ndarray | None = None
    power: np.ndarray | None = None
    freq_raw: np.ndarray | None = None
    power_raw: np.ndarray | None = None
    freq_binned: np.ndarray | None = None
    power_binned: np.ndarray | None = None
    freq_removed: np.ndarray | None = None
    power_removed: np.ndarray | None = None
    dof: float | None = None
    eta: np.ndarray | None = None
    numax: float | None = None
    dnu: float | None = None

    # Fit inputs
    priors: dict | None = None
    prior_samples: dict | None = None

    # Fit results
    fitdata: object | None = None
    fitresults: object | None = None

    # I/O
    respath: str | None = None
    fitdatafile: str | None = None
    fitresultsfile: str | None = None
    PDSfigname: str | None = None
    Cornerfigname: str | None = None
    priorCornerfigname: str | None = None
    priorpredictivePDSfigname: str | None = None
    PriorPostCornerfigname: str | None = None

    # Star data input
    PDSfile: str | None = None
    nyquist: float | None = None

    # Likelihood setup
    likelihood: object | None = None
    bagmean: float | None = None
    sigmoid: object | None = None
    nlive_enhance: float | None = None
    pulsmod: str | None = None  # 'Gaussian' or 'Peakbogging'
    sigwidth: float | None = None  # Peakbogging mixture coefficient width

    # Correlated inference
    corr_inference: bool = False

    # Stellar parameters for radius inference (new in Granalyse)
    teff: float | None = None
    teff_err: float | None = None
    feh: float | None = None
    feh_err: float | None = None
    mission: str = "Kepler"
