"""YAML run-configuration loading and validation."""

import os
from dataclasses import dataclass, field

import yaml

from .models import get_model
from .star import Star

VALID_PULSMODS = ("Gaussian", "Peakbogging")
VALID_MISSIONS = ("Kepler", "TESS")


@dataclass
class RunConfig:
    stars: list
    models: list                     # canonical model letters
    output_dir: str = "./results"
    pulsation_model: str = "Gaussian"
    correlated_inference: bool = True
    bin_data: bool = True
    bin_resolution: float = 0.5      # muHz
    bin_visuals: int = 10
    low_frequency_cut: float | None = 10.0  # muHz; None disables
    nlive_factor: float | None = None
    seed: int | None = None
    radius_inference: bool = True
    skip_replot: bool = False  # on resume, reuse existing figures instead of remaking
    star_kinds: dict = field(default_factory=dict)  # name -> 'star'|'simulation'


def load_config(path):
    """Read and validate a YAML run configuration; returns a RunConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping.")
    if "stars" not in raw or not raw["stars"]:
        raise ValueError("Config must list at least one star under 'stars'.")
    if "models" not in raw or not raw["models"]:
        raise ValueError("Config must list at least one background model under 'models'.")

    models = [get_model(m).name for m in raw["models"]]  # validates + canonicalises

    pulsmod = raw.get("pulsation_model", "Gaussian")
    if pulsmod not in VALID_PULSMODS:
        raise ValueError(f"pulsation_model must be one of {VALID_PULSMODS}, got '{pulsmod}'.")

    binning = raw.get("binning", {})
    sampling = raw.get("sampling", {})

    stars = []
    kinds = {}
    for entry in raw["stars"]:
        if "name" not in entry or "pds_file" not in entry:
            raise ValueError(f"Each star needs 'name' and 'pds_file'; got {entry}.")
        mission = entry.get("mission", "Kepler")
        if mission not in VALID_MISSIONS:
            raise ValueError(f"Star '{entry['name']}': mission must be one of "
                             f"{VALID_MISSIONS}, got '{mission}'.")
        kinds[entry["name"]] = entry.get("kind", "star")
        if kinds[entry["name"]] not in ("star", "simulation"):
            raise ValueError(f"Star '{entry['name']}': kind must be 'star' or 'simulation'.")
        name = str(entry["name"])
        # Reject names that could cause path traversal
        if "/" in name or os.sep in name or ".." in name:
            raise ValueError(
                f"Star name '{name}' contains path separators or '..'; "
                "use a plain identifier without slashes."
            )
        star = Star(
            name=name,
            pds_file=entry["pds_file"],
            numax=entry.get("numax"),
            logg=entry.get("logg"),
            teff=entry.get("teff"),
            teff_err=entry.get("teff_err"),
            feh=entry.get("feh"),
            feh_err=entry.get("feh_err"),
            mission=mission,
            nyquist=entry.get("nyquist"),
        )  # Star.__post_init__ enforces numax or (logg, teff)
        stars.append(star)

    cfg = RunConfig(
        stars=stars,
        models=models,
        output_dir=raw.get("output_dir", "./results"),
        pulsation_model=pulsmod,
        correlated_inference=raw.get("correlated_inference", True),
        bin_data=binning.get("enabled", True),
        bin_resolution=binning.get("resolution", 0.5),
        bin_visuals=binning.get("visual_bins", 10),
        low_frequency_cut=raw.get("low_frequency_cut", 10.0),
        nlive_factor=sampling.get("nlive_factor"),
        seed=sampling.get("seed"),
        radius_inference=raw.get("radius_inference", True),
        skip_replot=raw.get("skip_replot", False),
        star_kinds=kinds,
    )

    if cfg.radius_inference:
        for star in cfg.stars:
            missing = []
            if star.teff is None:
                missing.append("teff")
            if star.teff_err is None:
                missing.append("teff_err")
            # feh/feh_err are optional: a population prior handles missing feh
            if missing:
                raise ValueError(
                    f"Star '{star.name}': radius inference requires "
                    f"{', '.join(missing)} "
                    "(set radius_inference: false to skip)."
                )
    return cfg
