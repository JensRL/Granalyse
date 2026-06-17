"""YAML run-configuration loading and validation."""

import os
from dataclasses import dataclass, field

import yaml

from .models import get_model
from .star import Star

VALID_PULSMODS = ("Gaussian", "Peakbogging")
VALID_MISSIONS = ("Kepler", "TESS")

# Prefer the libyaml-backed safe loader: ~5x faster on large inline-star configs
# (each HPC node task parses the whole config to pick out its one star).
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


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
        raw = yaml.load(f, Loader=_YAML_LOADER)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping.")
    if "models" not in raw or not raw["models"]:
        raise ValueError("Config must list at least one background model under 'models'.")

    # Stars come from an inline 'stars' list or a 'stars_csv' table (+ optional
    # 'pds_dir' to match PDS files by name). stars_csv/pds_dir are resolved
    # relative to the config file so the run is portable across machines.
    config_dir = os.path.dirname(os.path.abspath(path))
    if raw.get("stars_csv"):
        if raw.get("stars"):
            raise ValueError("Config has both 'stars' and 'stars_csv'; use only one.")
        from .make_config import collect_star_records

        def _rel(p):
            return p if os.path.isabs(p) else os.path.join(config_dir, p)

        pds_dir = raw.get("pds_dir")
        star_entries = collect_star_records(
            _rel(raw["stars_csv"]),
            _rel(pds_dir) if pds_dir else None,
            raw.get("pds_pattern", "*.pow"),
            raw.get("mission", "Kepler"),
        )
    elif raw.get("stars"):
        star_entries = raw["stars"]
    else:
        raise ValueError("Config must list stars inline under 'stars' or via 'stars_csv'.")

    models = [get_model(m).name for m in raw["models"]]  # validates + canonicalises

    pulsmod = raw.get("pulsation_model", "Gaussian")
    if pulsmod not in VALID_PULSMODS:
        raise ValueError(f"pulsation_model must be one of {VALID_PULSMODS}, got '{pulsmod}'.")

    binning = raw.get("binning", {})
    sampling = raw.get("sampling", {})

    stars = []
    kinds = {}
    for entry in star_entries:
        if not entry.get("name") or not entry.get("pds_file"):
            raise ValueError(f"Each star needs 'name' and 'pds_file'; got {entry}.")
        mission = entry.get("mission") or "Kepler"
        if mission not in VALID_MISSIONS:
            raise ValueError(f"Star '{entry['name']}': mission must be one of "
                             f"{VALID_MISSIONS}, got '{mission}'.")
        kinds[str(entry["name"])] = entry.get("kind") or "star"
        if kinds[str(entry["name"])] not in ("star", "simulation"):
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
