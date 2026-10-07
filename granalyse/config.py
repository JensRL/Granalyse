"""YAML run-configuration loading and validation."""

import os
from dataclasses import dataclass, field

import yaml

from .models import get_model
from .radius.loader import RELATION_NUMAX_SPLIT, SHIPPED_RELATIONS, load_relations
from .star import Star

VALID_PULSMODS = ("Gaussian", "Peakbogging")
VALID_MISSIONS = ("Kepler", "TESS")

_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _data_setting(value, key, where):
    """Validate a bin resolution / low-frequency cut"""
    if value == "auto" or (value is None and key == "low_frequency_cut"):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    allowed = "'auto', null" if key == "low_frequency_cut" else "'auto'"
    raise ValueError(f"{where}: {key} must be {allowed} or a positive number [muHz], "
                     f"got {value!r}.")


@dataclass
class RunConfig:
    stars: list
    models: list                    
    output_dir: str = "./results"
    pulsation_model: str = "Gaussian"
    correlated_inference: bool = True
    bin_data: bool = True
    bin_resolution: float | str = "auto"            # muHz; "auto" = by numax
    bin_visuals: int = 10
    low_frequency_cut: float | str | None = "auto"  # muHz; "auto" = by numax, None disables
    nlive_factor: float | None = None
    seed: int | None = None
    radius_inference: bool = True
    relations: dict = field(default_factory=dict)       # {shipped set name: own relation-set directory}
    relation_numax_split: float = RELATION_NUMAX_SPLIT  # muHz; rgb below, ms_sgb above
    skip_replot: bool = False
    star_kinds: dict = field(default_factory=dict)


def load_config(path):
    """Read and validate a YAML run configuration; returns a RunConfig."""
    with open(path) as f:
        raw = yaml.load(f, Loader=_YAML_LOADER)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping.")
    if "models" not in raw or not raw["models"]:
        raise ValueError("Config must list at least one background model under 'models'.")

    config_dir = os.path.dirname(os.path.abspath(path))

    def _rel(p):
        return p if os.path.isabs(p) else os.path.join(config_dir, p)

    if raw.get("stars_csv"):
        if raw.get("stars"):
            raise ValueError("Config has both 'stars' and 'stars_csv'; use only one.")
        from .make_config import collect_star_records

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

    models = [get_model(m).name for m in raw["models"]]  

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
            bin_resolution=_data_setting(entry.get("bin_resolution", "auto"),
                                         "bin_resolution", f"Star '{name}'"),
            low_frequency_cut=_data_setting(entry.get("low_frequency_cut", "auto"),
                                            "low_frequency_cut", f"Star '{name}'"),
        )
        stars.append(star)

    cfg = RunConfig(
        stars=stars,
        models=models,
        output_dir=raw.get("output_dir", "./results"),
        pulsation_model=pulsmod,
        correlated_inference=raw.get("correlated_inference", True),
        bin_data=binning.get("enabled", True),
        bin_resolution=_data_setting(binning.get("resolution", "auto"),
                                     "resolution", "binning"),
        bin_visuals=binning.get("visual_bins", 10),
        low_frequency_cut=_data_setting(raw.get("low_frequency_cut", "auto"),
                                        "low_frequency_cut", "Config"),
        nlive_factor=sampling.get("nlive_factor"),
        seed=sampling.get("seed"),
        radius_inference=raw.get("radius_inference", True),
        relations={k: os.path.normpath(_rel(str(v)))
                   for k, v in (raw.get("relations") or {}).items()},
        relation_numax_split=float(raw.get("relation_numax_split", RELATION_NUMAX_SPLIT)),
        skip_replot=raw.get("skip_replot", False),
        star_kinds=kinds,
    )

    # Own relation sets are checked now rather than after hours of fitting
    rel_models = [m for m in cfg.models if get_model(m).has_radius_relation]
    for name, set_dir in cfg.relations.items():
        if name not in SHIPPED_RELATIONS:
            raise ValueError(f"relations: '{name}' is not one of {SHIPPED_RELATIONS}.")
        if cfg.radius_inference and rel_models:
            load_relations(set_dir, models=rel_models)

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
