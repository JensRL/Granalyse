"""Config-file generator: `granalyse make-config`.

Builds a human-readable YAML scaffold from a CSV/TSV star table, a directory
of PDS files, or both.  The emitted file is valid input for load_config when
all required fields are present; otherwise it includes explicit null
placeholders so the user knows exactly what to fill in.
"""

import csv
import glob
import io
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Name extraction (port of original framework's extract_kic_name)
# ---------------------------------------------------------------------------

def _extract_name(filename: str) -> str:
    """Extract a star identifier from a PDS filename."""
    base = os.path.basename(filename)
    if "KIC" in base:
        return base.split("_")[0].split("KIC")[1]
    if "kplr" in base:
        return base.split("_")[0].split("r")[1].lstrip("0")
    if "_" in base:
        return base.split("_")[0]
    return base.split(".")[0]


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

# Columns we know how to use; everything else is ignored.
_KNOWN_COLS = {
    "name", "pds_file", "numax", "logg", "teff", "teff_err",
    "feh", "feh_err", "mission", "kind", "nyquist",
}
_FLOAT_COLS = {"numax", "logg", "teff", "teff_err", "feh", "feh_err", "nyquist"}


def _read_csv(path: str) -> list[dict]:
    """Read a CSV or TSV, sniffing the delimiter.  Returns list of row dicts."""
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel          # fall back to comma
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV '{path}' contains no data rows.")

    # Warn about unknown columns
    header_cols = set(rows[0].keys())
    unknown = header_cols - _KNOWN_COLS
    if unknown:
        print(f"[make-config] Notice: ignoring unrecognised CSV columns: "
              f"{', '.join(sorted(unknown))}", file=sys.stderr)

    # Coerce types
    result = []
    for row in rows:
        cleaned: dict[str, Any] = {}
        for col in _KNOWN_COLS:
            val = row.get(col, "").strip()
            if val == "" or val.lower() in ("", "none", "null", "na", "n/a"):
                cleaned[col] = None
            elif col in _FLOAT_COLS:
                try:
                    cleaned[col] = float(val)
                except ValueError:
                    raise ValueError(
                        f"CSV column '{col}' has non-numeric value '{val}'."
                    )
            else:
                cleaned[col] = val
        result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Star record assembly
# ---------------------------------------------------------------------------

def _is_complete(star: dict, radius_inference: bool) -> tuple[bool, list[str]]:
    """Return (complete, list_of_missing_fields).

    A star is 'complete' when it has a pds_file, enough info for fitting,
    AND (if radius_inference is on) the required Teff fields.
    """
    missing = []

    if not star.get("pds_file"):
        missing.append("pds_file")

    has_numax = star.get("numax") is not None
    has_logg  = star.get("logg") is not None
    has_teff  = star.get("teff") is not None
    if not has_numax and not (has_logg and has_teff):
        if not has_numax:
            missing.append("numax")
        if not has_logg:
            missing.append("logg")
        if not has_teff:
            missing.append("teff")

    if radius_inference:
        if not has_teff and "teff" not in missing:
            missing.append("teff")
        if star.get("teff_err") is None:
            missing.append("teff_err")

    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# YAML emission helpers
# ---------------------------------------------------------------------------

def _yaml_val(v: Any) -> str:
    """Format a scalar value for YAML output."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # Avoid scientific notation for typical stellar values
        return f"{v:g}"
    return str(v)


def _star_block(star: dict, missing: list[str], placeholder_comment: bool) -> str:
    """Render one star entry as a YAML block string."""
    lines = []

    def line(key: str, value: Any, comment: str = "") -> None:
        val_str = _yaml_val(value)
        suffix = f"  # {comment}" if comment else ""
        lines.append(f"    {key}: {val_str}{suffix}")

    def placeholder(key: str, comment: str = "FILL IN") -> None:
        lines.append(f"    {key}: null  # {comment}")

    # Quote name if it looks numeric so YAML doesn't parse it as an integer.
    name_val = star["name"]
    try:
        float(name_val)  # catches '11111', '8006161', etc.
        name_repr = f'"{name_val}"'
    except (ValueError, TypeError):
        name_repr = name_val
    lines.append(f"  - name: {name_repr}")

    # pds_file
    if star.get("pds_file"):
        line("pds_file", star["pds_file"])
    else:
        placeholder("pds_file", "path to the power spectrum ASCII file")

    # mission / kind
    if star.get("mission"):
        line("mission", star["mission"])
    if star.get("kind") and star["kind"] != "star":
        line("kind", star["kind"])

    # numax vs logg/teff
    if star.get("numax") is not None:
        line("numax", star["numax"], "muHz")
    elif "numax" in missing:
        placeholder("numax", "muHz — or delete and supply logg + teff instead")

    if star.get("logg") is not None:
        line("logg", star["logg"])
    if star.get("teff") is not None:
        line("teff", star["teff"], "K")
    elif "teff" in missing:
        placeholder("teff", "K")

    if star.get("teff_err") is not None:
        line("teff_err", star["teff_err"], "K")
    elif "teff_err" in missing:
        placeholder("teff_err", "K")

    if star.get("feh") is not None:
        line("feh", star["feh"])
    if star.get("feh_err") is not None:
        line("feh_err", star["feh_err"])

    if star.get("nyquist") is not None:
        line("nyquist", star["nyquist"], "muHz")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def collect_star_records(
    stars_csv: str | None = None,
    pds_dir: str | None = None,
    pattern: str = "*.pow",
    mission: str = "Kepler",
) -> list[dict]:
    """Assemble star-record dicts from a CSV table and/or a PDS directory.

    Three modes (mirroring the make-config CLI):
      CSV + pds_dir : join rows to files by extracted name (fills pds_file)
      CSV only      : pds_file comes from the table
      pds_dir only  : names from filenames, stellar params left null

    Each record is a dict with the keys consumed by load_config (name,
    pds_file, numax, logg, teff, teff_err, feh, feh_err, mission, kind,
    nyquist). Shared by `granalyse make-config` and config.load_config.
    """
    csv_rows: list[dict] = _read_csv(stars_csv) if stars_csv else []

    pds_files: list[str] = []
    if pds_dir:
        globbed = sorted(glob.glob(os.path.join(pds_dir, pattern)))
        if not globbed:
            print(f"[make-config] Warning: no files matched '{pattern}' "
                  f"in '{pds_dir}'.", file=sys.stderr)
        pds_files = globbed

    if not csv_rows and not pds_files:
        raise ValueError("Provide at least one of stars_csv or pds_dir.")

    stars: list[dict] = []

    if csv_rows and pds_files:
        # Matching mode: join CSV rows to files by extracted name.
        file_map: dict[str, str] = {
            _extract_name(f).strip(): f for f in pds_files
        }
        matched_names: set[str] = set()
        for row in csv_rows:
            csv_name = str(row.get("name") or "").strip()
            if not csv_name:
                print("[make-config] Warning: CSV row has empty name; skipping.",
                      file=sys.stderr)
                continue
            matched_file = file_map.get(csv_name)
            entry = dict(row)
            if matched_file:
                matched_names.add(csv_name)
                if not entry.get("pds_file"):
                    entry["pds_file"] = matched_file
            else:
                print(f"[make-config] Warning: CSV star '{csv_name}' has no "
                      f"matching PDS file.", file=sys.stderr)
            if not entry.get("mission"):
                entry["mission"] = mission
            stars.append(entry)

        unmatched_files = set(file_map) - matched_names
        for name in sorted(unmatched_files):
            print(f"[make-config] Warning: PDS file for '{name}' has no "
                  f"matching CSV row.", file=sys.stderr)

    elif csv_rows:
        # CSV-only: pds_file must come from the table.
        for row in csv_rows:
            entry = dict(row)
            if not entry.get("name"):
                print("[make-config] Warning: CSV row has empty name; skipping.",
                      file=sys.stderr)
                continue
            if not entry.get("mission"):
                entry["mission"] = mission
            stars.append(entry)

    else:
        # pds-dir-only: names from filenames, no stellar params.
        for f in pds_files:
            name = _extract_name(f)
            stars.append({
                "name": name,
                "pds_file": f,
                "numax": None, "logg": None, "teff": None, "teff_err": None,
                "feh": None, "feh_err": None, "mission": mission,
                "kind": None, "nyquist": None,
            })

    if not stars:
        raise ValueError("No stars could be assembled from the provided inputs.")
    return stars


def generate_config(
    *,
    output: str = "config.yaml",
    output_dir: str = "./results",
    models: str = "H,J,T",
    mission: str = "Kepler",
    no_radius_inference: bool = False,
    stars_csv: str | None = None,
    pds_dir: str | None = None,
    pattern: str = "*.pow",
) -> tuple[str, bool]:
    """Generate a Granalyse YAML config scaffold.

    Parameters
    ----------
    output:               path to write the YAML file
    output_dir:           output_dir value written into the YAML
    models:               comma-separated model letters (e.g. 'H,J,T')
    mission:              fallback mission when not in CSV
    no_radius_inference:  if True, write radius_inference: false
    stars_csv:            path to a CSV/TSV with star metadata
    pds_dir:              directory to glob for PDS files
    pattern:              glob pattern within pds_dir

    Returns
    -------
    (output_path, all_complete) — all_complete is True when every star has
    all required fields (and load_config was called successfully).
    """
    radius_inference = not no_radius_inference

    # -----------------------------------------------------------------------
    # 1. Collect star records
    # -----------------------------------------------------------------------

    stars = collect_star_records(stars_csv, pds_dir, pattern, mission)

    # -----------------------------------------------------------------------
    # 2. Completeness check
    # -----------------------------------------------------------------------

    completeness: list[tuple[dict, list[str]]] = []
    all_complete = True
    for star in stars:
        complete, missing = _is_complete(star, radius_inference)
        completeness.append((star, missing))
        if not complete:
            all_complete = False

    incomplete = [(s, m) for s, m in completeness if m]
    if incomplete:
        print("\n[make-config] Incomplete stars — fill in the marked fields:",
              file=sys.stderr)
        col_w = max(len(s["name"]) for s, _ in incomplete)
        print(f"  {'Star':<{col_w}}  Missing fields", file=sys.stderr)
        print(f"  {'-'*col_w}  {'-'*30}", file=sys.stderr)
        for s, m in incomplete:
            print(f"  {s['name']:<{col_w}}  {', '.join(m)}", file=sys.stderr)
        print("", file=sys.stderr)

    # -----------------------------------------------------------------------
    # 3. Emit YAML
    # -----------------------------------------------------------------------

    model_list = ", ".join(m.strip() for m in models.split(","))

    buf = io.StringIO()

    buf.write("# Granalyse run configuration\n")
    buf.write("#\n")
    buf.write("# Run with:  granalyse run config.yaml\n")
    buf.write("# Generated by: granalyse make-config\n")
    buf.write("#\n")
    buf.write("# Fields marked  # FILL IN  are required before running.\n")
    buf.write("\n")

    buf.write(f"output_dir: {output_dir}\n")
    buf.write("\n")
    buf.write("# Background models: D/J/H/T or aliases harvey1/hybrid/harvey2/harvey3.\n")
    buf.write(f"models: [{model_list}]\n")
    buf.write("\n")
    buf.write("pulsation_model: Gaussian      # Gaussian (default) or Peakbogging\n")
    buf.write("correlated_inference: true     # granulation params via Kallinger scaling + scatter\n")
    buf.write("\n")
    buf.write("binning:\n")
    buf.write("  enabled: true\n")
    buf.write("  resolution: 0.5              # target bin resolution [muHz]\n")
    buf.write("  visual_bins: 10              # plot-only binning\n")
    buf.write("\n")
    buf.write("low_frequency_cut: 10          # discard frequencies below this [muHz]; null to keep\n")
    buf.write("\n")
    buf.write("sampling:\n")
    buf.write("  nlive_factor: null           # multiply default nlive; null = default\n")
    buf.write("  seed: null                   # RNG seed for reproducibility\n")
    buf.write("\n")
    ri_val = "false" if no_radius_inference else "true"
    buf.write(f"radius_inference: {ri_val}         # requires teff (and ideally feh) per star\n")
    buf.write("\n")
    buf.write("stars:\n")

    for star, missing in completeness:
        has_placeholders = bool(missing)
        buf.write(_star_block(star, missing, has_placeholders))
        buf.write("\n\n")

    yaml_text = buf.getvalue().rstrip("\n") + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        f.write(yaml_text)

    print(f"[make-config] Wrote {len(stars)} star(s) to '{output}'.",
          file=sys.stderr)

    # -----------------------------------------------------------------------
    # 4. Validate with load_config when all stars are complete
    # -----------------------------------------------------------------------

    if all_complete:
        try:
            from .config import load_config
            load_config(output)
            print(f"[make-config] Validation passed — '{output}' is ready to run.",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[make-config] Warning: load_config validation failed: {exc}",
                  file=sys.stderr)
            all_complete = False

    return output, all_complete


# ---------------------------------------------------------------------------
# CLI entry point (called from cli.py)
# ---------------------------------------------------------------------------

def add_subparser(subparsers) -> None:
    """Register the make-config subcommand on an argparse subparsers object."""
    p = subparsers.add_parser(
        "make-config",
        help="Generate a YAML config scaffold from a star table or PDS directory.",
        description=(
            "Build a human-readable Granalyse config from a CSV/TSV star table, "
            "a directory of PDS spectra, or both.  Fields that cannot be inferred "
            "are emitted as  null  # FILL IN  placeholders."
        ),
    )
    p.add_argument(
        "--stars",
        metavar="TABLE.CSV",
        help="CSV/TSV file with star metadata (delimiter sniffed automatically).  "
             "Recognised columns: name, pds_file, numax, logg, teff, teff_err, "
             "feh, feh_err, mission, kind, nyquist.",
    )
    p.add_argument(
        "--pds-dir",
        metavar="DIR",
        help="Directory containing PDS spectrum files.  Star names are extracted "
             "from filenames automatically.",
    )
    p.add_argument(
        "--pattern",
        default="*.pow",
        metavar="GLOB",
        help="Glob pattern for PDS files within --pds-dir (default: '*.pow').",
    )
    p.add_argument(
        "--output",
        default="config.yaml",
        metavar="FILE",
        help="Destination YAML file (default: config.yaml).",
    )
    p.add_argument(
        "--output-dir",
        default="./results",
        metavar="DIR",
        help="output_dir written into the YAML (default: ./results).",
    )
    p.add_argument(
        "--models",
        default="H,J,T",
        metavar="LETTERS",
        help="Comma-separated background models (default: H,J,T).",
    )
    p.add_argument(
        "--mission",
        default="Kepler",
        choices=["Kepler", "TESS"],
        help="Fallback mission when not specified in CSV (default: Kepler).",
    )
    p.add_argument(
        "--no-radius-inference",
        action="store_true",
        default=False,
        help="Set radius_inference: false (relaxes teff requirement).",
    )
