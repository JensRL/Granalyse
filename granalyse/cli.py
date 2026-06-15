"""Command-line interface: `granalyse run config.yaml`."""

import argparse
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="granalyse",
        description="Granulation Analysis with Nested Sampling for "
                    "Stellar Radius Inference",
    )
    parser.add_argument("--version", action="version", version=f"granalyse {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline from a YAML configuration.")
    run_p.add_argument("config", help="Path to the YAML run configuration.")

    from .make_config import add_subparser as _add_make_config
    _add_make_config(sub)

    args = parser.parse_args(argv)

    if args.command == "run":
        from .config import load_config
        from .pipeline import run_pipeline
        cfg = load_config(args.config)
        run_pipeline(cfg)

    elif args.command == "make-config":
        from .make_config import generate_config
        generate_config(
            output=args.output,
            output_dir=args.output_dir,
            models=args.models,
            mission=args.mission,
            no_radius_inference=args.no_radius_inference,
            stars_csv=args.stars,
            pds_dir=args.pds_dir,
            pattern=args.pattern,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
