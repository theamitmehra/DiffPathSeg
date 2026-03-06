import argparse

from .config import load_config
from .pipeline import run_pipeline
from .utils import seed_everything


def cmd_run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    metrics = run_pipeline(cfg)

    print("Pipeline completed")
    for k, v in metrics.items():
        print(f"{k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion-augmented segmentation starter app")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run generation + quality-control curation")
    run_p.add_argument("--config", required=True, help="Path to JSON config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args.config)


if __name__ == "__main__":
    main()
