import argparse
import os

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


def cmd_serve(host: str, port: int) -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Serve mode requires uvicorn installed") from exc

    uvicorn.run("app.server:app", host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info").lower())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion-augmented segmentation starter app")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run generation + quality-control curation")
    run_p.add_argument("--config", required=True, help="Path to JSON config")

    serve_p = sub.add_parser("serve", help="Run API server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Bind port")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args.config)
    elif args.command == "serve":
        cmd_serve(args.host, args.port)


if __name__ == "__main__":
    main()
