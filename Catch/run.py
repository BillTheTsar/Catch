import argparse
import importlib
import sys


def load_config(config_path: str):
    """
    Dynamically import a config module.
    Defaults to 'config.py' in current directory.
    """
    module_name = config_path.replace(".py", "").replace("/", ".").replace("\\", ".")
    return importlib.import_module(module_name)


def override_config(CONFIG, args):
    """
    Apply CLI overrides to CONFIG (mutating frozen dataclasses via __dict__).
    """
    # Runtime overrides
    if args.device_index is not None:
        object.__setattr__(CONFIG.runtime, "device_index", args.device_index)

    if args.camera_height is not None:
        object.__setattr__(CONFIG.runtime, "camera_height", args.camera_height)

    if args.telemetry is not None:
        object.__setattr__(CONFIG.telemetry, "enabled", args.telemetry)

    if args.write_csv is not None:
        object.__setattr__(CONFIG.runtime, "write_csv", args.write_csv)

    if args.show_print is not None:
        object.__setattr__(CONFIG.runtime, "show_print", args.show_print)

    # Annotator-specific
    if args.video is not None:
        object.__setattr__(CONFIG.paths, "video_path", args.video)


def main():
    parser = argparse.ArgumentParser(description="Run Catch system")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---------- MAIN ----------
    parser_main = subparsers.add_parser("main", help="Run live inference")

    # ---------- ANNOTATOR ----------
    parser_annot = subparsers.add_parser("annotator", help="Run video annotator")

    # ---------- SHARED FLAGS ----------
    parser.add_argument("--config", type=str, default="config",
                        help="Path to config module (default: config)")

    parser.add_argument("--device-index", type=int, help="Camera index override")
    parser.add_argument("--video", type=str, help="Input video path (annotator) override")

    parser.add_argument("--camera-height", type=float, help="Camera height override")

    parser.add_argument("--telemetry", type=lambda x: x.lower() == "true",
                        help="Enable/disable telemetry (true/false) override")

    parser.add_argument("--write-csv", type=lambda x: x.lower() == "true",
                        help="Enable/disable CSV output override")

    parser.add_argument("--show-print", type=lambda x: x.lower() == "true",
                        help="Enable/disable debug prints override")

    args = parser.parse_args()

    # ---------- LOAD CONFIG ----------
    config_module = load_config(args.config)
    CONFIG = config_module.CONFIG

    # ---------- APPLY OVERRIDES ----------
    override_config(CONFIG, args)

    # ---------- DISPATCH ----------
    if args.mode == "main":
        from main import run
        run(CONFIG)
    elif args.mode == "annotator":
        from annotator import run
        run(CONFIG)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()