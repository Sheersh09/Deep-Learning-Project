"""Command-line prediction using the saved baseline model."""

import argparse
import json
import sys
from collections.abc import Sequence

from .inference import load_baseline_model, predict_image


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify one tomato leaf image.")
    parser.add_argument("image_path", help="Path to a JPG, PNG, or BMP image")
    args = parser.parse_args(argv)

    try:
        model = load_baseline_model()
        result = predict_image(args.image_path, model)
    except (OSError, ValueError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
