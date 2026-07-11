from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from dotenv import load_dotenv

load_dotenv()

from pipeline.config import BENCHMARK, BENCHMARKS, CAMERA_TYPE, build_pipeline_config
from pipeline.main import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="VLM-guided manipulation pipeline")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default=BENCHMARK)
    parser.add_argument("--camera-type", choices=["gemini", "realsense"], default=CAMERA_TYPE)
    parser.add_argument("--camera-serial")
    parser.add_argument(
        "--skip-axis-reasoning",
        action="store_true",
        help="Use configured sampling constraints instead of asking a VLM to reason about sampling axes.",
    )
    parser.add_argument("--board-x-mj", type=float, default=-0.12)
    parser.add_argument("--board-y-mj", type=float, default=0.0)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable interactive previews and debug-only artifact capture.",
    )
    parsed = parser.parse_args()

    return build_pipeline_config(
        parsed.benchmark,
        camera_type=parsed.camera_type,
        camera_serial=parsed.camera_serial,
        axis_reasoning=not parsed.skip_axis_reasoning,
        debug=parsed.debug,
        board_x_mj=parsed.board_x_mj,
        board_y_mj=parsed.board_y_mj,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
