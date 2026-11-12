"""Compute window features for one experiment.

    python scripts/compute_windows.py --input_file <exp>_analyze/saved_data/instant_df.csv --n_jobs 16

Writes instant_df_processed.csv and time_windowed_df_processed.csv next to the input
(or into --output_dir). Window settings come from configs/pipeline.yaml.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sptnano import config, windows  # noqa: E402


def main():
    settings = config.load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", default=None, help="default: folder of --input_file")
    parser.add_argument("--n_jobs", type=int, default=-1, help="-1 = all cores")
    parser.add_argument("--chunk_size", type=int, default=settings["windows"]["chunk_size"])
    parser.add_argument("--time_between_frames", type=float, default=settings["acquisition"]["frame_interval_s"])
    args = parser.parse_args()

    input_file = config.fix_path(args.input_file)
    output_dir = config.fix_path(args.output_dir) if args.output_dir else input_file.parent
    windows.compute_windows(
        input_file,
        output_dir,
        window_params=windows.window_params_from_settings(settings),
        frame_interval_s=args.time_between_frames,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
