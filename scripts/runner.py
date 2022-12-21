#!/usr/bin/env python3

import re
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt

LOSS_PATTERN = re.compile(r"Loss: (\d+\.\d+)")
TIME_PATTERN = re.compile(r"time: (\d+\.\d+) sec")
BIN_DIR = Path.home() / "PP-final" / "experiment" / "bin"
IMAGE_DIR = Path.home() / "PP-final" / "experiment" / "image"

VERSION_NAME_MAP = {
    "s": "singleThread",
    "singleThread": "singleThread",
    "o": "openmp",
    "openmp": "openmp",
    "c": "cuda",
    "cuda": "cuda",
}


@dataclass
class RunArgument:
    version: Literal["singleThread", "s", "openmp", "o", "cuda", "c"]
    batch_size: int
    hidden_size: int
    epoch_count: int
    output: str


def configure_arguments() -> RunArgument:
    parser = ArgumentParser("runner")
    parser.add_argument(
        "-v",
        "--version",
        default="s",
        type=str,
    )
    parser.add_argument("-b", "--batch-size", default=64, type=int, dest="batch_size")
    parser.add_argument(
        "-s", "--hidden-size", default=300, type=int, dest="hidden_size"
    )
    parser.add_argument("-o", "--output", default="loss.png", type=str)
    parser.add_argument(
        "-e", "--epoch-count", default=100, type=int, dest="epoch_count"
    )
    return parser.parse_args()


def run_task(args: RunArgument):
    result = subprocess.check_output(
        [
            BIN_DIR / VERSION_NAME_MAP[args.version],
            "--batch-size",
            str(args.batch_size),
            "--hidden-size",
            str(args.hidden_size),
            "--epoch-count",
            str(args.epoch_count),
        ],
        encoding="ascii",
        input="-1\n",
    )

    print(result)

    matched = LOSS_PATTERN.finditer(result)

    loss = []
    if matched is not None:
        loss = [float(loss.group(1)) for loss in matched]

    matched = TIME_PATTERN.search(result)

    time = 0
    if matched is not None:
        time = float(matched.group(1))

    return loss, time


def draw_line_chart(loss: Iterable[float], output_filename: str) -> None:
    plt.plot(range(1, len(loss) + 1), loss)
    plt.title(output_filename.removesuffix(".png").replace("-", " ").capitalize())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(IMAGE_DIR / output_filename)


def main() -> None:
    args = configure_arguments()

    loss, time = run_task(args)

    print(f"time: {time:.4f} sec / epoch")

    draw_line_chart(loss, args.output)


if __name__ == "__main__":
    main()
