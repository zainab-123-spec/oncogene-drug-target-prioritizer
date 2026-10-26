"""Command line interface.

Examples
--------
python -m odtp build
python -m odtp train
python -m odtp analyze EGFR
python -m odtp all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from odtp.config import DISEASES, TARGETS
from odtp.http import HttpClient
from odtp.pipeline import analyze_target, build_dataset, train_models

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odtp",
        description=(
            "Multi-Disease Drug Target & Bioactivity Prioritizer. "
            "Collect Open Targets, ChEMBL, UniProt, and PDB evidence, "
            "then evaluate potency models with a scaffold split and a random split."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Download APIs, run QC, and write data/processed tables")
    _add_paths(build)

    train = sub.add_parser("train", help="Train and evaluate models, then write figures")
    _add_paths(train)
    train.add_argument(
        "--fast",
        action="store_true",
        help="Fewer trees for a smoke test. Do not use this for reported benchmarks.",
    )

    analyze = sub.add_parser("analyze", help="Write an HTML report for one gene, e.g. EGFR")
    analyze.add_argument("gene", help="HGNC symbol, for example EGFR, PCSK9, or BACE1")
    _add_paths(analyze)

    all_cmd = sub.add_parser(
        "all",
        help="Build the dataset, train, and write several target reports",
    )
    _add_paths(all_cmd)
    all_cmd.add_argument("--fast", action="store_true")
    return parser


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "build":
        http = HttpClient(args.cache_dir)
        build_dataset(args.processed_dir, http=http)
        return 0
    if args.command == "train":
        train_models(
            args.processed_dir,
            args.figure_dir,
            args.report_dir,
            fast=args.fast,
        )
        return 0
    if args.command == "analyze":
        analyze_target(args.gene, args.processed_dir, args.figure_dir, args.report_dir)
        return 0
    if args.command == "all":
        http = HttpClient(args.cache_dir)
        build_dataset(args.processed_dir, http=http)
        train_models(
            args.processed_dir,
            args.figure_dir,
            args.report_dir,
            fast=args.fast,
        )
        for symbol in ("EGFR", "PCSK9", "BACE1", "CCR5", "JAK1", "ACHE", "DPP4"):
            if any(target.symbol == symbol for target in TARGETS):
                analyze_target(symbol, args.processed_dir, args.figure_dir, args.report_dir)
        logger.info(
            "Finished. Diseases: %s. Targets: %s.",
            len(DISEASES),
            len(TARGETS),
        )
        return 0
    return 2
