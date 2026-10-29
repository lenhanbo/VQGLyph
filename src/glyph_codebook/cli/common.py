from __future__ import annotations

import argparse


def training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description); parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--resume"); parser.add_argument("--epochs", type=int); parser.add_argument("--max-steps", type=int)
    return parser
