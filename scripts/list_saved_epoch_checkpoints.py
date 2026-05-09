#!/usr/bin/env python3
"""List ep*_model.pth under <dvc>/log/checkpoints/<run_name>/ and print embedded epoch + ID."""
from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dvc", required=True, help="Experiment folder (--dvc), absolute path.")
    p.add_argument("--run-name", required=True, help="Training run id / checkpoint subfolder (same as --run_name).")
    args = p.parse_args()

    ckpt_dir = os.path.join(os.path.abspath(args.dvc), "log", "checkpoints", args.run_name)
    if not os.path.isdir(ckpt_dir):
        print(f"Not a directory: {ckpt_dir}", file=sys.stderr)
        return 1

    paths = sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("ep") and f.endswith("_model.pth")
    )
    if not paths:
        print(f"No ep*_model.pth in {ckpt_dir}")
        return 0

    for path in paths:
        try:
            ckpt = torch.load(path, map_location="cpu")
            ep = ckpt.get("epoch", "?")
            rid = ckpt.get("ID", "?")
            print(f"{path}\tepoch={ep}\tID={rid}")
        except Exception as ex:
            print(f"{path}\t(load error: {ex})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
