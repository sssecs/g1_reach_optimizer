#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("unitree_rl_mjlab")
args=ap.parse_args()

repo=Path(args.unitree_rl_mjlab).expanduser().resolve()
src=repo/"src/assets/robots/unitree_g1/xmls"
dst=Path("assets/g1")
dst.mkdir(parents=True,exist_ok=True)
shutil.copy2(src/"g1.xml",dst/"g1.xml")
if (dst/"assets").exists():
    shutil.rmtree(dst/"assets")
shutil.copytree(src/"assets",dst/"assets")
print("Copied G1 MJCF to",dst.resolve())
