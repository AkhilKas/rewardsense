#!/usr/bin/env python3
"""Fail if any src/model_pipeline module is below a required line coverage threshold."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_xml", type=Path)
    parser.add_argument("--threshold", type=float, default=80.0)
    args = parser.parse_args()

    if not args.coverage_xml.exists():
        print(f"Coverage XML not found: {args.coverage_xml}")
        return 2

    root = ET.parse(args.coverage_xml).getroot()
    below: list[tuple[str, float]] = []

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        if not filename.endswith(".py"):
            continue
        line_rate = float(cls.attrib.get("line-rate", "0"))
        pct = line_rate * 100.0
        if pct < args.threshold:
            below.append((filename, pct))

    below.sort(key=lambda x: x[1])

    if not below:
        print(f"PASS: all src/model_pipeline modules >= {args.threshold:.1f}%")
        return 0

    print(f"FAIL: {len(below)} modules below {args.threshold:.1f}%")
    for filename, pct in below:
        print(f"  - {filename}: {pct:.1f}%")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
