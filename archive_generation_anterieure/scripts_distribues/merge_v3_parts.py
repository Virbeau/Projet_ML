#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path


def merge_parts(parts_dir: Path, output_file: Path):
    part_files = sorted(parts_dir.glob("part_*.json"))
    if not part_files:
        raise FileNotFoundError(f"No part_*.json found in {parts_dir}")

    merged_instances = []
    part_metas = []

    for pf in part_files:
        with pf.open("r") as f:
            data = json.load(f)
        inst = data.get("instances", [])
        meta = data.get("metadata", {})
        merged_instances.extend(inst)
        part_metas.append({
            "file": pf.name,
            "n_instances": len(inst),
            "metadata": meta,
        })

    out_payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "merged_from": str(parts_dir),
            "n_parts": len(part_files),
            "n_instances": len(merged_instances),
            "parts": part_metas,
        },
        "instances": merged_instances,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"Merged {len(part_files)} parts -> {output_file}")
    print(f"Total instances: {len(merged_instances)}")


def main():
    parser = argparse.ArgumentParser(description="Merge distributed v3 part files")
    parser.add_argument("--parts-dir", required=True, help="Directory containing part_*.json")
    parser.add_argument("--output", required=True, help="Output merged dataset JSON")
    args = parser.parse_args()

    merge_parts(Path(args.parts_dir), Path(args.output))


if __name__ == "__main__":
    main()
