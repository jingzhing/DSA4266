from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _split_from_relpath(relpath: str) -> str:
    parts = relpath.lower().split("/")
    if "test" in parts:
        return "test"
    if "train" in parts:
        return "train"
    if "val" in parts:
        return "val"
    return "unknown"


def _choose_duplicate_keep(files: List[str]) -> Tuple[str, List[str], str]:
    """
    Deterministic duplicate policy:
    1) If duplicate spans test + train, keep test (protect evaluation integrity).
    2) Otherwise keep lexicographically first path.
    """
    ordered = sorted(files)
    test_files = [f for f in ordered if _split_from_relpath(f) == "test"]
    train_files = [f for f in ordered if _split_from_relpath(f) == "train"]

    if test_files and train_files:
        keep = test_files[0]
        reason = "cross_split_keep_test_drop_train"
    else:
        keep = ordered[0]
        reason = "same_split_keep_lexicographic_first"
    remove = [f for f in ordered if f != keep]
    return keep, remove, reason


def _move_to_quarantine(root: Path, relpath: str, quarantine_root: Path, bucket: str) -> Dict[str, str]:
    src = root / relpath
    dst = quarantine_root / bucket / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        if dst.exists():
            return {
                "status": "already_quarantined",
                "src": str(src),
                "dst": str(dst),
            }
        return {
            "status": "missing_source",
            "src": str(src),
            "dst": str(dst),
        }

    shutil.move(str(src), str(dst))
    return {
        "status": "moved",
        "src": str(src),
        "dst": str(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw dataset quality issues from Stage-1 inventory.")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    parser.add_argument("--inventory", default="", help="Path to dataset_inventory.json")
    parser.add_argument(
        "--quarantine-dir",
        default="_quarantine",
        help="Relative directory under dataset root for moved files",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    inventory_path = Path(args.inventory).resolve() if args.inventory else root / "dataset_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory file not found: {inventory_path}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    quarantine_root = root / args.quarantine_dir
    quarantine_root.mkdir(parents=True, exist_ok=True)

    corrupted = inventory.get("corrupted_files", [])
    duplicate_groups = inventory.get("duplicates_md5_groups", [])

    corrupted_actions: List[Dict[str, str]] = []
    for item in corrupted:
        relpath = str(item["relpath"])
        action = _move_to_quarantine(root, relpath, quarantine_root, "corrupted")
        action["reason"] = "corrupted_decode_failure"
        corrupted_actions.append(action)

    duplicate_actions: List[Dict[str, str]] = []
    policy_reasons: Dict[str, int] = {}
    for group in duplicate_groups:
        md5 = str(group["md5"])
        files = list(group["files"])
        keep, remove, reason = _choose_duplicate_keep(files)
        policy_reasons[reason] = policy_reasons.get(reason, 0) + 1
        for relpath in remove:
            action = _move_to_quarantine(root, relpath, quarantine_root, "duplicates")
            action["reason"] = reason
            action["md5"] = md5
            action["keep"] = keep
            duplicate_actions.append(action)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "inventory_source": str(inventory_path),
        "policy": {
            "corrupted": "move all corrupted files into _quarantine/corrupted",
            "duplicates": (
                "remove one file from each exact-MD5 duplicate group; "
                "if group spans train/test keep test and drop train, else keep lexicographically first"
            ),
        },
        "summary": {
            "corrupted_candidates": len(corrupted),
            "duplicates_groups": len(duplicate_groups),
            "corrupted_actions_moved": sum(1 for a in corrupted_actions if a["status"] == "moved"),
            "duplicate_actions_moved": sum(1 for a in duplicate_actions if a["status"] == "moved"),
            "policy_reason_counts": policy_reasons,
        },
        "corrupted_actions": corrupted_actions,
        "duplicate_actions": duplicate_actions,
    }

    report_path = root / "data_cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    policy_md = root / "duplicate_policy.md"
    policy_md.write_text(
        "\n".join(
            [
                "# Duplicate Policy",
                "",
                "1. Identify exact duplicates using MD5 groups from `dataset_inventory.json`.",
                "2. For each duplicate group:",
                "   - If duplicate spans `train` and `test`, keep the `test` file and quarantine the `train` copy.",
                "   - Otherwise keep the lexicographically first path and quarantine the other copy.",
                "3. Quarantined duplicates are moved to `_quarantine/duplicates/...` preserving relative paths.",
                "4. Corrupted files are moved to `_quarantine/corrupted/...`.",
                "",
                f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {policy_md}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
