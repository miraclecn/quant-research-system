#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqt.features import FEATURE_COLUMNS, build_factor_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate factor registry metadata for production-readiness.")
    parser.add_argument("--output", default="outputs/factor_registry_check.json", help="Output JSON report path")
    args = parser.parse_args()

    registry = build_factor_registry(FEATURE_COLUMNS)
    required_columns = [
        "feature",
        "family",
        "subfamily",
        "template",
        "expression",
        "params",
        "data_dependencies",
        "calc_source",
        "production_ready",
        "direction_hint",
        "status",
    ]
    missing_columns = [column for column in required_columns if column not in registry.columns]
    duplicated_features = sorted(registry["feature"][registry["feature"].duplicated()].unique().tolist())
    invalid_calc_source = sorted(
        registry.loc[~registry["calc_source"].isin(["internal"]), "feature"].tolist()
    )
    not_production_ready = sorted(
        registry.loc[registry["production_ready"] != True, "feature"].tolist()
    )

    report = {
        "feature_count": int(len(registry)),
        "missing_columns": missing_columns,
        "duplicated_features": duplicated_features,
        "invalid_calc_source": invalid_calc_source,
        "not_production_ready": not_production_ready,
        "status": "ok" if not any([missing_columns, duplicated_features, invalid_calc_source, not_production_ready]) else "error",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
