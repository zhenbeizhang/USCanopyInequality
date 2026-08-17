from __future__ import annotations

from uscanopy.paths import (analysis_store_root, external_input_root, manuscript_root, model_store_root, repo_root)

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = analysis_store_root() / "Result" / "Result4" / "versions"
ANALYSIS = ROOT / "leave_one_unit_out_benchmark"
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "04_neighbourhood_typology"
    / "run_leave_one_unit_out_benchmarks.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("r4_loo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()


def direct_expected(block: pd.DataFrame, event: str) -> float:
    g = block[f"{event}_g"].to_numpy(float)
    h = block["h"].to_numpy(float)
    v = block["v"].to_numpy(float)
    n = len(block)
    if n < 2:
        return float("nan")
    values = []
    for index in range(n):
        keep = np.arange(n) != index
        values.append(g[keep].mean() * h[keep].mean() * v[keep].mean())
    return float(np.sum(values))


def main() -> None:
    frame = pd.read_csv(mod.FRAME_PATH, dtype={"GEOID_str": str}, low_memory=False)
    frame["n"] = 1
    frame = mod.add_event_flags(frame)
    ranges = pd.read_csv(mod.RANGE_PATH).set_index("city_key")
    rng = np.random.default_rng(20260804)
    errors = []
    blocks_tested = 0
    for city, sub in frame.groupby("city_key", sort=True):
        coords = sub[["x_5070_m", "y_5070_m"]].to_numpy(float)
        anchor = coords.min(axis=0)
        size = float(ranges.loc[city, "main_block_size_km"]) * 1000
        labels = mod.assign_grid(coords, anchor, size, mod.OFFSETS[0])
        for _, block in sub.assign(block=labels).groupby("block", sort=False):
            if len(block) < 2 or rng.random() > 0.08:
                continue
            for event in mod.EVENTS:
                totals = block[mod.aggregate_columns(event)].sum()
                analytic = mod.loo_expected_count_from_totals(totals, event)
                brute = direct_expected(block, event)
                errors.append(abs(analytic - brute))
            blocks_tested += 1

    stored = pd.read_csv(ANALYSIS / "tables" / "result4_leave_one_unit_out_benchmarks.csv")
    primary = stored.loc[
        stored["event"].eq("joint_context")
        & stored["benchmark"].eq("main_block_leave_one_unit_out")
        & stored["grid_origin"].astype(str).eq("origin_mean")
    ].iloc[0]
    draws = pd.read_csv(ANALYSIS / "diagnostics" / "result4_leave_one_unit_out_bootstrap_draws.csv.gz")
    selected_draws = draws.loc[
        draws["event"].eq("joint_context")
        & draws["benchmark"].eq("main_block_leave_one_unit_out"),
        "ratio",
    ]
    interval = selected_draws.quantile([0.025, 0.975]).to_numpy(float)
    audit = {
        "status": "PASS" if max(errors, default=0.0) < 1e-12 and len(selected_draws) == 2000 else "FAIL",
        "random_blocks_directly_checked": blocks_tested,
        "event_block_checks": len(errors),
        "maximum_formula_vs_bruteforce_error": max(errors, default=0.0),
        "bootstrap_draws_recalculated": len(selected_draws),
        "stored_interval": [float(primary["bootstrap_p025"]), float(primary["bootstrap_p975"])],
        "recalculated_interval": interval.tolist(),
        "interval_max_absolute_error": float(
            np.max(np.abs(interval - np.asarray([primary["bootstrap_p025"], primary["bootstrap_p975"]], dtype=float)))
        ),
        "claim_audit": {
            "ratio_above_one_after_loo": bool(primary["observed_expected_ratio"] > 1),
            "interval_above_one": bool(primary["bootstrap_p025"] > 1),
            "formal_spatial_null_claim_permitted": False,
        },
    }
    (ANALYSIS / "qa" / "result4_independent_formula_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
