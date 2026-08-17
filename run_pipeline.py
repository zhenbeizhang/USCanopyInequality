"""Orchestrate the frozen analysis modules without embedding local paths."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

STAGES: dict[str, tuple[str, ...]] = {
    "1": (
        "01_canopy_mapping/aggregate_canopy_context.py",
        "01_canopy_mapping/validation/validate_annotation_patches.py",
        "01_canopy_mapping/validation/check_checkpoint_compatibility.py",
        "01_canopy_mapping/validation/audit_inference_seams.py",
        "01_canopy_mapping/validation/audit_overlap_inference.py",
        "01_canopy_mapping/validation/audit_geographic_split.py",
        "01_canopy_mapping/validation/summarize_imagery_timing.py",
    ),
    "2": (
        "02_social_inequality/build_boundary_population_frame.py",
        "02_social_inequality/run_primary_social_models.py",
        "02_social_inequality/complete_case_model_core.py",
        "02_social_inequality/audit_complete_case_models.py",
        "02_social_inequality/audit_complete_case_suppression.py",
        "02_social_inequality/run_cross_scale_transferability.py",
        "02_social_inequality/summarize_cross_scale_transferability.py",
        "02_social_inequality/audit_cross_scale_transferability.py",
        "02_social_inequality/run_boundary_sensitivity.py",
    ),
    "3": (
        "03_surface_heat/run_support_aligned_models.py",
        "03_surface_heat/run_spatial_block_bootstrap.py",
        "03_surface_heat/run_paired_model_bootstrap.py",
        "03_surface_heat/audit_nonlinearity.py",
    ),
    "4": (
        "04_neighbourhood_typology/build_neighbourhood_typology.py",
        "04_neighbourhood_typology/run_leave_one_unit_out_benchmarks.py",
        "04_neighbourhood_typology/run_threshold_sensitivity.py",
        "04_neighbourhood_typology/audit_neighbourhood_typology.py",
    ),
    "5": (
        "05_allocation/freeze_allocation_protocol.py",
        "05_allocation/verify_allocation_protocol.py",
        "05_allocation/run_equal_budget_allocation.py",
        "05_allocation/run_budget_sensitivity.py",
        "05_allocation/run_group_distribution_sensitivity.py",
        "05_allocation/audit_weighting_heterogeneity.py",
        "05_allocation/freeze_tradeoff_outcomes.py",
        "05_allocation/run_divergence_decomposition.py",
        "05_allocation/audit_divergence_decomposition.py",
    ),
    "transfer": (
        "05_allocation/build_transferability_predictors.py",
        "05_allocation/run_cross_city_transferability.py",
        "05_allocation/audit_cross_city_transferability.py",
        "05_allocation/assess_cross_city_transferability.py",
        "05_allocation/verify_cross_city_transferability.py",
    ),
    "figures": tuple(
        [f"06_main_figures/make_figure_{index}.py" for index in range(1, 6)]
        + [f"07_supplementary_figures/make_supplementary_figure_{index:02d}.py" for index in range(1, 17)]
    ),
    "qa": tuple(
        [
            "08_qa/audit_figure_1.py",
            "08_qa/audit_result_4.py",
            "08_qa/audit_figure_5.py",
            "08_qa/build_table_sources.py",
            "08_qa/audit_table_sources.py",
            "08_qa/audit_reporting_consistency.py",
        ]
        + [f"08_qa/audit_supplementary_figure_{index:02d}.py" for index in range(1, 17)]
    ),
}


def static_check() -> list[str]:
    problems: list[str] = []
    for script in sorted(SCRIPTS.rglob("*.py")):
        source = script.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            problems.append(f"syntax: {script.relative_to(ROOT)}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.match(r"^[A-Za-z]:[\\/]", node.value):
                    problems.append(f"absolute path: {script.relative_to(ROOT)}:{node.lineno}")
    for stage, names in STAGES.items():
        for name in names:
            if not (SCRIPTS / name).exists():
                problems.append(f"missing stage {stage} script: {name}")
    return problems


def command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    src = str(ROOT / "src")
    environment["PYTHONPATH"] = src + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(STAGES),
        default=["1", "2", "3", "4", "5", "transfer", "figures", "qa"],
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    problems = static_check()
    if problems:
        raise SystemExit("Static release check failed:\n" + "\n".join(problems))
    print("Static release check passed")
    if args.check_only:
        return

    commands = [SCRIPTS / name for stage in args.stages for name in STAGES[stage]]
    for index, script in enumerate(commands, start=1):
        command = [sys.executable, str(script)]
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT, env=command_environment(), check=False)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
