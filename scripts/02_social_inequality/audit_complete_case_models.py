from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "result2_audit", HERE / "complete_case_model_core.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not import the primary audit implementation")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


frame = audit.load_data()
complete, _ = audit.freeze_complete_case(frame)
complete = audit.standardise_within_city(complete, audit.ALL_MARGINAL)
formula = (
    "canopy_reference_0p6m ~ "
    + " + ".join(audit.JOINT_Z)
    + " + C(result_city_key)"
)
model = smf.wls(formula, data=complete, weights=complete["population"]).fit()
reference = pd.read_csv(audit.TABLES / "06_joint_complete_case_restandardized.csv")
rows = []
for predictor in audit.JOINT_Z:
    expected = float(reference.loc[reference["predictor"].eq(predictor), "estimate_pp_per_within_city_sd"].iloc[0])
    independent = float(model.params[predictor])
    rows.append(
        {
            "predictor": predictor,
            "primary_within_transform_estimate": expected,
            "independent_explicit_city_dummy_wls_estimate": independent,
            "absolute_difference": abs(expected - independent),
        }
    )
comparison = pd.DataFrame(rows)
comparison.to_csv(audit.QA / "independent_statsmodels_city_dummy_check.csv", index=False, encoding="utf-8-sig")
maximum = float(comparison["absolute_difference"].max())
result = {
    "implementation": "statsmodels WLS with 49 explicit city indicators and an intercept",
    "n_spatial_units": int(model.nobs),
    "n_cities": int(complete["result_city_key"].nunique()),
    "design_rank": int(model.model.rank),
    "maximum_absolute_slope_difference": maximum,
    "pass_tolerance": 1e-10,
    "pass": maximum < 1e-10,
}
(audit.QA / "independent_statsmodels_city_dummy_check.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
final_audit_path = audit.QA / "final_audit.json"
if final_audit_path.exists():
    final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    final_audit["independent_implementation_check"] = result
    final_audit.setdefault("audit_checks", {})[
        "independent_explicit_city_dummy_wls_reproduces_joint_slopes"
    ] = bool(result["pass"])
    final_audit_path.write_text(
        json.dumps(final_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
print(json.dumps(result, indent=2))
