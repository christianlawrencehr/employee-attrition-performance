from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("IBM-HR-Employee-Attrition-Performance.xlsx")
MODEL_PATH = Path("artifacts/attrition_model.joblib")
TOP_DRIVERS_PATH = Path("artifacts/top_3_drivers.csv")
ASSUMPTIONS_PATH = Path("roi_assumptions.json")
OUTPUT_DIR = Path("artifacts")
EXECUTIVE_SUMMARY_PATH = OUTPUT_DIR / "executive_summary.md"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_currency(amount: float) -> str:
    return f"${amount:,.0f}"


def select_high_risk_segment(df: pd.DataFrame, probs: pd.Series, feature: str) -> tuple[pd.Series, str]:
    series = df[feature]

    if pd.api.types.is_numeric_dtype(series):
        # Use quartiles for numeric features, then pick the highest-risk band.
        binned = pd.qcut(series, q=4, duplicates="drop")
        risk_by_bin = probs.groupby(binned, observed=False).mean().sort_values(ascending=False)
        top_bin = risk_by_bin.index[0]
        mask = binned == top_bin
        rule_text = f"{feature} in {top_bin}"
        return mask, rule_text

    risk_by_value = probs.groupby(series, observed=False).mean().sort_values(ascending=False)
    top_value = risk_by_value.index[0]
    mask = series == top_value
    rule_text = f"{feature} == {top_value}"
    return mask, rule_text


def get_driver_config(assumptions: dict, feature: str) -> dict:
    default_cfg = assumptions["default_intervention"]
    return assumptions.get("interventions", {}).get(feature, default_cfg)


def scenario_row(
    feature: str,
    scenario: str,
    cost_per_attrition: float,
    target_mask: pd.Series,
    probs: pd.Series,
    cfg: dict,
    targeting_rule: str,
) -> dict:
    coverage = float(cfg["coverage"])
    effect_size = float(cfg["effect_size"][scenario])
    fixed_cost = float(cfg["fixed_program_cost"])
    var_cost = float(cfg["variable_cost_per_employee"])

    targeted_employees = int(target_mask.sum())
    treated_employees = int(round(targeted_employees * coverage))

    prob_reduction = probs[target_mask] * coverage * effect_size
    avoided_leavers = float(prob_reduction.sum())
    gross_savings = avoided_leavers * cost_per_attrition
    program_cost = fixed_cost + (treated_employees * var_cost)
    net_savings = gross_savings - program_cost
    roi = (net_savings / program_cost) if program_cost > 0 else 0.0
    break_even_avoided = (program_cost / cost_per_attrition) if cost_per_attrition > 0 else 0.0

    return {
        "intervention": feature,
        "scenario": scenario,
        "targeting_rule": targeting_rule,
        "targeted_employees": targeted_employees,
        "coverage": coverage,
        "treated_employees": treated_employees,
        "effect_size": effect_size,
        "avoided_leavers": avoided_leavers,
        "cost_per_attrition": cost_per_attrition,
        "gross_savings_usd": gross_savings,
        "program_cost_usd": program_cost,
        "net_savings_usd": net_savings,
        "roi": roi,
        "break_even_avoided_leavers": break_even_avoided,
    }


def combined_program_row(
    scenario: str,
    drivers: list[str],
    masks: dict[str, pd.Series],
    probs: pd.Series,
    assumptions: dict,
) -> dict:
    cost_per_attrition = float(assumptions["cost_per_attrition"][scenario])
    adjusted_probs = probs.copy()
    total_program_cost = 0.0
    targeting_bits: list[str] = []
    treated_counts = 0
    targeted_counts = 0

    for driver in drivers:
        cfg = get_driver_config(assumptions, driver)
        coverage = float(cfg["coverage"])
        effect_size = float(cfg["effect_size"][scenario])
        fixed_cost = float(cfg["fixed_program_cost"])
        var_cost = float(cfg["variable_cost_per_employee"])
        mask = masks[driver]

        targeted = int(mask.sum())
        treated = int(round(targeted * coverage))
        targeted_counts += targeted
        treated_counts += treated
        total_program_cost += fixed_cost + (treated * var_cost)

        # Apply effects sequentially so combined impact compounds realistically.
        adjusted_probs.loc[mask] = adjusted_probs.loc[mask] * (1 - (coverage * effect_size))
        targeting_bits.append(driver)

    baseline_expected = float(probs.sum())
    scenario_expected = float(adjusted_probs.sum())
    avoided_leavers = baseline_expected - scenario_expected
    gross_savings = avoided_leavers * cost_per_attrition
    net_savings = gross_savings - total_program_cost
    roi = (net_savings / total_program_cost) if total_program_cost > 0 else 0.0
    break_even_avoided = (total_program_cost / cost_per_attrition) if cost_per_attrition > 0 else 0.0

    return {
        "intervention": "Combined Top 3 Program",
        "scenario": scenario,
        "targeting_rule": " + ".join(targeting_bits),
        "targeted_employees": targeted_counts,
        "coverage": None,
        "treated_employees": treated_counts,
        "effect_size": None,
        "avoided_leavers": avoided_leavers,
        "cost_per_attrition": cost_per_attrition,
        "gross_savings_usd": gross_savings,
        "program_cost_usd": total_program_cost,
        "net_savings_usd": net_savings,
        "roi": roi,
        "break_even_avoided_leavers": break_even_avoided,
    }


def write_summary(
    baseline_expected: float,
    top_drivers: list[str],
    driver_rules: dict[str, str],
    results_df: pd.DataFrame,
    overtime_ten_percent: dict[str, float],
) -> None:
    combined = results_df[results_df["intervention"] == "Combined Top 3 Program"].copy()
    combined = combined.sort_values("scenario")

    lines = [
        "# Retention ROI Scenario Summary",
        "",
        f"- Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Baseline expected leavers (model-based): {baseline_expected:.2f}",
        f"- Top 3 drivers analyzed: {', '.join(top_drivers)}",
        "",
        "## Targeting Definitions",
    ]
    for driver in top_drivers:
        lines.append(f"- {driver}: {driver_rules[driver]}")

    lines.extend(
        [
            "",
            "## Combined Program Outcomes (Top 3 Interventions Together)",
        ]
    )

    for row in combined.itertuples(index=False):
        lines.append(
            f"- {row.scenario.title()}: avoided leavers {row.avoided_leavers:.2f}, "
            f"gross savings {to_currency(row.gross_savings_usd)}, "
            f"program cost {to_currency(row.program_cost_usd)}, "
            f"net savings {to_currency(row.net_savings_usd)}, ROI {row.roi:.2f}x"
        )

    lines.extend(
        [
            "",
            "## Executive Q&A",
            "",
            "### Q: Based on your data, how can we save money by addressing Overtime?",
        ]
    )

    overtime_rows = results_df[results_df["intervention"] == "OverTime"].copy()
    overtime_rows = overtime_rows.sort_values("scenario")
    for row in overtime_rows.itertuples(index=False):
        lines.append(
            f"- {row.scenario.title()}: avoid {row.avoided_leavers:.2f} leavers, "
            f"gross savings {to_currency(row.gross_savings_usd)}, "
            f"net savings {to_currency(row.net_savings_usd)}, ROI {row.roi:.2f}x"
        )

    lines.extend(
        [
            "",
            "### Q: How much would we save if we reduced overtime by 10%?",
            f"- Expected avoided leavers: {overtime_ten_percent['avoided_leavers']:.2f}",
            f"- Gross savings (base cost per attrition): {to_currency(overtime_ten_percent['gross_savings_usd'])}",
            f"- Net savings at 80% coverage (base program assumptions): {to_currency(overtime_ten_percent['net_savings_80_usd'])}",
            f"- Net savings at 100% coverage (same assumptions): {to_currency(overtime_ten_percent['net_savings_100_usd'])}",
            "",
            "## Notes",
            "- Scenario assumptions are configurable in roi_assumptions.json.",
            "- Savings estimates are directional planning values, not accounting outcomes.",
            "- Driver interventions are modeled as risk reduction, not causal certainty.",
        ]
    )

    (OUTPUT_DIR / "roi_summary.md").write_text("\n".join(lines), encoding="utf-8")


def append_executive_qa_to_summary(results_df: pd.DataFrame, overtime_ten_percent: dict[str, float]) -> None:
    if not EXECUTIVE_SUMMARY_PATH.exists():
        return

    existing = EXECUTIVE_SUMMARY_PATH.read_text(encoding="utf-8")
    marker = "## Executive Q&A"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()

    overtime_rows = results_df[results_df["intervention"] == "OverTime"].copy()
    overtime_rows = overtime_rows.sort_values("scenario")

    qa_lines = [
        "",
        "## Executive Q&A",
        "",
        "### Q: Based on your data, how can we save money by addressing Overtime?",
    ]
    for row in overtime_rows.itertuples(index=False):
        qa_lines.append(
            f"- {row.scenario.title()}: avoid {row.avoided_leavers:.2f} leavers, "
            f"gross savings {to_currency(row.gross_savings_usd)}, "
            f"net savings {to_currency(row.net_savings_usd)}, ROI {row.roi:.2f}x"
        )

    qa_lines.extend(
        [
            "",
            "### Q: How much would we save if we reduced overtime by 10%?",
            f"- Expected avoided leavers: {overtime_ten_percent['avoided_leavers']:.2f}",
            f"- Gross savings (base cost per attrition): {to_currency(overtime_ten_percent['gross_savings_usd'])}",
            f"- Net savings at 80% coverage (base program assumptions): {to_currency(overtime_ten_percent['net_savings_80_usd'])}",
            f"- Net savings at 100% coverage (same assumptions): {to_currency(overtime_ten_percent['net_savings_100_usd'])}",
        ]
    )

    EXECUTIVE_SUMMARY_PATH.write_text(existing + "\n" + "\n".join(qa_lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = [p for p in [DATA_PATH, MODEL_PATH, TOP_DRIVERS_PATH, ASSUMPTIONS_PATH] if not p.exists()]
    if missing:
        missing_text = ", ".join(str(m) for m in missing)
        raise FileNotFoundError(f"Missing required files: {missing_text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    assumptions = load_json(ASSUMPTIONS_PATH)
    top_drivers_df = pd.read_csv(TOP_DRIVERS_PATH)
    top_drivers = top_drivers_df["raw_feature"].head(3).tolist()

    probs = pd.Series(model.predict_proba(df.drop(columns=["Attrition"]))[:, 1], index=df.index)
    baseline_expected = float(probs.sum())

    masks: dict[str, pd.Series] = {}
    driver_rules: dict[str, str] = {}
    for driver in top_drivers:
        if driver not in df.columns:
            continue
        mask, rule = select_high_risk_segment(df, probs, driver)
        masks[driver] = mask
        driver_rules[driver] = rule

    scenarios = ["low", "base", "high"]
    rows: list[dict] = []
    for scenario in scenarios:
        cost_per_attrition = float(assumptions["cost_per_attrition"][scenario])
        for driver in top_drivers:
            if driver not in masks:
                continue
            cfg = get_driver_config(assumptions, driver)
            rows.append(
                scenario_row(
                    feature=driver,
                    scenario=scenario,
                    cost_per_attrition=cost_per_attrition,
                    target_mask=masks[driver],
                    probs=probs,
                    cfg=cfg,
                    targeting_rule=driver_rules[driver],
                )
            )

        rows.append(
            combined_program_row(
                scenario=scenario,
                drivers=[d for d in top_drivers if d in masks],
                masks=masks,
                probs=probs,
                assumptions=assumptions,
            )
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "retention_roi_scenarios.csv", index=False)

    overtime_base_cfg = get_driver_config(assumptions, "OverTime")
    overtime_mask = masks.get("OverTime", pd.Series(False, index=df.index))
    base_cost_per_attrition = float(assumptions["cost_per_attrition"]["base"])
    targeted_overtime = int(overtime_mask.sum())
    fixed_cost = float(overtime_base_cfg["fixed_program_cost"])
    var_cost = float(overtime_base_cfg["variable_cost_per_employee"])
    effect_10 = 0.10
    avoided_100 = float((probs[overtime_mask] * effect_10).sum())
    gross_100 = avoided_100 * base_cost_per_attrition
    program_cost_80 = fixed_cost + int(round(targeted_overtime * 0.8)) * var_cost
    program_cost_100 = fixed_cost + targeted_overtime * var_cost
    avoided_80 = avoided_100 * 0.8
    gross_80 = avoided_80 * base_cost_per_attrition
    overtime_ten_percent = {
        "avoided_leavers": avoided_100,
        "gross_savings_usd": gross_100,
        "net_savings_80_usd": gross_80 - program_cost_80,
        "net_savings_100_usd": gross_100 - program_cost_100,
    }

    write_summary(
        baseline_expected=baseline_expected,
        top_drivers=[d for d in top_drivers if d in masks],
        driver_rules=driver_rules,
        results_df=results_df,
        overtime_ten_percent=overtime_ten_percent,
    )
    append_executive_qa_to_summary(results_df=results_df, overtime_ten_percent=overtime_ten_percent)

    print("ROI scenario analysis complete.")
    print("Saved:", (OUTPUT_DIR / "retention_roi_scenarios.csv").resolve())
    print("Saved:", (OUTPUT_DIR / "roi_summary.md").resolve())
    print(f"Baseline expected leavers: {baseline_expected:.2f}")


if __name__ == "__main__":
    main()
