"""Run A vs run B regression comparison.

The pitch feature: given two prediction runs on the same GT, report per-class
deltas and an overall PASS/REGRESSED verdict. A class regresses when its
mAP@0.5 delta is below -threshold (strict <, so exactly -0.05 does not flag).
Classes missing from one run get delta=None (undefined, reported not silent).
"""


def compare_runs(
    run_a_metrics: dict,
    run_b_metrics: dict,
    regression_threshold: float = 0.05,
) -> dict:
    """Compare two evaluation results.

    Args:
        run_a_metrics: evaluate() result for baseline (run A)
        run_b_metrics: evaluate() result for candidate (run B)
        regression_threshold: a class regresses when delta < -threshold

    Returns:
        {
          "verdict": "PASS" | "REGRESSED",
          "deltas": {cat_id: {"delta_map50": float | None,
                              "delta_f1": float | None,
                              "regressed": bool}},
          "summary": {"classes": n, "regressed_classes": [ids]},
        }
    """
    per_class_a = run_a_metrics["per_class"]
    per_class_b = run_b_metrics["per_class"]
    all_classes = sorted(set(per_class_a) | set(per_class_b))

    deltas: dict[int, dict] = {}
    regressed_classes: list[int] = []

    for cat_id in all_classes:
        a = per_class_a.get(cat_id)
        b = per_class_b.get(cat_id)

        if a is None or b is None:
            # Class present in one run only → undefined delta, not silently zero
            delta_map50 = None
            delta_f1 = None
        else:
            delta_map50 = b["ap50"] - a["ap50"]
            delta_f1 = b["f1"] - a["f1"]

        # A class with no GT in either run (ap50 == -1) is not evaluable
        if a is not None and b is not None and (a["ap50"] < 0 or b["ap50"] < 0):
            delta_map50 = None
            delta_f1 = None

        regressed = delta_map50 is not None and delta_map50 < -regression_threshold
        if regressed:
            regressed_classes.append(cat_id)

        deltas[cat_id] = {
            "delta_map50": delta_map50,
            "delta_f1": delta_f1,
            "regressed": regressed,
        }

    verdict = "REGRESSED" if regressed_classes else "PASS"

    return {
        "verdict": verdict,
        "deltas": deltas,
        "summary": {
            "classes": len(all_classes),
            "regressed_classes": regressed_classes,
        },
    }
