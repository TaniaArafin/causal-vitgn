"""Run every figure script in order and report success/failure.

Run:
    PYTHONPATH=. python scripts/figures/run_all.py
"""

import importlib
import sys
import time
import traceback


FIGURES = [
    ("fig1_architecture",       "Fig. 1 — Architecture block diagram"),
    ("fig2_cascade_tree",       "Fig. 2 — Example cascade tree"),
    ("fig3_calibration",        "Fig. 3 — Calibration / reliability plot"),
    ("fig4_uncertainty_virality","Fig. 4 — Posterior uncertainty vs. virality"),
    ("fig5_counterfactual",     "Fig. 5 — Counterfactual outcomes"),
    ("fig6_risk_coverage",      "Fig. 6 — Risk-coverage curve"),
    ("fig7_causal_dag",         "Fig. 7 — Causal adjacency heatmap"),
]


def main():
    successes, failures = [], []
    for module_name, label in FIGURES:
        print(f"\n=== {label} ===")
        start = time.time()
        try:
            mod = importlib.import_module(f"scripts.figures.{module_name}")
            mod.main()
            elapsed = time.time() - start
            print(f"  ✓ done in {elapsed:.1f}s")
            successes.append(label)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failures.append(label)

    print("\n" + "=" * 60)
    print(f"Summary: {len(successes)} succeeded, {len(failures)} failed")
    for s in successes:
        print(f"  ✓ {s}")
    for f in failures:
        print(f"  ✗ {f}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
