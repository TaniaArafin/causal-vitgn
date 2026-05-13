"""Fig. 5 — Counterfactual outcomes.


For a handful of test cascades, plot:
 - Baseline P(activated) on the left bar
 - Counterfactual P(activated) on the right bar
 - The arrow + percentage between them showing the predicted reduction


This is the central figure showing Pearl Rung 3 in action.


Run:
   PYTHONPATH=. python scripts/figures/fig5_counterfactual.py
"""


import matplotlib.pyplot as plt
import numpy as np
import torch


from scripts.figures._utils import load_model_and_test, save, setup_matplotlib
from src.utils.data_loader import collate_cascades




def pick_cascades(test_ds, n: int = 6):
   """Pick positive cascades with enough neighbours for a meaningful intervention."""
   selected = []
   for i in range(len(test_ds)):
       s = test_ds[i]
       if float(s["activated"]) < 0.5:
           continue
       nbrs = [int(u) for u in s["neighbor_nodes"].tolist() if int(u) != 0]
       if 5 <= len(nbrs) <= 30:
           selected.append((i, s))
       if len(selected) >= n:
           break


   # Fallback: if not enough cascades matched the size filter, accept any
   # positive cascade so we still produce a figure.
   if len(selected) < n:
       for i in range(len(test_ds)):
           s = test_ds[i]
           if float(s["activated"]) >= 0.5 and (i, s) not in selected:
               selected.append((i, s))
               if len(selected) >= n:
                   break
   return selected




def baseline_and_cf(model, sample, device):
   batch = collate_cascades([sample])
   batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


   with torch.no_grad():
       baseline = torch.sigmoid(model(batch)["intensity"]).item()


   # Intervene on z_0 -> 0 (same protocol as counterfactual_demo.py).
   idx = torch.tensor([0], device=device, dtype=torch.long)
   val = torch.zeros(1, device=device)
   cf_int, _ = model.counterfactual(batch, idx, val)
   cf = torch.sigmoid(cf_int).item()
   return baseline, cf




def main():
   setup_matplotlib()
   model, _, test_ds, device = load_model_and_test()


   rows = []
   for i, (idx, sample) in enumerate(pick_cascades(test_ds, n=6)):
       b, c = baseline_and_cf(model, sample, device)
       rows.append({"case": f"#{idx}", "baseline": b, "cf": c})


   if not rows:
       print("  No suitable cascades found; using whatever is available.")
       return


   fig, ax = plt.subplots(figsize=(9, 4.5))


   labels = [r["case"] for r in rows]
   baseline_vals = [r["baseline"] for r in rows]
   cf_vals = [r["cf"] for r in rows]
   x = np.arange(len(labels))
   w = 0.36


   bars1 = ax.bar(x - w / 2, baseline_vals, w, label="Baseline",
                  color="#5b9bd5", edgecolor="#1a4f76")
   bars2 = ax.bar(x + w / 2, cf_vals, w, label=r"Counterfactual  $do(z_0=0)$",
                  color="#e6743b", edgecolor="#9c4218")


   # Annotate each pair with the reduction %
   for i, (b, c) in enumerate(zip(baseline_vals, cf_vals)):
       red = 100.0 * (b - c) / max(b, 1e-6)
       y = max(b, c) + 0.02
       ax.annotate(
           f"{red:+.1f}%",
           xy=(x[i], y), ha="center", va="bottom",
           fontsize=9, fontweight="bold",
           color="#9c4218" if red > 0 else "#1a4f76",
       )


   ax.set_xticks(x)
   ax.set_xticklabels(labels)
   ax.set_ylim(0, max(max(baseline_vals), max(cf_vals)) * 1.25)
   ax.set_ylabel("P(activated)")
   ax.set_title("Baseline vs. counterfactual activation probability")
   ax.legend(loc="upper right", framealpha=0.95)


   save(fig, "fig5_counterfactual")
   plt.close(fig)


   for r in rows:
       red = 100.0 * (r["baseline"] - r["cf"]) / max(r["baseline"], 1e-6)
       print(f"  Case {r['case']:>6s}:  baseline={r['baseline']:.3f}  "
             f"cf={r['cf']:.3f}  reduction={red:+.1f}%")




if __name__ == "__main__":
   main()




