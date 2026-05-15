"""Streamlit demo: interactive counterfactual cascade reasoning.


Run locally:
   streamlit run scripts/demo.py


What the demo does:
 1. Loads the trained Causal-VITGN checkpoint.
 2. Lets the user pick a cascade from the test set.
 3. Shows the model's baseline prediction (Pearl Rung 1: association).
 4. Lets the user "remove" one or more participating users.
 5. Re-runs the model through abduction-action-prediction
    (Pearl Rungs 2 + 3: intervention + counterfactual).
 6. Compares baseline vs. counterfactual side-by-side.


This is the visual centerpiece of the defense: it shows the panel a
counterfactual cascade query running live, which is the contribution
no purely-predictive cascade model can produce.
"""


import json
import os
import sys
from pathlib import Path


import numpy as np
import streamlit as st
import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.models.causal_vitgn import CausalVITGN
from src.utils.data_loader import CascadeDataset, collate_cascades




# --------------------------------------------------------------------- #
#  Page setup
# --------------------------------------------------------------------- #
st.set_page_config(
   page_title="Causal-VITGN — Counterfactual Cascade Reasoning",
   page_icon="🔁",
   layout="wide",
)




PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "best_model.pt"




# --------------------------------------------------------------------- #
#  Caching
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained checkpoint…")
def load_model(config_path: str, ckpt_path: str):
   with open(config_path) as f:
       config = yaml.safe_load(f)


   metadata_path = Path(config["data"]["processed_dir"]) / "metadata.json"
   if metadata_path.exists():
       with open(metadata_path) as f:
           meta = json.load(f)
       config["model"]["num_nodes"] = int(meta["num_users"])


   device = torch.device("cpu")
   model = CausalVITGN(config).to(device)
   ckpt = torch.load(ckpt_path, map_location=device)
   model.load_state_dict(ckpt["model_state_dict"])
   model.eval()
   return model, config, device




@st.cache_resource(show_spinner="Loading test cascades…")
def load_test_set(processed_dir: str):
   test_path = Path(processed_dir) / "test.pkl"
   return CascadeDataset(str(test_path))




# --------------------------------------------------------------------- #
#  Inference helpers
# --------------------------------------------------------------------- #
def baseline_prediction(model, batch, device):
   """Pearl Rung 1: associational prediction."""
   with torch.no_grad():
       out = model(batch)
   prob = torch.sigmoid(out["intensity"]).item()
   return prob, out




def counterfactual_prediction(model, batch, removed_user_ids, device):
   """Pearl Rung 3: counterfactual via abduction-action-prediction.


   Maps each removed user id to a latent component using a deterministic
   hash (id % latent_dim), then sets that component to zero — i.e.
   'what if this user's influence vanished?'
   """
   if not removed_user_ids:
       return baseline_prediction(model, batch, device)


   latent_dim = model.config["model"]["latent_dim"]
   indices = sorted({int(u) % latent_dim for u in removed_user_ids})
   idx = torch.tensor(indices, device=device, dtype=torch.long)
   val = torch.zeros(len(indices), device=device)


   with torch.no_grad():
       cf_int, _z_cf = model.counterfactual(batch, idx, val)
   prob = torch.sigmoid(cf_int).item()
   return prob, None




# --------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------- #
def header():
   st.title("🔁 Counterfactual Cascade Reasoning")
   st.markdown(
       "**Causal-VITGN — Pearl's Ladder of Causation for information cascades.** "
       "Pick a cascade, remove a user, and see what the model thinks "
       "*would have happened* under the intervention."
   )




def sidebar(config_path: str, ckpt_path: str):
   st.sidebar.header("⚙️ Configuration")
   cfg_in = st.sidebar.text_input("Config path", value=config_path)
   ckpt_in = st.sidebar.text_input("Checkpoint path", value=ckpt_path)


   st.sidebar.markdown("---")
   st.sidebar.header("📊 About")
   st.sidebar.markdown(
       "Implements **Pearl's three rungs**:\n\n"
       "1. **Association** — P(Y \\| X)\n"
       "2. **Intervention** — P(Y \\| do(X=x))\n"
       "3. **Counterfactual** — P(Y_x \\| X')\n\n"
       "The Structural Causal Layer uses NOTEARS for differentiable "
       "DAG learning. Counterfactuals are computed via abduction → "
       "action → prediction on the learned SCM."
   )
   return cfg_in, ckpt_in




def cascade_card(idx: int, sample: dict, num_users: int):
   """Render compact info about the selected cascade."""
   neighbors = [int(u) for u in sample["neighbor_nodes"] if int(u) != 0]
   activated = float(sample["activated"])


   col1, col2, col3, col4 = st.columns(4)
   col1.metric("Cascade ID", f"#{idx}")
   col2.metric("Target user", int(sample["target_nodes"]))
   col3.metric("Observed reshares", len(neighbors))
   col4.metric(
       "Ground truth",
       "Activated ✅" if activated >= 0.5 else "Not activated ❌",
   )




def render_comparison(baseline_p: float, cf_p: float, removed: list[int]):
   """Side-by-side baseline vs counterfactual."""
   reduction_abs = baseline_p - cf_p
   reduction_pct = (
       100.0 * reduction_abs / max(baseline_p, 1e-6) if baseline_p > 0 else 0.0
   )


   c1, c2, c3 = st.columns(3)
   c1.metric(
       "Baseline P(activated)",
       f"{baseline_p:.3f}",
       help="Pearl Rung 1 — what the model predicts on the observed cascade.",
   )
   c2.metric(
       "Counterfactual P(activated)",
       f"{cf_p:.3f}",
       delta=f"{-reduction_abs:+.3f}",
       delta_color="inverse",
       help="Pearl Rung 3 — what the model predicts had the selected "
            "users not participated.",
   )
   c3.metric(
       "Predicted reduction",
       f"{reduction_pct:+.1f}%",
       help="Positive = the intervention shrinks the predicted cascade.",
   )


   # Bar chart comparison
   st.subheader("Baseline vs. Counterfactual")
   import pandas as pd


   df = pd.DataFrame(
       {
           "Scenario": ["Baseline", "Counterfactual"],
           "P(activated)": [baseline_p, cf_p],
       }
   )
   st.bar_chart(df, x="Scenario", y="P(activated)", height=300)


   if removed:
       st.caption(
           f"**Intervention:** removed users → "
           f"{', '.join(str(u) for u in removed)}"
       )




# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #
def main():
   header()
   cfg_path, ckpt_path = sidebar(str(DEFAULT_CONFIG), str(DEFAULT_CKPT))


   if not Path(cfg_path).exists():
       st.error(f"Config file not found: {cfg_path}")
       st.stop()
   if not Path(ckpt_path).exists():
       st.error(
           f"Checkpoint not found: {ckpt_path}\n\n"
           "Download `best_model.pt` from Google Drive (folder "
           "`causal-vitgn-ckpt`) and place it under `checkpoints/`."
       )
       st.stop()


   model, config, device = load_model(cfg_path, ckpt_path)
   test_ds = load_test_set(config["data"]["processed_dir"])


   # ----------------------------------------------------------------- #
   #  Cascade picker
   # ----------------------------------------------------------------- #
   st.subheader("1. Pick a cascade")


   max_idx = min(len(test_ds) - 1, 5000)
   idx = st.slider(
       "Cascade index in test set",
       min_value=0,
       max_value=max_idx,
       value=0,
       step=1,
       help="Each test cascade has an observed prefix + a target user. "
            "We predict whether the target activates.",
   )


   sample = test_ds[idx]
   batch = collate_cascades([sample])
   batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


   cascade_card(idx, sample, config["model"]["num_nodes"])


   # ----------------------------------------------------------------- #
   #  User removal
   # ----------------------------------------------------------------- #
   st.subheader("2. Choose users to remove (counterfactual intervention)")
   neighbors = sorted({int(u) for u in sample["neighbor_nodes"] if int(u) != 0})
   target = int(sample["target_nodes"])
   removable = [target] + [u for u in neighbors if u != target]


   st.caption(f"Debug: target={target}, n_neighbors={len(neighbors)}, removable={len(removable)} users")


   if not removable:
       st.warning(
           "No users available to intervene on for this cascade. "
           "Try moving the slider to a different cascade index."
       )
       st.stop()


   removed = st.multiselect(
       "Users participating in this cascade — select any subset to remove",
       options=removable,
       default=[],
       help="Selecting a user maps it to a latent component (user_id % "
            "latent_dim) and intervenes on that component via do(z_k = 0).",
   )


   # ----------------------------------------------------------------- #
   #  Compare baseline vs counterfactual
   # ----------------------------------------------------------------- #
   st.subheader("3. Compare baseline vs. counterfactual")


   baseline_p, _ = baseline_prediction(model, batch, device)
   cf_p, _ = counterfactual_prediction(model, batch, removed, device)
   render_comparison(baseline_p, cf_p, removed)


   # ----------------------------------------------------------------- #
   #  Sweep: remove each user one at a time
   # ----------------------------------------------------------------- #
   st.subheader("4. Per-user influence sweep")
   st.markdown(
       "Removes each participant **individually** and measures the predicted "
       "reduction in activation probability. Higher reduction → more "
       "influential user under the model's learned causal graph."
   )


   if st.button("Run influence sweep"):
       with st.spinner("Sweeping over all participants…"):
           rows = []
           for u in removable:
               cf_p_u, _ = counterfactual_prediction(model, batch, [u], device)
               rows.append(
                   {
                       "user_id": u,
                       "baseline": baseline_p,
                       "counterfactual": cf_p_u,
                       "reduction_pct": 100.0 * (baseline_p - cf_p_u)
                       / max(baseline_p, 1e-6),
                   }
               )
       import pandas as pd


       df = pd.DataFrame(rows).sort_values("reduction_pct", ascending=False)
       st.dataframe(
           df.style.format(
               {
                   "baseline": "{:.3f}",
                   "counterfactual": "{:.3f}",
                   "reduction_pct": "{:+.1f}%",
               }
           ),
           use_container_width=True,
           hide_index=True,
       )
       st.bar_chart(df.set_index("user_id")["reduction_pct"], height=300)


   # ----------------------------------------------------------------- #
   #  Footer
   # ----------------------------------------------------------------- #
   st.markdown("---")
   st.caption(
       "Causal-VITGN · CSE 756 Probabilistic Graphical Models · "
       "BRAC University · Tania Arafin"
   )




if __name__ == "__main__":
   main()


