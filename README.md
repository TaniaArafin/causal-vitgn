# Causal-VITGN: Counterfactual Cascade Reasoning on Evolving Social Networks

> Counterfactual cascade prediction using Pearl's Structural Causal Model framework on top of Variational Temporal Graph Networks.

## Overview

**Causal-VITGN** extends predictive cascade prediction with three levels of probabilistic reasoning (Pearl's Ladder of Causation):

1. **Associational** — *P(Y | X)* (standard prediction)
2. **Interventional** — *P(Y | do(X))* (intervention effect)
3. **Counterfactual** — *P(Y_x | X')* (alternate-world reasoning)

Built on Variational Temporal Graph Networks with a novel Structural Causal Model layer constrained to be acyclic via NOTEARS continuous DAG optimization.

## Project Structure

```
causal-vitgn/
├── config/default.yaml       # Hyperparameters
├── data/                     # Dataset download + preprocessing
├── notebooks/                # Colab training notebook
├── src/
│   ├── models/
│   │   ├── temporal_encoder.py
│   │   ├── variational_encoder.py
│   │   ├── causal_layer.py        # Structural Causal Model + NOTEARS
│   │   ├── hawkes_decoder.py
│   │   └── causal_vitgn.py
│   ├── inference/
│   │   └── counterfactual.py      # Pearl's abduction-action-prediction
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/                  # Train, evaluate, demo, figures
├── paper/                    # IEEE paper LaTeX source
└── tests/                    # Smoke tests
```

## Quickstart

### Local Setup

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python data/download.py
```

### Preprocess Data

```bash
python data/preprocess.py
```

### Train (Local)

```bash
python scripts/train.py --config config/default.yaml
```

### Train on Google Colab (Recommended)

Open `notebooks/train_colab.ipynb` in Google Colab, mount Google Drive, and run all cells.

### Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Live Counterfactual Demo

```bash
streamlit run scripts/demo.py
```

## Key Features

- **Counterfactual reasoning** — answer "what-if" questions on cascades
- **Calibrated uncertainty** — variational inference over latent influences
- **Evolving graph support** — Temporal Graph Network backbone
- **Interactive demo** — Streamlit dashboard for live simulation
- **DAG-constrained learning** — NOTEARS continuous optimization

## Citation

```bibtex
@misc{causalvitgn2026,
  author = {Umma Tania Arafin},
  title  = {Counterfactual Cascade Reasoning on Evolving Social Networks via Causal Variational Temporal Graph Networks},
  year   = {2026},
  note   = {CSE 756 Project, BRAC University}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Umma Tania Arafin** (ID: 24166024)
Department of Computer Science and Engineering
BRAC University, Dhaka, Bangladesh
