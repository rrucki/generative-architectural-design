# Curriculum-Guided Reinforcement Learning with Continuous Local Refinement for Constrained Building Footprint Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: >= 3.12](https://img.shields.io/badge/Python->=%203.12-blue.svg)](https://www.python.org/)
[![Gymnasium: v1.0.0](https://img.shields.io/badge/Gymnasium-v1.0.0-green.svg)](https://gymnasium.farama.org/)
[![Stable--Baselines3: v2.4.1](https://img.shields.io/badge/Stable--Baselines3-v2.4.1-orange.svg)](https://stable-baselines3.readthedocs.io/)

This repository contains the official codebase and replication package for the research paper: **"Curriculum-Guided Reinforcement Learning with Continuous Local Refinement for Constrained Building Footprint Synthesis"**.

The project implements a custom Gymnasium environment (`UrbanPlanner-v0`), trains a Proximal Policy Optimization (PPO) agent using a 4-stage curriculum with PBRS-inspired undiscounted potential-difference reward shaping, pairs the learned policy as an amortized neural warm-start with continuous local refinement (`CL-PPO + Refinement`, implemented via `run_hybrid_ppo.py`), and benchmarks it against classical nonlinear programming (`SLSQP`, `COBYLA`), evolutionary metaheuristics (`CMA-ES`, `Differential Evolution`), Monte Carlo baselines, and RL ablations.

Trained model weights and raw training/evaluation logs are archived on Zenodo at [https://doi.org/10.5281/zenodo.24336686](https://doi.org/10.5281/zenodo.24336686). Download and extract the weights into `models/` and logs into `logs/` before running evaluation or plotting scripts.

---

## 🛠️ Geometrical Constraints & Curriculum Stages

To resolve the severe gradient conflicts and boundary containment traps inherent in multi-constraint spatial design, the training framework utilizes **Curriculum Learning** across four sequential stages:

1. **Stage 1 — Placement (Containment):** All building footprint vertices must lie within the buildable setback envelope (`permitted_outline`).
2. **Stage 2 — Area:** Footprint area must satisfy the parcel coverage bounds (`area_range`).
3. **Stage 3 — Width (Tract Width):** Minimum functional tract width must satisfy `width_range`.
4. **Stage 4 — Angle (Orthogonality):** Interior corner angles must lie within $90^\circ \pm 5^\circ$ (signed interior orthogonality guarantees quadrilateral convexity and simplicity; under unsigned `arccos` angle checks, `OutlineAssessment` pairs this with an explicit non-self-intersection test).

---

## 📂 Repository Structure

```text
├── README.md                          # Project overview & replication guide
├── pyproject.toml                     # Build configuration & dependencies
├── requirements.txt                   # Package list
├── LICENSE                            # MIT License
├── scripts/                           # Training, benchmarking, ablations & plotting
│   ├── train-model.py                 # Curriculum PPO training pipeline (SB3)
│   ├── evaluate-trained-model.py      # Single-checkpoint quantitative evaluation
│   ├── run-model.py                   # Interactive Pygame simulation of trained agent
│   ├── run-environment.py             # Random/manual environment exploration
│   ├── view_logs.py                   # TensorBoard launcher helper
│   ├── run-comparative-eval.py        # 1,000-parcel 9-method comparative benchmark
│   ├── audit_polygon_containment.py   # Polygon.within & simplicity audit (1,000 test + OOD)
│   ├── run-ablation-eval.py           # Full-checkpoint stage-removal ablations
│   ├── run_fast_ablations.py          # Controlled 4-stage vs 2-stage PPO & CL-A2C suite
│   ├── run-synthetic-ood-eval.py      # Zero-shot Synthetic OOD morphological shift suite
│   ├── evaluate_diversity.py          # Stochastic policy diversity benchmark (K=20)
│   ├── visualize_cma_es.py            # Real-time Pygame visualizer for CMA-ES baseline
│   ├── generate_plots.py              # Main manuscript figure generator
│   ├── generate_supplementary_plots.py# Supplementary diagnostics & curve generator
│   ├── ma_data.py                     # Moving-average export helper
│   └── baselines/                     # Classical & hybrid optimization solvers
│       ├── objective.py               # Smooth continuous surrogate + discrete validator
│       ├── run_hybrid_ppo.py          # Amortized neural warm-start (CL-PPO + Polish)
│       ├── run_cma_es.py              # CMA-ES baseline
│       ├── run_constrained_opt.py     # Multi-start SLSQP & COBYLA baselines
│       └── run_de.py                  # Differential Evolution baseline
└── src/
    └── urban_planner/                 # Core Gymnasium package
        ├── config.py                  # Global dimensions, step budgets, scaling
        ├── types.py                   # Stage & ObsKeys definitions
        ├── domain/geometries.py       # Outline & Plot geometric models
        ├── envs/                      # UrbanPlanner-v0 env & generation manager
        ├── logic/                     # OutlineAssessment & potential-difference RewardCalculator
        ├── models/                    # Custom neural policy extractors
        ├── presentation/              # Pygame renderer
        ├── utils/                     # Parcel generator, geometry helpers & SB3 callbacks
        └── wrappers/                  # Gymnasium observation/action/reward wrappers
```

---

## 💻 Installation

This project requires **Python >= 3.12**. Set up a virtual environment and install the package in editable mode:

### Using `uv` (Recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 🚀 Getting Started

### 1. Explore the Environment
Verify Gymnasium and Pygame rendering in random or manual keyboard mode:

```bash
python scripts/run-environment.py --mode random --episodes 3 --delay 0.05
python scripts/run-environment.py --mode manual --episodes 1
```

### 2. Train or Inspect a Curriculum Agent
Train the sequential 4-stage curriculum or the single-stage `PPO (No-CL)` baseline, or inspect TensorBoard logs. Note on rollout buffer semantics: `scripts/train-model.py` divides `--n-steps` by `--n-envs` before passing it to Stable-Baselines3 `PPO(..., n_steps=args.n_steps // n_envs)`. Thus, `--n-steps 8192 --n-envs 8` produces **1,024 steps per environment** and a total rollout buffer of **8,192 environment steps per PPO update** (16 minibatches of size 512 for 2 epochs). In the manuscript, empirical 90% reward convergence across Stages 1–4 occurs at 168.8M cumulative active steps, while extended 1.024B-step runs in Stages 3–4 serve as long-horizon stability tests:

```bash
# Stage 1 — Containment (64M steps)
python scripts/train-model.py --run-name stage_1 --stage 1 --total-timesteps 64000000 \
  --n-envs 8 --n-steps 8192 --batch-size 512 --n-epochs 2 \
  --lr-schedule-type linear --lr-start 0.001 --lr-end 0.0005 \
  --clip-schedule-type linear --clip-start 0.2 --clip-end 0.11 --ent-coef 0.025

# Stage 2 — Area (64M steps, warm-started from Stage 1 best checkpoint)
python scripts/train-model.py --run-name stage_2 --stage 2 --resume-from-model "stage_1_<ts>/best_model.zip" \
  --total-timesteps 64000000 --n-envs 8 --n-steps 8192 --batch-size 512 --n-epochs 2 \
  --lr-schedule-type linear --lr-start 0.003 --lr-end 0.0008 \
  --clip-schedule-type linear --clip-start 0.2 --clip-end 0.11 --ent-coef 0.025

# Stage 3 — Tract Width (1.024B stability horizon, warm-started from Stage 2)
python scripts/train-model.py --run-name stage_3 --stage 3 --resume-from-model "stage_2_<ts>/final_model.zip" \
  --total-timesteps 1024000000 --n-envs 8 --n-steps 8192 --batch-size 512 --n-epochs 2 \
  --lr-schedule-type constant --lr-start 0.0026 \
  --clip-schedule-type linear --clip-start 0.2 --clip-end 0.11 --ent-coef 0.015

# Stage 4a — Brief initial warm-start from Stage 3 before LR tuning (~2.25M steps / 0.39 h)
python scripts/train-model.py --run-name stage_4 --stage 4 --resume-from-model "stage_3_<ts>/final_model.zip" \
  --total-timesteps 2250000 --n-envs 8 --n-steps 8192 --batch-size 512 --n-epochs 2 \
  --lr-schedule-type constant --lr-start 0.0026 \
  --clip-schedule-type linear --clip-start 0.2 --clip-end 0.11 --ent-coef 0.015

# Stage 4b — Main Stage 4 stability run reported in the paper (1.024B steps; total ~2.18B across Stages 1-4)
python scripts/train-model.py --run-name stage_4_tunning --stage 4 --resume-from-model "stage_4_<ts>/final_model.zip" \
  --total-timesteps 1024000000 --n-envs 8 --n-steps 8192 --batch-size 512 --n-epochs 2 \
  --lr-schedule-type constant --lr-start 0.008 \
  --clip-schedule-type linear --clip-start 0.2 --clip-end 0.11 --ent-coef 0.01

# Baseline: 1-Stage PPO (No-CL) trained from scratch on Stage 4 (64M steps)
python scripts/train-model.py --run-name ppo_no_cl --stage 4 --total-timesteps 64000000 \
  --n-envs 8 --n-steps 8192 --batch-size 512 --n-epochs 2 \
  --lr-schedule-type linear --lr-start 0.002 --lr-end 0.0007 \
  --clip-schedule-type linear --clip-start 0.2 --clip-end 0.11 --ent-coef 0.01

python scripts/view_logs.py
```

### 3. Evaluate & Visualize the Trained Model
Once model artifacts from Zenodo are placed under `models/`:

```bash
python scripts/evaluate-trained-model.py --model-path "models/stage_4_tunning_1757784606/best_model.zip" --n-eval-episodes 100
python scripts/run-model.py --model-path "models/stage_4_tunning_1757784606/best_model.zip" --render-mode human --episodes 5
```

---

## 📊 Paper Figure & Data Replication

All scripts automatically resolve paths relative to the repository root, use deterministic parcel generation seeds (`0–999` for the 1,000-parcel main benchmark, `2000–2059` per regime for Synthetic OOD shifts), and read/write intermediate JSON summaries in `logs/`. To reproduce exact pinned dependencies, install from `requirements.txt` or `pyproject.toml` under Python 3.12+.

### 1. Cross-Paradigm Comparative Benchmark (1,000 Parcels)
Evaluates `CL-PPO`, `CL-PPO + Polish`, `PPO (No-CL)`, `CMA-ES`, `SLSQP` (5-start), `COBYLA` (10-start), `Diff-Evolution`, `MC-Local`, and `MC-Global`:

```bash
# Quick smoke test (10 parcels)
python scripts/run-comparative-eval.py --n-episodes 10

# Full 1,000-parcel benchmark -> logs/comparative_evaluation_v2_results.json
python scripts/run-comparative-eval.py --n-episodes 1000

# Independent full Polygon.within & simplicity audit (1,000 test + 180 OOD parcels)
python scripts/audit_polygon_containment.py --n-parcels 1000 --audit-ood
```

### 2. Curriculum & Algorithmic Ablations
```bash
# Stage-removal ablations (Stage 2 and Stage 3 checkpoints)
python scripts/run-ablation-eval.py

# Controlled curriculum granularity (4-Stage vs 2-Stage) & A2C comparison
python scripts/run_fast_ablations.py --steps 32768 --n-eval 100
```

### 3. Synthetic OOD Shift & Stochastic Diversity
```bash
# Three synthetic distribution-shift regimes (60 parcels/regime)
python scripts/run-synthetic-ood-eval.py --n-per-regime 60

# Stochastic multi-option synthesis (50 parcels x 20 rollouts)
python scripts/evaluate_diversity.py --n-parcels 50 --k-rollouts 20
```

### 4. Generate Figures
```bash
python scripts/generate_plots.py
python scripts/generate_supplementary_plots.py
```

---

## 📄 License

This project is open-source and licensed under the MIT License — see [LICENSE](LICENSE) for details.
