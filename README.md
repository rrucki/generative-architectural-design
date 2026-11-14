# Shaping Building Footprints via Curriculum-Guided Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: >= 3.12](https://img.shields.io/badge/Python->=%203.12-blue.svg)](https://www.python.org/)
[![Gymnasium: v0.29](https://img.shields.io/badge/Gymnasium-v0.29-green.svg)](https://gymnasium.farama.org/)
[![Stable--Baselines3: v2.2](https://img.shields.io/badge/Stable--Baselines3-v2.2-orange.svg)](https://stable-baselines3.readthedocs.io/)

This repository contains the official codebase and replication package for the research paper: **"Shaping Building Footprints via Curriculum-Guided Reinforcement Learning"** (submitted to *Automation in Construction*).

The project implements a custom Gymnasium environment (`UrbanPlanner-v0`), trains a Proximal Policy Optimization (PPO) agent using a 4-stage Potential-Based Reward Shaping (PBRS) curriculum, pairs the learned policy as an amortized neural warm-start with continuous local refinement (`CL-PPO + Polish`), and benchmarks it against classical nonlinear programming (`SLSQP`, `COBYLA`), evolutionary metaheuristics (`CMA-ES`, `Differential Evolution`), Monte Carlo baselines, and RL ablations.

Trained model weights and raw training/evaluation logs are archived on Zenodo at [https://doi.org/10.5281/zenodo.21206902](https://doi.org/10.5281/zenodo.21206902). Download and extract the weights into `models/` and logs into `logs/` before running evaluation or plotting scripts.

---

## 🛠️ Geometrical Constraints & Curriculum Stages

To resolve the severe gradient conflicts and boundary containment traps inherent in multi-constraint spatial design, the training framework utilizes **Curriculum Learning** across four sequential stages:

1. **Stage 1 — Placement (Containment):** All building footprint vertices must lie within the buildable setback envelope (`permitted_outline`).
2. **Stage 2 — Area:** Footprint area must satisfy the parcel coverage bounds (`area_range`).
3. **Stage 3 — Width (Tract Width):** Minimum functional tract width must satisfy `width_range`.
4. **Stage 4 — Angle (Orthogonality):** Interior corner angles must lie within $90^\circ \pm 5^\circ$ (which guarantees strict quadrilateral convexity and simplicity).

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
        ├── logic/                     # OutlineAssessment & PBRS RewardCalculator
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
Train a curriculum stage (resuming from the preceding stage checkpoint) or inspect TensorBoard logs:

```bash
python scripts/train-model.py --stage 4 --total-timesteps 64000000 --n-envs 8
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

All scripts automatically resolve paths relative to the repository root and read/write intermediate JSON summaries in `logs/`.

### 1. Cross-Paradigm Comparative Benchmark (1,000 Parcels)
Evaluates `CL-PPO`, `CL-PPO + Polish`, `PPO (No-CL)`, `CMA-ES`, `SLSQP` (5-start), `COBYLA` (10-start), `Diff-Evolution`, `MC-Local`, and `MC-Global`:

```bash
# Quick smoke test (10 parcels)
python scripts/run-comparative-eval.py --n-episodes 10

# Full 1,000-parcel benchmark -> logs/comparative_evaluation_v2_results.json
python scripts/run-comparative-eval.py --n-episodes 1000
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
