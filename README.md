# Generative Architectural Design utilizing Curriculum-Guided Deep Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: >= 3.12](https://img.shields.io/badge/Python->=%203.12-blue.svg)](https://www.python.org/)
[![Gymnasium: v0.29](https://img.shields.io/badge/Gymnasium-v0.29-green.svg)](https://gymnasium.farama.org/)
[![Stable--Baselines3: v2.2](https://img.shields.io/badge/Stable--Baselines3-v2.2-orange.svg)](https://stable-baselines3.readthedocs.io/)

This repository contains the official codebase and replication package for the research paper: **"Learning to design building footprints via Curriculum-Guided Reinforcement Learning"** (submitted to *Automation in Construction*). 

The project implements a custom Gymnasium environment (`UrbanPlanner-v0`) and trains a Proximal Policy Optimization (PPO) agent utilizing Stable-Baselines3 to synthesize building footprints on continuous coordinate spaces while satisfying strict, competing zoning and geometric requirements.

---

## 🛠️ Geometrical Constraints & Curriculum Stages

To resolve the severe sparse-reward barrier inherent in multi-constraint spatial design, the training framework utilizes **Curriculum Learning** to guide the agent through four sequential stages of increasing complexity:

1. **Stage 1: Placement (Containment):** The agent must position all outline vertices completely within the boundaries of the buildable design envelope (`permitted_plot` setback boundaries).
2. **Stage 2: Area:** The building outline area must fall within a target range specified by the plot properties (`area_range`).
3. **Stage 3: Width (Tract Width):** The building outline width must meet the minimum tract width requirement (`width_range`) to ensure functional layout depth.
4. **Stage 4: Angle (Orthogonality):** All corner angles must approximate right angles ($90^\circ \pm 5^\circ$) to enforce orthogonal structural forms.

---

## 📂 Repository Structure

```
├── README.md               # Project documentation
├── pyproject.toml          # PEP 517 build configuration & dependencies
├── requirements.txt        # Frozen dependencies for environment reproduction
├── LICENSE                 # MIT License file
├── scripts/                # Execution and pipeline scripts
│   ├── run-environment.py  # Manual/random exploration of the Gymnasium environment
│   ├── train-model.py      # Main PPO agent training pipeline
│   ├── evaluate-trained-model.py # Quantitative evaluation of model success rates
│   ├── run-model.py        # Visual simulation of trained agent actions
│   ├── view_logs.py        # TensorBoard dashboard execution helper
│   ├── run-comparative-eval.py # Comparative evaluation run against MC-Global/MC-Local
│   ├── run-ablation-eval.py    # Ablation study evaluation across curriculum checkpoints
│   ├── generate_plots.py   # Paper Figure 1, 2, and 3 plotting script
│   ├── generate_supplementary_plots.py # Supplementary materials plotting script
│   └── ma_data.py          # Data smoothing utility (for LaTeX/TikZ plotting)
└── src/
    └── urban_planner/      # Core package source code
        ├── config.py       # Global environment & hyperparameters configuration
        ├── logger_config.py# Centralized logging configuration
        ├── types.py        # Shared enums and Stage definitions
        ├── domain/         # Outline and plot geometric representations
        ├── envs/           # UrbanPlannerEnv Gym environment definition
        ├── logic/          # Checklist assessment and RewardCalculator classes
        ├── models/         # Neural network custom policy architectures
        └── utils/          # Geometry helper utilities and EvalCallbacks
```

---

## 💻 Installation

This project requires **Python >= 3.12**. We recommend setting up a virtual environment.

### Using `uv` (Recommended)

If you have `uv` installed, you can set up the environment and install dependencies in one step:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 🚀 Getting Started

### 1. Test the Environment
Verify that the Gymnasium environment and Pygame rendering are working by running in random or manual input mode:

```bash
# Run in random mode for 3 episodes with a 50ms rendering delay
python scripts/run-environment.py --mode random --episodes 3 --delay 0.05

# Run in manual mode (allows vertex manipulation keyboard input)
python scripts/run-environment.py --mode manual --episodes 1
```

### 2. Train the PPO Agent
Train the agent through the curriculum stages. Hyperparameters can be adjusted via command-line arguments:

```bash
# Train PPO agent up to Stage 4 for 64,000,000 timesteps across 8 parallel envs
python scripts/train-model.py --stage 4 --total-timesteps 64000000 --n-envs 8
```

You can resume training from a specific model checkpoint using `--resume-from-model` or resume a run using `--resume-run-name`.

### 3. Monitor Training Progress
Run TensorBoard to visualize the agent's reward progression and stage success rates:

```bash
python scripts/view_logs.py
```

### 4. Evaluate a Trained Agent
Compute quantitative performance metrics (mean reward and percentage of successfully solved episodes) for a saved model:

```bash
# Evaluate a saved model over 100 episodes
python scripts/evaluate-trained-model.py --model-path "models/CL_PPO_best/best_model.zip" --n-eval-episodes 100
```

### 5. Simulate a Trained Agent
Watch a trained model generate architectural outlines step-by-step with visual feedback:

```bash
# Visualize model performance with rendering enabled
python scripts/run-model.py --model-path "models/CL_PPO_best/best_model.zip" --render-mode human --episodes 5
```

---

## 📊 Paper Figure & Data Replication

To replicate the evaluations and figures presented in the paper, execute the following commands in sequence:

### 1. Run Baseline and Comparative Evaluations
To run the evaluation trials for the PPO agent and the stochastic baselines (`MC-Global` and `MC-Local`):

```bash
python scripts/run-comparative-eval.py
```

### 2. Run Ablation Checkpoints Evaluation
To run the ablation trials assessing performance differences across curriculum training checkpoints:

```bash
python scripts/run-ablation-eval.py
```

### 3. Plot Main Paper Figures (Figures 1, 2, and 3)
To generate the high-resolution charts used in the main manuscript (saved in the `figures/` directory):

```bash
python scripts/generate_plots.py
```

### 4. Plot Supplementary Material Figures
To generate the training curves and diagnostic charts for the supplementary material:

```bash
python scripts/generate_supplementary_plots.py
```

---

## 📄 License

This project is open-source and licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

