# Urban Planner: Generative Architectural Design using Reinforcement Learning

This repository contains the codebase and replication package for the research paper on generative architectural and urban design using Reinforcement Learning (RL). The project implements a custom Gymnasium environment (`UrbanPlanner-v0`) and trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3.

The training utilizes **Curriculum Learning** across multiple stages of geometrical constraints to shape the agent's behavior from simple boundary alignment to complex layout optimization.

---

## Geometrical Constraints & Curriculum Stages

The learning task is divided into four stages of increasing complexity:

1. **Stage 1: Placement:** The agent must position all outline vertices completely within the boundaries of the permitted plot (`permitted_plot`).
2. **Stage 2: Area:** The building outline area must fall within a target range specified by the plot properties (`area_range`).
3. **Stage 3: Width:** The building outline width must meet the rectangular width limits (`width_range`).
4. **Stage 4: Angle:** All corner angles must approximate right angles ($90^\circ \pm 5^\circ$) to enforce orthogonal rectangular structures.

---

## Repository Structure

```
├── README.md               # Project documentation
├── pyproject.toml          # PEP 517 build configuration & dependencies
├── requirements.txt        # Frozen dependencies for environment reproduction
├── scripts/                # Execution and pipeline scripts
│   ├── run-environment.py  # Manual/random exploration of Gymnasium environment
│   ├── train-model.py      # Main PPO agent training pipeline
│   ├── evaluate-trained-model.py # Quantitative evaluation of model success rates
│   ├── run-model.py        # Visual simulation of trained agent actions
│   ├── view_logs.py        # TensorBoard dashboard execution helper
│   └── ma_data.py          # Data smoothing utility (for LaTeX/TikZ plotting)
└── src/
    └── urban_planner/      # Core package source code
        ├── config.py       # Global environment & hyperparameters configuration
        ├── logger_config.py# Centralized logging configuration
        ├── types.py        # Shared enums and Stage definitions
        ├── domain/         # Outline and plot geometric representations
        ├── envs/           # UrbanPlannerEnv Gym environment definition
        ├── logic/          # Checklist assessment and RewardCalculator classes
        ├── models/         # Commited/commented policy extractor templates
        └── utils/          # Geometry helper utilities and EvalCallbacks
```

---

## Installation

This project requires **Python >= 3.9**. We recommend setting up a virtual environment.

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

## Getting Started

### 1. Test the Environment
Verify that the Gymnasium environment is working by running it in random or manual input mode:

```bash
# Run in random mode for 3 episodes with a 50ms rendering delay
python scripts/run-environment.py --mode random --episodes 3 --delay 0.05

# Run in manual mode (allows vertex manipulation keyboard input)
python scripts/run-environment.py --mode manual --episodes 1
```

### 2. Train the PPO Agent
Train the agent through the curriculum stages. Hyperparameters can be adjusted via command-line arguments:

```bash
# Train PPO agent up to Stage 3 for 5,000,000 timesteps across 8 parallel envs
python scripts/train-model.py --stage 3 --total-timesteps 5000000 --n-envs 8
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
python scripts/evaluate-trained-model.py --model-path "stage_3_run_name/best_model.zip" --n-eval-episodes 100
```

### 5. Simulate a Trained Agent
Watch a trained model generate architectural outlines step-by-step:

```bash
# Visualize model performance with rendering enabled
python scripts/run-model.py --model-path "stage_3_run_name/best_model.zip" --render-mode human --episodes 5
```

---

## Reproduction of Figures (Paper Plots)
To smooth raw TensorBoard statistics and format them for TikZ/LaTeX plotting in the paper, use the moving average tool:

```bash
python scripts/ma_data.py
```

---

## License

This project is open-source and licensed under the MIT License.
