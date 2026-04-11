# Strategic Classification Experiments

This repository contains the implementation and experimental setup for analyzing **Strategic Classification**—specifically exploring the differences between "Transparent" and "Dark" deployment environments and evaluating the "Price of Opacity".

## Project Overview

In strategic classification, users (or players) manipulate their features to receive a favorable outcome from a deployed machine learning model. This project evaluates different algorithms—specifically standard Support Vector Machines (SVM) and Hardt's robust classification algorithm—under two main settings:

- **Transparent Setting**: The deployed model's parameters and decision boundaries are perfectly known to the strategic players.
- **Dark (Opaque) Setting**: The model is hidden. Users must act with limited information to guess the model and attempt their strategic modifications.

The codebase relies on cost-based feature manipulation bounds using convex optimization to compare model convergence, generalizability, and classification error across varying dataset sizes. 

## Repository Structure

```text
strategic_rep/
├── experiments/
│   ├── transparent_experiment.py   # Simulates the transparent setting
│   └── dark_experiment.py          # Simulates the dark/opaque setting
├── src/
│   ├── cost_functions.py           # Mathematical bounds for feature manipulation
│   ├── model.py                    # Implements the core classifiers (including HardtAlgo)
│   ├── strategic_players.py        # Contains optimization logic for users to manipulate features
│   ├── projected_visualization.py  # Utility functions for generating plots/boundaries
│   └── utills_and_consts.py        # Shared configuration and helpers
├── requirements.txt                # Core Python dependencies
└── report.tex                      # TeX code for your final academic report
```

## Setup Instructions

The project only relies on standard Python wheels. A standard Python virtual environment is recommended (Conda is not strictly necessary).

**1. Create and Activate a Virtual Environment:**
```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Experiments

To run the experiments, execute the experiment scripts from the repository root. When executed, they will generate visualization plots evaluating the model performances, comparing them side-by-side, and saving them to the results directory.

**Run the Transparent baseline experiment:**
```bash
python experiments/transparent_experiment.py
```

**Run the Dark opacity experiment:**
```bash
python experiments/dark_experiment.py
```

## Technical Stack

- **Optimization:** `cvxpy` handles the underlying convex optimization formulation for strategic manipulation logic.
- **Machine Learning:** `scikit-learn` drives the baseline Support Vector Machines along with matrix operations from `numpy` and `pandas`.
- **Visualization:** `matplotlib` handles creating visual analysis for dataset projections, decision boundaries, and convergence evaluations.
