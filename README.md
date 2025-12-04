# mcphases-rl

Offline reinforcement learning experiments for personalized menstrual health recommendations built on the mcPHASES longitudinal wearable/self-report dataset. Developed as a CS 238 group project.

## Overview
- `data_preprocessing.py` ingests the mcPHASES CSV tables, encodes categorical symptoms and hormones, aggregates sleep/HRV/stress/activity metrics, engineers symptom trajectory features, defines a 4-dimensional daily “treatment” action (exercise, sleep, stress, nutrition), and produces Markov transition datasets ready for RL.
- `menstrual_rl_project.py` loads the processed data, trains a RandomForest behavior cloning baseline, trains an offline Fitted Q-Iteration (FQI) agent with BC-style regularization, and compares the learned policies via simple diagnostics and weighted importance sampling (WIS) off-policy evaluation.
- `inspect_mcphases.py` is a lightweight utility to examine non-numeric columns in each raw table when preparing new datasets.

The project workflow is:
1. Place the mcPHASES CSVs under `mcPHASES/` (or update the path in the config).
2. Run `menstrual_rl_project.py` to build daily panels, split participants into train/val/test, generate Markov tuples, train/evaluate policies, and print diagnostics.

## Repository Layout
- `data_preprocessing.py`: data loading, encoding, panel construction, train/val/test split, and Markov dataset creation utilities.
- `menstrual_rl_project.py`: main experiment script containing BC, FQI, policy regularization, disagreement analysis, and WIS evaluation.
- `inspect_mcphases.py`: helper script for dataset auditing.
- `requirements.txt`: minimal Python dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn).

## Dataset Notes
- Expected CSV files: hormones/self-report, sleep, stress_score, heart_rate_variability_details, exercise, active_minutes, computed_temperature (or wrist_temperature), resting_heart_rate.
- The preprocessing pipeline encodes Likert-style symptom scales, menstrual phases, and flow volume, merges biometrics, computes symptom cost trends, and infers joint actions for 24 discrete treatment combinations (3 exercise intensities × sleep/stress/nutrition binary toggles).
- Rewards are defined as improvements in aggregated symptom burden between successive days, clipped to [-10, 10] and scaled by 1/5 for stability.

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline
```bash
python menstrual_rl_project.py
```
Key configuration knobs (see `menstrual_rl_project.py`):
- `DataConfig`: dataset directory, minimum days per participant, train/val/test split fractions, discount factor.
- `RLConfig`: RandomForest hyperparameters for behavior cloning and FQI, iteration count, BC regularization strength.

The script prints dataset stats, behavior cloning validation accuracy, FQI Bellman error across iterations, policy disagreement rates, and WIS estimates for BC vs. conservative FQI policies on the held-out test trajectories.

### Inspecting Raw Tables
If you add or modify mcPHASES tables, optionally run:
```bash
python inspect_mcphases.py
```
to list non-numeric columns, unique categorical values, and missing-value counts per file when adjusting encoders.
