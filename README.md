# mcPHASES-RL: Offline RL for Personalized Menstrual Health

**Final Project for CS 238 – Decision Making Under Uncertainty**  
**Authors:** Brooke Ballhaus, Nora Menon, Riya Narain  

Offline reinforcement learning experiments for personalized menstrual health recommendations, built on the mcPHASES longitudinal wearable/self-report dataset.  

The dataset can be accessed at [PhysioNet: mcPHASES](https://physionet.org/content/mcphases/1.0.0/).

---

## Overview

This project frames daily menstrual-health management as a sequential decision-making problem. Offline RL is used to recommend personalized lifestyle actions that aim to improve long-term symptom trajectories.  

The workflow includes:

- **Data preprocessing**: `data_preprocessing.py` ingests mcPHASES CSV tables, encodes categorical symptoms and hormones, aggregates sleep, HRV, stress, and activity metrics, engineers symptom trajectory features, and defines a 4-dimensional daily treatment action (exercise, sleep, stress, nutrition). The output is a Markov transition dataset ready for RL.  

- **RL experiments**: `menstrual_rl_project.py` loads the processed data, trains a RandomForest behavior cloning (BC) baseline, trains an offline Fitted Q-Iteration (FQI) agent with BC-style regularization, and evaluates learned policies using weighted importance sampling (WIS) and diagnostic analyses.

- **Dataset inspection**: `inspect_mcphases.py` provides utilities for examining non-numeric columns, unique values, and missing data in raw tables.

---

## Repository Layout

- `data_preprocessing.py`: data loading, encoding, panel construction, train/val/test split, and Markov dataset creation utilities  
- `menstrual_rl_project.py`: main experiment script implementing BC, FQI, BC-regularized policy, disagreement analysis, and WIS evaluation  
- `inspect_mcphases.py`: helper script for dataset auditing  
- `requirements.txt`: Python dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn)  

---

## Dataset Notes

Expected CSV files include: hormones/self-report, sleep, stress_score, heart_rate_variability_details, exercise, active_minutes, computed_temperature (or wrist_temperature), resting_heart_rate.  

The preprocessing pipeline:  
- Encodes Likert-style symptom scales, menstrual phases, and flow volume  
- Merges biometrics with daily self-report data  
- Computes symptom cost trends and history features  
- Infers joint treatment actions across 24 discrete options (3 exercise intensities × sleep/stress/nutrition binary toggles)  

Rewards are defined as day-to-day improvement in total symptom burden, clipped to [-10, 10] and scaled by 1/5 for numerical stability.

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
