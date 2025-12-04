from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import numpy as np
import pandas as pd

# CONFIG DATACLASS

@dataclass
class DataConfig:
    data_dir: str
    min_days_per_participant: int = 10
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_state: int = 42
    gamma: float = 0.95  # slightly less than 0.99 for more stable bootstrapping


# CATEGORICAL ENCODING CONSTANTS

# Likert-like 0–5 scales for symptoms and related ratings
LIKERT_COLS_0_5 = [
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]

# Map text + numeric strings to 0–5
LIKERT_MAP = {
    "Not at all": 0,
    "Very Low": 1,
    "Very Low/Little": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5,
    # numeric strings
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

# Flow volume as ordered severity
FLOW_VOLUME_ORDER = [
    "Not at all",
    "Spotting / Very Light",
    "Light",
    "Somewhat Light",
    "Moderate",
    "Somewhat Heavy",
    "Heavy",
    "Very Heavy",
]
FLOW_VOLUME_MAP = {val: i for i, val in enumerate(FLOW_VOLUME_ORDER)}

# Menstrual phase mapping
PHASE_MAP = {
    "Follicular": 0,
    "Fertility": 1,
    "Luteal": 2,
    "Menstrual": 3,
}

# Default state features used by RL (exported)
STATE_FEATURES_DEFAULT = [
    # hormones
    "lh", "estrogen", "pdg",
    # cycle phase + weekend
    "phase_encoded",
    "is_weekend_int",
    # self-reported symptoms & modifiers (encoded)
    "cramps_encoded",
    "fatigue_encoded",
    "headaches_encoded",
    "moodswing_encoded",
    "stress_encoded",
    "sleepissue_encoded",
    "appetite_encoded",
    "exerciselevel_encoded",
    "foodcravings_encoded",
    "bloating_encoded",
    "indigestion_encoded",
    "sorebreasts_encoded",
    # flow volume (encoded)
    "flow_volume_encoded",
    # symptom trajectory features (for Markov-ish state)
    "symptom_cost_today",
    "prev_symptom_cost",
    "symptom_trend_3day",
    "prev_action",
    # physiology
    "sleep_minutes", "sleep_efficiency", "time_in_bed",
    "rmssd_mean", "lf_mean", "hf_mean",
    "stress_score", "resting_hr",
    "nightly_temp", "baseline_temp_sd",
    # activity context
    "sedentary_min", "lightly_min", "moderately_min", "very_min",
    "ex_duration_ms", "ex_steps", "ex_avg_hr",

    "exercise_dim",
    "sleep_dim",
    "stress_dim",
    "nutrition_dim",
]


# =============================================================================
# 2. ENCODING: HORMONES + SELF-REPORT
# =============================================================================

def encode_hormones_and_selfreport(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and numerically encode non-numeric columns in hormones_and_selfreport.
    Creates new *_encoded columns and keeps originals.
    """
    df = df.copy()

    # Encode phase
    if "phase" in df.columns:
        df["phase_encoded"] = df["phase"].map(PHASE_MAP)

    # Encode flow_volume
    if "flow_volume" in df.columns:
        df["flow_volume_encoded"] = df["flow_volume"].map(FLOW_VOLUME_MAP)

    # Encode Likert-type symptom scales
    for col in LIKERT_COLS_0_5:
        if col in df.columns:
            df[col + "_encoded"] = df[col].astype(str).map(LIKERT_MAP)

    # is_weekend to int (for this table)
    if "is_weekend" in df.columns:
        df["is_weekend_int"] = df["is_weekend"].astype(int)

    return df


# LOAD TABLES

def load_mcphases_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load only the mcPHASES tables we care about.
    Applies encoding to hormones_and_selfreport.
    """

    def read(name: str, **kwargs) -> pd.DataFrame:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        return pd.read_csv(path, **kwargs)

    tables: Dict[str, pd.DataFrame] = {}

    # hormones + self-report
    hormones_df = read("hormones_and_selfreport")
    hormones_df = encode_hormones_and_selfreport(hormones_df)
    tables["hormones"] = hormones_df

    # other key tables
    tables["sleep"] = read("sleep")
    tables["stress"] = read("stress_score")
    tables["hrv"] = read("heart_rate_variability_details")
    tables["exercise"] = read("exercise")
    tables["active_minutes"] = read("active_minutes")
    tables["rhr"] = read("resting_heart_rate")

    # Temperature: prefer computed_temperature, else fallback to wrist_temperature
    comp_temp_path = os.path.join(data_dir, "computed_temperature.csv")
    wrist_temp_path = os.path.join(data_dir, "wrist_temperature.csv")

    if os.path.exists(comp_temp_path):
        tables["temp"] = pd.read_csv(comp_temp_path)
        tables["temp_source"] = "computed_temperature"  # type: ignore
    elif os.path.exists(wrist_temp_path):
        tables["temp"] = pd.read_csv(wrist_temp_path)
        tables["temp_source"] = "wrist_temperature"  # type: ignore
    else:
        raise FileNotFoundError(
            "Neither computed_temperature.csv nor wrist_temperature.csv found "
            f"in {data_dir}"
        )

    return tables


# DAILY PANEL CREATION

def build_daily_panel(
    tables: Dict[str, pd.DataFrame],
    min_days_per_participant: int = 10,
) -> pd.DataFrame:
    """
    Build a (id, day_in_study) → feature panel from the chosen tables.

    Returns a DataFrame with:
        id, day_in_study, encoded symptoms, hormones, sleep, HRV,
        stress, resting HR, temperature, activity, and action,
        plus symptom trajectory features.
    """
    # Start from hormones + self-report (already keyed by day_in_study)
    df = tables["hormones"].copy()
    assert {"id", "day_in_study"}.issubset(df.columns)

    # --- Sleep aggregates ---
    sleep = tables["sleep"].copy()
    if {"id", "sleep_start_day_in_study"}.issubset(sleep.columns):
        sleep_agg = (
            sleep.groupby(["id", "sleep_start_day_in_study"])
            .agg(
                sleep_minutes=("minutesasleep", "mean"),
                sleep_efficiency=("efficiency", "mean"),
                time_in_bed=("timeinbed", "mean"),
            )
            .reset_index()
            .rename(columns={"sleep_start_day_in_study": "day_in_study"})
        )
        df = df.merge(sleep_agg, on=["id", "day_in_study"], how="left")

    # --- HRV aggregates ---
    hrv = tables["hrv"].copy()
    if {"id", "day_in_study"}.issubset(hrv.columns):
        hrv_agg = (
            hrv.groupby(["id", "day_in_study"])
            .agg(
                rmssd_mean=("rmssd", "mean"),
                lf_mean=("low_frequency", "mean"),
                hf_mean=("high_frequency", "mean"),
            )
            .reset_index()
        )
        df = df.merge(hrv_agg, on=["id", "day_in_study"], how="left")

    # --- Stress score ---
    stress = tables["stress"].copy()
    if {"id", "day_in_study"}.issubset(stress.columns):
        stress_agg = (
            stress.groupby(["id", "day_in_study"])
            .agg(stress_score=("stress_score", "mean"))
            .reset_index()
        )
        df = df.merge(stress_agg, on=["id", "day_in_study"], how="left")

    # --- Resting heart rate ---
    rhr = tables["rhr"].copy()
    if {"id", "day_in_study"}.issubset(rhr.columns):
        rhr_agg = (
            rhr.groupby(["id", "day_in_study"])
            .agg(resting_hr=("value", "mean"))
            .reset_index()
        )
        df = df.merge(rhr_agg, on=["id", "day_in_study"], how="left")

    # --- Nightly skin temperature (computed or wrist) ---
    temp = tables["temp"].copy()
    temp_source = tables.get("temp_source", "computed_temperature")  # type: ignore

    if temp_source == "computed_temperature":
        if {"id", "sleep_start_day_in_study"}.issubset(temp.columns):
            temp_agg = (
                temp.groupby(["id", "sleep_start_day_in_study"])
                .agg(
                    nightly_temp=("nightly_temperature", "mean"),
                    baseline_temp_sd=(
                        "baseline_relative_sample_standard_deviation",
                        "mean",
                    ),
                )
                .reset_index()
                .rename(columns={"sleep_start_day_in_study": "day_in_study"})
            )
        else:
            temp_agg = pd.DataFrame(
                columns=["id", "day_in_study", "nightly_temp", "baseline_temp_sd"]
            )
    else:  # wrist_temperature
        if {"id", "day_in_study"}.issubset(temp.columns):
            temp_agg = (
                temp.groupby(["id", "day_in_study"])
                .agg(
                    nightly_temp=("temperature_diff_from_baseline", "mean"),
                    baseline_temp_sd=("temperature_diff_from_baseline", "std"),
                )
                .reset_index()
            )
        else:
            temp_agg = pd.DataFrame(
                columns=["id", "day_in_study", "nightly_temp", "baseline_temp_sd"]
            )

    df = df.merge(temp_agg, on=["id", "day_in_study"], how="left")

    # --- Daily exercise / activity to define actions ---
    df = add_action_from_activity(df, tables)

    # --- Symptom trajectory features (for Markov-ish state) ---
    symptom_cols_enc = [
        c for c in [
            "cramps_encoded",
            "fatigue_encoded",
            "headaches_encoded",
            "moodswing_encoded",
            "stress_encoded",
            "sleepissue_encoded",
            "bloating_encoded",
        ] if c in df.columns
    ]
    if symptom_cols_enc:
        df["symptom_cost_today"] = df[symptom_cols_enc].apply(
            lambda row: pd.to_numeric(row, errors="coerce"), axis=1
        ).fillna(0.0).sum(axis=1)
    else:
        df["symptom_cost_today"] = 0.0

    # Sort before computing lag features
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)

    # prev_symptom_cost
    df["prev_symptom_cost"] = (
        df.groupby("id")["symptom_cost_today"].shift(1).fillna(0.0)
    )

    # 3-day moving average of symptom cost
    symptom_trend = (
        df.groupby("id")["symptom_cost_today"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["symptom_trend_3day"] = symptom_trend

    # prev_action (after actions have been defined)
    df["prev_action"] = (
        df.groupby("id")["action"].shift(1).fillna(0).astype(int)
    )

    # Drop participants with too little data
    counts = df.groupby("id")["day_in_study"].nunique()
    keep_ids = counts[counts >= min_days_per_participant].index
    df = df[df["id"].isin(keep_ids)].copy()

    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    return df


def add_action_from_activity(
    df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Infer a 4D daily treatment action:

        exercise_dim  ∈ {0,1,2}   (low, moderate, intense)
        sleep_dim     ∈ {0,1}     (no sleep focus, sleep intervention)
        stress_dim    ∈ {0,1}     (no stress focus, stress intervention)
        nutrition_dim ∈ {0,1}     (no nutrition focus, nutrition/GI intervention)

    We then encode these into a single joint action integer in [0, 23]
    so that the rest of the RL pipeline can treat actions as discrete.
    """

    exercise = tables["exercise"].copy()
    active = tables["active_minutes"].copy()

    # ---------------------------
    # 1) Exercise aggregation
    # ---------------------------
    if {"id", "start_day_in_study"}.issubset(exercise.columns):
        ex_agg = (
            exercise.groupby(["id", "start_day_in_study"])
            .agg(
                ex_duration_ms=("duration", "sum"),
                ex_steps=("steps", "sum"),
                ex_avg_hr=("averageheartrate", "mean"),
            )
            .reset_index()
            .rename(columns={"start_day_in_study": "day_in_study"})
        )
    else:
        ex_agg = (
            exercise.groupby(["id", "day_in_study"])
            .agg(
                ex_duration_ms=("duration", "sum"),
                ex_steps=("steps", "sum"),
                ex_avg_hr=("averageheartrate", "mean"),
            )
            .reset_index()
        )

    if {"id", "day_in_study"}.issubset(active.columns):
        act_agg = (
            active.groupby(["id", "day_in_study"])
            .agg(
                sedentary_min=("sedentary", "sum"),
                lightly_min=("lightly", "sum"),
                moderately_min=("moderately", "sum"),
                very_min=("very", "sum"),
            )
            .reset_index()
        )
    else:
        act_agg = pd.DataFrame(columns=["id", "day_in_study"])

    merged = df.merge(ex_agg, on=["id", "day_in_study"], how="left")
    merged = merged.merge(act_agg, on=["id", "day_in_study"], how="left")

    # Fill NaNs with zeros for activity fields
    for col in [
        "ex_duration_ms", "ex_steps", "ex_avg_hr",
        "sedentary_min", "lightly_min", "moderately_min", "very_min",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    total_active = merged[["lightly_min", "moderately_min", "very_min"]].sum(axis=1)
    ex_hours = merged["ex_duration_ms"] / (1000 * 60 * 60)

    # Raw 0–3 action for exercise
    action_raw = np.zeros(len(merged), dtype=int)
    # intense
    action_raw[(total_active >= 60) | (ex_hours >= 1.0)] = 3
    # moderate
    mask_mod = (total_active >= 30) & (total_active < 60)
    action_raw[mask_mod] = np.maximum(action_raw[mask_mod], 2)
    # light
    mask_light = (total_active > 0) & (total_active < 30)
    action_raw[mask_light] = np.maximum(action_raw[mask_light], 1)

    # Collapse 0–3 → exercise_dim 0–2
    exercise_dim = np.zeros_like(action_raw)
    exercise_dim[action_raw == 0] = 0  # rest/none
    exercise_dim[action_raw == 1] = 0  # light → low
    exercise_dim[action_raw == 2] = 1  # moderate
    exercise_dim[action_raw == 3] = 2  # intense

    # ---------------------------
    # 2) Sleep treatment dimension
    # ---------------------------
    sleep_minutes = merged.get("sleep_minutes", pd.Series(index=merged.index, data=np.nan))
    sleep_eff   = merged.get("sleep_efficiency", pd.Series(index=merged.index, data=np.nan))
    sleep_issue = merged.get("sleepissue_encoded", pd.Series(index=merged.index, data=np.nan))

    sleep_minutes = sleep_minutes.fillna(480.0)  # assume 8h if missing
    sleep_eff = sleep_eff.fillna(90.0)
    sleep_issue = pd.to_numeric(sleep_issue, errors="coerce").fillna(0.0)

    sleep_dim = np.zeros(len(merged), dtype=int)
    bad_sleep_mask = (
        (sleep_minutes < 420) |   # < 7h
        (sleep_eff < 85.0) |
        (sleep_issue >= 3.0)
    )
    sleep_dim[bad_sleep_mask] = 1  # needs sleep intervention

    # ---------------------------
    # 3) Stress treatment dimension
    # ---------------------------
    stress_score = merged.get("stress_score", pd.Series(index=merged.index, data=np.nan))
    stress_enc   = merged.get("stress_encoded", pd.Series(index=merged.index, data=np.nan))

    stress_score = pd.to_numeric(stress_score, errors="coerce").fillna(0.0)
    stress_enc = pd.to_numeric(stress_enc, errors="coerce").fillna(0.0)

    stress_dim = np.zeros(len(merged), dtype=int)
    high_stress_mask = (
        (stress_enc >= 3.0) |    # self-reported high stress
        (stress_score >= 50.0)   # rough threshold for Fitbit stress score
    )
    stress_dim[high_stress_mask] = 1

    # ---------------------------
    # 4) Nutrition / GI treatment dimension
    # ---------------------------
    appetite_enc    = pd.to_numeric(merged.get("appetite_encoded", pd.Series(index=merged.index, data=np.nan)), errors="coerce").fillna(0.0)
    foodcravings_enc= pd.to_numeric(merged.get("foodcravings_encoded", pd.Series(index=merged.index, data=np.nan)), errors="coerce").fillna(0.0)
    indigestion_enc = pd.to_numeric(merged.get("indigestion_encoded", pd.Series(index=merged.index, data=np.nan)), errors="coerce").fillna(0.0)
    bloating_enc    = pd.to_numeric(merged.get("bloating_encoded", pd.Series(index=merged.index, data=np.nan)), errors="coerce").fillna(0.0)

    nutrition_dim = np.zeros(len(merged), dtype=int)
    bad_nutrition_mask = (
        (foodcravings_enc >= 3.0) |
        (indigestion_enc  >= 3.0) |
        (bloating_enc     >= 3.0) |
        (appetite_enc.isin([0.0, 5.0]))  # very low or very high appetite
    )
    nutrition_dim[bad_nutrition_mask] = 1

    # ---------------------------
    # 5) Encode joint action
    # ---------------------------
    # exercise_dim ∈ {0,1,2} (size 3)
    # sleep_dim    ∈ {0,1}   (size 2)
    # stress_dim   ∈ {0,1}   (size 2)
    # nutrition_dim∈ {0,1}   (size 2)
    #
    # index = exercise + 3 * (sleep + 2 * (stress + 2 * nutrition))

    joint_action = (
        exercise_dim
        + 3 * (sleep_dim + 2 * (stress_dim + 2 * nutrition_dim))
    )

    # Attach to dataframe
    merged["exercise_dim"] = exercise_dim
    merged["sleep_dim"] = sleep_dim
    merged["stress_dim"] = stress_dim
    merged["nutrition_dim"] = nutrition_dim
    merged["action"] = joint_action

    return merged


# PARTICIPANT-LEVEL TRAIN/VAL/TEST SPLIT

def split_by_participant(
    panel: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the daily panel into train/val/test *by participant ID*.
    No participant appears in more than one split.
    """
    pids = panel["id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(pids)

    n = len(pids)
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    # remaining go to test
    n_test = n - n_train - n_val

    train_pids = pids[:n_train]
    val_pids = pids[n_train:n_train + n_val]
    test_pids = pids[n_train + n_val:]

    train_df = panel[panel["id"].isin(train_pids)].copy()
    val_df = panel[panel["id"].isin(val_pids)].copy()
    test_df = panel[panel["id"].isin(test_pids)].copy()

    return train_df, val_df, test_df


# 6. MARKOV DATASET (s, a, r, s')

def make_markov_dataset(
    df: pd.DataFrame,
    gamma: float = 0.95,
    state_features: Optional[List[str]] = None,
    symptom_cols: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert the daily panel into (s_t, a_t, r_t, s_{t+1}, done) tuples.

    Reward definition (improvement-based):
        r_t = (symptom_cost_t - symptom_cost_{t+1})

    Then clipped to [-10, 10] and scaled by 1/5 for stability.
    """
    if state_features is None:
        state_features = [f for f in STATE_FEATURES_DEFAULT if f in df.columns]

    if symptom_cols is None:
        symptom_cols = [
            c for c in [
                "cramps_encoded",
                "fatigue_encoded",
                "headaches_encoded",
                "moodswing_encoded",
                "stress_encoded",
                "sleepissue_encoded",
                "bloating_encoded",
            ]
            if c in df.columns
        ]

    # Numeric-only version of the state features
    states_df = df[state_features].copy()
    for col in states_df.columns:
        states_df[col] = pd.to_numeric(states_df[col], errors="coerce")
    states_df = states_df.fillna(0.0)

    s_list, a_list, r_list, sp_list, done_list = [], [], [], [], []
    pid_list, day_list = [], []

    for pid, group in df.groupby("id"):
        # sort by day_in_study but keep original index
        g = group.sort_values("day_in_study")
        if len(g) < 2:
            continue

        g_states = states_df.loc[g.index].to_numpy(dtype=float)
        actions = g["action"].to_numpy(dtype=int)

        if symptom_cols:
            # Coerce symptoms to numeric and fill NaNs with 0 *before* scoring
            symptom_df = g[symptom_cols].apply(
                lambda col: pd.to_numeric(col, errors="coerce")
            ).fillna(0.0)
            symptom_array = symptom_df.to_numpy(dtype=float)
        else:
            symptom_array = np.zeros((len(g), 0))

        wellbeing_cost = symptom_array.sum(axis=1)  # higher = worse symptoms

        for t in range(len(g) - 1):
            s_t = g_states[t]
            a_t = actions[t]
            s_tp1 = g_states[t + 1]

            # Reward = improvement in symptom cost
            r_t = wellbeing_cost[t] - wellbeing_cost[t + 1]
            # Clip reward to [-10, 10]
            r_t = max(min(r_t, 10.0), -10.0)

            done = (t == len(g) - 2)

            s_list.append(s_t)
            a_list.append(a_t)
            r_list.append(r_t)
            sp_list.append(s_tp1)
            done_list.append(done)
            pid_list.append(pid)
            day_list.append(int(g["day_in_study"].iloc[t]))

    rewards = np.array(r_list, dtype=float) / 5.0  # scale to ~[-2,2]
    if np.isnan(rewards).any():
        print("⚠️ Warning: NaNs found in rewards after clipping/scaling. Replacing with 0.")
        rewards = np.nan_to_num(rewards, nan=0.0)

    data = {
        "states": np.vstack(s_list),
        "actions": np.array(a_list),
        "rewards": rewards,
        "next_states": np.vstack(sp_list),
        "dones": np.array(done_list, dtype=bool),
        "participant_ids": np.array(pid_list),
        "days": np.array(day_list),
        "state_features": np.array(state_features),
        "symptom_cols": np.array(symptom_cols),
        "gamma": gamma,
    }
    return data
