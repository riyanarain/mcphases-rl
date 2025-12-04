"""
menstrual_rl_project.py

Offline RL + behavioral cloning for menstrual health recommendations
using the mcPHASES dataset.

Data preprocessing (loading, encoding, panel construction, and Markov dataset
generation) is handled in data_preprocessing.py.

This file:
  - uses the preprocessed Markov datasets (train/val/test)
  - trains a behavior cloning baseline
  - trains a Fitted Q-Iteration offline RL agent with regularization
  - uses a BC-regularized FQI policy (closer to behavior policy)
  - runs simple off-policy evaluation + policy disagreement diagnostics
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from data_preprocessing import (
    DataConfig,
    load_mcphases_tables,
    build_daily_panel,
    split_by_participant,
    make_markov_dataset,
)


# RL HYPERPARAM CONFIG

@dataclass
class RLConfig:
    # Behavior cloning
    bc_n_estimators: int = 200
    bc_max_depth: Optional[int] = None

    # Fitted Q-Iteration
    fqi_iterations: int = 20
    fqi_n_estimators: int = 200
    fqi_max_depth: Optional[int] = None

    random_state: int = 42

    # BC-regularization strength for FQI policy
    bc_regularization_alpha: float = 0.1  # weight on log π_BC(a|s)


# BEHAVIOR CLONING (SUPERVISED BASELINE)

@dataclass
class BehaviorCloningModel:
    clf: RandomForestClassifier
    scaler: StandardScaler
    action_space: np.ndarray


def train_behavior_cloning(
    rl_cfg: RLConfig,
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[BehaviorCloningModel, Dict[str, float]]:
    """
    Supervised learning baseline: predict action from state.
    Trained ONLY on train_data, evaluated on val_data if provided.
    """
    X_train = train_data["states"]
    y_train = train_data["actions"]
    action_space = np.unique(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=rl_cfg.bc_n_estimators,
        max_depth=rl_cfg.bc_max_depth,
        random_state=rl_cfg.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    metrics: Dict[str, float] = {}

    if val_data is not None and len(val_data["states"]) > 0:
        X_val = val_data["states"]
        y_val = val_data["actions"]
        X_val_scaled = scaler.transform(X_val)
        y_pred = clf.predict(X_val_scaled)
        print("Behavior Cloning validation report:")
        print(classification_report(y_val, y_pred))
        metrics["val_accuracy"] = float((y_pred == y_val).mean())

    bc_model = BehaviorCloningModel(
        clf=clf,
        scaler=scaler,
        action_space=action_space,
    )

    return bc_model, metrics


def bc_policy(bc_model: BehaviorCloningModel) -> Callable[[np.ndarray], int]:
    """
    Deterministic greedy policy π(s) = argmax_a p(a|s) from the RF classifier.
    """
    def policy_fn(state: np.ndarray) -> int:
        state = np.asarray(state).reshape(1, -1)
        s_scaled = bc_model.scaler.transform(state)
        probs = bc_model.clf.predict_proba(s_scaled)[0]
        return int(bc_model.action_space[np.argmax(probs)])

    return policy_fn


def bc_action_probs(
    bc_model: BehaviorCloningModel, state: np.ndarray, target_action_space: np.ndarray
) -> np.ndarray:
    """
    Get behavior policy probabilities p_BC(a|s) aligned to the target_action_space.
    """
    state = np.asarray(state).reshape(1, -1)
    s_scaled = bc_model.scaler.transform(state)
    probs_full = bc_model.clf.predict_proba(s_scaled)[0]  # aligned with bc_model.action_space

    prob_vec = np.zeros(len(target_action_space), dtype=float)
    for j, a in enumerate(target_action_space):
        match = np.where(bc_model.action_space == a)[0]
        if len(match) > 0:
            prob_vec[j] = probs_full[match[0]]
        else:
            prob_vec[j] = 1e-3  # small non-zero to avoid log(0)

    # normalize to sum to 1 (just in case)
    s = prob_vec.sum()
    if s > 0:
        prob_vec /= s
    else:
        prob_vec = np.ones_like(prob_vec) / len(prob_vec)

    return prob_vec


# FITTED Q-ITERATION (OFFLINE BATCH Q-LEARNING)

@dataclass
class FQIModel:
    regressor: RandomForestRegressor
    scaler: StandardScaler
    action_space: np.ndarray
    gamma: float


def train_fitted_q_iteration(
    rl_cfg: RLConfig,
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    n_iterations: Optional[int] = None,
) -> FQIModel:
    """
    Fitted Q-Iteration (batch Q-learning) with RandomForestRegressor.
    Trained ONLY on train_data, optionally monitored on val_data for Bellman MSE.
    """
    if n_iterations is None:
        n_iterations = rl_cfg.fqi_iterations

    S = train_data["states"]
    A = train_data["actions"]
    R = train_data["rewards"]
    S_next = train_data["next_states"]
    gamma = float(train_data["gamma"])
    action_space = np.unique(A)

    n_actions = len(action_space)

    def encode_sa(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((len(actions), n_actions), dtype=float)
        for i, a in enumerate(actions):
            idx = np.where(action_space == a)[0][0]
            one_hot[i, idx] = 1.0
        return np.hstack([states, one_hot])

    X_sa = encode_sa(S, A)

    scaler = StandardScaler()
    X_sa_scaled = scaler.fit_transform(X_sa)

    reg = RandomForestRegressor(
        n_estimators=rl_cfg.fqi_n_estimators,
        max_depth=rl_cfg.fqi_max_depth,
        random_state=rl_cfg.random_state,
        n_jobs=-1,
        min_samples_leaf=5,   # regularization: avoid tiny leaves
        max_features="sqrt",  # feature subsampling
    )

    # Initialize: fit Q(s,a) ~ immediate reward R
    y = R.copy()
    reg.fit(X_sa_scaled, y)

    best_reg = deepcopy(reg)
    best_scaler = deepcopy(scaler)
    best_val_mse = None

    for it in range(n_iterations):
        # Compute max_a' Q(s', a') for each next state
        q_next = np.zeros((len(S_next), n_actions))
        for j, a_prime in enumerate(action_space):
            X_sprime_ap = encode_sa(S_next, np.full(len(S_next), a_prime))
            X_sprime_ap_scaled = scaler.transform(X_sprime_ap)
            q_next[:, j] = reg.predict(X_sprime_ap_scaled)

        target = R + gamma * q_next.max(axis=1)
        # Clip targets for additional stability
        target = np.clip(target, -10.0, 10.0)

        # Refit regressor on new Bellman targets (update scaler using X_sa again)
        X_sa_scaled = scaler.fit_transform(X_sa)
        reg.fit(X_sa_scaled, target)

        # Optional: validation Bellman MSE for early stopping/model selection
        if val_data is not None and len(val_data["states"]) > 0:
            S_val = val_data["states"]
            A_val = val_data["actions"]
            R_val = val_data["rewards"]
            S_next_val = val_data["next_states"]
            gamma_val = float(val_data["gamma"])

            X_val_sa = encode_sa(S_val, A_val)
            X_val_sa_scaled = scaler.transform(X_val_sa)
            q_val = reg.predict(X_val_sa_scaled)

            q_next_val = np.zeros((len(S_next_val), n_actions))
            for j, a_prime in enumerate(action_space):
                X_sprime_ap_val = encode_sa(S_next_val, np.full(len(S_next_val), a_prime))
                X_sprime_ap_val_scaled = scaler.transform(X_sprime_ap_val)
                q_next_val[:, j] = reg.predict(X_sprime_ap_val_scaled)

            target_val = R_val + gamma_val * q_next_val.max(axis=1)
            target_val = np.clip(target_val, -10.0, 10.0)
            bellman_mse = float(np.mean((q_val - target_val) ** 2))

            if best_val_mse is None or bellman_mse < best_val_mse:
                best_val_mse = bellman_mse
                best_reg = deepcopy(reg)
                best_scaler = deepcopy(scaler)

            print(f"[FQI] Iteration {it+1}/{n_iterations}, val Bellman MSE={bellman_mse:.4f}")
        else:
            print(f"[FQI] Iteration {it+1}/{n_iterations} complete.")

    # Use best validation model if available
    final_reg = best_reg if best_val_mse is not None else reg
    final_scaler = best_scaler if best_val_mse is not None else scaler

    return FQIModel(
        regressor=final_reg,
        scaler=final_scaler,
        action_space=action_space,
        gamma=gamma,
    )


def fqi_policy(model: FQIModel) -> Callable[[np.ndarray], int]:
    """
    Greedy policy π_Q(s) = argmax_a Q_hat(s,a).
    """
    def policy_fn(state: np.ndarray) -> int:
        state = np.asarray(state).reshape(1, -1)
        n_actions = len(model.action_space)
        sa_list = []
        for a in model.action_space:
            one_hot = np.zeros((1, n_actions))
            idx = np.where(model.action_space == a)[0][0]
            one_hot[0, idx] = 1.0
            sa_list.append(np.hstack([state, one_hot]))
        X = np.vstack(sa_list)
        X_scaled = model.scaler.transform(X)
        q_vals = model.regressor.predict(X_scaled)
        a_best = int(model.action_space[np.argmax(q_vals)])
        return a_best

    return policy_fn


def fqi_policy_bc_regularized(
    model: FQIModel,
    bc_model: BehaviorCloningModel,
    alpha: float = 0.1,
) -> Callable[[np.ndarray], int]:
    """
    Conservative FQI policy that adds a behavior-cloning prior:

        argmax_a [ Q(s,a) + α * log π_BC(a|s) ]

    This encourages the policy to stay closer to the behavior policy,
    which is important in offline RL to avoid out-of-support actions.
    """
    def policy_fn(state: np.ndarray) -> int:
        state = np.asarray(state).reshape(1, -1)
        n_actions = len(model.action_space)

        # Build [s, one_hot(a)] for each a
        sa_list = []
        for a in model.action_space:
            one_hot = np.zeros((1, n_actions))
            idx = np.where(model.action_space == a)[0][0]
            one_hot[0, idx] = 1.0
            sa_list.append(np.hstack([state, one_hot]))
        X = np.vstack(sa_list)
        X_scaled = model.scaler.transform(X)
        q_vals = model.regressor.predict(X_scaled)

        # Behavior policy probabilities over same actions
        p_bc = bc_action_probs(bc_model, state, model.action_space)

        q_adj = q_vals + alpha * np.log(p_bc + 1e-6)
        a_best = int(model.action_space[np.argmax(q_adj)])
        return a_best

    return policy_fn


# SIMPLE OFF-POLICY EVALUATION (WEIGHTED IS)

def estimate_return_weighted_importance_sampling(
    data: Dict[str, np.ndarray],
    behavior_policy: Callable[[np.ndarray], int],
    target_policy: Callable[[np.ndarray], int],
) -> float:
    """
    VERY rough Weighted Importance Sampling estimator of target policy return.

    Uses per-participant trajectories with cumulative importance weight.
    """
    S = data["states"]
    A = data["actions"]
    R = data["rewards"]
    pids = data["participant_ids"]

    returns: List[float] = []
    weights: List[float] = []

    for pid, idxs in _group_indices_by_pid(pids).items():
        idxs = sorted(idxs)
        w = 1.0
        G = 0.0

        for i in idxs:
            s = S[i]
            a = A[i]

            a_b = behavior_policy(s)
            a_t = target_policy(s)

            # Smoothed probabilities to avoid zero-division:
            eps = 1e-2
            p_b = 1.0 - eps if a == a_b else eps
            p_t = 1.0 - eps if a == a_t else eps

            w *= p_t / p_b
            G += R[i]

        returns.append(G)
        weights.append(w)

    weights_arr = np.array(weights)
    returns_arr = np.array(returns)

    if np.all(weights_arr == 0):
        return float("nan")

    wis = float(np.sum(weights_arr * returns_arr) / np.sum(weights_arr))
    return wis


def _group_indices_by_pid(pids: np.ndarray) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, pid in enumerate(pids):
        groups.setdefault(pid, []).append(i)
    return groups


# DIAGNOSTICS


def summarize_rewards(name: str, data: Dict[str, np.ndarray]) -> None:
    R = data["rewards"]
    print(f"[{name}] Reward summary:")
    print(f"  mean: {R.mean():.3f}")
    print(f"  std:  {R.std():.3f}")
    print(f"  min:  {R.min():.3f}")
    print(f"  max:  {R.max():.3f}")


def summarize_actions(name: str, data: Dict[str, np.ndarray]) -> None:
    A = data["actions"]
    unique, counts = np.unique(A, return_counts=True)
    print(f"[{name}] Action distribution:")
    for u, c in zip(unique, counts):
        print(f"  action {u}: {c} samples ({c/len(A):.1%})")


def compute_policy_disagreement(
    data: Dict[str, np.ndarray],
    policy_1: Callable[[np.ndarray], int],
    policy_2: Callable[[np.ndarray], int],
) -> Tuple[float, Dict[Tuple[int, int], int]]:
    """
    Compute how often two policies choose different actions on the same states.

    Returns:
      disagreement_rate: fraction of states where actions differ
      pair_counts: dict mapping (a1, a2) -> count
    """
    S = data["states"]
    n = len(S)

    diff_count = 0
    pair_counts: Dict[Tuple[int, int], int] = {}

    for i in range(n):
        s = S[i]
        a1 = policy_1(s)
        a2 = policy_2(s)

        pair_counts[(a1, a2)] = pair_counts.get((a1, a2), 0) + 1
        if a1 != a2:
            diff_count += 1

    disagreement_rate = diff_count / n if n > 0 else 0.0
    return disagreement_rate, pair_counts


# MAIN PIPELINE

def main():
    # ----- Data configuration -----
    data_cfg = DataConfig(
        data_dir="mcPHASES",          # <-- update if needed
        min_days_per_participant=10,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        random_state=42,
        gamma=0.95,
    )

    # ----- RL hyperparameters -----
    rl_cfg = RLConfig(
        bc_n_estimators=200,
        bc_max_depth=None,
        fqi_iterations=20,
        fqi_n_estimators=200,
        fqi_max_depth=None,
        random_state=42,
        bc_regularization_alpha=0.1,
    )

    print("Loading mcPHASES tables...")
    tables = load_mcphases_tables(data_cfg.data_dir)

    print("Building daily panel...")
    panel = build_daily_panel(tables, min_days_per_participant=data_cfg.min_days_per_participant)
    print(f"Daily panel shape: {panel.shape}")

    print("\nSplitting by participant into train/val/test...")
    train_df, val_df, test_df = split_by_participant(
        panel,
        train_frac=data_cfg.train_frac,
        val_frac=data_cfg.val_frac,
        test_frac=data_cfg.test_frac,
        random_state=data_cfg.random_state,
    )
    print(f"  Train rows: {len(train_df)}")
    print(f"  Val rows:   {len(val_df)}")
    print(f"  Test rows:  {len(test_df)}")

    print("\nConstructing Markov datasets...")
    train_data = make_markov_dataset(train_df, gamma=data_cfg.gamma)
    val_data = make_markov_dataset(val_df, gamma=data_cfg.gamma)
    test_data = make_markov_dataset(test_df, gamma=data_cfg.gamma)

    summarize_rewards("TRAIN", train_data)
    summarize_actions("TRAIN", train_data)
    summarize_rewards("VAL", val_data)
    summarize_actions("VAL", val_data)
    summarize_rewards("TEST", test_data)
    summarize_actions("TEST", test_data)

    print("\nTraining behavior cloning baseline on TRAIN, validating on VAL...")
    bc_model, bc_metrics = train_behavior_cloning(rl_cfg, train_data, val_data)
    bc_pi = bc_policy(bc_model)
    print("BC metrics:", bc_metrics)

    print("\nTraining Fitted Q-Iteration offline RL agent on TRAIN (with val monitoring)...")
    fqi_model = train_fitted_q_iteration(rl_cfg, train_data, val_data)
    fqi_pi_raw = fqi_policy(fqi_model)
    fqi_pi = fqi_policy_bc_regularized(
        fqi_model,
        bc_model,
        alpha=rl_cfg.bc_regularization_alpha,
    )

    print("\nComputing policy disagreement on TEST states (BC vs BC-regularized FQI)...")
    disagreement_rate, pair_counts = compute_policy_disagreement(
        test_data, bc_pi, fqi_pi
    )
    print(f"Policy disagreement rate (BC vs FQI) on TEST: {disagreement_rate:.3f}")
    print("Action pair counts (a_bc, a_fqi) on TEST:")
    for (a_bc, a_fqi), cnt in sorted(pair_counts.items()):
        print(f"  BC={a_bc}, FQI={a_fqi}: {cnt} states")

    print("\nOff-policy evaluation (WIS) on TEST trajectories...")
    wis_bc = estimate_return_weighted_importance_sampling(
        test_data,
        behavior_policy=bc_pi,   # approximate logging policy
        target_policy=bc_pi,     # BC evaluated against itself
    )
    wis_fqi = estimate_return_weighted_importance_sampling(
        test_data,
        behavior_policy=bc_pi,   # logging ≈ BC
        target_policy=fqi_pi,    # conservative FQI as target
    )

    print(f"Estimated return (WIS) BC policy on TEST:  {wis_bc:.3f}")
    print(f"Estimated return (WIS) FQI policy on TEST: {wis_fqi:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
