"""
Custom2 Experiment: Dark (Opaque) vs Transparent Strategic Classification

Setup:
  - x ~ N(0, I_2)  (2D standard normal)
  - Ground truth: h(x) = sign(w^T x),  w = [1, 2]
  - Linear manipulation cost: c(x,u) = max{0, a^T(u - x)},  a = [1, 1.5]
  - Train: 500 points, Test: 100 points

Protocol:
  1. Train classifier f on h(x) labels using SVM (and separately using Hardt).
  2. Dark case: sample m points from training data, label them with f,
     give (x, f(x)) to contestants who train f_hat via SVM.
     Contestants manipulate test inputs using f_hat.
     Jury evaluates f on manipulated inputs: error = Pr[h(x) != f(Delta_{f_hat}(x))]
  3. Transparent case: contestants know f exactly.
     They manipulate test inputs directly against f.
     Jury evaluates: error = Pr[h(x) != f(Delta_f(x))]
  4. Price of Opacity = error_dark - error_transparent

Sweep over m in {4, 8, 16, 32, 64, 128} (configurable).

Plots:
  1. Errors vs m  (dark SVM, dark Hardt, transparent SVM, transparent Hardt)
  2. Price of Opacity vs m  (SVM, Hardt)
"""

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from tqdm import tqdm
from model import HardtAlgo
from cost_functions import WeightedLinearCostFunction
from strategic_players import strategic_modify_using_known_clf
from utills_and_consts import safe_create_folder, result_folder_path

# Suppress noisy sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ── Hyperparameters (easy to change) ─────────────────────────────────────────
NUM_DIMENSIONS = 1                     # Toggle number of dimensions (e.g., 1 or 2)

if NUM_DIMENSIONS == 1:
    W_TRUE    = np.array([1.0])        # ground truth weight vector
    A_COST    = np.array([1.0])        # cost weight vector
elif NUM_DIMENSIONS == 2:
    W_TRUE    = np.array([1.0, 2.0])
    A_COST    = np.array([1.0, 1.5])
else:
    W_TRUE    = np.ones(NUM_DIMENSIONS)
    A_COST    = np.ones(NUM_DIMENSIONS)

FEATURE_LIST = [f'f{i}' for i in range(NUM_DIMENSIONS)]

COST_FACTOR = 1                        # scale of cost function

TRAIN_SIZE = 700
TEST_SIZE  = 100

M_VALUES = [4, 8, 16, 32, 64, 128, 256, 512]    # number of friend-labelled samples

NUM_DARK_REPEATS = 1                  # repeat dark experiment to reduce noise

SEED = 42


# ── Data generation ──────────────────────────────────────────────────────────

def create_dataset(n, w, seed=None):
    """
    Generate n points from N(0, I_d) with labels sign(w^T x).
    Returns a DataFrame with columns: f0, ..., fn, MemberKey, LoanStatus
    """
    rng = np.random.RandomState(seed)
    dim = len(w)
    
    # Support arbitrary dimensions
    X = rng.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=n)
    
    labels = np.sign(X @ w).astype(int)
    labels[labels == 0] = 1

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(dim)])
    df['MemberKey'] = [f's{i}' for i in range(n)]
    df['LoanStatus'] = labels
    return df


# ── Transparent evaluation ───────────────────────────────────────────────────

def compute_transparent_error(test_df, clf, true_labels, cost_vec, cost_factor):
    """
    Transparent case: contestants know clf perfectly.
    They manipulate, jury evaluates clf on manipulated inputs.
    Returns error = Pr[h(x) != f(Delta_f(x))]
    """
    cost_func = WeightedLinearCostFunction(cost_vec, cost_factor=cost_factor)
    modified_df = strategic_modify_using_known_clf(test_df, clf, FEATURE_LIST, cost_func)

    if hasattr(clf, 'coef_') and not isinstance(clf, HardtAlgo):
        preds = clf.predict(modified_df[FEATURE_LIST])
    else:
        preds = clf.predict(pd.DataFrame(modified_df[FEATURE_LIST]))

    error = np.mean(preds != true_labels)
    return error


# ── Dark evaluation ──────────────────────────────────────────────────────────

def compute_dark_error(train_df, test_df, clf, true_labels, m, cost_vec,
                       cost_factor, rng, desc="Eval"):
    """
    Dark case:
      1. Sample m points from train_df, label them with clf  -> (x, f(x))
      2. Train f_hat (SVM) on these m labelled points
      3. Contestants manipulate test_df using f_hat
      4. Jury evaluates clf on manipulated inputs
    Returns error = Pr[h(x) != f(Delta_{f_hat}(x))]
    """
    cost_func = WeightedLinearCostFunction(cost_vec, cost_factor=cost_factor)
    incorrect_count = 0
    n_test = len(test_df)

    for i in tqdm(range(n_test), desc=desc, leave=False):
        # Step 1: sample m friend-points and label with clf for THIS specific user
        # Use rejection sampling to ensure at least one example from each class
        while True:
            indices = rng.choice(len(train_df), size=m, replace=False)
            friend_X = train_df.iloc[indices][FEATURE_LIST].values

            if isinstance(clf, HardtAlgo):
                friend_labels = clf.predict(pd.DataFrame(friend_X, columns=FEATURE_LIST))
            else:
                friend_labels = clf.predict(friend_X)
                
            # Check if both classes are present
            if len(np.unique(friend_labels)) >= 2:
                break

        # Step 2: train individualized f_x via SVM on friend data
        f_x = LinearSVC(C=1000, random_state=42, max_iter=1000000)
        f_x.fit(friend_X, friend_labels)

        # Step 3: user manipulates ONLY their own point
        row_df = test_df.iloc[[i]].copy()
        modified_row = strategic_modify_using_known_clf(row_df, f_x, FEATURE_LIST, cost_func)

        # Step 4: jury evaluates the REAL classifier clf on manipulated data
        if isinstance(clf, HardtAlgo):
            pred = clf.predict(pd.DataFrame(modified_row[FEATURE_LIST]))
        else:
            pred = clf.predict(modified_row[FEATURE_LIST].values)

        pred_val = pred.iloc[0] if isinstance(pred, pd.Series) else pred[0]
        if pred_val != true_labels[i]:
            incorrect_count += 1

    error = incorrect_count / n_test
    return error


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_errors_vs_m(m_values, dark_svm_errors, dark_hardt_errors,
                     trans_svm_error, trans_hardt_error,
                     dark_svm_std, dark_hardt_std,
                     nonstrat_svm_error, nonstrat_hardt_error, save_path):
    """
    Plot all error curves vs m.
    Transparent and non-strategic errors are horizontal lines.
    Dark errors are plotted with error bars from repeated runs.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Dark errors with error bands
    ax.errorbar(m_values, dark_svm_errors, yerr=dark_svm_std,
                color='#2196F3', marker='o', linewidth=2.5, markersize=8,
                capsize=5, capthick=1.5, label='Dark SVM', zorder=5)
    ax.errorbar(m_values, dark_hardt_errors, yerr=dark_hardt_std,
                color='#9C27B0', marker='s', linewidth=2.5, markersize=8,
                capsize=5, capthick=1.5, label='Dark Hardt', zorder=5)

    # Transparent errors (horizontal baselines)
    ax.axhline(y=trans_svm_error, color='#2196F3', linewidth=2, linestyle='--',
               alpha=0.7, label=f'Transparent SVM ({trans_svm_error:.3f})')
    ax.axhline(y=trans_hardt_error, color='#9C27B0', linewidth=2, linestyle='--',
               alpha=0.7, label=f'Transparent Hardt ({trans_hardt_error:.3f})')

    # Non-strategic errors (no manipulation at all)
    ax.axhline(y=nonstrat_svm_error, color='#2196F3', linewidth=2, linestyle=':',
               alpha=0.5, label=f'Non-strategic SVM ({nonstrat_svm_error:.3f})')
    ax.axhline(y=nonstrat_hardt_error, color='#9C27B0', linewidth=2, linestyle=':',
               alpha=0.5, label=f'Non-strategic Hardt ({nonstrat_hardt_error:.3f})')

    # Shade the Price of Opacity region for Hardt
    trans_hardt_line = [trans_hardt_error] * len(m_values)
    ax.fill_between(m_values, dark_hardt_errors, trans_hardt_line,
                    color='#9C27B0', alpha=0.15, label='PoO (Hardt)', zorder=2)

    ax.set_xscale('log', base=2)
    ax.set_xticks(m_values)
    ax.set_xticklabels([str(m) for m in m_values])
    ax.set_xlabel('Number of friend samples ($m$)', fontsize=14)
    ax.set_ylabel('Error: $\\Pr[h(x) \\neq f(\\Delta(x))]$', fontsize=14)
    ax.set_title('Strategic Error vs Number of Friend Samples\n'
                 '(Dark vs Transparent vs Non-strategic)', fontsize=15)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot 1] Saved: {save_path}')


def plot_price_of_opacity(m_values, pop_svm, pop_hardt,
                          pop_svm_std, pop_hardt_std, save_path):
    """
    Plot Price of Opacity vs m for both SVM and Hardt.
    PoO = error_dark - error_transparent
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(m_values, pop_svm, yerr=pop_svm_std,
                color='#2196F3', marker='o', linewidth=2.5, markersize=8,
                capsize=5, capthick=1.5, label='Price of Opacity (SVM)', zorder=5)
    ax.errorbar(m_values, pop_hardt, yerr=pop_hardt_std,
                color='#9C27B0', marker='s', linewidth=2.5, markersize=8,
                capsize=5, capthick=1.5, label='Price of Opacity (Hardt)', zorder=5)

    ax.axhline(y=0, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)

    ax.set_xscale('log', base=2)
    ax.set_xticks(m_values)
    ax.set_xticklabels([str(m) for m in m_values])
    ax.set_xlabel('Number of friend samples ($m$)', fontsize=14)
    ax.set_ylabel('Price of Opacity\n(Error$_{dark}$ − Error$_{transparent}$)', fontsize=14)
    ax.set_title('Price of Opacity vs Number of Friend Samples', fontsize=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot 2] Saved: {save_path}')


def plot_combined_summary(m_values, dark_svm_errors, dark_hardt_errors,
                          trans_svm_error, trans_hardt_error,
                          dark_svm_std, dark_hardt_std,
                          pop_svm, pop_hardt,
                          pop_svm_std, pop_hardt_std,
                          nonstrat_svm_error, nonstrat_hardt_error, save_path):
    """
    Combined figure with two subplots: errors (left) and PoO (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: Errors vs m ──
    ax1.errorbar(m_values, dark_svm_errors, yerr=dark_svm_std,
                 color='#2196F3', marker='o', linewidth=2.5, markersize=8,
                 capsize=5, capthick=1.5, label='Dark SVM', zorder=5)
    ax1.errorbar(m_values, dark_hardt_errors, yerr=dark_hardt_std,
                 color='#9C27B0', marker='s', linewidth=2.5, markersize=8,
                 capsize=5, capthick=1.5, label='Dark Hardt', zorder=5)
    ax1.axhline(y=trans_svm_error, color='#2196F3', linewidth=2, linestyle='--',
                alpha=0.7, label=f'Transparent SVM ({trans_svm_error:.3f})')
    ax1.axhline(y=trans_hardt_error, color='#9C27B0', linewidth=2, linestyle='--',
                alpha=0.7, label=f'Transparent Hardt ({trans_hardt_error:.3f})')

    # Non-strategic errors
    ax1.axhline(y=nonstrat_svm_error, color='#2196F3', linewidth=2, linestyle=':',
                alpha=0.5, label=f'Non-strategic SVM ({nonstrat_svm_error:.3f})')
    ax1.axhline(y=nonstrat_hardt_error, color='#9C27B0', linewidth=2, linestyle=':',
                alpha=0.5, label=f'Non-strategic Hardt ({nonstrat_hardt_error:.3f})')

    # Shade the Price of Opacity region for Hardt
    trans_hardt_line = [trans_hardt_error] * len(m_values)
    ax1.fill_between(m_values, dark_hardt_errors, trans_hardt_line,
                     color='#9C27B0', alpha=0.15, label='PoO (Hardt)', zorder=2)

    ax1.set_xscale('log', base=2)
    ax1.set_xticks(m_values)
    ax1.set_xticklabels([str(m) for m in m_values])
    ax1.set_xlabel('Number of friend samples ($m$)', fontsize=14)
    ax1.set_ylabel('Error: $\\Pr[h(x) \\neq f(\\Delta(x))]$', fontsize=14)
    ax1.set_title('Strategic Error vs $m$', fontsize=15)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.05)

    # ── Right: Price of Opacity vs m ──
    ax2.errorbar(m_values, pop_svm, yerr=pop_svm_std,
                 color='#2196F3', marker='o', linewidth=2.5, markersize=8,
                 capsize=5, capthick=1.5, label='SVM', zorder=5)
    ax2.errorbar(m_values, pop_hardt, yerr=pop_hardt_std,
                 color='#9C27B0', marker='s', linewidth=2.5, markersize=8,
                 capsize=5, capthick=1.5, label='Hardt', zorder=5)
    ax2.axhline(y=0, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(m_values)
    ax2.set_xticklabels([str(m) for m in m_values])
    ax2.set_xlabel('Number of friend samples ($m$)', fontsize=14)
    ax2.set_ylabel('Price of Opacity', fontsize=14)
    ax2.set_title('Price of Opacity vs $m$', fontsize=15)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Custom2: Dark vs Transparent Strategic Classification',
                 fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot 3] Saved: {save_path}')


# ── Main experiment runner ───────────────────────────────────────────────────

def run_custom2_experiment():
    """Run the full custom2 dark-vs-transparent experiment."""
    np.random.seed(SEED)
    print('=' * 70)
    print(' Custom2 Experiment: Dark vs Transparent Strategic Classification')
    print('=' * 70)

    # ── Create output folder ─────────────────────────────────────────────
    exp_folder = safe_create_folder(result_folder_path, 'custom2_exp')

    # ── Generate data ────────────────────────────────────────────────────
    print(f'\n[1/5] Generating data: {TRAIN_SIZE} train + {TEST_SIZE} test ...')
    train_df = create_dataset(TRAIN_SIZE, W_TRUE, seed=SEED)
    test_df  = create_dataset(TEST_SIZE,  W_TRUE, seed=SEED + 1)

    true_labels = test_df['LoanStatus'].values
    print(f'  Train: +1={sum(train_df["LoanStatus"]==1)}, '
          f'-1={sum(train_df["LoanStatus"]==-1)}')
    print(f'  Test:  +1={sum(test_df["LoanStatus"]==1)}, '
          f'-1={sum(test_df["LoanStatus"]==-1)}')

    # ── Train classifiers on h(x) labels ─────────────────────────────────
    print('\n[2/5] Training classifiers on ground truth labels ...')

    # SVM
    svm_model = LinearSVC(C=1000, random_state=SEED, max_iter=100000)
    svm_model.fit(train_df[FEATURE_LIST], train_df['LoanStatus'])
    print(f'  SVM   coef: {svm_model.coef_[0]}, intercept: {svm_model.intercept_[0]:.4f}')

    # Hardt
    hardt_cost = WeightedLinearCostFunction(A_COST, cost_factor=COST_FACTOR)
    hardt_model = HardtAlgo(hardt_cost)
    hardt_model.fit(train_df[FEATURE_LIST], train_df['LoanStatus'])
    print(f'  Hardt cost vector a: {A_COST}, min_si: {hardt_model.min_si:.4f}')

    # ── Non-strategic errors (no manipulation) ────────────────────────────
    print('\n[3/6] Computing non-strategic errors ...')
    svm_pred_orig = svm_model.predict(test_df[FEATURE_LIST])
    hardt_pred_orig = hardt_model.predict(pd.DataFrame(test_df[FEATURE_LIST]))
    nonstrat_svm_error = np.mean(svm_pred_orig != true_labels)
    nonstrat_hardt_error = np.mean(hardt_pred_orig != true_labels)
    print(f'  Non-strategic SVM   error: {nonstrat_svm_error:.4f}')
    print(f'  Non-strategic Hardt error: {nonstrat_hardt_error:.4f}')

    # ── Transparent errors (computed once) ────────────────────────────────
    print('\n[4/6] Computing transparent errors ...')
    trans_svm_error = compute_transparent_error(
        test_df, svm_model, true_labels, A_COST, COST_FACTOR)
    trans_hardt_error = compute_transparent_error(
        test_df, hardt_model, true_labels, A_COST, COST_FACTOR)
    print(f'  Transparent SVM   error: {trans_svm_error:.4f}')
    print(f'  Transparent Hardt error: {trans_hardt_error:.4f}')

    # ── Dark errors for each m ────────────────────────────────────────────
    print(f'\n[5/6] Computing dark errors for m = {M_VALUES} ...')
    print(f'       (repeating {NUM_DARK_REPEATS} times per m to reduce noise)\n')

    dark_svm_errors_mean   = []
    dark_svm_errors_std    = []
    dark_hardt_errors_mean = []
    dark_hardt_errors_std  = []

    for m in M_VALUES:
        print(f'  m = {m}:')
        svm_errors_for_m   = []
        hardt_errors_for_m = []

        for repeat in range(NUM_DARK_REPEATS):
            rng = np.random.RandomState(SEED + 100 * m + repeat)

            # Dark SVM
            err_svm = compute_dark_error(
                train_df, test_df, svm_model, true_labels,
                m, A_COST, COST_FACTOR, rng,
                desc=f"Dark SVM (m={m}, rep={repeat+1}/{NUM_DARK_REPEATS})")
            svm_errors_for_m.append(err_svm)

            # Dark Hardt
            rng2 = np.random.RandomState(SEED + 100 * m + repeat + 1000)
            err_hardt = compute_dark_error(
                train_df, test_df, hardt_model, true_labels,
                m, A_COST, COST_FACTOR, rng2,
                desc=f"Dark Hardt (m={m}, rep={repeat+1}/{NUM_DARK_REPEATS})")
            hardt_errors_for_m.append(err_hardt)

        svm_mean   = np.mean(svm_errors_for_m)
        svm_std    = np.std(svm_errors_for_m)
        hardt_mean = np.mean(hardt_errors_for_m)
        hardt_std  = np.std(hardt_errors_for_m)

        dark_svm_errors_mean.append(svm_mean)
        dark_svm_errors_std.append(svm_std)
        dark_hardt_errors_mean.append(hardt_mean)
        dark_hardt_errors_std.append(hardt_std)

        print(f'    Dark SVM   error: {svm_mean:.4f} ± {svm_std:.4f}')
        print(f'    Dark Hardt error: {hardt_mean:.4f} ± {hardt_std:.4f}')

    # ── Compute Price of Opacity ──────────────────────────────────────────
    pop_svm   = [d - trans_svm_error   for d in dark_svm_errors_mean]
    pop_hardt = [d - trans_hardt_error  for d in dark_hardt_errors_mean]
    # Standard deviations carry through since transparent error is a constant
    pop_svm_std   = dark_svm_errors_std
    pop_hardt_std = dark_hardt_errors_std

    # ── Generate plots ────────────────────────────────────────────────────
    print('\n[6/6] Generating plots ...')

    # Plot 1: All errors vs m
    plot_errors_vs_m(
        M_VALUES,
        dark_svm_errors_mean, dark_hardt_errors_mean,
        trans_svm_error, trans_hardt_error,
        dark_svm_errors_std, dark_hardt_errors_std,
        nonstrat_svm_error, nonstrat_hardt_error,
        os.path.join(exp_folder, '1_errors_vs_m.png'))

    # Plot 2: Price of Opacity vs m
    plot_price_of_opacity(
        M_VALUES, pop_svm, pop_hardt,
        pop_svm_std, pop_hardt_std,
        os.path.join(exp_folder, '2_price_of_opacity.png'))

    # Plot 3: Combined summary
    plot_combined_summary(
        M_VALUES,
        dark_svm_errors_mean, dark_hardt_errors_mean,
        trans_svm_error, trans_hardt_error,
        dark_svm_errors_std, dark_hardt_errors_std,
        pop_svm, pop_hardt,
        pop_svm_std, pop_hardt_std,
        nonstrat_svm_error, nonstrat_hardt_error,
        os.path.join(exp_folder, '3_combined_summary.png'))

    # ── Print summary table ───────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(' RESULTS SUMMARY')
    print('=' * 70)
    print(f'  Ground truth w = {W_TRUE}')
    print(f'  Cost vector  a = {A_COST}')
    print(f'  Non-strategic SVM   error: {nonstrat_svm_error:.4f}')
    print(f'  Non-strategic Hardt error: {nonstrat_hardt_error:.4f}')
    print(f'  Transparent SVM   error: {trans_svm_error:.4f}')
    print(f'  Transparent Hardt error: {trans_hardt_error:.4f}')
    print()
    print(f'  {"m":>5}  {"Dark SVM":>14}  {"Dark Hardt":>14}  '
          f'{"PoO SVM":>10}  {"PoO Hardt":>10}')
    print(f'  {"-"*5}  {"-"*14}  {"-"*14}  {"-"*10}  {"-"*10}')
    for i, m in enumerate(M_VALUES):
        print(f'  {m:>5}  '
              f'{dark_svm_errors_mean[i]:>6.4f}+/-{dark_svm_errors_std[i]:.4f}  '
              f'{dark_hardt_errors_mean[i]:>6.4f}+/-{dark_hardt_errors_std[i]:.4f}  '
              f'{pop_svm[i]:>10.4f}  {pop_hardt[i]:>10.4f}')
    print(f'\n  All plots saved to: {exp_folder}')
    print('=' * 70)


if __name__ == '__main__':
    run_custom2_experiment()
