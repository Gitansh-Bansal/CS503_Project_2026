"""
Custom1 Experiment: Transparent Model (Hardt) on 2D Synthetic Data

Setup:
  - x ~ N(0, I_2)  (2D standard normal)
  - Ground truth: h(x) = sign(w^T x),  w = [1, 2]
  - Linear manipulation cost: c(x,u) = max{0, a^T(u - x)},  a = [1, 1.5]
  - Train: 500 points, Test: 100 points
  - Two classifiers: Hardt (strategic-aware) and SVM (non-strategic)

Visualizations:
  1. Generated data with true labels (color coded)
  2. SVM, Hardt, and ground truth decision boundaries in one graph
  3. Original vs strategic accuracy (bar chart)
  4. Movement of manipulated points (arrows) with both classifiers shown
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import LinearSVC
from model import HardtAlgo
from cost_functions import WeightedLinearCostFunction
from strategic_players import strategic_modify_using_known_clf
from utills_and_consts import safe_create_folder, result_folder_path

# ── Fixed parameters ──────────────────────────────────────────────────────────
W_TRUE = np.array([1.0, 2.0])          # ground truth weight vector
A_COST = np.array([1.0, 1.5])          # cost weight vector
COST_FACTOR = 1                        # scale of cost function
TRAIN_SIZE = 500
TEST_SIZE = 100
SEED = 42
FEATURE_LIST = ['f0', 'f1']


# ── Data generation ──────────────────────────────────────────────────────────

def create_2d_dataset(n, w, seed=None):
    """
    Generate n points from N(0, I_2) with labels sign(w^T x).
    Returns a DataFrame with columns: f0, f1, MemberKey, LoanStatus
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    X = rng.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=n)
    labels = np.sign(X @ w).astype(int)
    # Replace any 0 labels with +1 (edge case when w^T x == 0)
    labels[labels == 0] = 1

    df = pd.DataFrame(X, columns=['f0', 'f1'])
    df['MemberKey'] = [f's{i}' for i in range(n)]
    df['LoanStatus'] = labels
    return df


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _boundary_line_points(coef, intercept, xlim):
    """
    Given a linear classifier with coef=[w1,w2] and intercept b,
    the boundary is w1*x + w2*y + b = 0  =>  y = -(w1*x + b)/w2
    Returns (x_vals, y_vals) for plotting.
    """
    w1, w2 = coef
    x_vals = np.linspace(xlim[0], xlim[1], 300)
    if abs(w2) > 1e-12:
        y_vals = -(w1 * x_vals + intercept) / w2
    else:
        # vertical line at x = -b/w1
        x_vals = np.full(300, -intercept / w1)
        y_vals = np.linspace(-5, 5, 300)
    return x_vals, y_vals


def _ground_truth_line_points(w, xlim):
    """Ground truth boundary: w^T x = 0  =>  y = -w[0]/w[1] * x"""
    x_vals = np.linspace(xlim[0], xlim[1], 300)
    y_vals = -w[0] / w[1] * x_vals
    return x_vals, y_vals


# ── Visualization 1: Data with true labels ───────────────────────────────────

def plot_data_with_labels(train_df, test_df, w, save_path):
    """Scatter plot of all data, color-coded by true label, with ground truth line."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot training data
    pos_train = train_df[train_df['LoanStatus'] == 1]
    neg_train = train_df[train_df['LoanStatus'] == -1]
    ax.scatter(pos_train['f0'], pos_train['f1'], c='#2196F3', alpha=0.35,
               s=20, label='Train +1', edgecolors='none')
    ax.scatter(neg_train['f0'], neg_train['f1'], c='#FF5722', alpha=0.35,
               s=20, label='Train −1', edgecolors='none')

    # Plot test data (larger markers)
    pos_test = test_df[test_df['LoanStatus'] == 1]
    neg_test = test_df[test_df['LoanStatus'] == -1]
    ax.scatter(pos_test['f0'], pos_test['f1'], c='#2196F3', alpha=0.9,
               s=50, marker='D', label='Test +1', edgecolors='black', linewidths=0.5)
    ax.scatter(neg_test['f0'], neg_test['f1'], c='#FF5722', alpha=0.9,
               s=50, marker='D', label='Test −1', edgecolors='black', linewidths=0.5)

    # Ground truth boundary
    xlim = (train_df['f0'].min() - 0.5, train_df['f0'].max() + 0.5)
    gx, gy = _ground_truth_line_points(w, xlim)
    ax.plot(gx, gy, 'k--', linewidth=2, label=f'Ground truth h(x)=sign({w[0]:.0f}·x₁+{w[1]:.0f}·x₂)')

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Generated 2D Data with True Labels', fontsize=15)
    ax.legend(fontsize=9, loc='upper left')
    ylim = (train_df['f1'].min() - 0.5, train_df['f1'].max() + 0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot 1] Saved: {save_path}')


# ── Visualization 2: Decision boundary comparison ───────────────────────────

def plot_decision_boundaries(train_df, w, svm_model, hardt_model, save_path):
    """Show ground truth, SVM, and Hardt boundaries in one plot with regions."""
    fig, ax = plt.subplots(figsize=(8, 7))

    xlim = (train_df['f0'].min() - 0.5, train_df['f0'].max() + 0.5)
    ylim = (train_df['f1'].min() - 0.5, train_df['f1'].max() + 0.5)

    # Create background mesh for SVM region shading
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300),
                         np.linspace(ylim[0], ylim[1], 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Plot training data
    pos = train_df[train_df['LoanStatus'] == 1]
    neg = train_df[train_df['LoanStatus'] == -1]
    ax.scatter(pos['f0'], pos['f1'], c='#2196F3', alpha=0.3, s=15, edgecolors='none')
    ax.scatter(neg['f0'], neg['f1'], c='#FF5722', alpha=0.3, s=15, edgecolors='none')

    # Ground truth boundary
    gx, gy = _ground_truth_line_points(w, xlim)
    ax.plot(gx, gy, color='black', linewidth=2.5, linestyle='--',
            label=f'Ground Truth: $w$=[{w[0]:.0f},{w[1]:.0f}]')

    # SVM boundary
    svm_coef = svm_model.coef_[0]
    svm_inter = svm_model.intercept_[0]
    sx, sy = _boundary_line_points(svm_coef, svm_inter, xlim)
    ax.plot(sx, sy, color='#4CAF50', linewidth=2.5, linestyle='-',
            label=f'SVM: coef=[{svm_coef[0]:.2f},{svm_coef[1]:.2f}]')

    # Hardt boundary — Hardt uses a^T x >= threshold internally
    # The boundary is a^T x = min_si / cost_factor
    hardt_a = hardt_model.separable_cost.a
    hardt_cf = hardt_model.separable_cost.cost_factor
    hardt_threshold = hardt_model.min_si / hardt_cf
    # Boundary: a[0]*x + a[1]*y = threshold  =>  y = (threshold - a[0]*x) / a[1]
    hx = np.linspace(xlim[0], xlim[1], 300)
    hy = (hardt_threshold - hardt_a[0] * hx) / hardt_a[1]
    ax.plot(hx, hy, color='#9C27B0', linewidth=2.5, linestyle='-.',
            label=f'Hardt: $a$=[{hardt_a[0]:.1f},{hardt_a[1]:.1f}], thr={hardt_threshold:.2f}')

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Decision Boundary Comparison:\nGround Truth vs SVM vs Hardt', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot 2] Saved: {save_path}')


# ── Visualization 3: Accuracy comparison bar chart ───────────────────────────

def plot_accuracy_comparison(acc_dict, save_path):
    """Bar chart comparing original and strategic accuracies for SVM and Hardt."""
    labels = list(acc_dict.keys())
    values = [acc_dict[k] for k in labels]

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8, width=0.55)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Original vs Strategic Accuracy', fontsize=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot 3] Saved: {save_path}')


# ── Visualization 4: Point movement arrows ──────────────────────────────────

def plot_point_movements(test_df, modified_svm_df, modified_hardt_df,
                         w, svm_model, hardt_model, save_path):
    """
    Two subplots side-by-side showing arrows from original to manipulated positions.
    Only points that actually moved are shown.
    Both classifier boundaries are drawn on both subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    datasets = [
        ('SVM Manipulation', modified_svm_df, axes[0]),
        ('Hardt Manipulation', modified_hardt_df, axes[1]),
    ]

    for title, mod_df, ax in datasets:
        # Find points that moved
        orig_xy = test_df[FEATURE_LIST].values
        mod_xy = mod_df[FEATURE_LIST].values
        diffs = np.linalg.norm(orig_xy - mod_xy, axis=1)
        moved_mask = diffs > 1e-6

        moved_count = moved_mask.sum()

        # Draw arrows for moved points
        for i in np.where(moved_mask)[0]:
            ox, oy = orig_xy[i]
            mx, my = mod_xy[i]
            ax.annotate('', xy=(mx, my), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle='->', color='#E91E63',
                                        lw=0.8, mutation_scale=10))

        # Plot original positions of moved points
        if moved_count > 0:
            ax.scatter(orig_xy[moved_mask, 0], orig_xy[moved_mask, 1],
                       c='#FF5722', s=40, zorder=5, label='Original', edgecolors='black', linewidths=0.5)
            ax.scatter(mod_xy[moved_mask, 0], mod_xy[moved_mask, 1],
                       c='#4CAF50', s=40, zorder=5, marker='^', label='Manipulated', edgecolors='black', linewidths=0.5)

        # Determine axis limits from ALL points (original + manipulated)
        all_x = np.concatenate([orig_xy[:, 0], mod_xy[:, 0]])
        all_y = np.concatenate([orig_xy[:, 1], mod_xy[:, 1]])
        xlim = (all_x.min() - 0.5, all_x.max() + 0.5)
        ylim = (all_y.min() - 0.5, all_y.max() + 0.5)

        # Draw decision boundaries
        # Ground truth
        gx, gy = _ground_truth_line_points(w, xlim)
        ax.plot(gx, gy, 'k--', linewidth=1.5, label='Ground Truth')

        # SVM boundary
        svm_coef = svm_model.coef_[0]
        svm_inter = svm_model.intercept_[0]
        sx, sy = _boundary_line_points(svm_coef, svm_inter, xlim)
        ax.plot(sx, sy, color='#4CAF50', linewidth=2, linestyle='-', label='SVM')

        # Hardt boundary
        hardt_a = hardt_model.separable_cost.a
        hardt_cf = hardt_model.separable_cost.cost_factor
        hardt_threshold = hardt_model.min_si / hardt_cf
        hx = np.linspace(xlim[0], xlim[1], 300)
        hy = (hardt_threshold - hardt_a[0] * hx) / hardt_a[1]
        ax.plot(hx, hy, color='#9C27B0', linewidth=2, linestyle='-.', label='Hardt')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$x_1$', fontsize=13)
        ax.set_ylabel('$x_2$', fontsize=13)
        ax.set_title(f'{title}\n({moved_count} points moved)', fontsize=13)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Strategic Point Movements', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot 4] Saved: {save_path}')


# ── Visualization 5: Classification categories ─────────────────────────────

def _draw_boundaries_light(ax, svm_model, hardt_model, xlim):
    """Draw both classifier boundaries at low opacity."""
    # SVM boundary
    svm_coef = svm_model.coef_[0]
    svm_inter = svm_model.intercept_[0]
    sx, sy = _boundary_line_points(svm_coef, svm_inter, xlim)
    ax.plot(sx, sy, color='#4CAF50', linewidth=2, linestyle='-', alpha=0.3, label='SVM boundary')

    # Hardt boundary
    hardt_a = hardt_model.separable_cost.a
    hardt_cf = hardt_model.separable_cost.cost_factor
    hardt_threshold = hardt_model.min_si / hardt_cf
    hx = np.linspace(xlim[0], xlim[1], 300)
    hy = (hardt_threshold - hardt_a[0] * hx) / hardt_a[1]
    ax.plot(hx, hy, color='#9C27B0', linewidth=2, linestyle='-.', alpha=0.3, label='Hardt boundary')


def plot_classification_categories(test_df, pred_orig, pred_manip, clf_name,
                                   svm_model, hardt_model, save_path,
                                   orig_label='f(x)'):
    """
    Plot test points color-coded by 4 categories based on orig_label vs f(Δx).
    Both classifier boundaries are drawn at alpha=0.3.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    X = test_df[FEATURE_LIST].values

    # Build category masks  (using +1/-1, user notation: 1=positive, 0=negative)
    cat1 = (pred_orig == 1) & (pred_manip == -1)
    cat2 = (pred_orig == -1) & (pred_manip == -1)
    cat3 = (pred_orig == 1) & (pred_manip == 1)
    cat4 = (pred_orig == -1) & (pred_manip == 1)

    categories = [
        (cat1, '#E91E63', 's', f'{orig_label}=1, f(Δx)=0'),   # pink square
        (cat2, '#FF9800', 'o', f'{orig_label}=0, f(Δx)=0'),   # orange circle
        (cat3, '#2196F3', 'D', f'{orig_label}=1, f(Δx)=1'),   # blue diamond
        (cat4, '#4CAF50', '^', f'{orig_label}=0, f(Δx)=1'),   # green triangle
    ]

    for mask, color, marker, label in categories:
        count = mask.sum()
        if count > 0:
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=60, marker=marker,
                       edgecolors='black', linewidths=0.5, zorder=5,
                       label=f'{label}  ({count})', alpha=0.85)

    # Draw boundaries at low opacity
    xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    _draw_boundaries_light(ax, svm_model, hardt_model, xlim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'{clf_name}: {orig_label} vs f(Δx)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot - {clf_name}] Saved: {save_path}')


# ── Main experiment runner ───────────────────────────────────────────────────

def run_custom1_experiment():
    """Run the full custom1 experiment."""
    np.random.seed(SEED)
    print('=' * 60)
    print(' Custom1 Experiment: Transparent Model on 2D Synthetic Data')
    print('=' * 60)

    # ── Create output folder ─────────────────────────────────────────────
    exp_folder = safe_create_folder(result_folder_path, 'custom1_exp')

    # ── Generate data ────────────────────────────────────────────────────
    print(f'\n[1/6] Generating data: {TRAIN_SIZE} train + {TEST_SIZE} test points ...')
    train_df = create_2d_dataset(TRAIN_SIZE, W_TRUE, seed=SEED)
    test_df = create_2d_dataset(TEST_SIZE, W_TRUE, seed=SEED + 1)
    print(f'  Train label distribution: +1={sum(train_df["LoanStatus"]==1)}, '
          f'-1={sum(train_df["LoanStatus"]==-1)}')
    print(f'  Test  label distribution: +1={sum(test_df["LoanStatus"]==1)}, '
          f'-1={sum(test_df["LoanStatus"]==-1)}')

    # ── Train SVM ────────────────────────────────────────────────────────
    print('\n[2/6] Training SVM classifier ...')
    svm_model = LinearSVC(C=1000, random_state=SEED, max_iter=100000)
    svm_model.fit(train_df[FEATURE_LIST], train_df['LoanStatus'])
    print(f'  SVM coef: {svm_model.coef_[0]}, intercept: {svm_model.intercept_[0]:.4f}')

    # ── Train Hardt ──────────────────────────────────────────────────────
    print('\n[3/6] Training Hardt classifier ...')
    hardt_cost = WeightedLinearCostFunction(A_COST, cost_factor=COST_FACTOR)
    hardt_model = HardtAlgo(hardt_cost)
    hardt_model.fit(train_df[FEATURE_LIST], train_df['LoanStatus'])
    print(f'  Hardt cost vector a: {A_COST}, min_si: {hardt_model.min_si:.4f}')

    # ── Original accuracies (no manipulation) ────────────────────────────
    svm_pred_orig = svm_model.predict(test_df[FEATURE_LIST])
    hardt_pred_orig = hardt_model.predict(pd.DataFrame(test_df[FEATURE_LIST]))
    true_labels = test_df['LoanStatus'].values

    svm_acc_orig = np.mean(svm_pred_orig == true_labels)
    hardt_acc_orig = np.mean(hardt_pred_orig == true_labels)
    print(f'\n  SVM   original accuracy: {svm_acc_orig:.4f}')
    print(f'  Hardt original accuracy: {hardt_acc_orig:.4f}')

    # ── Strategic manipulation (full info / transparent) ──────────────────
    print('\n[4/6] Computing strategic manipulation against SVM ...')
    cost_func_svm = WeightedLinearCostFunction(A_COST, cost_factor=COST_FACTOR)
    modified_svm_df = strategic_modify_using_known_clf(test_df, svm_model, FEATURE_LIST, cost_func_svm)

    print('\n[5/6] Computing strategic manipulation against Hardt ...')
    cost_func_hardt = WeightedLinearCostFunction(A_COST, cost_factor=COST_FACTOR)
    modified_hardt_df = strategic_modify_using_known_clf(test_df, hardt_model, FEATURE_LIST, cost_func_hardt)

    # ── Strategic accuracies ─────────────────────────────────────────────
    # Strategic accuracy = Pr[h(x) == f(Delta_f(x))]
    # h is the ground truth; after manipulation the true label stays the same
    # but the classifier may or may not classify the manipulated point correctly

    # For SVM: test points were manipulated against SVM.
    # We evaluate the *ground truth* label vs *SVM prediction on manipulated input*
    svm_pred_strat = svm_model.predict(modified_svm_df[FEATURE_LIST])
    svm_acc_strat = np.mean(svm_pred_strat == true_labels)

    # For Hardt: test points were manipulated against Hardt.
    hardt_pred_strat = hardt_model.predict(pd.DataFrame(modified_hardt_df[FEATURE_LIST]))
    hardt_acc_strat = np.mean(hardt_pred_strat == true_labels)

    print(f'\n  SVM   strategic accuracy: {svm_acc_strat:.4f}')
    print(f'  Hardt strategic accuracy: {hardt_acc_strat:.4f}')

    # ── Generate all 4 visualizations ────────────────────────────────────
    print('\n[6/6] Generating visualizations ...')

    # Plot 1: Data with true labels
    plot_data_with_labels(train_df, test_df, W_TRUE,
                          os.path.join(exp_folder, '1_data_with_labels.png'))

    # Plot 2: Decision boundary comparison
    plot_decision_boundaries(train_df, W_TRUE, svm_model, hardt_model,
                             os.path.join(exp_folder, '2_decision_boundaries.png'))

    # Plot 3: Accuracy comparison
    acc_dict = {
        'SVM\nOriginal': svm_acc_orig,
        'SVM\nStrategic': svm_acc_strat,
        'Hardt\nOriginal': hardt_acc_orig,
        'Hardt\nStrategic': hardt_acc_strat,
    }
    plot_accuracy_comparison(acc_dict, os.path.join(exp_folder, '3_accuracy_comparison.png'))

    # Plot 4: Point movements
    plot_point_movements(test_df, modified_svm_df, modified_hardt_df,
                         W_TRUE, svm_model, hardt_model,
                         os.path.join(exp_folder, '4_point_movements.png'))

    # Plot 5: Classification categories (SVM)
    plot_classification_categories(
        test_df, svm_pred_orig, svm_pred_strat, 'SVM',
        svm_model, hardt_model,
        os.path.join(exp_folder, '5_categories_svm.png'))

    # Plot 6: Classification categories (Hardt)
    plot_classification_categories(
        test_df, hardt_pred_orig, hardt_pred_strat, 'Hardt',
        svm_model, hardt_model,
        os.path.join(exp_folder, '6_categories_hardt.png'))

    # Plot 7: Ground truth h(x) vs SVM f(Δx) categories
    plot_classification_categories(
        test_df, true_labels, svm_pred_strat, 'SVM',
        svm_model, hardt_model,
        os.path.join(exp_folder, '7_categories_hx_svm.png'),
        orig_label='h(x)')

    # Plot 8: Ground truth h(x) vs Hardt f(Δx) categories
    plot_classification_categories(
        test_df, true_labels, hardt_pred_strat, 'Hardt',
        svm_model, hardt_model,
        os.path.join(exp_folder, '8_categories_hx_hardt.png'),
        orig_label='h(x)')

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' RESULTS SUMMARY')
    print('=' * 60)
    print(f'  Ground truth w = {W_TRUE}')
    print(f'  Cost vector  a = {A_COST}')
    print(f'  SVM   original accuracy:  {svm_acc_orig:.4f}')
    print(f'  SVM   strategic accuracy: {svm_acc_strat:.4f}')
    print(f'  Hardt original accuracy:  {hardt_acc_orig:.4f}')
    print(f'  Hardt strategic accuracy: {hardt_acc_strat:.4f}')
    print(f'\n  All plots saved to: {exp_folder}')
    print('=' * 60)


if __name__ == '__main__':
    run_custom1_experiment()
