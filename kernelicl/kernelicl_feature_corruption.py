import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    precision_recall_curve, auc,
)

CORRUPTION_RESULTS_PATH = 'corruption_results.csv'
METRIC_KEYS = ['Balanced Accuracy', 'Precision (PPV)', 'Sensitivity',
               'Specificity', 'F1', 'MCC', 'AUROC', 'AUPRC']


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    'Logistic Regression': {'model': l2_model,       'threshold': l2_threshold,
                            'raw': False},
    'XGBoost':             {'model': xgb_model,      'threshold': xgb_threshold,
                            'raw': False},
    'TabICLv2':            {'model': tabiclv2_model, 'threshold': tabiclv2_threshold,
                            'raw': True},     # ← fitted on X_train_raw
}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    """All evaluation metrics, or None if the labels are degenerate."""
    if len(np.unique(y_true)) < 2:
        return None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec, rec, _   = precision_recall_curve(y_true, y_prob)

    return {
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision (PPV)'  : precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity'      : recall_score(y_true, y_pred, zero_division=0),
        'Specificity'      : tn / (tn + fp) if (tn + fp) else 0.0,
        'F1'               : f1_score(y_true, y_pred, zero_division=0),
        'MCC'              : matthews_corrcoef(y_true, y_pred),
        'AUROC'            : roc_auc_score(y_true, y_prob),
        'AUPRC'            : auc(rec, prec),
    }


def evaluate_models(X_raw, y_true, preprocessor, model_registry):
    """
    Evaluate every model on one version of the test set.

    Models flagged 'raw' receive X_raw directly; the rest receive it
    transformed by the fitted preprocessor. The transform runs once and
    is shared, so it isn't repeated per model.
    """
    needs_transform = any(not m['raw'] for m in model_registry.values())
    X_proc = preprocessor.transform(X_raw) if needs_transform else None

    out = {}
    for name, spec in model_registry.items():
        X      = X_raw if spec['raw'] else X_proc
        y_prob = spec['model'].predict_proba(X)[:, 1]
        y_pred = (y_prob >= spec['threshold']).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        if metrics is not None:
            out[name] = metrics
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CORRUPTION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

AGE_CUT  = 50
SEASONS  = [('Spring (Mar–May)', [3, 4, 5]),
            ('Summer (Jun–Aug)', [6, 7, 8]),
            ('Autumn (Sep–Nov)', [9, 10, 11]),
            ('Winter (Dec–Feb)', [12, 1, 2])]
DAYS     = [('Weekdays (Mon–Fri)', [0, 1, 2, 3, 4]),
            ('Weekends (Sat–Sun)', [5, 6])]
YEARS    = [2023, 2024, 2025]
EYE      = ['eye_condition_normal', 'eye_condition_conjunctivitis',
            'eye_condition_cataract']
ACUITY   = ['acuity_status_no_unmet_need', 'acuity_status_poor_distance',
            'acuity_status_poor_distance_and_near']
SITES    = ['Govt. City Hospital', 'DHQ Chakwal', 'RHC Lawa', 'RHC Buchaal']
DIST_COL = 'Travel Distance (km)'


def build_corruptions(X, min_n=100, verbose=True):
    """
    Define all feature corruption experiments.

    Values are derived from X at runtime so the design transfers to another
    cohort. Categorical targets are validated against the observed data and
    skipped with a reason if absent or too rare.

    Each corruption: {name, experiment, feature, value, enabled, flag}
    where `feature` may be a list of columns for grouped corruptions.
    """
    out, skipped = [], []

    def add(name, exp, feature, value, n=None, warn_below=None):
        flag = f'⚠ n={n:,}' if (n and warn_below and n < warn_below) else None
        out.append({'name': name, 'experiment': exp, 'feature': feature,
                    'value': value, 'enabled': True, 'flag': flag})

    def category_n(col, val):
        """n for a category, or None with a logged reason if unusable."""
        if col not in X.columns:
            skipped.append(f'{col} — column not found')
            return None
        n = int((X[col] == val).sum())
        if n < min_n:
            skipped.append(f'{col} → {val} — n={n} < {min_n}')
            return None
        return n

    def window_mode(col, values, label):
        """Modal observed value within a window, in the column's own dtype."""
        mask = X[col].isin(values)
        if mask.sum() < min_n:
            skipped.append(f'{col} {label} — n={int(mask.sum())} < {min_n}')
            return None, None
        return X.loc[mask, col].mode().iloc[0], int(mask.sum())

    # ── 1. Demographic ────────────────────────────────────────────────────────
    for val in sorted(X['Gender'].dropna().unique()):
        n = category_n('Gender', val)
        if n:
            add(f'Gender → {val}', 'Demographic', 'Gender', val, n, 500)

    for label, mask in [(f'≥{AGE_CUT}', X['Age'] >= AGE_CUT),
                        (f'<{AGE_CUT}',  X['Age'] <  AGE_CUT)]:
        sub = X['Age'][mask].dropna()
        if len(sub) < min_n:
            skipped.append(f'Age {label} — n={len(sub)} < {min_n}')
            continue
        add(f'Age → {label} (median = {sub.median():.0f})',
            'Demographic', 'Age', float(sub.median()), len(sub), 500)

    # ── 2. Temporal ───────────────────────────────────────────────────────────
    observed_years = set(X['Planned_year'].dropna().astype(int))
    for yr in [y for y in YEARS if y in observed_years]:
        n = int((X['Planned_year'] == yr).sum())
        if n < min_n:
            skipped.append(f'Planned_year → {yr} — n={n} < {min_n}')
            continue
        add(f'Year → {yr}', 'Temporal', 'Planned_year', float(yr), n, 300)

    for label, months in SEASONS:
        mode, n = window_mode('Planned_month', months, label)
        if mode is not None:
            add(f'Month → {label} (= {mode})',
                'Temporal', 'Planned_month', mode, n, 500)

    for label, days in DAYS:
        mode, n = window_mode('Planned_dayofweek', days, label)
        if mode is not None:
            add(f'Day → {label} (= {mode})',
                'Temporal', 'Planned_dayofweek', mode, n, 500)

    # ── 3. Clinical ───────────────────────────────────────────────────────────
    for col, values in [('Eye Conditions', EYE), ('Acuity Status', ACUITY)]:
        for val in values:
            n = category_n(col, val)
            if n:
                add(f'{col} → {val}', 'Clinical', col, val, n, 300)

    # Grouped: corrupting one indicator alone lets the model recover testing
    # status from the others, understating the effect.
    cutoff = [c for c in X.columns if 'Cutoff' in c and 'Missing' in c]
    if cutoff:
        for val, label in [(1, 'not performed'), (0, 'performed')]:
            add(f'Cutoff testing → {label}', 'Clinical', cutoff, val)
    else:
        skipped.append('Cutoff missingness indicators — none found')

    # ── 4. Geographical ───────────────────────────────────────────────────────
    if DIST_COL in X.columns:
        d = X[DIST_COL].dropna()
        for label, q in [('short (P10)', 0.10), ('typical (P50)', 0.50),
                         ('long (P90)', 0.90)]:
            v = float(d.quantile(q))
            add(f'Distance → {label} = {v:.1f} km',
                'Geographical', DIST_COL, v)
    else:
        skipped.append(f'{DIST_COL} — column not found (v4 only)')

    for site in SITES:
        n = category_n('Location_y', site)
        if n:
            add(f'Location → {site}', 'Geographical', 'Location_y', site, n, 200)

    # ── Report ────────────────────────────────────────────────────────────────
    if verbose:
        counts = pd.Series([c['experiment'] for c in out]).value_counts()
        print('── Corruptions defined ──')
        for exp, n in counts.items():
            print(f'  {exp:<15}: {n}')
        print(f"  {'Total':<15}: {len(out)}")
        if skipped:
            print(f'\n  Skipped ({len(skipped)}):')
            for s in skipped:
                print(f'    {s}')

    return out


def apply_corruption(X, feature, value):
    """
    Return a copy of X with `feature` set to `value`, preserving each
    column's original dtype. Assigning a float into an object column would
    otherwise silently recast it and break the categorical pipeline.
    """
    out  = X.copy()
    cols = feature if isinstance(feature, (list, tuple)) else [feature]

    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise KeyError(f'columns not found: {missing}')

    for c in cols:
        v     = value[c] if isinstance(value, dict) else value
        dtype = out[c].dtype
        out[c] = v
        if out[c].dtype != dtype:
            out[c] = out[c].astype(dtype)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_corruption_experiments(X_test_raw, y_test, preprocessor,
                               model_registry, corruptions,
                               results_path=CORRUPTION_RESULTS_PATH):
    """
    Evaluate every model on every corrupted version of the test set.

    The fitted models are reused throughout — nothing retrains — and the test
    set keeps its size and labels, so gaps from baseline are comparable across
    all corruptions.
    """
    y_test  = np.asarray(y_test)
    enabled = [c for c in corruptions if c['enabled']]

    print(f"\n{'='*64}")
    print(f'  Feature Corruption Experiments')
    print(f'  Corruptions : {len(enabled)}')
    print(f'  Models      : {len(model_registry)} '
          f"({sum(m['raw'] for m in model_registry.values())} on raw input)")
    print(f'  Evaluations : {len(enabled) * len(model_registry)}')
    print(f'  n_test      : {len(y_test):,} (constant throughout)')
    print(f"{'='*64}")

    def row(corruption, model_name, metrics, baseline):
        feat = corruption['feature']
        r = {
            'Corruption': corruption['name'],
            'Experiment': corruption['experiment'],
            'Feature'   : ', '.join(feat) if isinstance(feat, (list, tuple)) else feat,
            'Value'     : str(corruption['value']),
            'Model'     : model_name,
            'Flag'      : corruption['flag'] or '',
            'n_test'    : len(y_test),
        }
        for k in METRIC_KEYS:
            r[k]           = round(metrics[k], 4)
            r[f'Gap_{k}'] = round(metrics[k] - baseline[k], 4) if baseline else 0.0
        return r

    # ── Baseline ──────────────────────────────────────────────────────────────
    print('\n  Baseline (uncorrupted)')
    baseline = evaluate_models(X_test_raw, y_test, preprocessor, model_registry)

    results = []
    for name, metrics in baseline.items():
        results.append(row(
            {'name': 'Baseline (uncorrupted)', 'experiment': 'Baseline',
             'feature': None, 'value': None, 'flag': None},
            name, metrics, None))
        print(f'    {name:<22} MCC={metrics["MCC"]:.4f}  '
              f'AUROC={metrics["AUROC"]:.4f}  AUPRC={metrics["AUPRC"]:.4f}')

    # ── Corruptions ───────────────────────────────────────────────────────────
    for i, corruption in enumerate(enabled, 1):
        print(f'\n  [{i}/{len(enabled)}] {corruption["name"]}'
              + (f'   {corruption["flag"]}' if corruption['flag'] else ''))

        try:
            X_corrupt = apply_corruption(X_test_raw, corruption['feature'],
                                         corruption['value'])
            evaluated = evaluate_models(X_corrupt, y_test, preprocessor,
                                        model_registry)
        except Exception as e:
            print(f'    Failed: {str(e)[:100]}')
            continue

        for name, metrics in evaluated.items():
            if name not in baseline:
                continue
            results.append(row(corruption, name, metrics, baseline[name]))
            gap_mcc = metrics['MCC']   - baseline[name]['MCC']
            gap_auroc = metrics['AUROC'] - baseline[name]['AUROC']
            gap_auprc  = metrics['AUPRC'] - baseline[name]['AUPRC']
            print(f'    {name:<22} MCC={metrics["MCC"]:.4f} ({gap_mcc:+.4f})  '
                  f'AUROC={metrics["AUROC"]:.4f} ({gap_auroc:+.4f})  AUPRC={metrics["AUPRC"]:.4f} ({gap_auprc:+.4f})')

        pd.DataFrame(results).to_csv(results_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*64}")
    print(f'  Complete — {len(results_df):,} rows → {results_path}')
    print(f"{'='*64}")
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# USAGE
# ══════════════════════════════════════════════════════════════════════════════

corruptions = build_corruptions(X_test_raw, min_n=100)

results_df = run_corruption_experiments(
    X_test_raw     = X_test_raw,
    y_test         = y_test,
    preprocessor   = preprocessor,
    model_registry = MODELS,
    corruptions    = corruptions,
)

# One corruption per experiment group — the most damaging in each
model_name = 'TabICLv2'
gap = 'Gap_MCC'
worst = (results_df[(results_df['Model'] == model_name) &
                    (results_df['Experiment'] != 'Baseline')]
         .sort_values(gap)
         .groupby('Experiment', as_index=False).first())

chosen = [c for c in corruptions
          if c['name'] in set(worst['Corruption'])][:4]
print('Selected:', [c['name'] for c in chosen])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — WHAT EACH CORRUPTION DOES TO THE EVIDENCE
# ══════════════════════════════════════════════════════════════════════════════
# The tables above say how much accuracy each corruption costs. They cannot say why.
# This figure answers the why for the four worst corruptions, one per row:
#
#   column 1  where the cohort lands in the learned embedding, and what it gets
#             called there — test cases coloured by the model's *prediction*
#   column 2  which training cases one particular patient's prediction rests on
#   column 3  the same evidence, laid over the embedding
#
# Everything is held constant except the corrupted column. The training embedding is
# fixed by construction (column statistics attend to training rows, and the kernel
# keys are the training context), so one projection is fitted once and reused for all
# twelve panels — a point that moves between rows moved because of the corruption, not
# because the frame was redrawn. The same patient is followed down every row for the
# same reason; set CASE to pin a different one.

import textwrap

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

if 'fit_explainer' not in globals():
    import sys
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from kernelicl_clinical import fit_explainer

SEED = 0
CASE = None            # test row to follow down the rows; None picks one at random
FINETUNED = None       # path to a kernelicl_finetune checkpoint, or None
CLASS_LABELS = globals().get('CLASS_LABELS', {})   # e.g. {0: 'Adherence', 1: 'Non-adherence'}

SAVE_FIGURES = None    # e.g. 'figures'
FIG_DPI = 300
FIG_FORMATS = ('png', 'pdf')

C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8984'
SURFACE, GRID = '#fcfcfb', '#e8e7e3'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK_3, 'axes.labelcolor': INK_2, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'font.size': 10,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'figure.dpi': 130,
})


def finish(fig, name):
    """Show a figure, and write a publication copy when SAVE_FIGURES is set."""
    if SAVE_FIGURES:
        from pathlib import Path
        directory = Path(SAVE_FIGURES)
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in FIG_FORMATS:
            fig.savefig(directory / f'{name}.{suffix}', dpi=FIG_DPI,
                        bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    return ax


def limits(v, margin=0.06):
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * margin or 1.0
    return lo - pad, hi + pad


# ── The explainer, and the fixed reference frame ─────────────────────────────────
if 'ex' not in globals():
    ex = fit_explainer(X_train_raw, y_train, X_test_raw, finetuned=FINETUNED)

head, clf = ex.head, ex.clf
y_train_enc = clf.y_encoder_.transform(ex.y_train)
n_classes = len(clf.classes_)

# Only the first three palette slots clear the all-pairs colourblind gate, so any
# further classes fold into one neutral rather than inventing a fourth hue.
keep = list(np.argsort(-np.bincount(y_train_enc, minlength=n_classes))[:3])
class_colors = [C_BLUE, C_ORANGE, C_AQUA]


def class_color(c):
    return class_colors[keep.index(c)] if c in keep else INK_3


def class_name(c):
    raw = clf.classes_[c]
    return f'{CLASS_LABELS.get(raw, raw)} ({raw})' if raw in CLASS_LABELS else f'class {raw}'


with torch.no_grad():
    H_train = head.embed(ex.E_train)[0].cpu().numpy().astype(np.float64)

# Fitted on training cases alone, with every test set transformed in afterwards. A
# training row sits in its own context and finds a perfect self-match among the keys;
# that offset is systematic and would dominate a jointly fitted layout.
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=SEED).fit(H_train)
    PROJ_NAME = 'UMAP'
except ImportError:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, svd_solver='full').fit(H_train)
    PROJ_NAME = 'PCA'
    print('umap-learn not installed; using PCA, which shows only linear structure')

with np.errstate(all='ignore'):
    proj = np.asarray(reducer.transform(H_train))
order = np.argsort(proj[:, 0])
XLIM, YLIM = limits(proj[:, 0]), limits(proj[:, 1])

if CASE is None:
    CASE = int(np.random.RandomState(SEED).randint(ex.w.shape[0]))
print(f'\nFigure 9 — following test row {CASE} across {len(chosen)} corruptions')


# ── Re-score under each corruption, reusing the fitted model and its scale ───────
rows = []
for corruption in chosen[:4]:
    X_corrupt = apply_corruption(X_test_raw, corruption['feature'], corruption['value'])
    ex_c = ex.for_test_set(X_corrupt)
    with torch.no_grad():
        H_test = head.embed(ex_c.E_test)[0].cpu().numpy().astype(np.float64)
    with np.errstate(all='ignore'):
        proj_test = np.asarray(reducer.transform(H_test))
    rows.append({'name': corruption['name'], 'ex': ex_c, 'proj_test': proj_test,
                 'changed': float(np.mean(ex_c.pred != ex.pred))})
    print(f"  {corruption['name']:<44} {rows[-1]['changed']:6.1%} of predictions changed, "
          f'evidence base {ex_c.evidence_cases[CASE]:.0f} of {ex.n:,} cases')

# One weight scale across all rows: normalising per row would draw a weight of 0.006 as
# large as one of 0.89, and the point of the figure is that the rows differ.
peak = max(float(r['ex'].w[CASE].max()) for r in rows)


# ── The figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.6 * len(rows)),
                         gridspec_kw={'hspace': 0.42, 'wspace': 0.18})
axes = np.atleast_2d(axes)

for r, row in enumerate(rows):
    ex_c, proj_test = row['ex'], row['proj_test']
    pred_enc = clf.y_encoder_.transform(ex_c.pred)
    wt = ex_c.w[CASE]

    # (1) the corrupted cohort in the embedding, coloured by what it was called.
    ax = axes[r, 0]
    ax.scatter(proj[:, 0], proj[:, 1], s=4, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        at = pred_enc == c
        if at.any():
            ax.scatter(proj_test[at, 0], proj_test[at, 1], s=7, color=class_color(c),
                       alpha=0.55, linewidths=0, zorder=2)
    bare(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_title(f'Test cohort, coloured by prediction\n'
                 f"{row['changed']:.1%} of predictions changed", fontsize=10,
                 color=INK_2, loc='left', pad=8)

    # (2) which training cases this one patient's prediction rests on.
    ax = axes[r, 1]
    ax.vlines(np.arange(len(order)), 0, wt[order], color=C_BLUE, linewidth=0.7)
    ax.set_ylim(0, peak * 1.08)
    ax.set_xlim(0, len(order))
    ax.set_ylabel('contribution')
    ax.set_xlabel('training cases, ordered by embedding')
    ax.set_title(f'Evidence for test row {CASE}\n'
                 f'{ex_c.evidence_cases[CASE]:.0f} of {ex.n:,} cases carry it',
                 fontsize=10, color=INK_2, loc='left', pad=8)
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # (3) the same evidence, in the embedding, over recorded training outcomes.
    ax = axes[r, 2]
    heavy = wt > wt.max() * 0.02
    ax.scatter(proj[~heavy, 0], proj[~heavy, 1], s=4, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        at = heavy & (y_train_enc == c)
        if at.any():
            ax.scatter(proj[at, 0], proj[at, 1], s=8 + 260 * wt[at] / peak,
                       color=class_color(c), alpha=0.7, linewidths=0.5,
                       edgecolors=SURFACE, zorder=2)
    # An out-of-frame case is worth seeing, but letting one point rescale the axes
    # would squash the cloud, so clamp it to the frame and say so.
    tx, ty = proj_test[CASE]
    inside = XLIM[0] <= tx <= XLIM[1] and YLIM[0] <= ty <= YLIM[1]
    ax.scatter(np.clip(tx, *XLIM), np.clip(ty, *YLIM), marker='X', s=150,
               color=INK if inside else SURFACE, edgecolors=INK, linewidths=1.5, zorder=3)
    if not inside:
        ax.text(0.5, 0.02, 'case sits outside the training cloud', transform=ax.transAxes,
                ha='center', fontsize=8, color=INK_2, style='italic')
    bare(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    predicted = CLASS_LABELS.get(ex_c.pred[CASE], ex_c.pred[CASE])
    ax.set_title(f'Evidence in the embedding\npredicted {predicted}, '
                 f'agreement {100 * ex_c.probs[CASE].max():.0f}%',
                 fontsize=10, color=INK_2, loc='left', pad=8)

    # Corruption names run long; wrapped, they stay clear of the panel letters.
    axes[r, 0].text(-0.13, 0.5, textwrap.fill(row['name'], 26),
                    transform=axes[r, 0].transAxes, rotation=90, va='center',
                    ha='center', fontsize=10, color=INK)

for ax, letter in zip(axes.ravel(), 'abcdefghijkl'):
    ax.text(-0.04, 1.14, f'({letter})', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='right', color=INK)

handles = [Line2D([], [], marker='o', linestyle='', markersize=8, color=class_color(c),
                  label=class_name(c)) for c in range(n_classes)]
handles += [Line2D([], [], marker='o', linestyle='', markersize=5, color=GRID,
                   label='training case, negligible contribution'),
            Line2D([], [], marker='X', linestyle='', markersize=9, color=INK,
                   label=f'test row {CASE}')]
fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, -0.015))
fig.suptitle(f'How each corruption moves the evidence ({PROJ_NAME} of the learned '
             f'embedding)\nColumn 1 colours test cases by prediction; column 3 colours '
             f'training cases by recorded outcome.',
             color=INK, fontsize=11, x=0.01, y=0.995, ha='left', va='top')
finish(fig, 'figure9_corruption_evidence')

# Notes
# -----
# CASE is drawn at random so the figure is not cherry-picked, but any single patient
# can be uninformative. Two worth trying instead:
#
#   CASE = int(np.argmax(rows[0]['ex'].pred != ex.pred))    a case the corruption flipped
#   CASE = int(np.argmin(ex.evidence_cases))                the thinnest evidence base
#
# Read the rows against each other, not on their own. A corruption that costs accuracy
# while column 1 stays put and column 2 keeps its shape has broken the labels without
# the model noticing — errors are arriving unannounced. A corruption that visibly moves
# the cohort and rebuilds the evidence is one the model registered, and those errors
# are the ones triage() will flag.
