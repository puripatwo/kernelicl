"""Tables and figures: T1-T4 and F1/F3/F4/F7.

Paste kernelicl_clinical.py first -- fit_explainer and its helpers live there.
Also needs X_train, y_train, X_test, y_test in the session.

See README.md for what each table and figure answers and how to read it.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import relative_perplexity

# Works whether kernelicl_clinical.py was pasted into the session or is importable.
if "fit_explainer" not in globals():
    import os
    import sys

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from kernelicl_clinical import (GAMMA_GRID, K_GRID, _embed, _make_clf, _make_folds,
                                    _take, fit_explainer)

SEED = 0
KERNELS = ("gaussian", "dot", "knn")
COMPACTNESS_K = 5
FINETUNED = None   # path to a kernelicl_finetune checkpoint, or None

# "accuracy" reproduces the paper, whose benchmark datasets are roughly balanced.
# For screening it flatters the majority class. This feeds the calibration too, so it
# changes which scale is selected, not only what is reported.
METRIC = "balanced_accuracy"   # "accuracy" | "balanced_accuracy"

FEATURE_NAMES = list(X_train.columns) if hasattr(X_train, "columns") else None
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()


def evaluate(y_true, y_pred) -> float:
    if METRIC == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if METRIC == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"unknown METRIC {METRIC!r}")


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
# Categorical slots 1-3 of a validated palette, unmodified: these are scatter and
# line charts, where every pair of hues must be distinguishable, and only the first
# three slots clear that gate. A fourth would put yellow beside orange and fail.
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"
SERIES = {"gaussian": C_BLUE, "dot": C_ORANGE, "knn": C_AQUA}
LABEL = {"gaussian": "KernelICL-Gaussian", "dot": "KernelICL-Dot", "knn": "KernelICL-kNN"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130,
})


def tidy(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    return ax


def bare(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, color=INK_2, fontsize=9, loc="left")
    return ax


def limits(v, margin=0.06):
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * margin or 1.0
    return lo - pad, hi + pad


# --------------------------------------------------------------------------- #
# Setup: embed once, reuse for every table and figure
# --------------------------------------------------------------------------- #
ex = fit_explainer(X_train, y_train, X_test, feature_names=FEATURE_NAMES,
                   finetuned=FINETUNED)
clf, head, n_classes = ex.clf, ex.head, ex.clf.n_classes_
DEVICE = ex.head.proj.weight.device.type


def score(kernel, scale):
    """(metric, mean relative perplexity, weights) for one kernel and scale."""
    head.kernel = kernel
    y_ctx = torch.from_numpy(
        clf.y_encoder_.transform(y_train)).float().to(ex.E_train.device)[None]
    with torch.no_grad():
        probs, w = head(ex.E_train, ex.E_test, y_ctx, num_classes=n_classes, gamma=scale)
    pred = clf.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
    return evaluate(y_test, pred), relative_perplexity(w).mean().item(), w


# Scales for all three kernels, chosen on one held-out split. The clinical module
# does 5-fold for the scale you deploy with; a single split is enough to mark a
# sensible operating point on these charts, and costs one extra embedding pass.
def calibrate_all_kernels(val_size=0.2, accuracy_tolerance=0.01):
    folds = _make_folds(y_train, 1, val_size, SEED)
    tr_idx, va_idx = folds[0]
    cal = _make_clf(DEVICE, "none", SEED)
    cal.fit(_take(X_train, tr_idx), y_train[tr_idx])
    E_tr, E_va = _embed(cal, _take(X_train, va_idx))
    y_ctx = torch.from_numpy(
        cal.y_encoder_.transform(y_train[tr_idx])).float().to(E_tr.device)[None]
    y_val = y_train[va_idx]

    chosen = {}
    for kernel in KERNELS:
        head.kernel = kernel
        rows = []
        for scale in (K_GRID if kernel == "knn" else GAMMA_GRID):
            with torch.no_grad():
                probs, w = head(E_tr, E_va, y_ctx, num_classes=cal.n_classes_, gamma=scale)
            pred = cal.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
            rows.append((scale, evaluate(y_val, pred), float(relative_perplexity(w).mean())))
        # Accuracy is near-flat across much of the grid while the evidence base varies
        # by orders of magnitude, so take the sparsest scale within tolerance.
        best = max(r[1] for r in rows)
        chosen[kernel] = min([r for r in rows if r[1] >= best - accuracy_tolerance],
                             key=lambda r: r[2])[0]
    return chosen


BEST = calibrate_all_kernels()
print("calibrated scales:", BEST)


# --------------------------------------------------------------------------- #
# T1 - kernel x scale
# --------------------------------------------------------------------------- #
# The scale column runs in opposite directions for the two kernel families: for
# gaussian/dot it is gamma, where larger is sharper and so gives FEWER effective
# neighbours; for knn it is k itself, where larger gives more. Read rel.PPL%.
print(f"\n{'kernel':>10} {'scale':>8} {METRIC[:10]:>12} {'rel.PPL%':>10} {'kernel ms':>11}")
T1 = []
for kernel in KERNELS:
    for scale in (K_GRID if kernel == "knn" else GAMMA_GRID):
        started = time.perf_counter()
        metric, perplexity, _ = score(kernel, scale)
        elapsed_ms = (time.perf_counter() - started) * 1000
        T1.append(dict(kernel=kernel, scale=scale, metric=metric, ppl=perplexity))
        star = " *" if scale == BEST[kernel] else ""
        print(f"{kernel:>10} {scale:>8} {metric:>12.4f} {100 * perplexity:>10.2f} "
              f"{elapsed_ms:>11.1f}{star}")
print("\n* = scale selected on held-out data")


# --------------------------------------------------------------------------- #
# T2 - method comparison
# --------------------------------------------------------------------------- #
# TabICL-MLP is absent on purpose: in the paper it is the same architecture
# fine-tuned with an MLP head, so without fine-tuning it is bit-identical to
# TabICL (single). Reporting it would be inventing a number.
clf_ensemble = TabICLClassifier(n_estimators=8, device=DEVICE, random_state=SEED)
clf_ensemble.fit(X_train, y_train)

started = time.perf_counter()
metric_ensemble = evaluate(y_test, clf_ensemble.predict(X_test))
time_ensemble = time.perf_counter() - started

started = time.perf_counter()
metric_single = evaluate(y_test, clf.predict(X_test))
time_single = time.perf_counter() - started

T2 = [("TabICL (ensemble, n=8)", metric_ensemble, None, time_ensemble),
      ("TabICL (single)", metric_single, None, time_single),
      ("TabICL-MLP", None, None, None)]
for kernel in KERNELS:
    metric, perplexity, _ = score(kernel, BEST[kernel])
    T2.append((LABEL[kernel], metric, perplexity, None))

print(f"\n{'method':>24} {METRIC[:10]:>12} {'rel.PPL%':>10} {'time (s)':>10}")
for name, metric, perplexity, seconds in T2:
    print(f"{name:>24} {f'{metric:.4f}' if metric is not None else '-':>12} "
          f"{f'{100 * perplexity:.2f}' if perplexity is not None else '-':>10} "
          f"{f'{seconds:.1f}' if seconds is not None else '-':>10}")
print("\nKernelICL times omitted: all three share one embedding pass here.")
print("TabICL-MLP is identical to TabICL (single) until the model is fine-tuned.")


# --------------------------------------------------------------------------- #
# T3 / F4 - what the learned metric treats as similar
# --------------------------------------------------------------------------- #
emphasis = ex.with_kernel("knn", gamma=COMPACTNESS_K).feature_emphasis(k=COMPACTNESS_K)

TOP = min(12, len(emphasis) // 2)
show = list(range(TOP)) + list(range(len(emphasis) - TOP, len(emphasis)))
print(f"\nneighbourhood compactness, k={COMPACTNESS_K}")
print("both columns are normalized within their own method, so they show relative")
print("emphasis rather than absolute closeness\n")
print(f"{'feature':>26} {'plain kNN':>10} {'KernelICL':>10} {'rel.diff':>9}")
for i in show:
    row = emphasis.iloc[i]
    print(f"{row['feature'][:26]:>26} {row['plain_knn']:>10.2f} {row['kernelicl']:>10.2f} "
          f"{100 * row['rel_diff']:>8.0f}%")

fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(show) + 1.4))
values = emphasis.iloc[show]["rel_diff"].to_numpy() * 100
positions = np.arange(len(show))[::-1]
ax.barh(positions, values, color=np.where(values >= 0, C_BLUE, C_RED), height=0.72)
ax.axvline(0, color=INK_3, linewidth=1)
ax.set_yticks(positions)
ax.set_yticklabels([emphasis.iloc[i]["feature"][:28] for i in show], fontsize=8)
tidy(ax, f"F4  Which features the learned metric tightens on (k={COMPACTNESS_K})",
     "<- looser than plain kNN      relative difference (%)      tighter ->")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# T4 - learned metric versus input-space metric
# --------------------------------------------------------------------------- #
# Relative perplexity is k/n for both methods, so equal k means equal
# inspectability and the only difference is which neighbours get chosen.
Xtr_num = np.asarray(clf.X_encoder_.transform(X_train), dtype=float)
Xte_num = np.asarray(clf.X_encoder_.transform(X_test), dtype=float)
median = np.nan_to_num(np.nanmedian(Xtr_num, axis=0))
Xtr_num = np.where(np.isnan(Xtr_num), median, Xtr_num)
Xte_num = np.where(np.isnan(Xte_num), median, Xte_num)
mean, sd = Xtr_num.mean(0), Xtr_num.std(0)
sd = np.where(sd == 0, 1.0, sd)
Xtr_s, Xte_s = (Xtr_num - mean) / sd, (Xte_num - mean) / sd

print(f"\n{'k':>6} {'rel.PPL%':>10} {'KernelICL':>11} {'plain kNN':>11} {'delta pp':>10}")
T4 = []
for k in K_GRID:
    if k >= len(y_train):
        continue
    metric_kernel, perplexity, _ = score("knn", k)
    metric_plain = evaluate(
        y_test, KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, y_train).predict(Xte_s))
    T4.append(dict(k=k, ppl=perplexity, kernelicl=metric_kernel, plain=metric_plain))
    print(f"{k:>6} {100 * perplexity:>10.2f} {metric_kernel:>11.4f} {metric_plain:>11.4f} "
          f"{100 * (metric_kernel - metric_plain):>10.2f}")


# --------------------------------------------------------------------------- #
# F1 - accuracy against inspectability
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for kernel in KERNELS:
    points = sorted((100 * r["ppl"], r["metric"]) for r in T1 if r["kernel"] == kernel)
    ax.plot(*zip(*points), color=SERIES[kernel], linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL[kernel], zorder=3)

ax.plot(*zip(*sorted((100 * r["ppl"], r["plain"]) for r in T4)), color=INK_3,
        linewidth=1.5, linestyle="--", label="plain kNN (input space)", zorder=2)
ax.axhline(metric_single, color=INK_3, linewidth=1, linestyle=":", zorder=1)
ax.annotate("TabICL (single)", xy=(1.0, metric_single), xycoords=("axes fraction", "data"),
            xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=9)

# Labels float above the curves with a stagger: the three variants often sit within a
# fraction of a point of each other and would otherwise collide.
for kernel, dy in zip(KERNELS, (20, 12, 4)):
    x, y = sorted((100 * r["ppl"], r["metric"]) for r in T1 if r["kernel"] == kernel)[0]
    ax.annotate(LABEL[kernel].replace("KernelICL-", ""), xy=(x, y), xytext=(6, dy),
                textcoords="offset points", color=SERIES[kernel], fontsize=9, weight="bold")

ax.set_xscale("log")
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels(["0.1", "1", "10", "100"])
tidy(ax, f"F1  {METRIC.replace('_', ' ').capitalize()} against inspectability",
     "relative perplexity (%, log scale) - lower is more inspectable",
     f"test {METRIC.replace('_', ' ')}")
ax.legend(loc="center right", fontsize=9)
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# Cases to inspect, and a projection to place them in
# --------------------------------------------------------------------------- #
_, _, w_soft = score("gaussian", BEST["gaussian"])
weights = w_soft[0].cpu()
head.kernel = "gaussian"
with torch.no_grad():
    y_ctx = torch.from_numpy(
        clf.y_encoder_.transform(y_train)).float().to(ex.E_train.device)[None]
    probs_soft, _ = head(ex.E_train, ex.E_test, y_ctx, num_classes=n_classes,
                         gamma=BEST["gaussian"])
pred_soft = clf.y_encoder_.inverse_transform(probs_soft.argmax(-1)[0].cpu().numpy())
row_ppl = relative_perplexity(weights).numpy()

# Two extremes of the evidence spectrum, then one error in each direction. A missed
# positive and a false alarm have very different costs, so which one you see should
# not be luck. The rarer training class is treated as positive.
classes, counts = np.unique(y_train, return_counts=True)
POSITIVE = classes[counts.argmin()]
missed = np.flatnonzero((y_test == POSITIVE) & (pred_soft != POSITIVE))
false_alarm = np.flatnonzero((y_test != POSITIVE) & (pred_soft == POSITIVE))

candidates = [(int(row_ppl.argmin()), "most concentrated"),
              (int(row_ppl.argmax()), "most diffuse")]
# Within each direction take the most concentrated error: a confidently wrong
# prediction built on few cases is far more diagnostic than a hedged one.
if len(missed):
    candidates.append((int(missed[np.argmin(row_ppl[missed])]),
                       f"missed {POSITIVE} (false negative)"))
if len(false_alarm):
    candidates.append((int(false_alarm[np.argmin(row_ppl[false_alarm])]),
                       f"false {POSITIVE} (false positive)"))
if len(candidates) == 2:
    candidates.append((int(np.argsort(row_ppl)[len(row_ppl) // 2]), "median (no errors)"))

seen, PICKS, PICK_TAGS = set(), [], []
for idx, tag in candidates:
    if idx not in seen:
        seen.add(idx)
        PICKS.append(idx)
        PICK_TAGS.append(tag)

print(f"\npositive class = {POSITIVE} (rarer in training) | "
      f"{len(missed)} missed, {len(false_alarm)} false alarms")
for idx, tag in zip(PICKS, PICK_TAGS):
    print(f"  test row {idx:>5}  {tag}")

with torch.no_grad():
    H_train = head.embed(ex.E_train)[0].cpu().numpy().astype(np.float64)
    H_test = head.embed(ex.E_test)[0].cpu().numpy().astype(np.float64)

# Fitted on training cases with test cases transformed in, not on the union. A
# training row appears in its own context and so finds a perfect self-match among the
# keys; that offset is systematic and would dominate a joint layout.
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=SEED).fit(H_train)
    PROJ_NAME = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    # svd_solver="full": the default randomized solver emitted spurious overflow
    # warnings on well-conditioned input.
    reducer = PCA(n_components=2, svd_solver="full").fit(H_train)
    PROJ_NAME = "PCA"
    print("umap-learn not installed; using PCA, which shows only linear structure")

with np.errstate(all="ignore"):
    proj = np.asarray(reducer.transform(H_train))
    proj_test = np.asarray(reducer.transform(H_test))
assert np.isfinite(proj).all() and np.isfinite(proj_test).all()
order = np.argsort(proj[:, 0])


# --------------------------------------------------------------------------- #
# F3 - where each prediction's evidence sits
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(len(PICKS), 1, figsize=(7.2, 1.5 * len(PICKS) + 1.2), sharex=True)
axes = np.atleast_1d(axes)
for ax, idx, tag in zip(axes, PICKS, PICK_TAGS):
    ax.vlines(np.arange(len(order)), 0, weights[idx].numpy()[order], color=C_BLUE, linewidth=0.6)
    ax.set_ylabel("weight")
    ax.margins(x=0.01)
    ax.text(0.995, 0.88,
            f"test row {idx} - {tag}   rel. PPL {100 * row_ppl[idx]:.1f}%   "
            f"true {y_test[idx]} / pred {pred_soft[idx]}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_2)
axes[0].set_title("F3  Which training cases each prediction uses", color=INK,
                  fontsize=11, loc="left", pad=10)
axes[-1].set_xlabel(f"training cases, ordered by {PROJ_NAME} dimension 1")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# F7 - the weights in embedding space
# --------------------------------------------------------------------------- #
y_train_enc = clf.y_encoder_.transform(y_train)
# Only the first three palette slots clear the all-pairs colourblind gate, so further
# classes fold into one neutral rather than inventing a fourth hue.
PALETTE_CAP = 3
keep = list(np.argsort(-np.bincount(y_train_enc, minlength=n_classes))[:PALETTE_CAP])
class_colors = [C_BLUE, C_ORANGE, C_AQUA]
if n_classes > PALETTE_CAP:
    print(f"{n_classes} classes: showing the {PALETTE_CAP} most frequent separately")


def class_style(c):
    if c in keep:
        return class_colors[keep.index(c)], f"class {clf.classes_[c]}"
    return INK_3, "other classes"


# One area scale shared across panels: normalizing per panel would draw a weight of
# 0.006 as large as one of 0.89 and make a diffuse prediction look decisive. Which
# rows to colour stays per panel, since that is a per-prediction question.
SIZE_MAX = float(weights[PICKS].max())
XLIM, YLIM = limits(proj[:, 0]), limits(proj[:, 1])

fig, axes = plt.subplots(1, len(PICKS), figsize=(3.8 * len(PICKS), 4.0), sharex=True, sharey=True)
for ax, idx, tag in zip(np.atleast_1d(axes), PICKS, PICK_TAGS):
    wt = weights[idx].numpy()
    heavy = wt > wt.max() * 0.02
    ax.scatter(proj[~heavy, 0], proj[~heavy, 1], s=3, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        selected = heavy & (y_train_enc == c)
        if selected.any():
            ax.scatter(proj[selected, 0], proj[selected, 1],
                       s=8 + 260 * wt[selected] / SIZE_MAX, color=class_style(c)[0],
                       alpha=0.7, linewidths=0.5, edgecolors=SURFACE, zorder=2)

    # An out-of-frame case is worth seeing, but letting one point rescale the axes
    # would squash the cloud, so clamp it to the frame and say so.
    tx, ty = proj_test[idx]
    inside = XLIM[0] <= tx <= XLIM[1] and YLIM[0] <= ty <= YLIM[1]
    ax.scatter(np.clip(tx, *XLIM), np.clip(ty, *YLIM), marker="X", s=150,
               color=INK if inside else SURFACE, edgecolors=INK, linewidths=1.5, zorder=3)
    if not inside:
        ax.text(0.5, 0.02, "case sits outside the training cloud", transform=ax.transAxes,
                ha="center", fontsize=8, color=INK_2, style="italic")

    bare(ax, f"test row {idx}\n{tag} - rel. PPL {100 * row_ppl[idx]:.1f}%\n"
             f"true {y_test[idx]} / pred {pred_soft[idx]}")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

handles, labels_seen = [], set()
for c in range(n_classes):
    color, label = class_style(c)
    if label not in labels_seen:
        labels_seen.add(label)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=7,
                              color=color, label=label))
handles += [Line2D([], [], marker="o", linestyle="", markersize=4, color=GRID,
                   label="negligible weight"),
            Line2D([], [], marker="X", linestyle="", markersize=9, color=INK,
                   label="the case being decided")]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"F7  Training cases in {PROJ_NAME} space, sized by contribution",
             color=INK, fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
# Paste kernelicl_clinical.py first, then this file. Set METRIC above before running.
#
# Read F1 first: it sets the operating point every other artifact depends on.
# Then T4 as an honesty check, then T3/F4 for the finding about your features.
#
# To evaluate a fine-tuned model, set FINETUNED at the top of this file.
