"""What the learned representation looks like: T5, T6 and E1-E4.

Post-hoc: inspects the representation rather than the reasoning. A projection cannot
say what a given prediction used -- that is what the weights are for -- but it
answers whether the representation separates your outcomes at all, which is the
question to settle before paying for fine-tuning.

Paste kernelicl_clinical.py first. Also needs X_train, y_train, X_test, y_test.

See README.md, in particular the note on label leakage in train-side embeddings.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

# Works whether kernelicl_clinical.py was pasted into the session or is importable.
# The candidates cover %run from anywhere, pasting from the repo root, and pasting
# from this directory.
if "fit_explainer" not in globals():
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
    for _candidate in (_here, os.getcwd(), os.path.join(os.getcwd(), "kernelicl")):
        if _candidate and _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    from kernelicl_clinical import V1_CHECKPOINT, V2_CHECKPOINT, fit_explainer

SEED = 0
PURITY_K = (1, 5, 10, 20, 50)
FINETUNED = None   # path to a kernelicl_finetune checkpoint, or None
CHECKPOINT = None  # None = TabICL's default (v2); or V1_CHECKPOINT

FEATURE_NAMES = list(X_train.columns) if hasattr(X_train, "columns") else None
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"
CLASS_COLORS = [C_BLUE, C_ORANGE, C_AQUA]

# One hue light to dark for magnitude, from a validated blue ramp rather than a
# rainbow map.
SEQ = LinearSegmentedColormap.from_list(
    "kicl_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 130,
})


def bare(ax, title=None):
    """Projections have no meaningful axis units, so drop the furniture."""
    ax.set_xticks([])
    ax.set_yticks([])
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
# Setup: three spaces to compare
# --------------------------------------------------------------------------- #
ex = fit_explainer(X_train, y_train, X_test, feature_names=FEATURE_NAMES,
                   keep_row_repr=True, finetuned=FINETUNED,
                   checkpoint_version=CHECKPOINT)

with torch.no_grad():
    H_train = ex.head.embed(ex.E_train)[0].cpu().numpy().astype(np.float64)
    H_test = ex.head.embed(ex.E_test)[0].cpu().numpy().astype(np.float64)
    # The row stage: column embedding then row interaction, before any label enters.
    ROW_train = ex.R_train[0].cpu().numpy().astype(np.float64)
    ROW_test = ex.R_test[0].cpu().numpy().astype(np.float64)

# Raw baseline, given the same treatment as the input-space kNN in T4.
Xtr_num = np.asarray(ex.clf.X_encoder_.transform(X_train), dtype=float)
Xte_num = np.asarray(ex.clf.X_encoder_.transform(X_test), dtype=float)
median = np.nan_to_num(np.nanmedian(Xtr_num, axis=0))
Xtr_num = np.where(np.isnan(Xtr_num), median, Xtr_num)
Xte_num = np.where(np.isnan(Xte_num), median, Xte_num)
mean, sd = Xtr_num.mean(0), Xtr_num.std(0)
sd = np.where(sd == 0, 1.0, sd)
RAW_train, RAW_test = (Xtr_num - mean) / sd, (Xte_num - mean) / sd

y_enc = ex.clf.y_encoder_.transform(y_train)
y_test_enc = ex.clf.y_encoder_.transform(y_test)
n_classes = ex.clf.n_classes_
print(f"embedding {H_train.shape} | row stage {ROW_train.shape} | raw {RAW_train.shape}")


# --------------------------------------------------------------------------- #
# T5 - neighbourhood purity
# --------------------------------------------------------------------------- #
# Measured on TEST cases and in the FULL space. Train-side purity is 1.000 for any k
# because a training row sits in its own context and reads its own label off its own
# key; and purity computed on the 2-D projection would measure UMAP, not the model.
def purity(P, labels, k: int) -> float:
    """Fraction of each point's k nearest neighbours sharing its label."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


base_rate = max(np.bincount(y_test_enc) / len(y_test_enc))
print(f"\n{'k':>5} {'raw features':>14} {'learned embedding':>19} {'gain':>8}")
T5 = []
for k in PURITY_K:
    if k >= len(y_test_enc):
        continue
    raw = purity(RAW_test, y_test_enc, k)
    learned = purity(H_test, y_test_enc, k)
    T5.append(dict(k=k, raw=raw, embedding=learned))
    print(f"{k:>5} {raw:>14.3f} {learned:>19.3f} {learned - raw:>+8.3f}")
print(f"\nchance (majority class share in test): {base_rate:.3f}")
print(f"train-side embedding purity at k=5, for contrast: "
      f"{purity(H_train, y_enc, 5):.3f}  (label leakage, not performance)")


# --------------------------------------------------------------------------- #
# T6 - which stage creates the separation
# --------------------------------------------------------------------------- #
# TabICL has three stages and only the last sees labels. TF_col is per-cell so it
# needs different treatment; TF_row is per-case and label-free, which makes it the
# honest "before in-context learning" baseline.
print(f"\n{'stage':<22}{'train-side':>11}{'test-side':>11}   note")
T6 = []
for name, P_train, P_test, note in [
    ("raw features", RAW_train, RAW_test, ""),
    ("TF_row (no labels)", ROW_train, ROW_test, "honest on both sides"),
    ("TF_icl (in-context)", H_train, H_test, "train side leaks labels"),
]:
    train_purity = purity(P_train, y_enc, 5)
    test_purity = purity(P_test, y_test_enc, 5)
    T6.append(dict(stage=name, train=train_purity, test=test_purity))
    print(f"{name:<22}{train_purity:>11.3f}{test_purity:>11.3f}   {note}")


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #
# Fitted on training cases with test cases transformed in, never on the union: the
# train/test offset from that self-match is systematic and would dominate the layout.
def project(fit_on, apply_to):
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=SEED).fit(fit_on)
        name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, svd_solver="full").fit(fit_on)
        name = "PCA"

    # numpy emits spurious overflow warnings from the projection matmul in some
    # environments; the outputs are asserted finite.
    with np.errstate(all="ignore"):
        base = np.asarray(reducer.transform(fit_on))
        others = [np.asarray(reducer.transform(a)) for a in apply_to]
    assert np.isfinite(base).all(), "projection produced non-finite coordinates"
    return name, base, others


PROJ_NAME, P_emb, (P_emb_test,) = project(H_train, [H_test])
_, P_raw, (P_raw_test,) = project(RAW_train, [RAW_test])
_, P_row, (P_row_test,) = project(ROW_train, [ROW_test])
if PROJ_NAME == "PCA":
    print("\numap-learn not installed; using PCA, which shows only linear structure")


def class_style(c):
    if c < len(CLASS_COLORS):
        return CLASS_COLORS[c], str(ex.clf.classes_[c])
    return INK_3, "other"


# --------------------------------------------------------------------------- #
# E1 - the three stages side by side
# --------------------------------------------------------------------------- #
# Coloured points are test cases; training cases are the gray backdrop. Colouring
# training cases by outcome would draw perfect separation in the third panel for the
# label-leakage reason above.
stage_purity = {r["stage"]: r["test"] for r in T6}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
for ax, P_tr, P_te, label, key in [
    (axes[0], P_raw, P_raw_test, "1. Raw features", "raw features"),
    (axes[1], P_row, P_row_test, "2. TF_row (features only)", "TF_row (no labels)"),
    (axes[2], P_emb, P_emb_test, "3. TF_icl (in-context)", "TF_icl (in-context)"),
]:
    ax.scatter(P_tr[:, 0], P_tr[:, 1], s=4, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        selected = y_test_enc == c
        if selected.any():
            ax.scatter(P_te[selected, 0], P_te[selected, 1], s=16, color=class_style(c)[0],
                       alpha=0.75, linewidths=0.3, edgecolors=SURFACE, zorder=2)
    bare(ax, f"{label}\n   test purity@5 = {stage_purity[key]:.3f}")

handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=class_style(c)[0],
                  label=class_style(c)[1]) for c in range(min(n_classes, len(CLASS_COLORS)))]
handles.append(Line2D([], [], marker="o", linestyle="", markersize=5, color=GRID,
                      label="training case"))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, -0.03), title="test-case outcome", title_fontsize=9)
fig.suptitle(f"E1  Where the outcomes sit, stage by stage ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# E2 - test cases against the training population, and where errors are
# --------------------------------------------------------------------------- #
XLIM, YLIM = limits(P_emb[:, 0]), limits(P_emb[:, 1])
errors = ex.pred != y_test

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
axes[0].scatter(P_emb[:, 0], P_emb[:, 1], s=6, color=GRID, linewidths=0, label="training")
axes[0].scatter(P_emb_test[:, 0], P_emb_test[:, 1], s=10, color=C_BLUE, alpha=0.7,
                linewidths=0, label="test")
bare(axes[0], "Test cases over the training population")
axes[0].legend(loc="upper right", fontsize=9)

axes[1].scatter(P_emb_test[~errors, 0], P_emb_test[~errors, 1], s=8, color=GRID,
                linewidths=0, label="correct")
axes[1].scatter(P_emb_test[errors, 0], P_emb_test[errors, 1], s=34, color=C_RED, alpha=0.85,
                linewidths=0.5, edgecolors=SURFACE, label=f"error ({errors.sum()})")
bare(axes[1], f"Where the {errors.sum()} errors are")
axes[1].legend(loc="upper right", fontsize=9)

for ax in axes:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
fig.suptitle(f"E2  Test cases in the learned embedding ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# E3 - how much evidence each region gets
# --------------------------------------------------------------------------- #
# Expect dark in dense regions and light in sparse ones: a soft kernel spreads weight
# across many close cases. So a light point inside the cloud is a decisive
# prediction, while a light point away from it has few comparable cases at all.
fig, ax = plt.subplots(figsize=(6.8, 5.0))
ax.scatter(P_emb[:, 0], P_emb[:, 1], s=5, color=GRID, linewidths=0, zorder=1)
scatter = ax.scatter(P_emb_test[:, 0], P_emb_test[:, 1], c=ex.evidence_cases, cmap=SEQ,
                     s=22, linewidths=0.3, edgecolors=SURFACE, zorder=2)
colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
colorbar.set_label("effective number of training cases behind the prediction",
                   fontsize=9, color=INK_2)
colorbar.outline.set_visible(False)
bare(ax, f"gray = training cases   -   median evidence base "
         f"{np.median(ex.evidence_cases):.0f} of {len(y_train):,}")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
fig.suptitle(f"E3  Evidence base across the embedding ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# E4 - does the embedding organise by the features T3 flagged?
# --------------------------------------------------------------------------- #
# An independent cross-check: T3 works from neighbour sets, this from the geometry.
emphasis = ex.feature_emphasis()
names = [str(n) for n in (FEATURE_NAMES if FEATURE_NAMES is not None
                          else [f"feature {i}" for i in range(Xtr_num.shape[1])])]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
for ax, position, tag in [(axes[0], 0, "most emphasised"), (axes[1], -1, "least emphasised")]:
    feature = emphasis.iloc[position]["feature"]
    values = RAW_train[:, names.index(feature)]
    lo, hi = np.percentile(values, [2, 98])
    scatter = ax.scatter(P_emb[:, 0], P_emb[:, 1], c=np.clip(values, lo, hi), cmap=SEQ,
                         s=7, linewidths=0, vmin=lo, vmax=hi)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.set_label("standardized value", fontsize=8, color=INK_2)
    bare(ax, f"{tag}: {feature}")
fig.suptitle(f"E4  Feature gradients across the embedding ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
# Paste kernelicl_clinical.py first, then this file.
#
# T5 and T6 decide whether to fine-tune. If test-side embedding purity clearly beats
# raw purity, the pretrained representation transfers and everything downstream is
# trustworthy. If they are close, no kernel or threshold will fix that.
#
# To compare a fine-tuned model against the pretrained one, set FINETUNED at the top
# and re-run: T5 and T6 are the numbers that should move.
