"""Corruption testing: did the model notice?

The usual corruption test compares accuracy on a clean test set against a corrupted
one. A drop tells you the model is sensitive. It does not tell you the thing that
matters in deployment -- whether the model knew it was in trouble.

Degrading from 96% to 81% while flagging most of the new errors is a model behaving
well. Degrading identically while staying confident is dangerous, because in the field
nobody catches the difference. Accuracy cannot separate the two; a kernel head can,
because every prediction carries an evidence base, an agreement share, and a distance
to the nearest comparable training case.

Corrupting the test set cannot move the training embeddings -- column statistics
attend to training rows only, and the ICL keys are the training context -- so the
reference library is fixed and every difference is attributable to the corruption.

Paste kernelicl_clinical.py first. Also needs X_train, y_train, X_test, y_test.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "fit_explainer" not in globals():
    import os
    import sys

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from kernelicl_clinical import fit_explainer

SEED = 0
TOP_K = 5          # neighbourhood size for the evidence-overlap measure
FINETUNED = None   # path to a kernelicl_finetune checkpoint, or None

# Where to write publication copies, and at what resolution. None shows only.
SAVE_FIGURES = None   # e.g. "figures"
FIG_DPI = 300
FIG_FORMATS = ("png", "pdf")


FEATURE_NAMES = list(X_train.columns) if hasattr(X_train, "columns") else None
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"

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


def finish(fig, name: str):
    """Show a figure, and write a publication copy when SAVE_FIGURES is set."""
    if SAVE_FIGURES:
        from pathlib import Path

        directory = Path(SAVE_FIGURES)
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in FIG_FORMATS:
            fig.savefig(directory / f"{name}.{suffix}", dpi=FIG_DPI,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


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
# Corruptions
# --------------------------------------------------------------------------- #
# Each returns a copy; the original is never modified. ``column`` is a name for a
# DataFrame or an integer index for an array. Supply your own if the failure you want
# to model is not one of these.
#
# The four differ in what they destroy, and that changes what a drop in accuracy means:
#
#   collapse / constant  the variable carries no information AND its distribution
#                        changes. "What if everyone were recorded as male."
#   shuffle              the distribution is untouched, only the per-row link breaks.
#                        "What if this field were filled in at random."
#   replace              a value-level edit; other categories survive.
#                        "What if females were recorded as male."
#   copy                 one field overwritten by another. A data-entry mix-up.
#
# Shuffle is the cleaner test of "does the model use this feature", because collapse
# changes two things at once and a drop could be either.
def _column(X, column):
    return X[column].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, column]


def _with_column(X, column, values):
    out = X.copy()
    if hasattr(out, "columns"):
        out[column] = values
    else:
        out[:, column] = values
    return out


def _typical(values):
    """Median for numbers, most common value otherwise.

    A categorical column has no median, so a single fill value cannot be chosen by
    dtype-blind code without crashing on strings.
    """
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.median()
    mode = series.mode()
    return mode.iloc[0] if len(mode) else series.iloc[0]


def corrupt_constant(X, column, value=None):
    """Overwrite a column with one value: a field defaulted, a sensor stuck.

    ``value=None`` uses the median for a numeric column and the most common value
    otherwise -- so on a Gender column it makes everyone whichever value is commonest,
    which is the categorical version of this test.
    """
    if value is None:
        value = _typical(_column(X, column))
    return _with_column(X, column, value)


def corrupt_replace(X, column, mapping):
    """Rewrite specific values, leaving the rest alone.

    ``corrupt_replace(X, "Gender", {"female": "male"})`` records every female as male
    while any other category survives. Narrower than a collapse, and closer to how
    miscoding actually happens.
    """
    return _with_column(X, column, pd.Series(_column(X, column)).replace(mapping).to_numpy())


def corrupt_shuffle(X, column, seed=SEED):
    """Permute a column, breaking its link to the row but keeping its distribution.

    Isolates whether the model uses the column's *information* or just its presence:
    the marginal statistics are identical, only the per-row correspondence is gone.
    """
    values = _column(X, column)
    return _with_column(X, column, np.random.RandomState(seed).permutation(values))


def corrupt_copy(X, source, target):
    """Write one column's values into another: the classic data-entry mix-up."""
    return _with_column(X, target, _column(X, source))


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def detection_table(ex_clean, ex_corrupt, y_true) -> pd.DataFrame:
    """The 2x2 of correct/wrong against flagged/not-flagged, clean versus corrupted.

    A case is "flagged" when triage would not route it as routine -- either the
    comparable cases disagreed, or no comparable case existed.

    ``silent failures`` is the operational number: wrong, with no warning attached.
    ``detection rate`` is the answer to "did the model notice": of the errors it made,
    what share did it flag. Holding up under corruption means graceful degradation;
    collapsing means errors are arriving unannounced.
    """
    rows = {}
    for name, ex in (("clean", ex_clean), ("corrupted", ex_corrupt)):
        wrong = ex.pred != y_true
        flagged = (ex.triage().sort_index()["action"] != "routine").to_numpy()
        n = len(y_true)
        rows[name] = {
            "accuracy": float((~wrong).mean()),
            "flagged for review": float(flagged.mean()),
            "silent failures": float((wrong & ~flagged).mean()),
            "caught failures": float((wrong & flagged).mean()),
            "detection rate": float(flagged[wrong].mean()) if wrong.any() else float("nan"),
            "mean agreement": float(ex.agreement.mean()),
            "mean evidence cases": float(ex.evidence_cases.mean()),
        }
    table = pd.DataFrame(rows)
    table["change"] = table["corrupted"] - table["clean"]
    return table


def evidence_overlap(ex_clean, ex_corrupt, k=TOP_K) -> np.ndarray:
    """Fraction of each case's top-k evidence that survived the corruption.

    1.0 means the same past cases were consulted; 0.0 means the corruption sent the
    model to an entirely different part of the training set. This is the most direct
    measure of whether the corrupted feature was actually being used.
    """
    before = np.argsort(-ex_clean.w, axis=1)[:, :k]
    after = np.argsort(-ex_corrupt.w, axis=1)[:, :k]
    return np.array([len(set(a) & set(b)) / k for a, b in zip(before, after)])


def shift_table(ex_clean, ex_corrupt, k=TOP_K) -> pd.DataFrame:
    """Per-case before/after deltas, summarised."""
    overlap = evidence_overlap(ex_clean, ex_corrupt, k)
    deltas = {
        "agreement": ex_corrupt.agreement - ex_clean.agreement,
        "evidence cases": ex_corrupt.evidence_cases - ex_clean.evidence_cases,
        "distance to nearest": ex_corrupt._nn_test - ex_clean._nn_test,
    }
    rows = {name: {"mean": float(d.mean()), "median": float(np.median(d)),
                   "worst": float(d.min() if name != "distance to nearest" else d.max())}
            for name, d in deltas.items()}
    rows[f"evidence overlap@{k}"] = {"mean": float(overlap.mean()),
                                     "median": float(np.median(overlap)),
                                     "worst": float(overlap.min())}
    return pd.DataFrame(rows).T


def per_case(ex_clean, ex_corrupt, y_true, k=TOP_K) -> pd.DataFrame:
    """One row per case, for finding the ones the corruption broke silently."""
    wrong_clean = ex_clean.pred != y_true
    wrong_corrupt = ex_corrupt.pred != y_true
    flagged = (ex_corrupt.triage().sort_index()["action"] != "routine").to_numpy()
    return pd.DataFrame({
        "record": ex_clean.test_ids,
        "truth": y_true,
        "pred_clean": ex_clean.pred,
        "pred_corrupt": ex_corrupt.pred,
        "broke": ~wrong_clean & wrong_corrupt,
        "silent": ~wrong_clean & wrong_corrupt & ~flagged,
        "agreement_clean": ex_clean.agreement,
        "agreement_corrupt": ex_corrupt.agreement,
        "evidence_overlap": evidence_overlap(ex_clean, ex_corrupt, k),
        "flagged_corrupt": flagged,
    })


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
# Choose the corruption. The default overwrites whichever feature the model leans on
# most, which is the version of this test most likely to hurt; swap in your own.
ex_clean = fit_explainer(X_train, y_train, X_test, feature_names=FEATURE_NAMES,
                         finetuned=FINETUNED)

emphasis = ex_clean.feature_emphasis()
TARGET = emphasis.iloc[0]["feature"]
print(f"\ncorrupting {TARGET!r} (the most emphasised feature per T3)")

TARGET_COLUMN = TARGET if hasattr(X_test, "columns") else (
    list(FEATURE_NAMES).index(TARGET) if FEATURE_NAMES else 0)
X_test_corrupt = corrupt_constant(X_test, TARGET_COLUMN)

# Same model, same scale, same thresholds -- only the cases differ.
ex_corrupt = ex_clean.for_test_set(X_test_corrupt, test_ids=ex_clean.test_ids)


# --------------------------------------------------------------------------- #
# S1 - did the model notice?
# --------------------------------------------------------------------------- #
S1 = detection_table(ex_clean, ex_corrupt, y_test)
print(f"\n{'':<22}{'clean':>10}{'corrupted':>12}{'change':>10}")
for name, row in S1.iterrows():
    print(f"{name:<22}{row['clean']:>10.3f}{row['corrupted']:>12.3f}{row['change']:>+10.3f}")

verdict = ("the model noticed: it flagged a similar share of its errors"
           if S1.loc["detection rate", "change"] > -0.1 else
           "SILENT DEGRADATION: errors rose faster than warnings did")
print(f"\n{verdict}")


# --------------------------------------------------------------------------- #
# S2 - what happened to the evidence
# --------------------------------------------------------------------------- #
S2 = shift_table(ex_clean, ex_corrupt)
print(f"\n{'':<24}{'mean':>9}{'median':>9}{'worst':>9}")
for name, row in S2.iterrows():
    print(f"{name:<24}{row['mean']:>9.3f}{row['median']:>9.3f}{row['worst']:>9.3f}")

CASES = per_case(ex_clean, ex_corrupt, y_test)
broke, silent = int(CASES["broke"].sum()), int(CASES["silent"].sum())
print(f"\n{broke} cases went from correct to wrong; {silent} of those unflagged")
if silent:
    worst = CASES[CASES["silent"]].nlargest(5, "agreement_corrupt")
    print("\nmost confident silent failures")
    print(worst[["record", "truth", "pred_corrupt", "agreement_corrupt",
                 "evidence_overlap"]].to_string(index=False))


# --------------------------------------------------------------------------- #
# G1 - the safety plot
# --------------------------------------------------------------------------- #
# Unfamiliarity against confidence, coloured by correctness. Silent failures are the
# wrong points in the confident-and-familiar corner: the model had no signal that
# anything was off. Read the figure by whether that corner fills up after corruption.
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
for ax, ex, label in [(axes[0], ex_clean, "clean"), (axes[1], ex_corrupt, "corrupted")]:
    wrong = ex.pred != y_test
    ax.scatter(ex._nn_test[~wrong], ex.agreement[~wrong], s=10, color=GRID,
               linewidths=0, label="correct")
    ax.scatter(ex._nn_test[wrong], ex.agreement[wrong], s=30, color=C_RED, alpha=0.85,
               linewidths=0.5, edgecolors=SURFACE, label=f"wrong ({int(wrong.sum())})")
    ax.axhline(ex.min_agreement, color=INK_3, linewidth=1, linestyle="--")
    ax.axvline(ex.novelty_threshold, color=INK_3, linewidth=1, linestyle="--")
    tidy(ax, f"{label}   accuracy {(~wrong).mean():.3f}",
         "distance to nearest comparable case", "agreement")
    ax.legend(loc="lower left", fontsize=9)
axes[0].annotate("wrong points below-left of the\ndashed lines are silent failures",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8,
                 color=INK_2, style="italic")
fig.suptitle("Did the model have any warning?", color=INK, fontsize=11,
             x=0.01, ha="left")
plt.tight_layout()
finish(fig, "warning_signals")


# --------------------------------------------------------------------------- #
# G2 - paired confidence shift
# --------------------------------------------------------------------------- #
# One point per case. Below the diagonal means the corruption cost confidence. What
# you want is for the cases that broke to sit well below it -- lost accuracy showing
# up as lost confidence rather than as unchanged certainty.
fig, ax = plt.subplots(figsize=(5.6, 5.4))
broke_mask = CASES["broke"].to_numpy()
lo = min(ex_clean.agreement.min(), ex_corrupt.agreement.min())
ax.plot([lo, 1], [lo, 1], color=INK_3, linewidth=1, linestyle="--", zorder=1)
ax.scatter(ex_clean.agreement[~broke_mask], ex_corrupt.agreement[~broke_mask], s=12,
           color=GRID, linewidths=0, zorder=2, label="unchanged verdict")
ax.scatter(ex_clean.agreement[broke_mask], ex_corrupt.agreement[broke_mask], s=34,
           color=C_RED, alpha=0.85, linewidths=0.5, edgecolors=SURFACE, zorder=3,
           label=f"broke ({broke})")
tidy(ax, "Confidence before and after corruption",
     "agreement, clean data", "agreement, corrupted data")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
finish(fig, "confidence_shift")


# --------------------------------------------------------------------------- #
# G3 - where the cohort moved
# --------------------------------------------------------------------------- #
# Fitted on training cases only, both test sets transformed in, so the reference frame
# is the one the model actually compares against.
import torch  # noqa: E402  (kept local; only this figure needs it)

with torch.no_grad():
    H_train = ex_clean.head.embed(ex_clean.E_train)[0].cpu().numpy().astype(np.float64)
    H_clean = ex_clean.head.embed(ex_clean.E_test)[0].cpu().numpy().astype(np.float64)
    H_corrupt = ex_corrupt.head.embed(ex_corrupt.E_test)[0].cpu().numpy().astype(np.float64)

try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=SEED).fit(H_train)
    PROJ_NAME = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, svd_solver="full").fit(H_train)
    PROJ_NAME = "PCA"
    print("\numap-learn not installed; using PCA, which shows only linear structure")

with np.errstate(all="ignore"):
    P_train = np.asarray(reducer.transform(H_train))
    P_clean = np.asarray(reducer.transform(H_clean))
    P_corrupt = np.asarray(reducer.transform(H_corrupt))

fig, ax = plt.subplots(figsize=(6.6, 5.4))
ax.scatter(P_train[:, 0], P_train[:, 1], s=4, color=GRID, linewidths=0, zorder=1,
           label="training cases")
sample = np.random.RandomState(SEED).choice(len(P_clean), min(60, len(P_clean)),
                                            replace=False)
for i in sample:
    ax.annotate("", xy=P_corrupt[i], xytext=P_clean[i], zorder=2,
                arrowprops=dict(arrowstyle="->", color=INK_3, linewidth=0.6, alpha=0.6))
ax.scatter(P_clean[:, 0], P_clean[:, 1], s=12, color=C_BLUE, alpha=0.7, linewidths=0,
           zorder=3, label="clean")
ax.scatter(P_corrupt[:, 0], P_corrupt[:, 1], s=12, color=C_ORANGE, alpha=0.7,
           linewidths=0, zorder=3, label="corrupted")
bare(ax)
ax.set_xlim(*limits(np.vstack([P_train, P_clean, P_corrupt])[:, 0]))
ax.set_ylim(*limits(np.vstack([P_train, P_clean, P_corrupt])[:, 1]))
ax.legend(loc="upper right", fontsize=9)
fig.suptitle(f"Where the corruption moved the cohort ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
finish(fig, "cohort_movement")


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
# Paste kernelicl_clinical.py first, then this file. It corrupts the most emphasised
# feature by default; edit the "Run" section for your own corruption:
#
#   X_test_corrupt = corrupt_constant(X_test, "Gender")             # everyone the mode
#   X_test_corrupt = corrupt_replace(X_test, "Gender", {"female": "male"})
#   X_test_corrupt = corrupt_shuffle(X_test, "Gender")              # keeps the mix
#   X_test_corrupt = corrupt_copy(X_test, "Gender", "Age")          # data-entry mix-up
#   ex_corrupt = ex_clean.for_test_set(X_test_corrupt, test_ids=ex_clean.test_ids)
#
# Read S1 first. "detection rate" is the answer to whether the model noticed: if it
# holds up, degradation is graceful; if it collapses, errors arrive unannounced.
# "silent failures" is what that costs you per cohort.
#
# Then S2: an evidence overlap near 1.0 means the corruption barely changed which past
# cases were consulted, so the model was not relying on that feature. Near 0 means it
# was, and the neighbours it found instead are why the predictions changed.
#
# Compare corruptions of a high- and a low-emphasis feature from T3 to check that the
# emphasis ranking predicts fragility -- the two measurements are independent.
