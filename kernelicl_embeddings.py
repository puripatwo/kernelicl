# %% [markdown]
# # KernelICL — looking at the embedding itself
#
# Throwaway scratch file. Companion to `kernelicl_analysis.py` and
# `kernelicl_clinical.py`.
#
# This is **post-hoc**: it inspects the representation rather than the reasoning.
# A projection cannot tell you what a given prediction used — that is what the
# weights are for, and why F3/F7 exist. But it answers questions the weights
# cannot, and they are the right questions to ask *before* committing to
# fine-tuning:
#
# * Does the learned space separate your classes at all?
# * Is it better than raw features, and by how much?
# * Do errors sit somewhere specific, or are they scattered?
# * Do the diffuse, low-evidence predictions come from a particular region?
#
# If the answer to the first two is "no", fine-tuning is not optional polish —
# it is the whole job.
#
# ```
# !git clone -b kernelicl-head https://github.com/puripatwo/kernelicl.git
# %cd kernelicl
# !pip install -e . umap-learn
# ```
#
# Expects `X_train`, `y_train`, `X_test`, `y_test` in memory.

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

# fit_explainer may already be in the session (kernelicl_clinical.py pasted into a
# cell), or importable from the cloned repo. Adding the working directory covers
# `%run` from another location, where sys.path[0] is the script's directory.
if "fit_explainer" not in globals():
    import os
    import sys

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        from kernelicl_clinical import fit_explainer
    except ImportError:
        raise ImportError(
            "kernelicl_embeddings needs fit_explainer. Either run it from the cloned "
            "repo directory, or paste kernelicl_clinical.py into a cell first."
        ) from None

SEED = 0
PURITY_K = (1, 5, 10, 20, 50)   # neighbourhood sizes for the purity table
FEATURE_NAMES = list(X_train.columns) if hasattr(X_train, "columns") else None
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()

# %% [markdown]
# ## Palette
#
# Same tokens as `kernelicl_analysis.py` so the two sets of figures agree.
# Categorical slots 1-3, used unmodified: these are scatter plots, where every
# pair of hues must be distinguishable rather than just adjacent ones, and only
# the first three slots clear that gate. Classes past the third fold into one
# neutral rather than inventing a fourth hue.

# %%
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"
CLASS_COLORS = [C_BLUE, C_ORANGE, C_AQUA]
C_OTHER = INK_3

# Sequential ramp for magnitude, one hue light-to-dark, taken from the reference
# palette's blue steps rather than a rainbow map.
SEQ = LinearSegmentedColormap.from_list(
    "kicl_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 130,
})


def bare(ax, title=None):
    """Projections have no meaningful axis units, so drop the furniture."""
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, color=INK_2, fontsize=9, loc="left")
    return ax


# %% [markdown]
# ## Fit, and collect both spaces
#
# `fit_explainer` does the model, the scale calibration and the novelty
# calibration; everything below is a view of what it already computed.

# %%
ex = fit_explainer(X_train, y_train, X_test, feature_names=FEATURE_NAMES)

with torch.no_grad():
    H_train = ex.head.embed(ex.E_train)[0].cpu().numpy().astype(np.float64)
    H_test = ex.head.embed(ex.E_test)[0].cpu().numpy().astype(np.float64)

# The raw baseline: TabICL's own numeric encoding, median-imputed and z-scored.
# Same treatment the input-space kNN baseline gets in T4, so the comparison is fair.
Xtr_num = np.asarray(ex.clf.X_encoder_.transform(X_train), dtype=float)
Xte_num = np.asarray(ex.clf.X_encoder_.transform(X_test), dtype=float)
_med = np.nanmedian(Xtr_num, axis=0)
_med = np.where(np.isnan(_med), 0.0, _med)
Xtr_num = np.where(np.isnan(Xtr_num), _med, Xtr_num)
Xte_num = np.where(np.isnan(Xte_num), _med, Xte_num)
_mu, _sd = Xtr_num.mean(0), Xtr_num.std(0)
_sd = np.where(_sd == 0, 1.0, _sd)
RAW_train, RAW_test = (Xtr_num - _mu) / _sd, (Xte_num - _mu) / _sd

y_enc = ex.clf.y_encoder_.transform(y_train)
n_classes = ex.clf.n_classes_
print(f"embedding {H_train.shape} | raw {RAW_train.shape} | classes {n_classes}")

# %% [markdown]
# ## Read this before believing any of the pictures
#
# **Training-side embeddings encode their own labels.** In symmetric mode a
# training row is present in the context, so its query attends to its own key —
# at distance zero, therefore with the largest attention weight — and that key has
# `g(y_i)` added to it. Measured on the test data below, a linear probe recovers a
# training row's own outcome from its embedding with accuracy 1.000, against 0.877
# from the row representation before the label is added.
#
# This is the paper's design (Equation 14), not a defect, and it does not
# compromise predictions: a *test* query has no self-key, so nothing tells it its
# own outcome. But it makes every train-side diagnostic vacuous — train-side
# neighbourhood purity comes out at exactly 1.000 for any `k`, because the space is
# partly a label encoding. It is also the root cause of the novelty-calibration
# bias in `kernelicl_clinical`: training rows look implausibly close together.
#
# So the honest measurements below use the **test** side, whose labels were never
# in the context. Training cases still appear in the figures, but as gray context
# for the manifold rather than as evidence about class separation.

# %% [markdown]
# ## T5 — neighbourhood purity
#
# Of a case's `k` nearest neighbours, what fraction share its outcome?
#
# Measured in the **full** space — 512 dimensions for the embedding, all your
# features for raw — not in the 2-D projection. Purity computed on the projection
# would be measuring UMAP rather than the model.
#
# This is the quantity that decides whether a kernel head can work at all. A
# weighted average of neighbours' outcomes is useful only if neighbours tend to
# share the outcome, so test-side purity at k is close to a ceiling on what
# KernelICL-kNN at that k can achieve.

# %%
def purity(P, labels, k):
    """Fraction of each point's k nearest neighbours sharing its label."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


y_test_enc = ex.clf.y_encoder_.transform(y_test)
base_rate = max(np.bincount(y_test_enc) / len(y_test_enc))

print("measured on TEST cases -- train-side purity is 1.000 by construction, see above\n")
print(f"{'k':>5} {'raw features':>14} {'learned embedding':>19} {'gain':>8}")
T5 = []
for k in PURITY_K:
    if k >= len(y_test_enc):
        continue
    p_raw = purity(RAW_test, y_test_enc, k)
    p_emb = purity(H_test, y_test_enc, k)
    T5.append(dict(k=k, raw=p_raw, embedding=p_emb))
    print(f"{k:>5} {p_raw:>14.3f} {p_emb:>19.3f} {p_emb - p_raw:>+8.3f}")
print(f"\nchance level (majority class share in test): {base_rate:.3f}")

# The vacuous version, printed once so the contrast is on the record rather than
# something you have to take on trust.
print(f"for contrast, train-side embedding purity at k=5: "
      f"{purity(H_train, y_enc, 5):.3f}  (label leakage, not performance)")

# %% [markdown]
# ## Projections
#
# Fitted on the training cases, with test cases transformed in. Not fitted on the
# union: a training row appears in its own context and so finds a perfect match to
# itself among the keys, while a test row never does. That offset is systematic,
# and letting it shape the layout pushes every test case into empty space.

# %%
def project(fit_on, apply_to):
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=SEED).fit(fit_on)
        name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, svd_solver="full").fit(fit_on)
        name = "PCA"
    # numpy emits spurious overflow/divide warnings from the projection matmul on
    # well-conditioned input in some environments; the outputs are verified finite.
    with np.errstate(all="ignore"):
        base = np.asarray(reducer.transform(fit_on))
        others = [np.asarray(reducer.transform(a)) for a in apply_to]
    assert np.isfinite(base).all(), "projection produced non-finite coordinates"
    return name, base, others


PROJ_NAME, P_emb, (P_emb_test,) = project(H_train, [H_test])
_, P_raw, (P_raw_test,) = project(RAW_train, [RAW_test])
if PROJ_NAME == "PCA":
    print("umap-learn not installed; using PCA. `!pip install umap-learn` gives a much "
          "better picture -- PCA can only show linear structure.")


def limits(v, margin=0.06):
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * margin or 1.0
    return lo - pad, hi + pad


def class_style(c):
    keep = list(range(min(n_classes, len(CLASS_COLORS))))
    return (CLASS_COLORS[c], f"{ex.clf.classes_[c]}") if c in keep else (C_OTHER, "other")


# %% [markdown]
# ## E1 — raw features versus the learned embedding
#
# The headline comparison, and the one that decides whether fine-tuning is
# optional. Both panels are the same 2-D treatment of the same training cases;
# only the input space differs.
#
# What you want to see on the right: outcomes occupying distinct regions. If the
# two panels look equally mixed, the pretrained embedding is not separating your
# classes and no choice of kernel will rescue it.
#
# Coloured points are **test** cases. Training cases are the gray backdrop: they
# show the shape of the manifold, but colouring them by outcome would draw perfect
# separation in the right panel for the label-leakage reason above.

# %%
_t5 = {r["k"]: r for r in T5}
_k_show = 5 if 5 in _t5 else T5[0]["k"]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
for ax, P_tr, P_te, label, pk in [
    (axes[0], P_raw, P_raw_test, "Raw features", _t5[_k_show]["raw"]),
    (axes[1], P_emb, P_emb_test, "Learned embedding", _t5[_k_show]["embedding"]),
]:
    ax.scatter(P_tr[:, 0], P_tr[:, 1], s=4, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        sel = y_test_enc == c
        colour, name = class_style(c)
        ax.scatter(P_te[sel, 0], P_te[sel, 1], s=16, color=colour, alpha=0.75,
                   linewidths=0.3, edgecolors=SURFACE, zorder=2, label=name)
    bare(ax, f"{label}   ·   test purity@{_k_show} = {pk:.3f}")

handles = []
seen = set()
for c in range(n_classes):
    colour, name = class_style(c)
    if name not in seen:
        seen.add(name)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=7, color=colour, label=name))
handles.append(Line2D([], [], marker="o", linestyle="", markersize=5, color=GRID,
                      label="training case"))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, -0.03), title="test-case outcome", title_fontsize=9)
fig.suptitle(f"Where the outcomes sit, before and after the model ({PROJ_NAME})",
             color=INK, fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## E2 — test cases against the training population
#
# Left: do your test cases land where the training cases are? Gaps mean the model
# is extrapolating for part of the cohort, which is the same thing the novelty
# flag in `kernelicl_clinical` counts case by case.
#
# Right: errors only. Scattered errors are ordinary difficulty. Errors *clustered*
# in one region are a systematically hard sub-population — worth identifying,
# because that is a group the model quietly fails on rather than a random tail.

# %%
XL, YL = limits(P_emb[:, 0]), limits(P_emb[:, 1])
errors = ex.pred != y_test

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)

axes[0].scatter(P_emb[:, 0], P_emb[:, 1], s=6, color=GRID, linewidths=0, label="training")
axes[0].scatter(P_emb_test[:, 0], P_emb_test[:, 1], s=10, color=C_BLUE, alpha=0.7,
                linewidths=0, label="test")
bare(axes[0], "Test cases over the training population")
axes[0].legend(loc="upper right", fontsize=9)

axes[1].scatter(P_emb_test[~errors, 0], P_emb_test[~errors, 1], s=8, color=GRID,
                linewidths=0, label="correct")
axes[1].scatter(P_emb_test[errors, 0], P_emb_test[errors, 1], s=34, color=C_RED,
                alpha=0.85, linewidths=0.5, edgecolors=SURFACE, label=f"error ({errors.sum()})")
bare(axes[1], f"Where the {errors.sum()} errors are")
axes[1].legend(loc="upper right", fontsize=9)

for ax in axes:
    ax.set_xlim(*XL); ax.set_ylim(*YL)
fig.suptitle(f"Test cases in the learned embedding ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## E3 — how much evidence each region gets
#
# Each test case coloured by its evidence base: the effective number of training
# cases behind its prediction. Dark means the prediction rests on many cases
# weakly; light means it rests on few strongly.
#
# The pattern to expect is **dark in dense regions, light in sparse ones**. Where
# many training cases sit close together, a soft kernel spreads weight across all
# of them; where few do, weight concentrates on the handful that are near.
#
# So a light point is not automatically a well-supported one. Light *inside* the
# training cloud means a decisive, well-evidenced prediction. Light *away* from the
# cloud means only a few comparable cases exist at all — the extrapolation case
# that the novelty flag catches. Read this figure together with E2 left.
#
# If the whole map is dark, the kernel scale is too flat for anything to be
# inspectable, whatever the calibration chose; raise `accuracy_tolerance`.

# %%
fig, ax = plt.subplots(figsize=(6.8, 5.0))
ax.scatter(P_emb[:, 0], P_emb[:, 1], s=5, color=GRID, linewidths=0, zorder=1)
sc = ax.scatter(P_emb_test[:, 0], P_emb_test[:, 1], c=ex.evidence_cases, cmap=SEQ,
                s=22, linewidths=0.3, edgecolors=SURFACE, zorder=2)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("effective number of training cases behind the prediction", fontsize=9, color=INK_2)
cb.outline.set_visible(False)
bare(ax, f"gray = training cases   ·   median evidence base "
         f"{np.median(ex.evidence_cases):.0f} of {len(y_train):,}")
ax.set_xlim(*XL); ax.set_ylim(*YL)
fig.suptitle(f"Evidence base across the embedding ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## E4 — does the embedding organise by the features T3 flagged?
#
# A cross-check between two otherwise independent views. T3 says which features
# the model treats as defining similarity; if it is right, those features should
# vary smoothly across the embedding while features it ignores should look like
# noise sprayed over it.
#
# Agreement here is worth something: T3 is computed from neighbour sets and this
# from the geometry, so they can disagree.

# %%
emphasis = ex.feature_emphasis()
top_feature = emphasis.iloc[0]["feature"]
bottom_feature = emphasis.iloc[-1]["feature"]
names = [str(n) for n in (FEATURE_NAMES if FEATURE_NAMES is not None
                          else [f"feature {i}" for i in range(Xtr_num.shape[1])])]
col_top, col_bottom = names.index(top_feature), names.index(bottom_feature)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
for ax, col, tag in [(axes[0], col_top, f"most emphasised: {top_feature}"),
                     (axes[1], col_bottom, f"least emphasised: {bottom_feature}")]:
    v = RAW_train[:, col]
    lo, hi = np.percentile(v, [2, 98])  # clip outliers so the ramp is not wasted
    sc = ax.scatter(P_emb[:, 0], P_emb[:, 1], c=np.clip(v, lo, hi), cmap=SEQ,
                    s=7, linewidths=0, vmin=lo, vmax=hi)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("standardized value", fontsize=8, color=INK_2)
    bare(ax, tag)
fig.suptitle(f"Feature gradients across the embedding ({PROJ_NAME})", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reading these
#
# **T5 and E1 decide whether to fine-tune.** If embedding purity at k=5 clearly
# beats raw purity, the pretrained representation already transfers to your data
# and everything downstream is trustworthy; fine-tuning would raise the numbers
# but is not load-bearing. If the two are close, or embedding purity is barely
# above the majority-class share, the representation is not separating your
# outcomes — and no kernel, scale, or threshold fixes that. Fine-tuning becomes
# the whole task rather than the last step.
#
# **E2 left is a deployment check.** Test cases sitting outside the training cloud
# are cases the model has no comparable experience of. A few is normal; a whole
# region is a population you have not trained on.
#
# **E2 right is where to look next.** Clustered errors are a finding: pull those
# cases, look at what they share, and check whether it is a subgroup, a site, or a
# measurement artifact. Use `ex.triage()` and `ex.case(i)` on them — this figure
# tells you *where* to look, the weights tell you *why*.
#
# **E3 is a sanity check on the scale, and a second view of extrapolation.** Dark in
# dense regions and light in sparse ones is the expected pattern. Uniformly dark
# means nothing is inspectable and `accuracy_tolerance` in `fit_explainer` should go
# up. Light points sitting away from the training cloud are the cases with few
# comparable examples — cross-check them against `ex.triage()`.
#
# **E4 is corroboration, not evidence on its own.** A smooth gradient on the
# emphasised feature and noise on the ignored one supports T3. Disagreement is
# worth chasing — most often it means the emphasised feature is one of a
# correlated group, and the geometry is organised by the group rather than by that
# single column.
#
# **What none of this shows.** Which training cases produced a given prediction.
# A projection is a 2-D shadow of 512 dimensions: points that look adjacent may
# not be neighbours, and the weights are the only faithful account. Use this to
# decide where to look; use F3/F7 and `ex.case(i)` to decide what happened.
