# %% [markdown]
# # KernelICL — tables and figures
#
# Throwaway scratch file, companion to `kernelicl_colab.py`. Self-contained: it
# redoes the setup so it can run on its own.
#
# Builds T1-T4 and F1/F3/F4/F7 on your dataset, with **no fine-tuning** (`W = I`,
# frozen backbone). Read every number as "what the pretrained TabICL embedding
# gives you for free", not as a reproduction of the paper.
#
# ```
# !git clone -b kernelicl-head https://github.com/puripatwo/kernelicl.git
# %cd kernelicl
# !pip install -e . umap-learn
# ```
#
# Expects `X_train`, `X_test`, `y_train`, `y_test` in memory as numpy arrays.

# %%
import time
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import KernelHead, relative_perplexity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# Scale grids. The Gaussian/dot grids extend past the paper's Table 7 because an
# untrained W leaves the embeddings on a different scale -- see kernelicl_colab.py.
GAMMA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 5.0, 10.0]
K_GRID = [1, 4, 5, 16, 32, 64, 128, 256, 512, 1024]
KERNELS = ("gaussian", "dot", "knn")
COMPACTNESS_K = 5  # neighbourhood size for T3/F4, following the paper's Pima study

# X may be a DataFrame with string/categorical columns and NaNs. y is coerced to a
# plain array because weights are indexed positionally against training rows, and a
# Series with a non-default index would silently do label lookup instead.
FEATURE_NAMES = list(X_train.columns) if hasattr(X_train, "columns") else None
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()

print(f"device={DEVICE} | train {X_train.shape} | test {X_test.shape}")
_cls, _cnt = np.unique(y_train, return_counts=True)
print("classes:", dict(zip(_cls, _cnt)))
if _cnt.min() / _cnt.max() < 0.2:
    print("! imbalanced -- accuracy in T1/T2/T4 will flatter the majority class.")

# %% [markdown]
# ## Plot style
#
# One place for the palette so every figure agrees. Categorical slots 1-3 of the
# reference palette, used unmodified — that set is documented as clearing the
# colorblind-safety gates in both light and dark mode, including the all-pairs
# check that the scatter in F7 needs. Do not add a fourth categorical hue here:
# slot 4 puts yellow next to orange and fails all-pairs. Series beyond three
# become gray context lines instead.

# %%
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_RED = "#e34948"                     # diverging pole, F4 only
SERIES = {"gaussian": C_BLUE, "dot": C_ORANGE, "knn": C_AQUA}
LABEL = {"gaussian": "KernelICL-Gaussian", "dot": "KernelICL-Dot", "knn": "KernelICL-kNN"}

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


# %% [markdown]
# ## Setup — embed once, reuse everywhere
#
# Every table and figure below is a different view of the same two tensors, so
# the expensive stage runs exactly once. `n_estimators=1` and no shuffling, so a
# weight index maps straight back to a row of `X_train`.

# %%
def make_clf(n_estimators=1, single_norm=True):
    return TabICLClassifier(
        n_estimators=n_estimators,
        norm_methods=["none"] if single_norm else None,
        feat_shuffle_method="none" if single_norm else "latin",
        class_shuffle_method="none" if single_norm else "shift",
        device=DEVICE, random_state=SEED, kv_cache=False,
    )


def embed(fitted_clf, X_query):
    """Symmetric in-context embeddings for a fitted classifier and a query set.

    The ensemble generator sits *after* TabICL's numeric encoder, so the query has
    to be encoded first -- this mirrors what `predict_proba` does internally.
    Passing raw X works for all-numeric arrays and raises on string columns.
    """
    encoded = fitted_clf.X_encoder_.transform(X_query)
    X_ens, y_ens = next(iter(fitted_clf.ensemble_generator_.transform(encoded, mode="both").values()))
    X_t = torch.from_numpy(np.asarray(X_ens)).float().to(DEVICE)
    y_t = torch.from_numpy(np.asarray(y_ens)).float().to(DEVICE)
    m = fitted_clf.model_
    with torch.no_grad():
        R = m.row_interactor(
            m.col_embedder(X_t, y_train=y_t, mgr_config=fitted_clf.inference_config_.COL_CONFIG),
            mgr_config=fitted_clf.inference_config_.ROW_CONFIG,
        )
        E_train, E_test = m.icl_predictor.embed(R, y_t, symmetric=True)
    return E_train, E_test, y_t


clf = make_clf()
clf.fit(X_train, y_train)
n_classes = clf.n_classes_
d_model = clf.model_.icl_predictor.decoder[0].in_features

t0 = time.perf_counter()
E_train, E_test, y_t = embed(clf, X_test)
EMBED_SECONDS = time.perf_counter() - t0

head = KernelHead(d_model=d_model, d_k=d_model, identity_init=True).to(DEVICE)
print(f"d_model={d_model} | classes={n_classes} | embedding took {EMBED_SECONDS:.1f}s")
print(f"E_train {tuple(E_train.shape)} | E_test {tuple(E_test.shape)}")


def score(kernel, scale, E_tr=E_train, E_te=E_test, y_ctx=y_t, truth=None):
    """(accuracy, mean relative perplexity, weights) for one kernel/scale."""
    head.kernel = kernel
    with torch.no_grad():
        probs, w = head(E_tr, E_te, y_ctx, num_classes=n_classes, gamma=scale)
    pred = clf.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
    truth = y_test if truth is None else truth
    return (pred == truth).mean(), relative_perplexity(w).mean().item(), w


# %% [markdown]
# ## Calibration
#
# Scales are chosen on a split carved out of the training data — never on
# `y_test`. Ties on accuracy are broken toward the sparser (lower perplexity)
# scale; without that, the soft kernels select a near-uniform weighting that is
# accurate but carries no information. That tie-break is not in the paper; it
# compensates for the untrained `W`, which flattens the accuracy-vs-scale curve.

# %%
Xa, Xb, ya, yb = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
cal = make_clf()
cal.fit(Xa, ya)
E_ca_tr, E_ca_va, y_ca = embed(cal, Xb)

cal_head = KernelHead(d_model=d_model, d_k=d_model, identity_init=True).to(DEVICE)


def calibrate():
    chosen, curves = {}, {}
    for kernel in KERNELS:
        cal_head.kernel = kernel
        rows = []
        for scale in (K_GRID if kernel == "knn" else GAMMA_GRID):
            with torch.no_grad():
                probs, w = cal_head(E_ca_tr, E_ca_va, y_ca, num_classes=n_classes, gamma=scale)
            pred = cal.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
            rows.append((scale, (pred == yb).mean(), relative_perplexity(w).mean().item()))
        curves[kernel] = rows
        chosen[kernel] = max(rows, key=lambda r: (round(r[1], 3), -r[2]))[0]
    return chosen, curves


BEST, VAL_CURVES = calibrate()
print("calibrated scales:", BEST)

# %% [markdown]
# ## T1 — kernel × scale
#
# The paper's Table 3 analog. Kernel time is reported separately from embedding
# time because the scale never touches the embedding: one embedding pass serves
# the whole grid, which is what makes calibration cheap here.

# %%
print(f"embedding pass: {EMBED_SECONDS:.2f}s (shared by every row below)\n")
print(f"{'kernel':>10} {'scale':>8} {'test acc':>10} {'rel.PPL%':>10} {'kernel ms':>11}")
T1 = []
for kernel in KERNELS:
    for scale in (K_GRID if kernel == "knn" else GAMMA_GRID):
        t0 = time.perf_counter()
        acc, ppl, _ = score(kernel, scale)
        ms = (time.perf_counter() - t0) * 1000
        T1.append(dict(kernel=kernel, scale=scale, acc=acc, ppl=ppl, ms=ms))
        star = " *" if scale == BEST[kernel] else ""
        print(f"{kernel:>10} {scale:>8} {acc:>10.4f} {100*ppl:>10.2f} {ms:>11.1f}{star}")
print("\n* = scale selected on the validation split")

# %% [markdown]
# ## T2 — method comparison
#
# **TabICL-MLP is not available here.** In the paper it is the same architecture
# as KernelICL but fine-tuned with an MLP head instead of a kernel — the control
# that isolates the kernel's effect. Without fine-tuning it *is* TabICL (single):
# identical weights, identical predictions. Reporting it as a separate row would
# be inventing a number. It arrives with step D.
#
# KernelICL timings include the embedding pass, so they are standalone costs. The
# three variants share that pass in this notebook, which is why T1's kernel-only
# times are milliseconds.

# %%
clf_ens = make_clf(n_estimators=8, single_norm=False)
clf_ens.fit(X_train, y_train)

t0 = time.perf_counter()
acc_ens = (clf_ens.predict(X_test) == y_test).mean()
t_ens = time.perf_counter() - t0

t0 = time.perf_counter()
acc_single = (clf.predict(X_test) == y_test).mean()
t_single = time.perf_counter() - t0

T2 = [
    ("TabICL (ensemble, n=8)", acc_ens, None, t_ens),
    ("TabICL (single)", acc_single, None, t_single),
    ("TabICL-MLP", None, None, None),
]
for kernel in KERNELS:
    t0 = time.perf_counter()
    E_tr2, E_te2, y2 = embed(clf, X_test)          # standalone cost, not the shared pass
    head.kernel = kernel
    with torch.no_grad():
        probs, w = head(E_tr2, E_te2, y2, num_classes=n_classes, gamma=BEST[kernel])
    elapsed = time.perf_counter() - t0
    pred = clf.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
    T2.append((LABEL[kernel], (pred == y_test).mean(), relative_perplexity(w).mean().item(), elapsed))
    del E_tr2, E_te2

print(f"{'method':>24} {'accuracy':>10} {'rel.PPL%':>10} {'time (s)':>10}")
for name, acc, ppl, secs in T2:
    a = f"{acc:.4f}" if acc is not None else "—"
    p = f"{100*ppl:.2f}" if ppl is not None else "—"
    s = f"{secs:.1f}" if secs is not None else "—"
    print(f"{name:>24} {a:>10} {p:>10} {s:>10}")
print("\nTabICL-MLP: identical to TabICL (single) until the model is fine-tuned.")

# %% [markdown]
# ## T4 — learned metric vs input-space metric
#
# The paper's claim is that KernelICL-kNN beats plain kNN by ~5 points at matched
# sparsity. Matching is exact for the kNN kernel: relative perplexity is `k/n` for
# both methods, so equal `k` means equal inspectability and the only difference is
# *which* neighbours get chosen — input-space Euclidean versus learned embedding.

# %%
# The input-space baseline needs a purely numeric, NaN-free matrix, which raw X
# may not be. Reuse TabICL's own numeric encoder so string and categorical columns
# map the same way they do for the model, then median-impute -- sklearn's kNN
# cannot handle NaN, though TabICL itself can. Imputation touches only this
# baseline, never the KernelICL path.
Xtr_num = np.asarray(clf.X_encoder_.transform(X_train), dtype=float)
Xte_num = np.asarray(clf.X_encoder_.transform(X_test), dtype=float)
med = np.nanmedian(Xtr_num, axis=0)
med = np.where(np.isnan(med), 0.0, med)
Xtr_num = np.where(np.isnan(Xtr_num), med, Xtr_num)
Xte_num = np.where(np.isnan(Xte_num), med, Xte_num)

mu, sd = Xtr_num.mean(0), Xtr_num.std(0)
sd = np.where(sd == 0, 1.0, sd)
Xtr_s, Xte_s = (Xtr_num - mu) / sd, (Xte_num - mu) / sd

print(f"{'k':>6} {'rel.PPL%':>10} {'KernelICL':>11} {'std kNN':>10} {'delta pp':>10}")
T4 = []
for k in K_GRID:
    if k > len(X_train):
        continue
    acc_kicl, ppl, _ = score("knn", k)
    acc_std = (KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, y_train).predict(Xte_s) == y_test).mean()
    T4.append(dict(k=k, ppl=ppl, kicl=acc_kicl, std=acc_std))
    print(f"{k:>6} {100*ppl:>10.2f} {acc_kicl:>11.4f} {acc_std:>10.4f} {100*(acc_kicl-acc_std):>10.2f}")

# %% [markdown]
# ## F1 — accuracy vs inspectability
#
# The paper's Figure 6 analog, and the chart to read first: it tells you how sparse
# the weights can get before accuracy suffers, which sets the operating point every
# inspection below depends on.
#
# Three KernelICL variants carry the categorical hues because they are the subject.
# Standard kNN and stock TabICL are drawn as gray reference lines — they are
# context, not competing identities, so they do not consume a categorical slot.

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.6))

for kernel in KERNELS:
    pts = sorted([(100 * r["ppl"], r["acc"]) for r in T1 if r["kernel"] == kernel])
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=SERIES[kernel], linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL[kernel], zorder=3)

std_pts = sorted([(100 * r["ppl"], r["std"]) for r in T4])
ax.plot(*zip(*std_pts), color=INK_3, linewidth=1.5, linestyle="--",
        label="standard kNN (input space)", zorder=2)
ax.axhline(acc_single, color=INK_3, linewidth=1, linestyle=":", zorder=1)
ax.annotate("TabICL (single)", xy=(1.0, acc_single), xycoords=("axes fraction", "data"),
            xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=9)

# Direct-label each series at its sparsest point, so identity never rests on color
# alone. Labels float above the curves with a stagger, because the three variants
# often sit within a fraction of a point of each other and would otherwise collide.
for kernel, dy in zip(KERNELS, (20, 12, 4)):
    pts = sorted([(100 * r["ppl"], r["acc"]) for r in T1 if r["kernel"] == kernel])
    x, y = pts[0]
    ax.annotate(LABEL[kernel].replace("KernelICL-", ""), xy=(x, y), xytext=(6, dy),
                textcoords="offset points", color=SERIES[kernel], fontsize=9, weight="bold")

spread = max(r["acc"] for r in T1) - min(r["acc"] for r in T1 if r["ppl"] < 0.9)
if spread < 0.005:
    ax.annotate("all three kernels coincide at this scale", xy=(0.02, 0.06),
                xycoords="axes fraction", color=INK_2, fontsize=9, style="italic")

ax.set_xscale("log")
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels(["0.1", "1", "10", "100"])
tidy(ax, "Accuracy against inspectability", "relative perplexity (%, log scale) — lower is more inspectable",
     "test accuracy")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Projection for F3 and F7
#
# UMAP of the projected embeddings `h_D`. Falls back to PCA if `umap-learn` is
# missing — the layout differs but every conclusion below is computed from the
# weights, not from the projection, which only orders and positions the points.

# %%
with torch.no_grad():
    H_train = head.embed(E_train)[0].cpu().numpy()

try:
    from umap import UMAP
    proj = UMAP(n_components=2, random_state=SEED).fit_transform(H_train)
    PROJ_NAME = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    proj = PCA(n_components=2, random_state=SEED).fit_transform(H_train)
    PROJ_NAME = "PCA"
    print("umap-learn not installed; using PCA. `!pip install umap-learn` for the paper's layout.")

order = np.argsort(proj[:, 0])  # training samples sorted along the first component
print(f"{PROJ_NAME} projection: {proj.shape}")

# %% [markdown]
# ## Pick test points to inspect
#
# Rather than an arbitrary index, take three that are actually worth looking at:
# the most concentrated prediction, the most diffuse one, and a mistake.

# %%
_, _, W_GAUSS = score("gaussian", BEST["gaussian"])
w_g = W_GAUSS[0].cpu()
probs_g = None
head.kernel = "gaussian"
with torch.no_grad():
    probs_g, _ = head(E_train, E_test, y_t, num_classes=n_classes, gamma=BEST["gaussian"])
pred_g = clf.y_encoder_.inverse_transform(probs_g.argmax(-1)[0].cpu().numpy())

row_ppl = relative_perplexity(w_g).numpy()
errors = np.flatnonzero(pred_g != y_test)
PICKS = [int(row_ppl.argmin()), int(row_ppl.argmax())]
PICKS.append(int(errors[np.argmin(row_ppl[errors])]) if len(errors) else int(np.argsort(row_ppl)[len(row_ppl) // 2]))
PICK_TAGS = ["most concentrated", "most diffuse", "misclassified" if len(errors) else "median"]
print("inspecting test rows:", dict(zip(PICK_TAGS, PICKS)))

# %% [markdown]
# ## F3 — where each prediction's evidence sits
#
# Figure 2's "sample space" panel. One row per test point; the x-axis is the
# training set ordered along the projection's first component, the height is that
# training sample's weight. A single tall spike means the prediction rests on one
# neighbour; a low hedge across the width means it rests on everything, which is
# what an uncalibrated scale produces.

# %%
fig, axes = plt.subplots(len(PICKS), 1, figsize=(7.2, 1.5 * len(PICKS) + 1.2), sharex=True)
for ax, idx, tag in zip(np.atleast_1d(axes), PICKS, PICK_TAGS):
    ax.vlines(np.arange(len(order)), 0, w_g[idx].numpy()[order], color=C_BLUE, linewidth=0.6)
    ax.set_ylabel("weight")
    ax.margins(x=0.01)
    ax.text(0.995, 0.88, f"test row {idx} — {tag}   rel. PPL {100*row_ppl[idx]:.1f}%   "
                         f"true {y_test[idx]} / pred {pred_g[idx]}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_2)
np.atleast_1d(axes)[0].set_title("Which training samples each prediction uses", color=INK,
                                 fontsize=11, loc="left", pad=10)
np.atleast_1d(axes)[-1].set_xlabel(f"training samples, ordered by {PROJ_NAME} dimension 1")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## T3 & F4 — what the learned metric considers similar
#
# The paper's Table 5 analog, and the one artifact that says something about *your
# domain* rather than about the method. For each test point, measure the mean
# distance to its `k` neighbours along each feature in standardized space, then
# normalize by the method's own mean so the two methods are comparable.
#
# A **tighter** neighbourhood on a feature means the metric insists neighbours
# agree on it — the model treats it as important for similarity. Standard kNN is
# roughly isotropic by construction; anywhere KernelICL deviates is the pretrained
# embedding expressing a preference.
#
# Relative difference is `(standard - kernelicl) / standard`; positive means
# KernelICL is tighter. Note the paper's own Table 5 does not reproduce exactly
# from its rounded values under any simple formula, so this is my reading of its
# prose rather than a verified match.

# %%
def compactness(neighbour_idx):
    """Mean per-feature distance from each test point to its neighbours, normalized."""
    diff = np.abs(Xtr_s[neighbour_idx] - Xte_s[:, None, :])  # (m, k, F)
    per_feature = diff.mean(axis=(0, 1))
    return per_feature / per_feature.mean()


_, _, w_knn = score("knn", COMPACTNESS_K)
idx_kicl = w_knn[0].topk(COMPACTNESS_K, dim=-1).indices.cpu().numpy()
idx_std = NearestNeighbors(n_neighbors=COMPACTNESS_K).fit(Xtr_s).kneighbors(Xte_s, return_distance=False)

comp_kicl, comp_std = compactness(idx_kicl), compactness(idx_std)
rel_diff = (comp_std - comp_kicl) / comp_std

names = FEATURE_NAMES if FEATURE_NAMES is not None else [f"feature {i}" for i in range(Xtr_num.shape[1])]
names = [str(n) for n in names]
ranked = np.argsort(-rel_diff)
TOP = min(12, len(ranked) // 2)
show = np.concatenate([ranked[:TOP], ranked[-TOP:]])

print(f"neighbourhood compactness, k={COMPACTNESS_K} (normalized by method mean)\n")
print(f"{'feature':>28} {'standard':>10} {'KernelICL':>11} {'rel.diff':>10}")
for i in show:
    print(f"{names[i][:28]:>28} {comp_std[i]:>10.2f} {comp_kicl[i]:>11.2f} {100*rel_diff[i]:>9.0f}%")

# %%
fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(show) + 1.4))
vals = rel_diff[show] * 100
ypos = np.arange(len(show))[::-1]
ax.barh(ypos, vals, color=np.where(vals >= 0, C_BLUE, C_RED), height=0.72)
ax.axvline(0, color=INK_3, linewidth=1)
ax.set_yticks(ypos)
ax.set_yticklabels([names[i][:28] for i in show], fontsize=8)
tidy(ax, f"Which features the learned metric tightens on (k={COMPACTNESS_K})",
     "← looser than standard kNN      relative difference (%)      tighter →", None)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## F7 — the weights in embedding space
#
# Figure 2's middle panel. Each dot is a training sample positioned by the
# projection and colored by its class; dot area is the weight it contributes to
# the marked test point. Gray dots carry negligible weight.
#
# This is the figure that shows *why* a prediction came out as it did: if the
# heavy dots cluster in one class region, the prediction is well-supported; if
# they straddle a boundary, it is not.

# %%
y_train_enc = clf.y_encoder_.transform(y_train)

# A scatter needs every pair of hues distinguishable, not just adjacent ones, and
# only the first three slots of the palette clear that gate. Beyond three classes,
# the rest fold into one neutral "other" rather than inventing a fourth hue --
# generating extra hues is what makes a chart unreadable under colorblindness.
PALETTE_CAP = 3
class_colors = [C_BLUE, C_ORANGE, C_AQUA][:min(n_classes, PALETTE_CAP)]
C_OTHER = "#8a8984"
if n_classes > PALETTE_CAP:
    print(f"{n_classes} classes: showing the {PALETTE_CAP} most frequent separately, "
          f"the rest as 'other'. Facet by class if you need all of them.")
    keep = list(np.argsort(-np.bincount(y_train_enc, minlength=n_classes))[:PALETTE_CAP])
else:
    keep = list(range(n_classes))


def class_style(c):
    """(color, label) for an encoded class index."""
    return (class_colors[keep.index(c)], f"class {clf.classes_[c]}") if c in keep else (C_OTHER, "other classes")

# Two separate decisions, deliberately not conflated:
#   * which rows to color -- per panel, since "did this prediction use row i" is a
#     per-prediction question;
#   * how big to draw them -- one scale shared across panels, so a dot's size means
#     the same weight everywhere.
# Normalizing size per panel would draw a weight of 0.006 as large as one of 0.89,
# making a diffuse prediction look as decisive as a concentrated one. Shared sizing
# means a diffuse panel shows *where* its evidence sits while its small dots say
# honestly that no single row carries much.
SIZE_MAX = float(w_g[PICKS].max())

fig, axes = plt.subplots(1, len(PICKS), figsize=(3.8 * len(PICKS), 4.0), sharex=True, sharey=True)
for ax, idx, tag in zip(np.atleast_1d(axes), PICKS, PICK_TAGS):
    wt = w_g[idx].numpy()
    heavy = wt > wt.max() * 0.02
    ax.scatter(proj[~heavy, 0], proj[~heavy, 1], s=3, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        sel = heavy & (y_train_enc == c)
        if sel.any():
            ax.scatter(proj[sel, 0], proj[sel, 1], s=8 + 260 * wt[sel] / SIZE_MAX,
                       color=class_style(c)[0], alpha=0.7, linewidths=0.5,
                       edgecolors=SURFACE, zorder=2)
    ax.set_title(f"test row {idx}\n{tag} · rel. PPL {100*row_ppl[idx]:.1f}%\n"
                 f"true {y_test[idx]} / pred {pred_g[idx]}",
                 color=INK_2, fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)

seen, handles = set(), []
for c in range(n_classes):
    color, label = class_style(c)
    if label not in seen:
        seen.add(label)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=7, color=color, label=label))
handles.append(Line2D([], [], marker="o", linestyle="", markersize=4, color=GRID, label="negligible weight"))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"Training samples in {PROJ_NAME} space, sized by contribution", color=INK,
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reading these
#
# **F1 first.** It sets the operating point. If the kNN curve stays flat down to
# 1% relative perplexity, you can inspect predictions in terms of a handful of
# training rows at no accuracy cost, and everything below is trustworthy. If
# accuracy falls off a cliff before you reach a readable sparsity, the pretrained
# embedding is not separating your classes well and fine-tuning (step D) is the
# fix, not a different kernel.
#
# **T4 is the honesty check.** If KernelICL-kNN does not beat plain kNN at matched
# `k`, the learned embedding is contributing nothing over Euclidean distance on
# standardized inputs, and you should be using `sklearn` instead. The paper reports
# roughly a 5-point gap. A *much* larger gap is not necessarily good news — it
# usually means the input-space baseline is crippled by many uninformative
# features rather than that KernelICL is doing something exceptional. Check how
# plain kNN alone compares to a tuned GBM before reading a large gap as a win.
#
# **F3 and F7 are the same information twice**, and worth reading together: F3
# shows *how much* weight each training row carries, F7 shows *where* those rows
# sit relative to the class structure. A prediction with one tall spike in F3 and
# a single large dot inside its own class region in F7 is well-supported. One
# whose heavy dots straddle a boundary in F7 is a prediction to distrust, whatever
# its softmax score says. The misclassified panel is usually the instructive one:
# it tells you whether the model was misled by genuinely similar training rows or
# is extrapolating from somewhere it should not.
#
# **T3/F4 is the finding.** Features where KernelICL is much tighter are the ones
# the model treats as defining similarity. Check them against domain knowledge:
# agreement is evidence the embedding learned something real, disagreement is
# either a discovery or a data problem, and both are worth knowing.
#
# **T1 and T2 are bookkeeping** — they justify the operating point and record the
# cost. Expect KernelICL to be roughly 2x stock TabICL's time: symmetric mode runs
# the training rows through the ICL transformer a second time.
