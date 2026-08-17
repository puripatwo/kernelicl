# %% [markdown]
# # KernelICL on your own dataset — Colab scratch notebook
#
# Throwaway scratch file, not part of the package. Delete it before merging anything.
#
# Turns a TabICL prediction into a transparent weighted average of your training
# rows, so you can point at any test prediction and name the training cases that
# produced it. **No fine-tuning happens here** — this is Path A from the plan:
# frozen pretrained weights, identity projection. Accuracy will be close to stock
# TabICL; what you gain is the weight vector.
#
# Setup cell (run first):
#
# ```
# !git clone -b kernelicl-head https://github.com/puripatwo/kernelicl.git
# %cd kernelicl
# !pip install -e .
# ```
#
# Note the `-b kernelicl-head` — the kernel head lives on that branch, not `main`.
# Set the runtime to a GPU (Runtime → Change runtime type → T4 or better) *before*
# installing. Restart the session after `pip install -e .` if it upgraded numpy.

# %%
import numpy as np
import torch

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import KernelHead, relative_perplexity

# Expected inputs, already in memory:
#   X_train (11690, 176)   y_train (11690,)
#   X_test   (2923, 176)   y_test   (2923,)
#
# X may be a DataFrame with string/categorical columns and NaNs -- TabICL handles
# all three. y is coerced to a plain array because everything below indexes it
# positionally (weight index i -> training row i), and a pandas Series with a
# non-default index would silently do label lookup instead and return wrong rows.
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()

print("X_train", X_train.shape, "| X_test", X_test.shape)
classes, counts = np.unique(y_train, return_counts=True)
print("classes:", dict(zip(classes, counts)))
if counts.min() / counts.max() < 0.2:
    print("! imbalanced -- accuracy below will flatter the majority class; "
          "consider balanced accuracy or per-class recall instead.")

# 176 features is outside the 5-100 range the synthetic prior was trained on.
# TabICL handles it (the TALENT benchmark goes to 970 features), but it is
# extrapolation, so treat the numbers below as a working baseline rather than a
# best case.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Leave this at None. Chunking caches the context K/V and streams queries against
# it, which only pays off when queries vastly outnumber context rows. At your
# shape (n=11690, m=2923) the cache costs more than it saves -- measured at
# 12 blocks x 2 x n x 512 floats = ~570 MB of cache versus ~650 MB for the whole
# unchunked pass. Set it to e.g. 2048 only if you later score a very large test
# set against a small context.
QUERY_CHUNK = None
print("device:", DEVICE)

# %% [markdown]
# ## 1. Preprocessing
#
# Reuse the estimator's preprocessing rather than reimplementing it, but with a
# **single ensemble member**. TabICL defaults to 8 members averaged over
# normalizations and feature/class shuffles; averaging destroys the per-sample
# attribution that is the entire point here. The paper compares against
# "TabICL (single)" for the same reason.
#
# `fit()` does not train anything — it loads the checkpoint and fits the
# preprocessing. The model is only run at predict time.

# %%
clf = TabICLClassifier(
    n_estimators=1,
    norm_methods=["none"],  # try "power" or "quantile" if accuracy is poor
    feat_shuffle_method="none",
    class_shuffle_method="none",
    device=DEVICE,
    random_state=0,
    kv_cache=False,
)
clf.fit(X_train, y_train)

# The ensemble generator runs *after* TabICL's numeric encoder, so it must be fed
# encoded input -- this is exactly what `predict_proba` does internally. Passing
# raw X works for all-numeric arrays and raises on string or categorical columns.
data = clf.ensemble_generator_.transform(clf.X_encoder_.transform(X_test), mode="both")
norm_method = next(iter(data))
X_ens, y_ens = data[norm_method]  # (1, n+m, H), (1, n)

n_train, n_test = len(X_train), len(X_test)
print(f"norm={norm_method} | X_ens {X_ens.shape} | y_ens {y_ens.shape}")

# Row order survives preprocessing, which is what lets a weight index be read
# back as a training row. Assert it rather than trusting it.
assert X_ens.shape[1] == n_train + n_test
assert np.array_equal(y_ens[0], clf.y_encoder_.transform(y_train))

X_t = torch.from_numpy(np.asarray(X_ens)).float().to(DEVICE)
y_t = torch.from_numpy(np.asarray(y_ens)).float().to(DEVICE)

# %% [markdown]
# ## 2. Attach the kernel head
#
# `d_model` comes from the checkpoint, not from the defaults — read it off the
# decoder you are replacing. It is 512 for the current v2 checkpoint, matching
# the paper's `embed_dim * row_num_cls = 128 * 4`.
#
# `identity_init=True` makes `W` the identity, so the kernel acts directly on the
# pretrained in-context embeddings. That is the honest no-training baseline;
# fine-tuning `W` is what step D buys you.

# %%
model = clf.model_
d_model = model.icl_predictor.decoder[0].in_features
n_classes = clf.n_classes_
print("d_model:", d_model, "| n_classes:", n_classes)

model.kernel_head = KernelHead(
    d_model=d_model, d_k=d_model, kernel="gaussian", identity_init=True
).to(DEVICE)


def run(kernel, gamma=None, chunk=QUERY_CHUNK):
    """Predict with a given kernel/scale.

    Returns (accuracy, predicted labels, weights, mean relative perplexity).
    """
    model.kernel_head.kernel = kernel
    with torch.no_grad():
        probs, w = model.forward_kernel(
            X_t, y_t,
            gamma=gamma,
            query_chunk_size=chunk,
            inference_config=clf.inference_config_,
        )
    pred = clf.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
    return (pred == y_test).mean(), pred, w, relative_perplexity(w).mean().item()


acc, _, w, ppl = run("gaussian")
print(f"accuracy {acc:.4f} | weights {tuple(w.shape)} | rel. perplexity {100*ppl:.2f}%")

# %% [markdown]
# ## 3. Calibrate the kernel scale
#
# Expect that first relative perplexity to come back near 100%, i.e. the weights
# are almost uniform: accurate, but useless for inspection. The paper's default
# `gamma = 1/(2*sqrt(d_k))` assumes embeddings whose `W` was *trained* to a scale
# where that is selective. With `W = I` on the raw pretrained embeddings, typical
# squared distances are far larger, so the default kernel is far too flat.
#
# Calibration therefore is not optional in the untrained setting — it is what
# makes the head inspectable at all.
#
# The paper uses 5-fold CV on training data. Below is a cheaper single held-out
# split, which is enough to pick a scale at this dataset size. Note this sweeps
# on a *validation* split carved out of training data, never on `y_test`.

# %%
from sklearn.model_selection import train_test_split

GAMMA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 5.0]  # Table 7 + tail
K_GRID = [1, 4, 5, 16, 32, 64, 128, 256, 512, 1024]

Xa, Xb, ya, yb = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)

cal = TabICLClassifier(
    n_estimators=1, norm_methods=["none"], feat_shuffle_method="none",
    class_shuffle_method="none", device=DEVICE, random_state=0, kv_cache=False,
)
cal.fit(Xa, ya)
Xc_ens, yc_ens = next(iter(cal.ensemble_generator_.transform(
    cal.X_encoder_.transform(Xb), mode="both").values()))
Xc = torch.from_numpy(np.asarray(Xc_ens)).float().to(DEVICE)
yc = torch.from_numpy(np.asarray(yc_ens)).float().to(DEVICE)

# Embed once; the kernel scale never touches the embedding, so the whole grid is
# swept over cached embeddings. This is the difference between seconds and hours.
with torch.no_grad():
    R = cal.model_.row_interactor(
        cal.model_.col_embedder(Xc, y_train=yc, mgr_config=cal.inference_config_.COL_CONFIG),
        mgr_config=cal.inference_config_.ROW_CONFIG,
    )
    E_tr, E_va = cal.model_.icl_predictor.embed(R, yc, symmetric=True, query_chunk_size=QUERY_CHUNK)

head = KernelHead(d_model=d_model, d_k=d_model, identity_init=True).to(DEVICE)
print(f"mean embedding L2 norm: {E_tr.norm(dim=-1).mean():.3f}")
print(f"\n{'kernel':>10} {'scale':>8} {'val acc':>9} {'rel.PPL%':>10}")

results = []
for kernel, grid in (("gaussian", GAMMA_GRID), ("dot", GAMMA_GRID), ("knn", K_GRID)):
    head.kernel = kernel
    for g in grid:
        with torch.no_grad():
            probs, wv = head(E_tr, E_va, yc, num_classes=n_classes, gamma=g)
        pred = cal.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
        a = (pred == yb).mean()
        p = relative_perplexity(wv).mean().item()
        results.append((kernel, g, a, p))
        print(f"{kernel:>10} {g:>8} {a:>9.4f} {100*p:>10.2f}")

# Accuracy alone is the wrong selection rule here. Soft kernels tend to tie
# across most of the grid, and a plain argmax then returns the flattest scale --
# the one with ~100% perplexity and no inspectability at all. Break ties toward
# the sparser kernel: round accuracy to a resolution the validation split can
# actually resolve, then prefer lower perplexity.
TIE = 3  # decimal places; with ~2.3k validation rows, differences below this are noise


def selection_key(r):
    _, _, accuracy, perplexity = r
    return (round(accuracy, TIE), -perplexity)


best = {k: max((r for r in results if r[0] == k), key=selection_key) for k in ("gaussian", "dot", "knn")}
print("\nbest (accuracy, ties broken toward sparsity):")
for k, (_, scale, a, p) in best.items():
    print(f"  {k:>8}: scale={scale}, val acc={a:.4f}, rel.PPL={100*p:.2f}%")

# %% [markdown]
# ## 4. Test-set results at the calibrated scales
#
# Pay attention to the accuracy/perplexity pair, not accuracy alone. On a small
# synthetic check the kNN kernel reached the same accuracy as the soft kernels at
# ~1% relative perplexity versus ~50% — that is the paper's finding that kNN
# dominates in the sparse regime, and it is what makes the weights actually
# readable.

# %%
print(f"{'kernel':>10} {'scale':>8} {'test acc':>10} {'rel.PPL%':>10}")
final = {}
for kernel, (_, scale, _, _) in best.items():
    a, pred, w_k, p = run(kernel, gamma=scale)
    final[kernel] = dict(scale=scale, pred=pred, w=w_k, acc=a, ppl=p)
    print(f"{kernel:>10} {scale:>8} {a:>10.4f} {100*p:>10.2f}")

# Baseline for comparison. This runs the full stock model, so it is the slowest
# cell here — skip it if you are just iterating on the kernel.
print("\nstock TabICL (single, n_estimators=1):", (clf.predict(X_test) == y_test).mean())

# %% [markdown]
# ## 5. Inspect a prediction
#
# The payoff. For any test row, list the training rows that produced its
# prediction, with their weights. Weight index `i` is `X_train[i]` — verified by
# the assertion in section 1.

# %%
KERNEL = "knn"  # the sparse one is the readable one
w_cpu = final[KERNEL]["w"][0].cpu()  # (n_test, n_train); ~137 MB, keep it off GPU
pred = final[KERNEL]["pred"]

TEST_IDX = 0
row_w = w_cpu[TEST_IDX]
top = row_w.topk(10)

print(f"test row {TEST_IDX}: true={y_test[TEST_IDX]}, predicted={pred[TEST_IDX]}")
print(f"relative perplexity for this row: {100*relative_perplexity(row_w).item():.2f}%")

print(f"\n{'train row':>10} {'weight':>10} {'label':>8}")
for weight, idx in zip(top.values.tolist(), top.indices.tolist()):
    if weight > 0:
        print(f"{idx:>10} {weight:>10.4f} {y_train[idx]:>8}")

# Class mass: how much total weight backs each class for this prediction.
for c in np.unique(y_train):
    mass = row_w[torch.from_numpy(y_train == c)].sum()
    print(f"class {c}: total weight {mass:.4f}")

# %% [markdown]
# ## 6. Aggregate views worth a look
#
# Which training rows are influential overall, and which test predictions rest on
# a narrow base of evidence.

# %%
influence = w_cpu.sum(0)  # total weight each training row supplies
print("most influential training rows:", influence.topk(10).indices.tolist())
print("never used:", int((influence == 0).sum()), f"of {n_train}")

# Note: for kNN this is constant by construction (always k/n), so it only tells
# you something for the soft kernels -- there it identifies predictions resting
# on a narrow base of evidence, which are the ones worth reviewing by hand.
per_row_ppl = relative_perplexity(final["gaussian"]["w"][0].cpu())
print(f"\nper-prediction relative perplexity (gaussian): "
      f"min {100*per_row_ppl.min():.2f}%, median {100*per_row_ppl.median():.2f}%, "
      f"max {100*per_row_ppl.max():.2f}%")

# %% [markdown]
# ## Notes for your scale
#
# **Memory.** Symmetric mode issues `2*11690 + 2923 = 26,303` queries against
# 11,690 keys, roughly doubling the ICL transformer's cost versus stock TabICL.
# Measured on CPU at exactly your shape, the ICL stage costs ~650 MB — not a
# problem on any GPU. The heavy stages are column embedding and row interaction
# over 14,613 rows x 176 features, and `forward_kernel` routes those through
# TabICL's own `InferenceManager` via `inference_config`, exactly as the stock
# forward does. So you inherit TabICL's memory management for free.
#
# `query_chunk_size` exists for the opposite regime (huge test set, small
# context). At your shape it is counterproductive: the cached K/V for 12 blocks
# is ~570 MB, more than the unchunked pass it replaces. Hence `QUERY_CHUNK=None`.
#
# The weight matrix itself is `2923 x 11690` floats ≈ 137 MB. Fine on GPU, but
# section 5 moves it to CPU because you will be slicing it repeatedly.
#
# **The slow part** is column embedding, not the kernel. Section 3 embeds once
# and sweeps the whole grid over cached embeddings, which is why calibration is
# cheap here and expensive in the paper's Table 3.
#
# **If the chosen gamma lands at the top of the grid**, extend it upward. With
# `W = I` the useful range sits well above the paper's Table 7 grid, which was
# calibrated for a trained projection. That is a symptom of the missing
# fine-tuning, not a bug.
#
# **What is not done.** No fine-tuning (step D) — `W` is the identity and the
# backbone is frozen. Expect accuracy at or slightly below stock TabICL. The
# paper's numbers come from fine-tuning the embedding module and `W` end-to-end
# on 5,000 batches of synthetic prior data, which needs an A100.
