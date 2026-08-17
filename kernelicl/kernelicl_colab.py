"""Quickstart: the raw KernelICL mechanics on your own data, in about 60 lines.

Shows the whole path explicitly -- preprocess, embed, apply a kernel, read the
weights -- with no abstraction in the way. For real work use kernelicl_clinical.py,
which adds cross-validated calibration, triage, auditing and equity checks.

Independent: needs only X_train, y_train, X_test, y_test in the session.
"""

import numpy as np
import torch

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import KernelHead, relative_perplexity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KERNEL, SCALE = "knn", 5   # 5 nearest neighbours: the most legible starting point
CHECKPOINT = None          # None = TabICL's default (v2)
                           # "tabicl-classifier-v1-20250208.ckpt" for the paper's

# Weights are indexed positionally against training rows, so a pandas Series with a
# non-default index would silently do label lookup and report the wrong rows.
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()

print(f"device={DEVICE} | train {X_train.shape} | test {X_test.shape}")

# One ensemble member and no shuffling: TabICL averages 8 by default, and averaging
# destroys the per-case attribution this is all for.
clf = TabICLClassifier(n_estimators=1, norm_methods=["none"], feat_shuffle_method="none",
                       class_shuffle_method="none", device=DEVICE, random_state=0,
                       **({"checkpoint_version": CHECKPOINT} if CHECKPOINT else {}))
clf.fit(X_train, y_train)

# The ensemble generator sits after TabICL's numeric encoder, so encode first;
# passing raw X raises on string columns.
encoded = clf.X_encoder_.transform(X_test)
X_ens, y_ens = next(iter(clf.ensemble_generator_.transform(encoded, mode="both").values()))
assert np.array_equal(y_ens[0], clf.y_encoder_.transform(y_train)), "row order changed"

X_t = torch.from_numpy(np.asarray(X_ens)).float().to(DEVICE)
y_t = torch.from_numpy(np.asarray(y_ens)).float().to(DEVICE)

d_model = clf.model_.icl_predictor.decoder[0].in_features
clf.model_.kernel_head = KernelHead(d_model=d_model, d_k=d_model, kernel=KERNEL,
                                    identity_init=True).to(DEVICE)

with torch.no_grad():
    probs, w = clf.model_.forward_kernel(X_t, y_t, gamma=SCALE,
                                         inference_config=clf.inference_config_)

pred = clf.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
weights = w[0].cpu()
perplexity = relative_perplexity(weights).numpy()

print(f"\nkernel={KERNEL} scale={SCALE}")
print(f"accuracy        {(pred == y_test).mean():.4f}")
print(f"weights         {tuple(weights.shape)}  (test cases x training cases)")
print(f"evidence base   {perplexity.mean() * len(y_train):.1f} of {len(y_train):,} cases")

# Every prediction is a weighted average of training outcomes, and w holds the
# weights. Reading one row of w is the whole interpretability story.
CASE = 0
top = weights[CASE].topk(min(SCALE, weights.shape[1]))
print(f"\ntest case {CASE}: true={y_test[CASE]}, predicted={pred[CASE]}")
print(f"{'train row':>10} {'weight':>9} {'outcome':>10}")
for weight, row in zip(top.values.tolist(), top.indices.tolist()):
    if weight > 0:
        print(f"{row:>10} {weight:>9.4f} {y_train[row]:>10}")

influence = weights.sum(0)
print(f"\ntraining rows never used by any prediction: "
      f"{int((influence == 0).sum()):,} of {len(y_train):,}")


# --------------------------------------------------------------------------- #
# Next steps
# --------------------------------------------------------------------------- #
# The scale above is a guess. Calibrate it, and get triage, label auditing,
# equity checks and readable case cards, with kernelicl_clinical.py:
#
#   ex = fit_explainer(X_train, y_train, X_test)
#   print(ex.with_kernel("knn", gamma=5).case(0))
#   ex.triage()
#
# kernelicl_analysis.py    tables and figures  (T1-T4, F1/F3/F4/F7)
# kernelicl_embeddings.py  what the representation looks like
# kernelicl_finetune.py    train the embedding for the kernel
