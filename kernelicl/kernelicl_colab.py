"""Quickstart: the raw KernelICL mechanics on your own data.

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

# Gaussian, so weights are graded and visibly differ between cases. The kNN kernel
# gives every selected case exactly 1/k by definition, which reads well on a finished
# case card but makes this demo look broken -- the whole point here is that the
# weights vary.
KERNEL, SCALE = "gaussian", 5.0
TOP_K = 5

# Weights are indexed positionally against training rows, so a pandas Series with a
# non-default index would silently do label lookup and report the wrong rows.
y_train = (y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)).ravel()
y_test = (y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)).ravel()

print(f"device={DEVICE} | train {X_train.shape} | test {X_test.shape}")

# One ensemble member and no shuffling: TabICL averages 8 by default, and averaging
# destroys the per-case attribution this is all for.
clf = TabICLClassifier(n_estimators=1, norm_methods=["none"], feat_shuffle_method="none",
                       class_shuffle_method="none", device=DEVICE, random_state=0)
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

# The weights genuinely differ per case. Worth confirming rather than assuming,
# because on an imbalanced cohort the top neighbours often share an outcome and the
# output can look constant when it is not.
neighbour_sets = [tuple(sorted(weights[c].topk(TOP_K).indices.tolist()))
                  for c in range(len(y_test))]
print(f"distinct top-{TOP_K} neighbour sets: {len(set(neighbour_sets)):,} "
      f"across {len(neighbour_sets):,} test cases")

# Pick cases worth looking at rather than the first few: with an imbalanced cohort
# the leading rows are nearly all majority class, so an outcome column full of one
# value says more about the ordering than about the model.
picks = [int(weights.max(1).values.argmax())]          # most concentrated prediction
for cls in np.unique(pred):
    matching = np.flatnonzero(pred == cls)
    if len(matching):
        picks.append(int(matching[0]))                 # one of each predicted class
picks = list(dict.fromkeys(picks))[:3]

# Every prediction is a weighted average of training outcomes, and w holds the
# weights. Reading one row of w is the whole interpretability story.
for case in picks:
    top = weights[case].topk(min(TOP_K, weights.shape[1]))
    print(f"\ntest case {case}: true={y_test[case]}, predicted={pred[case]}, "
          f"evidence base {perplexity[case] * len(y_train):.1f} cases")
    print(f"{'train row':>10} {'weight':>9} {'outcome':>10}")
    for weight, row in zip(top.values.tolist(), top.indices.tolist()):
        if weight > 0:
            print(f"{row:>10} {weight:>9.4f} {y_train[row]:>10}")
    # Total weight per outcome. This is what the prediction actually rests on, and
    # explains an all-one-value column above: it means the evidence is one-sided.
    mass = {str(c): float(weights[case][torch.from_numpy(y_train == c)].sum())
            for c in np.unique(y_train)}
    print("   weight by outcome:", {k: round(v, 3) for k, v in mass.items()})

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
