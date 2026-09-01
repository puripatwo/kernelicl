# KernelICL on TabICL

An implementation of **KernelICL** — *Interpretable Tabular Foundation Models via
In-Context Kernel Regression* (Miftachov, Charron & Valentin, arXiv:2602.02162) — on
top of this fork of TabICL, plus the tooling to apply it to a real dataset.

The paper has no public code. Everything here is an implementation of its equations,
not a port, so treat agreement with its numbers as unverified.

New to this? [SUMMARY.md](SUMMARY.md) explains what it is and what we
found, in plain language and with no background assumed.

## What KernelICL is

TabICL predicts by passing an in-context embedding through an MLP. KernelICL replaces
that MLP with a kernel, so every prediction becomes a weighted average of training
outcomes:

```
prediction(x) = Σᵢ wᵢ · outcome(xᵢ)        wᵢ ≥ 0,  Σᵢ wᵢ = 1
```

The `wᵢ` are not an estimate of influence. They are the coefficients in the
arithmetic that produced the number, so the explanation *is* the computation. That is
the difference from post-hoc attribution: a SHAP value can be unfaithful to the
model, this cannot be.

The cost is small. The paper reports 0.2 accuracy points across 55 benchmark
datasets; what you gain is a named set of past cases behind every prediction.

## Package changes

Two additions to `src/tabicl`, with 154 tests passing:

| Where | What |
|---|---|
| [`_model/kernel_head.py`](../src/tabicl/_model/kernel_head.py) | `KernelHead` — projection `W` (Eq. 15) then a Gaussian, dot-product or kNN kernel (Eq. 16–18); `relative_perplexity` (Eq. 19) |
| [`_model/learning.py`](../src/tabicl/_model/learning.py) | `ICLearning.embed()` — symmetric in-context embeddings (Eq. 14), with optional query chunking |
| [`_model/tabicl.py`](../src/tabicl/_model/tabicl.py) | `TabICL.forward_kernel()` — the KernelICL forward pass; `forward()` is untouched |

**Symmetric embeddings** are the non-obvious part. TabICL's attention derives keys
from `q[..., :train_size, :]`, so every position is a query but only the leading ones
are context. Training and test rows therefore get different treatment for identical
inputs, which leaves distance-based kernels meaningless. Symmetric mode concatenates a
second, label-free copy of the training rows into the query stream:

```
[ R_train + g(y_train) | R_train | R_test ]
  \______ context _____/ \_____ queries ____/
```

Training embeddings must *not* be read off the context positions instead — those carry
their own label, so a row's outcome would leak into the embedding used to retrieve it.
Cost: `2n + m` query positions instead of `n + m`, matching the paper's ~2× ceiling.

## The five scratch files

Written to be pasted into Colab cells. Only one dependency between them.

| File | What it does | Needs |
|---|---|---|
| `kernelicl_quickstart.py` | quickstart: the raw mechanics, no abstraction | data |
| `kernelicl_clinical.py` | **the core.** `fit_explainer` + `ClinicalExplainer` | — |
| `kernelicl_analysis.py` | T1–T4, F1/F3/F4/F7/F8 | clinical, data |
| `kernelicl_embeddings.py` | T5, T6, E1–E4 | clinical, data |
| `kernelicl_shift.py` | corruption testing: S1–S2, G1–G3 | clinical, data |
| `kernelicl_feature_corruption.py` | per-feature corruption sweep, F9 | clinical, data, fitted models |
| `kernelicl_finetune.py` | trains the embedding for the kernel | — |

### Setup

Set the runtime to a GPU **before** installing — changing runtime type afterwards
destroys the VM.

```
!git clone -b kernelicl-head https://github.com/puripatwo/kernelicl.git
%cd kernelicl
!pip install -q -e ".[pretrain]" umap-learn
```

Then define `X_train`, `y_train`, `X_test`, `y_test` and paste whichever file you
want. `X` may be a DataFrame with string columns and NaNs. Each file's usage block
sits at its bottom, commented out.

## Reading the output

### The four derived numbers

| | Meaning |
|---|---|
| **agreement** | Share of evidence weight on the predicted class. Also *is* the predicted probability — under a kernel head the probability is a share of evidence, not a squashed logit. |
| **evidence base** | Effective number of past cases behind a prediction. Perplexity (Eq. 19) in absolute units, because "6 comparable cases" is legible and "0.05% relative perplexity" is not. |
| **nearest distance** | How far the case sits from the closest thing the model has seen. Drives the novelty flag. |
| **rel.PPL %** | Evidence base as a fraction of the training set. The paper's units. |

### Tables and figures

Set `SAVE_FIGURES = "figures"` at the top of any of the three analysis files to write
every figure as PNG *and* PDF at 300 dpi as it is drawn. The PDF is vector, so it is
the one to place in a thesis. Titles are written for a reader who has not seen this
README — the `F1`/`E1`/`G1` codenames below are for navigating the files only and do
not appear on the plots.

| | Answers | Read it for |
|---|---|---|
| **T1** | kernel × scale → metric, sparsity, time | how expensive sparsity is on your data |
| **T2** | KernelICL vs stock TabICL | what interpretability costs |
| **T3 / F4** | which features the metric tightens on | the finding about *your* domain |
| **T4** | learned vs input-space metric at matched `k` | whether the embedding earns its keep |
| **F1** | metric vs inspectability | **read first** — sets the operating point |
| **F3** | per-prediction weight distribution | how much evidence, and how concentrated |
| **F7** | weights in embedding space | where that evidence sits |
| **F8** | the six-panel evidence figure | one figure for a write-up |
| **F9** | four corruptions x three views of the evidence | *why* a corruption costs accuracy |
| **T5 / T6 / E1** | purity, raw vs `TF_row` vs `TF_icl` | whether to fine-tune |
| **E2** | test cases over training, and error locations | extrapolation, and clustered failure |
| **E3** | evidence base across the space | sanity check on the scale |
| **E4** | feature gradients | cross-check on T3 |
| **S1** | detection rate, clean vs corrupted | **did the model notice?** |
| **S2** | how far the evidence moved | whether the feature was being used |
| **G1–G3** | confidence vs unfamiliarity, paired shift, cohort movement | where silent failures sit |

**T2's TabICL-MLP row needs a checkpoint.** In the paper it is the same architecture
fine-tuned with an MLP head, so without fine-tuning it is bit-identical to TabICL
(single) and prints as `-`. Set `FINETUNED` to a `kernelicl_finetune` checkpoint and
the row fills in; the kernel rows then use those weights too, so the comparison is
like-for-like. Note that `FINETUNED` changes what "TabICL-MLP" and the KernelICL rows
mean but never the two stock TabICL rows, which are always built from the released
checkpoint.

**T2 reports seven metrics plus wall-clock time.** On imbalanced binary outcomes read
**MCC** first: it is the only one of these that a majority-class predictor cannot
inflate, and it moves only when both classes are handled. **Balanced accuracy** stays
as the headline because it is the paper's metric and drives the calibration.
**AUROC/AUPRC** grade the ranking rather than the decision, so they answer "could a
different threshold do better" — AUPRC is the one to watch when positives are rare.
**Sensitivity/specificity** are the two numbers a clinical reader will ask for
directly. **F1** is included because it is conventional, but it ignores true negatives
and depends on which class you call positive, so MCC is the better summary.

**T3's columns are normalized within each method**, so they show relative *emphasis*,
not absolute closeness. KernelICL neighbourhoods are typically *wider* in raw feature
space — it selects on the embedding, trading raw closeness for closeness on whatever
it considers diagnostic. The output prints the absolute level alongside.

**T1's scale column runs in opposite directions** for the two kernel families: for
Gaussian and dot-product it is `γ`, where larger is sharper and gives *fewer*
effective neighbours; for kNN it is `k` itself. Read `rel.PPL %`.

## Corruption testing

`kernelicl_shift.py` compares a clean test set against a corrupted one. The usual
version of this test reports an accuracy drop, which tells you the model is sensitive
but not whether it **knew**. Degrading from 96% to 81% while flagging most of the new
errors is graceful; degrading identically while staying confident is dangerous,
because in the field nobody catches the difference.

So the headline is a 2×2 of correct/wrong against flagged/not-flagged:

- **detection rate** — of the errors it made, what share did it flag. Holding up under
  corruption means graceful degradation; collapsing means errors arrive unannounced.
- **silent failures** — wrong, with no warning attached. The operational cost.
- **evidence overlap@k** — how much of each case's evidence survived. Near 1.0 means
  the corruption barely changed which past cases were consulted, so the model was not
  relying on that feature; near 0 means it was.

Four corruptions ship, and they destroy different things, which changes what a drop
means. `corrupt_constant` collapses a column to one value — for a Gender column that
records everyone as the commonest value. `corrupt_replace` rewrites specific values
(`{"female": "male"}`) and leaves other categories intact. `corrupt_shuffle` permutes
a column, keeping the distribution but breaking the per-row link. `corrupt_copy`
overwrites one field with another. Shuffle is the cleaner test of "does the model use
this feature", because a collapse changes both the information and the distribution
and a drop could be either.

This is valid because **corrupting the test set cannot move the training embeddings**:
column statistics attend to training rows only (`embed_with_test=False`) and the ICL
keys are the training context, so the reference library is fixed and every difference
is attributable to the corruption. `for_test_set()` asserts that invariant rather than
trusting it, and carries `gamma` and the novelty threshold across unchanged —
recalibrating on the corrupted set would change two things at once.

## Findings from building this

Things measurement contradicted, kept here so they are not rediscovered.

**Train-side embeddings encode their own labels.** A training row sits in its own
context, so its query attends to its own key — at distance zero, maximum attention —
and that key carries `g(yᵢ)`. A linear probe recovers a training row's own outcome at
**1.000**, against 0.877 from the row representation before labels. This is the
paper's Eq. 14, not a defect, and test predictions are unaffected. But it makes every
train-side diagnostic vacuous: train-side purity is 1.000 at any `k`. It is also why
novelty cannot be calibrated on training-internal distances.

**Novelty needs a held-out reference.** Calibrated against training-internal
nearest-neighbour distances, 65% of a cohort was flagged: a training point has
neighbours in a set it belongs to, a held-out point does not. Median held-out distance
(0.561) exceeded the training 99th percentile (0.518). With a held-out reference:
1.0%.

**Almost all separation comes from `TF_icl`.** Test-side purity@5 measured 0.829 for
raw features, **0.829** for `TF_row`, and 0.972 for `TF_icl`. The feature encoder makes
outcomes no more locally separable than plain standardized features; the geometry a
kernel reads from is manufactured by in-context learning against a labelled context.

**Calibrating on accuracy alone is actively harmful.** Accuracy is near-flat across
most of the grid while the evidence base varies by orders of magnitude, so `argmax`
returns the flattest kernel — accurate, and averaging over the whole training set, so
nothing explains anything. `accuracy_tolerance` (default 0.01) takes the sparsest
scale within one point of the best. On one run that moved perplexity from 98.9% to
30.6% at no cost. **The paper does not specify a tie-break**; ours compensates for an
untrained `W` and may be unnecessary after fine-tuning.

**Label auditing must not filter by influence.** A record whose stored outcome
contradicts its neighbourhood is pushed *away* in the label-conditioned embedding, so
it carries less weight than average — 0.128 against 0.300 measured. Filtering at one
average record's influence cut recovery of seeded bad labels from 41/60 to 5/60.
Ranking by error *share* rather than raw weight-on-errors doubled recovery again
(21/30 vs 10/30, chance 2.6).

**Chunked queries do not help at typical scale.** Measured at n=11,690 / m=2,923: the
unchunked ICL stage costs ~650 MB and chunking is *worse*, because the 12-block K/V
cache (~570 MB) outweighs the activations it replaces. It pays off only when queries
vastly outnumber context rows. `query_chunk_size` defaults to None.

**`d` is nearly unusable.** Column feature grouping asserts `d is None`, which is why
TabICLv2 training ignores per-dataset feature counts and treats padded columns as
real. `forward_kernel` accepts `d` but needs `col_feature_group=False`.

## Fine-tuning

`W = I` reads a geometry optimised for an MLP head, which has no reason to make
Euclidean distance mean similarity. Fine-tuning reshapes it.

One step shows the model 64 **separate synthetic tabular problems** from TabICL's
prior, each with its own context/query split, and asks that a plain weighted average
of context outcomes get the queries right. Across thousands of unrelated problems that
produces a representation where nearest-neighbour voting works in general. Your own
data is never involved.

### Starting from TabICLv2

`checkpoint` alone is not the whole change. `d_model` (512 in both), the class count
and every module path are read from the checkpoint, so the model side needs nothing.
But the two were pretrained on **different priors**, and `prior_type` should follow:

| | checkpoint | prior_type |
|---|---|---|
| paper / v1 | `tabicl-classifier-v1-20250208.ckpt` | `mlp_scm` — Appendix A's "random MLP functions" |
| v2 | `tabicl-classifier-v2-20260212.ckpt` | `graph_scm` |

```python
cfg = FinetuneConfig(**{**PRESETS["medium"].__dict__,
                        "checkpoint": V2_CHECKPOINT, "prior_type": "graph_scm"})
```

Their configs otherwise differ in ways that matter only inside TabICL: v1 has
`col_feature_group=False` and `col_target_aware=False`, v2 has grouping on, target-aware
column embedding, SSMax and non-interleaved RoPE. Both verified end to end here.

Do not copy v2's stage-2/3 recipe (`max_seq_len` 10240/60000, `seq_len_per_gp=True`):
those are its long-context stages, and `seq_len_per_gp` returns nested tensors the
training loop here does not handle.

| preset | GPU | for |
|---|---|---|
| `paper` | A100 / 40 GB | Appendix A verbatim, 5000 × 64 |
| `medium` | 24 GB | a real run |
| `small` | T4 / 16 GB | does the loss move at all |

Run `smoke_test()` first — two steps on tiny batches, and it caught three real bugs
during development.

The best checkpoint is rewritten **every time validation improves**, atomically (temp
file then rename), so an interrupted run keeps whatever it reached and a half-written
file cannot replace a good one. Point `out_path` at mounted Drive on Colab. If a
session dies, `resume_from` continues from the saved checkpoint — a warm restart,
since optimiser state and the schedule are not stored.

### If it is slow

Every log line reports `% waiting on data` and hours remaining, both over the window
since the last log rather than since the start — worker spin-up makes a cumulative
average understate throughput badly in the first few hundred steps.

1. **High `% waiting on data`** → generation is the bottleneck. Raise `prior_n_jobs`
   and `prefetch_factor`. Batches are already built in background dataloader workers
   and prefetched, one batch per step split into micro-batches afterwards, mirroring
   TabICL's own trainer; before that they were generated inline with a fresh process
   pool per micro-batch, which measured 0.05 step/s on an A100 against 0.21 after.
2. **Near-zero `% waiting on data`** → the GPU is the bottleneck, and `micro_batch` is
   the lever: fewer, larger launches for the same effective batch.
   `benchmark_micro_batch()` times a few steps at each size and reports peak memory,
   so you can pick the fastest that fits rather than guessing.
3. **Only then** reduce `max_features` — column embedding cost scales with it, and it
   is the largest deviation from Appendix A you can make for the smallest loss.

`recompute=True` trades speed for memory, so leave it off unless you OOM.

Three details that matter:

- **NLL, not cross-entropy.** The head returns probabilities; `F.cross_entropy` would
  apply a second softmax and flatten the gradients.
- **Watch validation loss, ignore train loss.** Every step draws a fresh random
  problem of varying difficulty. In one verification run train loss rose 0.75 → 0.90
  while validation fell 0.5721 → 0.5536 → 0.5512. A baseline validation runs before
  training so each later number carries `(+x vs baseline)` — without it a falling
  curve is only relative to the first checkpoint taken, not to the pretrained model.
- **`rel.PPL` falling is the geometry sharpening.** Training holds the kernel scale
  fixed, so the embedding has to adapt to it; a fall from ~100% toward 60% and below
  is the projection learning to make distance mean something. It is not directly
  comparable to the paper's 28.6%, which is measured after per-dataset calibration.
- **kNN is never trained** — it is non-differentiable, so train Gaussian and swap the
  kernel at evaluation, as §4.1 does.

### Run it once

Fine-tuning is **dataset-independent** — it samples synthetic problems from TabICL's
prior and never sees your data. The result is a general-purpose backbone, exactly like
the original pretraining, so it is a one-time cost: keep the ~110 MB checkpoint and
reuse it for any dataset, in any later session, indefinitely.

What still runs each session is `fit_explainer`, which embeds *your* data and
cross-validates the kernel scale. Minutes, not hours.

Re-run the fine-tune only to change what it produced: a different `d_k`, a
dot-product head (kNN reuses the Gaussian one), a different starting checkpoint, or
more steps if the validation curve had not flattened. `describe_checkpoint(path)`
prints what produced a saved file, which a checkpoint tends to outlive the memory of.

Then `fit_explainer(..., finetuned=path)` uses it everywhere, recalibrating the scale.
That recalibration is not optional: on a short run the calibrated evidence base moved
from ~130 cases to ~1, so scales do not transfer between geometries.

## Clinical use

`ClinicalExplainer` exists because a weight matrix is not usable by a clinician. Six
things it supports, and the six that map onto them:

| Inspection | Output |
|---|---|
| Case-based reasoning | `case(i)`, F3, F7 |
| Triage by evidence | `triage()`, `triage_summary()` |
| Extrapolation detection | the novelty flag in `triage()`, E2 |
| Label auditing | `audit_labels()` |
| Equity checking | `equity()` |
| Validating the similarity metric | `feature_emphasis()`, T3/F4 |

Only the first and last appear in the paper (Figure 2, Table 5). The other four are
things the kernel form enables but the paper does not explore — the paper computes
perplexity and then only ever averages it.

**Use `knn` for cards, `gaussian` for everything else.** `with_kernel()` switches for
the cost of a matmul. kNN's selected cases carry *all* of the evidence, so a card
reads "these 5 past cases"; a soft kernel may spread weight so thinly that the five
most similar cases explain under 10% of the decision. But a sparse kernel never
selects most records, so it cannot support the label audit at all.

**Thresholds are yours to set.** `min_agreement=0.80` and `novelty_quantile=0.99` are
starting points, not validated values. They decide how many cases reach a human. In
screening a missed positive usually costs far more than a review, which argues for a
higher agreement threshold. Changing them re-runs no model.

**Name your classes.** `CLASS_LABELS = {0: "Adherence", 1: "Non-adherence"}` in the
analysis file puts those words on axes, legends and titles instead of `0` and `1`.
`POSITIVE_LABEL` tells the asymmetric metrics (sensitivity, F1, AUPRC) which class is
the event; it defaults to the rarer one, which is usually right.

**Use balanced accuracy, not accuracy.** `METRIC` in the analysis file. On an 85/15
split, low-γ configurations scored 0.500 balanced where plain accuracy read 0.85 —
they were predicting the majority class for everything. The metric feeds calibration,
so it changes which scale is selected.

## Limits

- **Faithful is not correct.** A wrong prediction gets a clean, readable, wrong
  explanation.
- **Associational, not causal.** "Similar to these cases", never "because of X".
- **The leave-one-out identity `ŷ − ŷ₋ⱼ = wⱼ(yⱼ − ŷ)/(1 − wⱼ)` is exact for the kernel
  step**, first-order end to end — removing a row also perturbs the embeddings.
- **Correlated features get arbitrary credit in T3.** Read groups, not ranks.
- **Not validated clinical software.** An analysis tool for deciding whether the
  approach is worth pursuing.
- **Numbers quoted above come from synthetic data** on CPU at reduced scale, and the
  fine-tuning loop is verified to learn over 60 steps, not to reproduce the paper.
