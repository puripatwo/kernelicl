"""Clinician-facing layer over KernelICL weights.

The weight matrix from :meth:`tabicl._model.tabicl.TabICL.forward_kernel` is the
raw material for six things a deployed screening model needs: an auditable
per-case explanation, a triage signal, extrapolation detection, label auditing,
equity checking, and pre-deployment validation of what the model treats as
similar. This module turns the matrix into those six outputs, in record IDs and
plain language rather than tensors.

Nothing here re-runs the model. Everything is a view of weights already computed,
so it is cheap and can be re-run with different thresholds interactively.

Usage
-----
Paste this whole file into a Colab cell (or ``%run -i kernelicl_clinical.py``),
then one call does everything -- fits the model, calibrates the kernel scale on a
held-out split, and calibrates the novelty threshold::

    ex = fit_explainer(
        X_train, y_train, X_test,
        train_ids=patient_ids_train,     # optional; defaults to row numbers
        test_ids=patient_ids_test,
    )

Every method **returns** a value rather than printing one, so assign it and then
display or print it. Two return types, and they are displayed differently:

*Text* -- ``case()`` and ``report()`` return a string, so ``print()`` them, or the
newlines show up as ``\\n``::

    cards = ex.with_kernel("knn", gamma=5)   # see below
    print(cards.case(0))                     # one case, as a readable card
    print(ex.report(n_cases=3))              # caseload summary + worked examples

*Tables* -- everything else returns a ``pandas.DataFrame``. In a notebook, put one
on the last line of a cell to render it, or call ``display()`` for several in the
same cell::

    triage = ex.triage()                 # one row per case, review priority first
    triage.head(20)                      # last line of the cell -> renders

    summary = ex.triage_summary()        # how the caseload splits by action
    audit   = ex.audit_labels(y_test)    # candidate mislabelled training records
    equity  = ex.equity(site_train, site_test)
    emphasis = ex.feature_emphasis()     # what the model treats as "similar"

    from IPython.display import display  # to show several at once
    display(summary, audit.head(10), equity)

    paths = ex.export("report", y_test=y_test)   # writes CSVs, returns the paths
    print(paths)

For the per-case cards a clinician reads, switch to the kNN kernel. It costs a
matrix multiply, not another model run, and it is the difference between "these
5 past cases account for the decision" and "the 5 most similar cases account for
9% of it"::

    cards = ex.with_kernel("knn", gamma=5)
    print(cards.case(0))

``y_test`` is needed only for the label audit and the CSV export -- prediction,
triage, equity and feature emphasis never see it.

Returns at a glance
-------------------
===============================  ==========================================
``fit_explainer(...)``           ``ClinicalExplainer``
``ex.with_kernel(...)``          ``ClinicalExplainer``
``ex.case(i)``                   ``str``  -- print it
``ex.report(...)``               ``str``  -- print it
``ex.triage()``                  ``DataFrame``, one row per test case
``ex.triage_summary()``          ``DataFrame``, one row per action
``ex.neighbours(i)``             ``DataFrame``, the cases behind prediction i
``ex.audit_labels(y_test)``      ``DataFrame``; ``.attrs["n_errors"]`` holds the
                                 number of mistakes it had to work from
``ex.influence()``               ``DataFrame``, one row per training record
``ex.equity(gtr, gte)``          ``DataFrame``, one row per group
``ex.feature_emphasis()``        ``DataFrame``, one row per feature
``ex.export(dir, y_test=...)``   ``list[str]`` of written CSV paths
===============================  ==========================================

If you already have a fitted classifier and embeddings (say from
``kernelicl_analysis.py``), construct :class:`ClinicalExplainer` directly instead;
``fit_explainer`` is only a convenience wrapper around it.

Scope and limits
----------------
The weights are *faithful*: each is the actual coefficient in the weighted
average that produced the prediction, not an estimate of influence. They are not
causal — "similar to these cases", never "because of this feature". And a
confidently wrong prediction produces a clean, readable, wrong explanation;
faithfulness is not correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import KernelHead, relative_perplexity, squared_distances

__all__ = ["ClinicalExplainer", "fit_explainer"]

# Scale grids for calibration. Wider than the paper's Table 7 at the top end
# because an untrained projection leaves the embeddings on a different scale, so
# the useful range sits higher.
GAMMA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 5.0, 10.0]
K_GRID = [1, 4, 5, 16, 32, 64, 128, 256, 512, 1024]

_BAR = "█"


def _bar(share: float, width: int = 12) -> str:
    return (_BAR * int(round(share * width))).ljust(width, "·")


def _as_array(a):
    return (a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)).ravel()


def _embed(fitted_clf, X_query, return_row_repr: bool = False):
    """Symmetric in-context embeddings for a fitted classifier and a query set.

    The ensemble generator sits after TabICL's numeric encoder, so the query has
    to be encoded first -- this mirrors what ``predict_proba`` does internally.
    Passing raw X works for all-numeric arrays and raises on string columns.

    With ``return_row_repr`` also returns the row-stage representation, the output
    of column embedding plus row interaction before any label is added. That is the
    model's view of a case from its features alone, and unlike the in-context
    embedding it is free of per-row label leakage, which makes it the honest
    "before in-context learning" baseline.
    """
    encoded = fitted_clf.X_encoder_.transform(X_query)
    X_ens, y_ens = next(iter(fitted_clf.ensemble_generator_.transform(encoded, mode="both").values()))
    device = next(fitted_clf.model_.parameters()).device
    X_t = torch.from_numpy(np.asarray(X_ens)).float().to(device)
    y_t = torch.from_numpy(np.asarray(y_ens)).float().to(device)
    m = fitted_clf.model_
    with torch.no_grad():
        R = m.row_interactor(
            m.col_embedder(X_t, y_train=y_t, mgr_config=fitted_clf.inference_config_.COL_CONFIG),
            mgr_config=fitted_clf.inference_config_.ROW_CONFIG,
        )
        E_train, E_test = m.icl_predictor.embed(R, y_t, symmetric=True)
    if return_row_repr:
        return E_train, E_test, R
    return E_train, E_test


def _make_clf(device, norm_method, random_state):
    """One ensemble member, no shuffling -- so a weight index is a training row.

    TabICL defaults to averaging 8 members over different normalizations and
    feature/class shuffles. Averaging destroys the per-case attribution that this
    whole module exists to provide, so it is switched off.
    """
    return TabICLClassifier(
        n_estimators=1, norm_methods=[norm_method],
        feat_shuffle_method="none", class_shuffle_method="none",
        device=device, random_state=random_state, kv_cache=False,
    )


def fit_explainer(
    X_train,
    y_train,
    X_test,
    *,
    kernel: str = "gaussian",
    gamma: Optional[float] = None,
    train_ids: Optional[Sequence] = None,
    test_ids: Optional[Sequence] = None,
    feature_names: Optional[Sequence] = None,
    device: Optional[str] = None,
    norm_method: str = "none",
    calibrate: bool = True,
    keep_row_repr: bool = False,
    accuracy_tolerance: float = 0.01,
    val_size: float = 0.2,
    random_state: int = 0,
    verbose: bool = True,
    **explainer_kwargs,
) -> "ClinicalExplainer":
    """Fit everything from raw data and return a ready explainer.

    Runs three things you would otherwise have to wire together: the TabICL
    embedding, calibration of the kernel scale on a held-out split, and
    calibration of the novelty threshold on that same split.

    Parameters
    ----------
    X_train, y_train, X_test : array-like
        Your data. ``X`` may be a DataFrame with string or categorical columns
        and NaNs -- TabICL handles all three. ``y_test`` is deliberately not a
        parameter: nothing here may see it.

    kernel : {"gaussian", "dot", "knn"}, default="gaussian"
        ``"gaussian"`` grades neighbours by similarity, so a case card can rank
        them. ``"knn"`` weights its k selected cases equally, which reads as
        "these exact k cases" -- cleaner to state, less informative to inspect,
        and it cannot support the label audit because most records are never
        selected at all.

    gamma : float, optional
        Kernel scale. Left as None it is calibrated on held-out data, which is
        the right default: the scale controls how many past cases each prediction
        draws on, and an uncalibrated one typically spreads weight over the whole
        training set, giving predictions that are accurate but carry no usable
        evidence.

    keep_row_repr : bool, default=False
        Also keep the row-stage representation as ``ex.R_train`` / ``ex.R_test``:
        column embedding plus row interaction, before any label is added. Costs
        about 30 MB per 15,000 cases and nothing in compute, since it is computed
        on the way to the in-context embedding regardless.

    calibrate : bool, default=True
        Whether to hold out a split. Turning this off skips one embedding pass,
        but then ``gamma`` must be supplied and the novelty flag falls back to a
        cohort-relative reading rather than a training-range one.

    accuracy_tolerance : float, default=0.01
        How much held-out accuracy to trade for inspectability. Among scales
        within this much of the best, the sparsest is chosen. This is the single
        most consequential knob in the module: at 0.0 you get the most accurate
        model, whose predictions typically average over the entire training set
        and cannot be explained by any small set of past cases. At 0.01 you give
        up at most one accuracy point and usually get an evidence base one or two
        orders of magnitude smaller. Raise it if cards are still too diffuse to
        read; set it to 0.0 if accuracy is all you care about, in which case you
        probably do not need this module.

    norm_method : str, default="none"
        TabICL feature normalization: ``"none"``, ``"power"``, ``"quantile"``,
        ``"quantile_rtdl"``, ``"robust"``. With many real-world features
        ``"power"`` or ``"quantile"`` often matters more than the kernel choice;
        it does not affect interpretability, since row identity is untouched.

    Returns
    -------
    ClinicalExplainer
    """
    def say(msg):
        if verbose:
            print(msg)

    y_train = _as_array(y_train)
    if feature_names is None and hasattr(X_train, "columns"):
        feature_names = list(X_train.columns)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not calibrate and gamma is None:
        raise ValueError("gamma must be supplied when calibrate=False")

    say(f"fitting on {len(y_train):,} training cases, scoring {len(X_test):,} "
        f"(device={device}, kernel={kernel})")
    clf = _make_clf(device, norm_method, random_state)
    clf.fit(X_train, y_train)
    d_model = clf.model_.icl_predictor.decoder[0].in_features
    head = KernelHead(d_model=d_model, d_k=d_model, kernel=kernel, identity_init=True).to(device)

    reference_distances = None
    if calibrate:
        from sklearn.model_selection import train_test_split

        say("calibrating on a held-out split (one extra embedding pass)...")
        strat = y_train if len(np.unique(y_train)) < len(y_train) // 2 else None
        Xa, Xb, ya, yb = train_test_split(X_train, y_train, test_size=val_size,
                                          random_state=random_state, stratify=strat)
        cal = _make_clf(device, norm_method, random_state)
        cal.fit(Xa, ya)
        E_cal_train, E_cal_val = _embed(cal, Xb)
        y_cal = torch.from_numpy(cal.y_encoder_.transform(_as_array(ya))).float().to(device)[None]

        # The scale never touches the embedding, so the whole grid is swept over
        # embeddings computed once. This is what makes calibration cheap.
        if gamma is None:
            rows = []
            for scale in (K_GRID if kernel == "knn" else GAMMA_GRID):
                with torch.no_grad():
                    probs, w = head(E_cal_train, E_cal_val, y_cal,
                                    num_classes=cal.n_classes_, gamma=scale)
                pred = cal.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
                rows.append((scale, float((pred == _as_array(yb)).mean()),
                             float(relative_perplexity(w).mean())))
            # Picking the most accurate scale is the wrong rule here. Accuracy is
            # typically flat across much of the grid while the evidence base
            # varies by orders of magnitude, so a plain argmax returns a kernel
            # that averages over the whole training set -- accurate, and useless
            # to a clinician, since no small set of past cases explains anything.
            # Instead: among scales within `accuracy_tolerance` of the best, take
            # the sparsest. This buys inspectability at a bounded, stated cost.
            best_acc = max(r[1] for r in rows)
            candidates = [r for r in rows if r[1] >= best_acc - accuracy_tolerance]
            gamma, acc, ppl = min(candidates, key=lambda r: r[2])
            n_ctx = len(ya)
            say(f"  scale={gamma}  held-out accuracy={acc:.3f} "
                f"(best on grid {best_acc:.3f}, gave up {best_acc - acc:.3f} "
                f"within tolerance {accuracy_tolerance})")
            say(f"  evidence base ≈ {ppl * n_ctx:.0f} of {n_ctx:,} cases per prediction")

        reference_distances = ClinicalExplainer.reference_distances_from(head, E_cal_train, E_cal_val)
        say(f"  novelty threshold calibrated on {len(reference_distances):,} held-out cases")

    say("embedding the full training context...")
    R_train = R_test = None
    if keep_row_repr:
        E_train, E_test, R = _embed(clf, X_test, return_row_repr=True)
        n_ctx = len(y_train)
        R_train, R_test = R[:, :n_ctx], R[:, n_ctx:]
    else:
        E_train, E_test = _embed(clf, X_test)

    explainer = ClinicalExplainer(
        clf, head, E_train, E_test, y_train,
        kernel=kernel, gamma=gamma,
        train_ids=train_ids, test_ids=test_ids,
        X_train=X_train, X_test=X_test, feature_names=feature_names,
        reference_distances=reference_distances,
        **explainer_kwargs,
    )
    # The row-stage representation, for comparing what the features alone give
    # against what in-context learning adds (see kernelicl_embeddings.py).
    explainer.R_train, explainer.R_test = R_train, R_test
    return explainer


class ClinicalExplainer:
    """Turns a KernelICL weight matrix into clinician-usable outputs.

    Most users should call :func:`fit_explainer` instead of constructing this
    directly; it wires up the model, the calibration and the thresholds.

    Parameters
    ----------
    clf : TabICLClassifier
        The fitted classifier, used for its label encoder and class names.

    head : KernelHead
        The kernel head that produced the weights.

    E_train, E_test : Tensor
        Symmetric in-context embeddings, shape (1, n, d) and (1, m, d).

    y_train : array
        Training labels in their original form, length n.

    kernel : str, default="gaussian"
        Which kernel to explain with. ``"gaussian"`` gives graded similarity, so
        a case card can rank neighbours by how much each contributed.
        ``"knn"`` gives every selected case equal weight, which reads as "these
        exact k cases" — cleaner to state, less informative to inspect.

    gamma : float, optional
        Kernel scale. Use the value calibrated on held-out data.

    train_ids, test_ids : sequence, optional
        Record identifiers, so a clinician can look a case up in the source
        system. Defaults to row numbers.

    top_k : int, default=5
        How many comparable cases a card lists.

    min_agreement : float, default=0.80
        Below this share of evidence supporting the predicted class, a case is
        flagged for review. 0.80 is a starting point, not a validated threshold —
        set it from your own review capacity and tolerance for missed cases.

    reference_distances : array, optional
        Nearest-neighbour distances for *held-out* cases against the same
        training context, from :meth:`reference_distances_from`. Supplying this
        is what allows the novelty flag to mean "outside the population the
        model was fitted on".

        Without it the flag falls back to a cohort-relative reading — "unusual
        compared with the other cases in this batch" — which is well-defined and
        self-calibrating but is a different claim. The tempting shortcut,
        comparing against training-internal nearest-neighbour distances, is
        badly biased: a training point has neighbours in a set it belongs to and
        a held-out point does not, so almost every real case looks novel. On a
        700/200 split the median held-out distance already exceeded the 99th
        percentile of training-internal distances, flagging 65% of the cohort.

    novelty_quantile : float, default=0.99
        Quantile of the reference distribution above which a case is flagged.
    """

    def __init__(
        self,
        clf,
        head,
        E_train: torch.Tensor,
        E_test: torch.Tensor,
        y_train,
        *,
        kernel: str = "gaussian",
        gamma: Optional[float] = None,
        train_ids: Optional[Sequence] = None,
        test_ids: Optional[Sequence] = None,
        top_k: int = 5,
        min_agreement: float = 0.80,
        reference_distances: Optional[np.ndarray] = None,
        novelty_quantile: float = 0.99,
        X_train=None,
        X_test=None,
        feature_names: Optional[Sequence] = None,
    ):
        self.clf, self.head = clf, head
        self.kernel, self.gamma = kernel, gamma
        self.top_k, self.min_agreement = top_k, min_agreement
        # Kept so feature_emphasis() can be called with no arguments.
        self.X_train, self.X_test = X_train, X_test
        self.feature_names = feature_names

        self.y_train = _as_array(y_train)
        self.n, self.m = E_train.shape[1], E_test.shape[1]
        self.train_ids = np.asarray(train_ids if train_ids is not None else np.arange(self.n))
        self.test_ids = np.asarray(test_ids if test_ids is not None else np.arange(self.m))

        head.kernel = kernel
        y_ctx = torch.from_numpy(clf.y_encoder_.transform(self.y_train)).float().to(E_train.device)[None]
        with torch.no_grad():
            probs, w = head(E_train, E_test, y_ctx, num_classes=clf.n_classes_, gamma=gamma)
            H_train, H_test = head.embed(E_train), head.embed(E_test)
            # How far each case sits from the closest thing the model has seen.
            self._nn_test = squared_distances(H_test, H_train)[0].min(dim=-1).values.sqrt().cpu().numpy()

        if reference_distances is not None:
            self.novelty_basis = "held-out reference"
            self._novelty_label = "outside the range seen in training"
            ref = np.asarray(reference_distances).ravel()
        else:
            self.novelty_basis = "cohort-relative"
            self._novelty_label = "unusual compared with the rest of this cohort"
            ref = self._nn_test
        self.novelty_threshold = float(np.quantile(ref, novelty_quantile))
        # Kept so with_kernel() can re-derive everything without re-running the
        # model -- the embeddings are the expensive part, the kernel is a matmul.
        self.E_train, self.E_test = E_train, E_test
        self._reference_distances = reference_distances
        self._novelty_quantile = novelty_quantile
        self.w = w[0].cpu().numpy()
        self.probs = probs[0].cpu().numpy()
        self.pred = clf.y_encoder_.inverse_transform(self.probs.argmax(-1))
        self.agreement = self.probs.max(-1)

        # Perplexity in absolute units: the effective number of training cases
        # behind a prediction. "6 comparable cases" is legible in a way that
        # "relative perplexity 0.05%" is not.
        self.evidence_cases = relative_perplexity(torch.from_numpy(self.w)).numpy() * self.n
        self.is_novel = self._nn_test > self.novelty_threshold

    def with_kernel(self, kernel: str, gamma=None, **overrides) -> "ClinicalExplainer":
        """The same fitted model and embeddings, read through a different kernel.

        Costs a matrix multiply, not a model run, so switching is instant. Use it
        because the two kernels are good at different jobs:

        * ``"knn"`` for **case cards**. Every selected case carries equal weight
          and together they account for all of the evidence, so a card reads
          "these 5 past cases, 4 of which were referred" -- which is what a
          clinician can actually check. A soft kernel often spreads weight so
          thinly that the five most similar cases explain under 10% of the
          decision, which is honest but unusable at the bedside.

        * ``"gaussian"`` for **triage, label auditing and equity**. Graded weights
          give a smooth agreement score rather than multiples of 1/k, and every
          training record participates -- a sparse kernel never selects most
          records, so they cannot be audited or attributed to a subgroup at all.

        Example::

            ex = fit_explainer(X_train, y_train, X_test)   # gaussian
            cards = ex.with_kernel("knn", gamma=5)
            print(cards.case(0))                           # legible card
            ex.audit_labels(y_test)                        # soft weights
        """
        kwargs = dict(
            kernel=kernel, gamma=gamma,
            train_ids=self.train_ids, test_ids=self.test_ids,
            top_k=self.top_k, min_agreement=self.min_agreement,
            reference_distances=self._reference_distances,
            novelty_quantile=self._novelty_quantile,
            X_train=self.X_train, X_test=self.X_test, feature_names=self.feature_names,
        )
        kwargs.update(overrides)
        return ClinicalExplainer(self.clf, self.head, self.E_train, self.E_test,
                                 self.y_train, **kwargs)

    @staticmethod
    def reference_distances_from(head, E_ref_train: torch.Tensor, E_ref_holdout: torch.Tensor) -> np.ndarray:
        """Nearest-neighbour distances for held-out cases, to calibrate novelty.

        Run this on the same split you used to calibrate the kernel scale: embed
        it, then pass the result as ``reference_distances``. It is the difference
        between "unusual for this batch" and "outside what the model was fitted
        on" -- only the second justifies routing a case to a clinician on the
        grounds that the model has no relevant experience.
        """
        with torch.no_grad():
            d2 = squared_distances(head.embed(E_ref_holdout), head.embed(E_ref_train))[0]
            return d2.min(dim=-1).values.sqrt().cpu().numpy()

    # ------------------------------------------------------------------ #
    # 1. Auditable per-case explanation
    # ------------------------------------------------------------------ #
    def neighbours(self, i: int) -> pd.DataFrame:
        """The training cases behind test case ``i``, most influential first."""
        row = self.w[i]
        idx = np.argsort(-row)[: self.top_k]
        idx = idx[row[idx] > 0]
        return pd.DataFrame({
            "record": self.train_ids[idx],
            "outcome": self.y_train[idx],
            "evidence_share": row[idx],
            "row": idx,
        })

    def case(self, i: int) -> str:
        """A single case as a readable card."""
        nb = self.neighbours(i)
        action, reasons = self._action(i)
        width = 66
        pred, agree = self.pred[i], self.agreement[i]

        lines = [
            "─" * width,
            f" CASE {self.test_ids[i]}".ljust(28) + f"PREDICTION: {pred}".ljust(24) + f"ACTION: {action}",
            "─" * width,
            f" Evidence base    equivalent of {self.evidence_cases[i]:.1f} comparable cases "
            f"(of {self.n:,})",
            f" Agreement        {agree:.0%} of the evidence supports \"{pred}\"",
            f" Nearest match    {self._novelty_label.upper() if self.is_novel[i] else 'within the usual range'}",
            "",
            f" Most similar past cases",
            f"   {'record':<14}{'outcome':<14}{'share of evidence':<20}",
        ]
        for _, r in nb.iterrows():
            lines.append(f"   {str(r['record']):<14}{str(r['outcome']):<14}"
                         f"{_bar(r['evidence_share'] / max(nb['evidence_share'].max(), 1e-12))}  "
                         f"{r['evidence_share']:.1%}")
        lines.append(f"   these {len(nb)} cases carry {nb['evidence_share'].sum():.0%} of the total evidence")

        if reasons:
            lines += ["", " Flagged because:"] + [f"   · {r}" for r in reasons]
        lines.append("─" * width)
        return "\n".join(lines)

    def _action(self, i: int):
        reasons = []
        if self.is_novel[i]:
            reasons.append(f"nearest comparable case is far away — {self._novelty_label} "
                           f"({self.novelty_basis} basis)")
        if self.agreement[i] < self.min_agreement:
            reasons.append(f"comparable cases disagree ({self.agreement[i]:.0%} agreement, "
                           f"threshold {self.min_agreement:.0%})")
        if not reasons:
            return "routine", reasons
        return ("clinician required" if self.is_novel[i] else "review"), reasons

    # ------------------------------------------------------------------ #
    # 2 & 3. Triage, and extrapolation detection
    # ------------------------------------------------------------------ #
    def triage(self) -> pd.DataFrame:
        """Every test case with its evidence base and a recommended action.

        Sorted so the cases most in need of a human are first. In a programme
        where reviewer time is the binding constraint, this column is the point:
        it separates predictions with a real evidence base from predictions that
        merely have a confident-looking score.
        """
        actions, reasons = zip(*(self._action(i) for i in range(self.m)))
        out = pd.DataFrame({
            "record": self.test_ids,
            "prediction": self.pred,
            "agreement": self.agreement,
            "evidence_cases": self.evidence_cases,
            "nearest_distance": self._nn_test,
            "outside_training_range": self.is_novel,
            "action": actions,
            "reason": ["; ".join(r) for r in reasons],
        })
        priority = {"clinician required": 0, "review": 1, "routine": 2}
        return out.sort_values(["action", "agreement"],
                               key=lambda s: s.map(priority) if s.name == "action" else s)

    def triage_summary(self) -> pd.DataFrame:
        t = self.triage()
        return (t.groupby("action")
                 .agg(cases=("record", "size"),
                      share=("record", lambda s: len(s) / self.m),
                      mean_agreement=("agreement", "mean"))
                 .reset_index())

    # ------------------------------------------------------------------ #
    # 4. Label auditing
    # ------------------------------------------------------------------ #
    def audit_labels(self, y_test, min_influence_ratio: float = 0.0) -> pd.DataFrame:
        """Training records that keep showing up behind *wrong* predictions.

        A ranked re-review list. A record scoring high is one whose stored outcome
        repeatedly pulls comparable cases the wrong way — what a transcription
        error or an inconsistent grading looks like from the model's side. It is
        a hypothesis to check against the source record, not a verdict.

        Ranked by ``error_share``: the fraction of a record's total influence
        that went to wrong answers. Ranking by raw weight-on-errors instead just
        surfaces the globally influential records, which is a different and much
        less useful list — on a seeded test with 60 corrupted labels, error_share
        recovered 21 of the top 30 against 10 for raw weight (chance: 2.6).

        Power is bounded by how many errors you observed: a record can only be
        implicated by mistakes it actually contributed to. The returned frame
        carries ``n_errors`` in its ``.attrs`` so you can judge that. Prefer a
        soft kernel here even if your case cards use kNN — a sparse kernel never
        selects most records, so they cannot be audited at all.

        ``min_influence_ratio`` filters out records below this multiple of an
        average record's influence. **It defaults to 0 (no filter) on purpose.**
        The obvious instinct is to require decent influence before trusting a
        ratio, but that is backwards here: a record whose stored outcome
        contradicts its neighbourhood gets pushed away in the label-conditioned
        embedding, so it attracts *less* weight than an average record. On the
        seeded test, corrupted records averaged 0.128 influence against 0.300 for
        clean ones, and filtering at one average-record's worth of influence cut
        recovery from 41-of-60 to 5-of-60 by discarding the very records being
        hunted. Low influence combined with a high error share is the strongest
        signal available, not a reason to exclude a record.
        """
        y_test = np.asarray(y_test).ravel()
        wrong = self.pred != y_test
        cols = ["record", "stored_outcome", "error_share", "weight_on_errors",
                "total_influence", "cases_influenced"]
        if not wrong.any():
            out = pd.DataFrame(columns=cols)
            out.attrs["n_errors"] = 0
            return out

        on_errors = self.w[wrong].sum(0)
        total = self.w.sum(0)
        share = np.divide(on_errors, total, out=np.zeros_like(total), where=total > 0)

        # An average record carries m/n of the total weight; keep those at least
        # that influential so a share computed from a sliver of weight cannot win.
        keep = total >= min_influence_ratio * (self.m / self.n)

        out = (pd.DataFrame({
                   "record": self.train_ids[keep],
                   "stored_outcome": self.y_train[keep],
                   "error_share": share[keep],
                   "weight_on_errors": on_errors[keep],
                   "total_influence": total[keep],
                   "cases_influenced": (self.w[:, keep] > 0).sum(0),
               })
               .sort_values("error_share", ascending=False)
               .reset_index(drop=True))
        out.attrs["n_errors"] = int(wrong.sum())
        return out

    def influence(self) -> pd.DataFrame:
        """How much each training record contributes across all predictions.

        Records with zero influence are never used by any prediction — often a
        large fraction of the training set, and useful to know before investing
        in collecting more of the same.
        """
        total = self.w.sum(0)
        return (pd.DataFrame({
                    "record": self.train_ids,
                    "outcome": self.y_train,
                    "total_influence": total,
                    "cases_influenced": (self.w > 0).sum(0),
                })
                .sort_values("total_influence", ascending=False)
                .reset_index(drop=True))

    # ------------------------------------------------------------------ #
    # 5. Equity checking
    # ------------------------------------------------------------------ #
    def equity(self, train_groups, test_groups) -> pd.DataFrame:
        """Where each subgroup's predictions draw their evidence from.

        ``within_group_evidence`` is the share of a group's evidence that comes
        from training records in the same group. A low value means predictions
        for that group are being extrapolated from a different population —
        which can happen even when accuracy looks equal across groups, and is
        invisible without weight inspection.
        """
        train_groups = np.asarray(train_groups).ravel()
        test_groups = np.asarray(test_groups).ravel()
        if len(train_groups) != self.n or len(test_groups) != self.m:
            raise ValueError(f"expected {self.n} train and {self.m} test group labels, "
                             f"got {len(train_groups)} and {len(test_groups)}")

        levels = np.unique(np.concatenate([train_groups, test_groups]))
        mass = np.stack([self.w[:, train_groups == g].sum(1) for g in levels], axis=1)  # (m, G)

        rows = []
        for g in levels:
            sel = test_groups == g
            if not sel.any():
                continue
            own = levels.tolist().index(g)
            rows.append({
                "group": g,
                "test_cases": int(sel.sum()),
                "train_records": int((train_groups == g).sum()),
                "within_group_evidence": float(mass[sel, own].mean()),
                "mean_agreement": float(self.agreement[sel].mean()),
                "mean_evidence_cases": float(self.evidence_cases[sel].mean()),
                "outside_training_range": float(self.is_novel[sel].mean()),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 6. Pre-deployment validation of the similarity metric
    # ------------------------------------------------------------------ #
    def feature_emphasis(self, X_train=None, X_test=None, feature_names=None,
                         k: Optional[int] = None) -> pd.DataFrame:
        """Which features the model insists comparable cases agree on.

        Compares neighbourhood tightness per feature against plain Euclidean
        distance on standardized inputs. Positive ``rel_diff`` means the learned
        metric is tighter, i.e. it treats that feature as defining similarity.

        Take the top of this table to a clinician before deploying. Agreement
        with domain knowledge is independent evidence the model learned real
        structure rather than a site or device artifact.

        Call with no arguments when the explainer came from :func:`fit_explainer`.
        String and categorical columns are mapped through TabICL's own numeric
        encoder, so they appear here on the same footing as numeric ones; NaNs are
        median-imputed for the plain-distance baseline only, since sklearn's kNN
        cannot take them although TabICL can.
        """
        from sklearn.neighbors import NearestNeighbors

        k = k or self.top_k
        X_train = self.X_train if X_train is None else X_train
        X_test = self.X_test if X_test is None else X_test
        if X_train is None or X_test is None:
            raise ValueError("feature_emphasis needs X_train and X_test; pass them here, "
                             "or build the explainer with fit_explainer() which keeps them.")
        feature_names = feature_names if feature_names is not None else self.feature_names

        Xtr = np.asarray(self.clf.X_encoder_.transform(X_train), dtype=float)
        Xte = np.asarray(self.clf.X_encoder_.transform(X_test), dtype=float)
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)

        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

        idx_model = np.argsort(-self.w, axis=1)[:, :k]
        idx_plain = NearestNeighbors(n_neighbors=k).fit(Xtr_s).kneighbors(Xte_s, return_distance=False)

        def compact(idx):
            d = np.abs(Xtr_s[idx] - Xte_s[:, None, :]).mean(axis=(0, 1))
            return d / d.mean()

        c_model, c_plain = compact(idx_model), compact(idx_plain)
        names = feature_names if feature_names is not None else [f"feature {i}" for i in range(Xtr.shape[1])]
        return (pd.DataFrame({
                    "feature": [str(n) for n in names],
                    "plain_knn": c_plain,
                    "kernelicl": c_model,
                    "rel_diff": (c_plain - c_model) / c_plain,
                })
                .sort_values("rel_diff", ascending=False)
                .reset_index(drop=True))

    # ------------------------------------------------------------------ #
    def export(self, directory: str, y_test=None) -> list[str]:
        """Write the tables a programme actually hands round, as CSVs."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, frame in [("triage", self.triage()),
                            ("triage_summary", self.triage_summary()),
                            ("influence", self.influence())]:
            frame.to_csv(out / f"{name}.csv", index=False)
            written.append(str(out / f"{name}.csv"))
        if y_test is not None:
            audit = self.audit_labels(y_test)
            audit.to_csv(out / "label_audit.csv", index=False)
            written.append(str(out / "label_audit.csv"))
        return written

    def report(self, n_cases: int = 3, y_test=None) -> str:
        """A short standing summary: how the caseload splits, and worked examples."""
        s = self.triage_summary()
        lines = ["KernelICL screening report", "=" * 66, "",
                 f"{self.m:,} cases scored against {self.n:,} training records",
                 f"kernel: {self.kernel}  scale: {self.gamma}", "",
                 "Caseload split", "-" * 66]
        for _, r in s.iterrows():
            lines.append(f"  {r['action']:<22} {r['cases']:>6,} cases  ({r['share']:.1%})  "
                         f"mean agreement {r['mean_agreement']:.0%}")
        lines += ["", f"Novelty basis: {self.novelty_basis}"]
        if self.novelty_basis == "cohort-relative":
            lines.append("  (pass reference_distances to flag cases outside the training "
                         "population rather than merely unusual within this batch)")
        # triage() preserves positional index through sort_values, so the frame's
        # index is directly the case index -- no lookup by record id, which would
        # break if ids repeat.
        top = self.triage().index[:n_cases]
        lines += ["", f"Top {len(top)} cases by review priority", "-" * 66]
        for pos in top:
            lines.append(self.case(int(pos)))
        return "\n".join(lines)
