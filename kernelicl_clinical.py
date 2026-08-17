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
::

    from kernelicl_clinical import ClinicalExplainer

    ex = ClinicalExplainer(
        clf, head, E_train, E_test, y_train,
        kernel="gaussian", gamma=BEST["gaussian"],
        train_ids=train_record_ids,      # optional, defaults to row numbers
        test_ids=test_record_ids,
    )

    print(ex.case(0))                    # one case, as a readable card
    ex.triage()                          # every case, ranked by review priority
    ex.audit_labels(y_test)              # candidate mislabelled training records
    ex.equity(train_site, test_site)     # evidence sources by subgroup
    ex.feature_emphasis(X_tr_num, X_te_num, names)
    ex.export("kernelicl_report")        # CSVs

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

from tabicl._model.kernel_head import relative_perplexity, squared_distances

__all__ = ["ClinicalExplainer"]

_BAR = "█"


def _bar(share: float, width: int = 12) -> str:
    return (_BAR * int(round(share * width))).ljust(width, "·")


class ClinicalExplainer:
    """Turns a KernelICL weight matrix into clinician-usable outputs.

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
    ):
        self.clf, self.head = clf, head
        self.kernel, self.gamma = kernel, gamma
        self.top_k, self.min_agreement = top_k, min_agreement

        self.y_train = np.asarray(y_train).ravel()
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
        self.w = w[0].cpu().numpy()
        self.probs = probs[0].cpu().numpy()
        self.pred = clf.y_encoder_.inverse_transform(self.probs.argmax(-1))
        self.agreement = self.probs.max(-1)

        # Perplexity in absolute units: the effective number of training cases
        # behind a prediction. "6 comparable cases" is legible in a way that
        # "relative perplexity 0.05%" is not.
        self.evidence_cases = relative_perplexity(torch.from_numpy(self.w)).numpy() * self.n
        self.is_novel = self._nn_test > self.novelty_threshold

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
    def feature_emphasis(self, X_train_num, X_test_num, feature_names=None,
                         k: Optional[int] = None) -> pd.DataFrame:
        """Which features the model insists comparable cases agree on.

        Compares neighbourhood tightness per feature against plain Euclidean
        distance on standardized inputs. Positive ``rel_diff`` means the learned
        metric is tighter, i.e. it treats that feature as defining similarity.

        Take the top of this table to a clinician before deploying. Agreement
        with domain knowledge is independent evidence the model learned real
        structure rather than a site or device artifact.
        """
        from sklearn.neighbors import NearestNeighbors

        k = k or self.top_k
        Xtr = np.asarray(X_train_num, dtype=float)
        Xte = np.asarray(X_test_num, dtype=float)
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
