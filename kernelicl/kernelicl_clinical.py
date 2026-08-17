"""Clinician-facing views over a KernelICL weight matrix.

Turns the weights from TabICL.forward_kernel into per-case evidence cards, a
triage list, extrapolation detection, label auditing, equity checks, and
validation of what the model treats as similar.

See README.md for what each output means and how to read it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import KernelHead, relative_perplexity, squared_distances

__all__ = ["ClinicalExplainer", "fit_explainer", "as_array", "standardized_numeric",
           "V1_CHECKPOINT", "V2_CHECKPOINT"]

# Wider at the top end than the paper's Table 7: an untrained projection leaves the
# embeddings on a scale where the useful region sits higher.
GAMMA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 5.0, 10.0]
K_GRID = [1, 4, 5, 16, 32, 64, 128, 256, 512, 1024]

# TabICLClassifier defaults to v2, so every file here does too unless told otherwise.
# v1 is what the KernelICL paper built on.
V1_CHECKPOINT = "tabicl-classifier-v1-20250208.ckpt"
V2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bar(share: float, width: int = 12) -> str:
    return ("█" * int(round(share * width))).ljust(width, "·")


def as_array(a) -> np.ndarray:
    """A flat numpy array. Series are coerced because weights index rows positionally,
    and a non-default index would silently turn that into a label lookup."""
    return (a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)).ravel()


def standardized_numeric(clf, X_train, X_test):
    """Both matrices through TabICL's numeric encoder, median-imputed and z-scored.

    The input-space baselines need a numeric, NaN-free matrix, which raw X may not be.
    Reusing the model's own encoder maps string and categorical columns the way the
    model sees them. Imputation is for sklearn's benefit: TabICL handles NaN, its
    neighbour baselines do not.
    """
    train = np.asarray(clf.X_encoder_.transform(X_train), dtype=float)
    test = np.asarray(clf.X_encoder_.transform(X_test), dtype=float)
    median = np.nan_to_num(np.nanmedian(train, axis=0))
    train = np.where(np.isnan(train), median, train)
    test = np.where(np.isnan(test), median, test)
    mean, sd = train.mean(0), train.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (train - mean) / sd, (test - mean) / sd


def _take(X, idx):
    """Positional row selection for DataFrames and arrays alike."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _embed(clf, X_query, return_row_repr: bool = False):
    """Symmetric in-context embeddings, optionally with the row-stage representation.

    The ensemble generator sits after TabICL's numeric encoder, so the query must be
    encoded first; passing raw X raises on string columns.
    """
    encoded = clf.X_encoder_.transform(X_query)
    X_ens, y_ens = next(iter(clf.ensemble_generator_.transform(encoded, mode="both").values()))
    device = next(clf.model_.parameters()).device
    X_t = torch.from_numpy(np.asarray(X_ens)).float().to(device)
    y_t = torch.from_numpy(np.asarray(y_ens)).float().to(device)

    model = clf.model_
    with torch.no_grad():
        R = model.row_interactor(
            model.col_embedder(X_t, y_train=y_t, mgr_config=clf.inference_config_.COL_CONFIG),
            mgr_config=clf.inference_config_.ROW_CONFIG,
        )
        E_train, E_test = model.icl_predictor.embed(R, y_t, symmetric=True)

    return (E_train, E_test, R) if return_row_repr else (E_train, E_test)


def _make_folds(y, n_folds: int, val_size: float, random_state: int):
    """(train_idx, val_idx) pairs: stratified k-fold, or a single split as fallback."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    n = len(y)
    counts = np.unique(y, return_counts=True)[1]
    if n_folds > 1 and counts.min() >= n_folds:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(n), y))

    if n_folds > 1:
        print(f"  rarest class has {counts.min()} members, fewer than {n_folds} folds; "
              f"using a single held-out split")
    strat = y if counts.min() >= 2 else None
    return [train_test_split(np.arange(n), test_size=val_size,
                             random_state=random_state, stratify=strat)]


def _make_clf(device, norm_method: str, random_state: int,
              checkpoint_version: Optional[str] = None) -> TabICLClassifier:
    """One ensemble member, no shuffling, so a weight index is a training row.

    TabICL averages 8 members by default; averaging destroys per-case attribution.
    """
    extra = {"checkpoint_version": checkpoint_version} if checkpoint_version else {}
    return TabICLClassifier(
        n_estimators=1,
        norm_methods=[norm_method],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        device=device,
        random_state=random_state,
        kv_cache=False,
        **extra,
    )


def _load_finetuned(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("config", "state_dict", "kernel_head", "head_config")
               if k not in payload]
    if missing:
        raise ValueError(f"{path} is missing {missing}; expected a checkpoint from "
                         f"kernelicl_finetune.finetune()")
    return payload


def _swap_in_finetuned(clf, payload: dict, device):
    """Replace a fitted classifier's network with fine-tuned weights.

    Rebuilt from the checkpoint's own config, since a fine-tune may have started
    from a different TabICL version than the classifier downloaded.
    """
    from tabicl._model.tabicl import TabICL

    model = TabICL(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    clf.model_ = model.to(device).eval()
    clf.model_config_ = payload["config"]
    return clf


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
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
    checkpoint_version: Optional[str] = None,
    finetuned: Optional[str] = None,
    calibrate: bool = True,
    n_folds: int = 5,
    keep_row_repr: bool = False,
    accuracy_tolerance: float = 0.01,
    val_size: float = 0.2,
    random_state: int = 0,
    verbose: bool = True,
    **explainer_kwargs,
) -> "ClinicalExplainer":
    """Fit the model, calibrate, and return a ready explainer.

    Parameters
    ----------
    X_train, y_train, X_test : array-like
        X may be a DataFrame with string columns and NaNs. ``y_test`` is
        deliberately absent: nothing here may see it.
    kernel : {"gaussian", "dot", "knn"}
        "gaussian" grades neighbours by similarity; "knn" weights k cases equally.
        Use "gaussian" for triage and auditing, "knn" for case cards --
        see :meth:`ClinicalExplainer.with_kernel`.
    gamma : float, optional
        Kernel scale. Calibrated on held-out data when None.
    checkpoint_version : str, optional
        Which pretrained TabICL to load, e.g. ``V1_CHECKPOINT``. None uses
        TabICLClassifier's own default, currently v2. Ignored when ``finetuned`` is
        given, since the network is then rebuilt from the checkpoint's own config.
    finetuned : str, optional
        Checkpoint from ``kernelicl_finetune.finetune()``. Swapped into every
        classifier including the calibration folds, since a scale calibrated on the
        pretrained geometry does not transfer to a fine-tuned one.
    n_folds : int, default=5
        Stratified k-fold for the scale, per the paper. One embedding pass per fold.
    accuracy_tolerance : float, default=0.01
        Held-out accuracy to trade for sparsity. Among scales within this much of
        the best, the sparsest is chosen. The most consequential knob here: at 0.0
        predictions typically average over the whole training set and no small set
        of cases explains them.
    keep_row_repr : bool, default=False
        Also store ``ex.R_train`` / ``ex.R_test``, the representation before labels
        are added. Free -- it is computed on the way regardless.
    norm_method : str, default="none"
        TabICL normalization: "none", "power", "quantile", "quantile_rtdl", "robust".
    """
    def say(msg):
        if verbose:
            print(msg)

    y_train = as_array(y_train)
    if feature_names is None and hasattr(X_train, "columns"):
        feature_names = list(X_train.columns)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not calibrate and gamma is None:
        raise ValueError("gamma must be supplied when calibrate=False")

    payload = _load_finetuned(finetuned) if finetuned else None

    def fit_clf(X, y):
        clf = _make_clf(device, norm_method, random_state, checkpoint_version)
        clf.fit(X, y)
        return _swap_in_finetuned(clf, payload, device) if payload else clf

    say(f"fitting on {len(y_train):,} training cases, scoring {len(X_test):,} "
        f"(device={device}, kernel={kernel}{', fine-tuned' if payload else ''})")
    clf = fit_clf(X_train, y_train)

    if payload:
        head_config = payload["head_config"]
        d_model = head_config["d_model"]
        head = KernelHead(d_model=d_model, d_k=head_config["d_k"], kernel=kernel).to(device)
        head.load_state_dict(payload["kernel_head"])
        say(f"  fine-tuned projection {d_model}->{head_config['d_k']}, "
            f"trained with kernel={head_config['kernel']}, "
            f"validation loss {payload.get('val_loss', float('nan')):.4f}")
        # A v1-derived model compared against a v2 baseline is not a like-for-like
        # comparison, and the mismatch is otherwise invisible.
        trained_from = payload.get("finetune_config", {}).get("checkpoint")
        if trained_from and checkpoint_version and trained_from != checkpoint_version:
            say(f"  ! fine-tuned from {trained_from} but checkpoint_version is "
                f"{checkpoint_version}; baselines will not be like-for-like")
    else:
        d_model = clf.model_.icl_predictor.decoder[0].in_features
        head = KernelHead(d_model=d_model, d_k=d_model, kernel=kernel,
                          identity_init=True).to(device)

    reference_distances = None
    if calibrate:
        grid = K_GRID if kernel == "knn" else GAMMA_GRID
        folds = _make_folds(y_train, n_folds, val_size, random_state)
        say(f"calibrating over {len(folds)} fold(s), one embedding pass each...")

        scores = {scale: [] for scale in grid}
        perplexities = {scale: [] for scale in grid}
        ref_chunks = []

        for i, (tr_idx, va_idx) in enumerate(folds, 1):
            cal = fit_clf(_take(X_train, tr_idx), y_train[tr_idx])
            E_cal_train, E_cal_val = _embed(cal, _take(X_train, va_idx))
            y_val = y_train[va_idx]
            y_ctx = torch.from_numpy(
                cal.y_encoder_.transform(y_train[tr_idx])).float().to(device)[None]
            ref_chunks.append(
                ClinicalExplainer.reference_distances_from(head, E_cal_train, E_cal_val))

            # The scale never touches the embedding, so the whole grid is swept over
            # one embedding per fold rather than one per (fold, scale).
            if gamma is None:
                for scale in grid:
                    with torch.no_grad():
                        probs, w = head(E_cal_train, E_cal_val, y_ctx,
                                        num_classes=cal.n_classes_, gamma=scale)
                    pred = cal.y_encoder_.inverse_transform(probs.argmax(-1)[0].cpu().numpy())
                    scores[scale].append(float((pred == y_val).mean()))
                    perplexities[scale].append(float(relative_perplexity(w).mean()))
            say(f"  fold {i}/{len(folds)} done")

        if gamma is None:
            rows = [(scale, float(np.mean(scores[scale])), float(np.mean(perplexities[scale])))
                    for scale in grid]
            # Accuracy is near-flat across much of the grid while the evidence base
            # varies by orders of magnitude, so a plain argmax returns a kernel that
            # averages over everything. Take the sparsest scale within tolerance.
            best = max(r[1] for r in rows)
            candidates = [r for r in rows if r[1] >= best - accuracy_tolerance]
            gamma, accuracy, perplexity = min(candidates, key=lambda r: r[2])
            n_ctx = len(folds[0][0])
            say(f"  scale={gamma}  cross-validated accuracy={accuracy:.3f} "
                f"(+/-{np.std(scores[gamma]):.3f} across folds; best on grid {best:.3f})")
            say(f"  evidence base ~{perplexity * n_ctx:.0f} of {n_ctx:,} cases per prediction")

        reference_distances = np.concatenate(ref_chunks)
        say(f"  novelty threshold calibrated on {len(reference_distances):,} held-out cases")

    say("embedding the full training context...")
    R_train = R_test = None
    if keep_row_repr:
        E_train, E_test, R = _embed(clf, X_test, return_row_repr=True)
        R_train, R_test = R[:, :len(y_train)], R[:, len(y_train):]
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
    explainer.R_train, explainer.R_test = R_train, R_test
    return explainer


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
class ClinicalExplainer:
    """Views over a weight matrix. Prefer :func:`fit_explainer` to build one.

    Parameters
    ----------
    clf : TabICLClassifier
        Fitted, for its label encoder and class names.
    head : KernelHead
        The head that produced the weights.
    E_train, E_test : Tensor
        Symmetric in-context embeddings, (1, n, d) and (1, m, d).
    y_train : array
        Training outcomes in their original form.
    train_ids, test_ids : sequence, optional
        Record identifiers, so a clinician can look a case up. Default row numbers.
    top_k : int, default=5
        How many comparable cases a card lists.
    min_agreement : float, default=0.80
        Below this share of agreeing evidence, a case is flagged for review. A
        starting point, not a validated threshold.
    reference_distances : array, optional
        Nearest-neighbour distances for held-out cases, from
        :meth:`reference_distances_from`. Required for the novelty flag to mean
        "outside the training population"; without it the flag is cohort-relative.
        Do not substitute training-internal distances -- a training row appears in
        its own context and so sits implausibly close to its neighbours.
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
        self.X_train, self.X_test = X_train, X_test
        self.feature_names = feature_names
        self.y_train = as_array(y_train)
        self.n, self.m = E_train.shape[1], E_test.shape[1]
        self.train_ids = np.asarray(train_ids if train_ids is not None else np.arange(self.n))
        self.test_ids = np.asarray(test_ids if test_ids is not None else np.arange(self.m))
        self.E_train, self.E_test = E_train, E_test

        head.kernel = kernel
        y_ctx = torch.from_numpy(
            clf.y_encoder_.transform(self.y_train)).float().to(E_train.device)[None]
        with torch.no_grad():
            probs, w = head(E_train, E_test, y_ctx, num_classes=clf.n_classes_, gamma=gamma)
            H_train, H_test = head.embed(E_train), head.embed(E_test)
            self._nn_test = squared_distances(H_test, H_train)[0].min(-1).values.sqrt().cpu().numpy()

        if reference_distances is not None:
            self.novelty_basis = "held-out reference"
            self._novelty_label = "outside the range seen in training"
            ref = np.asarray(reference_distances).ravel()
        else:
            self.novelty_basis = "cohort-relative"
            self._novelty_label = "unusual compared with the rest of this cohort"
            ref = self._nn_test

        self.novelty_threshold = float(np.quantile(ref, novelty_quantile))
        self._reference_distances = reference_distances
        self._novelty_quantile = novelty_quantile

        self.w = w[0].cpu().numpy()
        self.probs = probs[0].cpu().numpy()
        self.pred = clf.y_encoder_.inverse_transform(self.probs.argmax(-1))
        self.agreement = self.probs.max(-1)
        # Perplexity in absolute units: the effective number of cases behind a
        # prediction. Legible in a way a percentage is not.
        self.evidence_cases = relative_perplexity(torch.from_numpy(self.w)).numpy() * self.n
        self.is_novel = self._nn_test > self.novelty_threshold

    # -- construction ------------------------------------------------------- #
    @staticmethod
    def reference_distances_from(head, E_ref_train: torch.Tensor,
                                 E_ref_holdout: torch.Tensor) -> np.ndarray:
        """Nearest-neighbour distances for held-out cases, to calibrate novelty."""
        with torch.no_grad():
            d2 = squared_distances(head.embed(E_ref_holdout), head.embed(E_ref_train))[0]
            return d2.min(-1).values.sqrt().cpu().numpy()

    def with_kernel(self, kernel: str, gamma=None, **overrides) -> "ClinicalExplainer":
        """The same embeddings read through a different kernel. Costs a matmul.

        Use "knn" for case cards: its selected cases carry all of the evidence, so a
        card reads "these 5 past cases". Use "gaussian" for triage, auditing and
        equity, where graded weights and full record coverage matter -- a sparse
        kernel never selects most records, so they cannot be audited at all.
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

    # -- per-case explanation ----------------------------------------------- #
    def neighbours(self, i: int) -> pd.DataFrame:
        """The training cases behind test case ``i``, most influential first."""
        row = self.w[i]
        idx = np.argsort(-row)[:self.top_k]
        idx = idx[row[idx] > 0]
        return pd.DataFrame({
            "record": self.train_ids[idx],
            "outcome": self.y_train[idx],
            "evidence_share": row[idx],
            "row": idx,
        })

    def _action(self, i: int) -> tuple[str, list[str]]:
        reasons = []
        if self.is_novel[i]:
            reasons.append(f"nearest comparable case is far away - {self._novelty_label} "
                           f"({self.novelty_basis} basis)")
        if self.agreement[i] < self.min_agreement:
            reasons.append(f"comparable cases disagree ({self.agreement[i]:.0%} agreement, "
                           f"threshold {self.min_agreement:.0%})")
        if not reasons:
            return "routine", reasons
        return ("clinician required" if self.is_novel[i] else "review"), reasons

    def case(self, i: int) -> str:
        """One case as a readable card. Returns a string -- print it."""
        neighbours = self.neighbours(i)
        action, reasons = self._action(i)
        width = 66
        prediction = self.pred[i]

        lines = [
            "-" * width,
            f" CASE {self.test_ids[i]}".ljust(28)
            + f"PREDICTION: {prediction}".ljust(24)
            + f"ACTION: {action}",
            "-" * width,
            f" Evidence base    equivalent of {self.evidence_cases[i]:.1f} comparable cases "
            f"(of {self.n:,})",
            f" Agreement        {self.agreement[i]:.0%} of the evidence supports "
            f'"{prediction}"',
            f" Nearest match    "
            f"{self._novelty_label.upper() if self.is_novel[i] else 'within the usual range'}",
            "",
            " Most similar past cases",
            f"   {'record':<14}{'outcome':<14}{'share of evidence':<20}",
        ]
        peak = max(neighbours["evidence_share"].max(), 1e-12) if len(neighbours) else 1.0
        for _, row in neighbours.iterrows():
            lines.append(f"   {str(row['record']):<14}{str(row['outcome']):<14}"
                         f"{_bar(row['evidence_share'] / peak)}  {row['evidence_share']:.1%}")
        lines.append(f"   these {len(neighbours)} cases carry "
                     f"{neighbours['evidence_share'].sum():.0%} of the total evidence")

        if reasons:
            lines += ["", " Flagged because:"] + [f"   - {r}" for r in reasons]
        lines.append("-" * width)
        return "\n".join(lines)

    # -- triage and extrapolation ------------------------------------------- #
    def triage(self) -> pd.DataFrame:
        """Every case with its evidence base and a recommended action.

        Sorted so the cases most needing a human come first.
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
        return out.sort_values(
            ["action", "agreement"],
            key=lambda s: s.map(priority) if s.name == "action" else s,
        )

    def triage_summary(self) -> pd.DataFrame:
        """How the caseload splits by recommended action."""
        return (self.triage()
                .groupby("action")
                .agg(cases=("record", "size"),
                     share=("record", lambda s: len(s) / self.m),
                     mean_agreement=("agreement", "mean"))
                .reset_index())

    # -- auditing ----------------------------------------------------------- #
    def audit_labels(self, y_test, min_influence_ratio: float = 0.0) -> pd.DataFrame:
        """Training records that keep appearing behind wrong predictions.

        Ranked by ``error_share``, the fraction of a record's influence that went to
        mistakes. Power is bounded by the number of errors observed;
        ``.attrs["n_errors"]`` reports it. Prefer a soft kernel here -- a sparse one
        never selects most records, so they cannot be audited.

        ``min_influence_ratio`` defaults to 0 deliberately. A record whose outcome
        contradicts its neighbourhood is pushed away in the label-conditioned
        embedding and so carries *less* weight than average, so filtering by
        influence discards exactly the records being hunted.
        """
        y_test = as_array(y_test)
        wrong = self.pred != y_test
        columns = ["record", "stored_outcome", "error_share", "weight_on_errors",
                   "total_influence", "cases_influenced"]
        if not wrong.any():
            out = pd.DataFrame(columns=columns)
            out.attrs["n_errors"] = 0
            return out

        on_errors = self.w[wrong].sum(0)
        total = self.w.sum(0)
        share = np.divide(on_errors, total, out=np.zeros_like(total), where=total > 0)
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
        """Total contribution of each training record across all predictions."""
        return (pd.DataFrame({
                    "record": self.train_ids,
                    "outcome": self.y_train,
                    "total_influence": self.w.sum(0),
                    "cases_influenced": (self.w > 0).sum(0),
                })
                .sort_values("total_influence", ascending=False)
                .reset_index(drop=True))

    # -- equity ------------------------------------------------------------- #
    def equity(self, train_groups, test_groups) -> pd.DataFrame:
        """Where each subgroup's predictions draw their evidence from.

        ``within_group_evidence`` is the share coming from training records in the
        same group. A low value means predictions for that group are extrapolated
        from a different population, which can happen at equal accuracy.
        """
        train_groups, test_groups = as_array(train_groups), as_array(test_groups)
        if len(train_groups) != self.n or len(test_groups) != self.m:
            raise ValueError(f"expected {self.n} train and {self.m} test group labels, "
                             f"got {len(train_groups)} and {len(test_groups)}")

        levels = np.unique(np.concatenate([train_groups, test_groups]))
        mass = np.stack([self.w[:, train_groups == g].sum(1) for g in levels], axis=1)

        rows = []
        for position, group in enumerate(levels):
            selected = test_groups == group
            if not selected.any():
                continue
            rows.append({
                "group": group,
                "test_cases": int(selected.sum()),
                "train_records": int((train_groups == group).sum()),
                "within_group_evidence": float(mass[selected, position].mean()),
                "mean_agreement": float(self.agreement[selected].mean()),
                "mean_evidence_cases": float(self.evidence_cases[selected].mean()),
                "outside_training_range": float(self.is_novel[selected].mean()),
            })
        return pd.DataFrame(rows)

    # -- similarity metric -------------------------------------------------- #
    def feature_emphasis(self, X_train=None, X_test=None, feature_names=None,
                         k: Optional[int] = None) -> pd.DataFrame:
        """Which features the model insists comparable cases agree on.

        Neighbourhood tightness per feature against plain Euclidean distance on
        standardized inputs. Positive ``rel_diff`` means the learned metric is
        tighter, i.e. treats that feature as defining similarity.

        Both columns are normalized within their own method, so they show relative
        emphasis rather than absolute closeness -- KernelICL neighbourhoods are
        typically wider in raw feature space, since it selects on the embedding.

        Call with no arguments when built by :func:`fit_explainer`.
        """
        from sklearn.neighbors import NearestNeighbors

        k = k or self.top_k
        X_train = self.X_train if X_train is None else X_train
        X_test = self.X_test if X_test is None else X_test
        if X_train is None or X_test is None:
            raise ValueError("feature_emphasis needs X_train and X_test; pass them here "
                             "or build the explainer with fit_explainer()")
        feature_names = feature_names if feature_names is not None else self.feature_names

        Xtr_s, Xte_s = standardized_numeric(self.clf, X_train, X_test)

        idx_model = np.argsort(-self.w, axis=1)[:, :k]
        idx_plain = NearestNeighbors(n_neighbors=k).fit(Xtr_s).kneighbors(
            Xte_s, return_distance=False)

        def compactness(idx):
            gap = np.abs(Xtr_s[idx] - Xte_s[:, None, :]).mean(axis=(0, 1))
            return gap / gap.mean()

        model_c, plain_c = compactness(idx_model), compactness(idx_plain)
        names = feature_names if feature_names is not None else [
            f"feature {i}" for i in range(Xtr_s.shape[1])]
        return (pd.DataFrame({
                    "feature": [str(n) for n in names],
                    "plain_knn": plain_c,
                    "kernelicl": model_c,
                    "rel_diff": (plain_c - model_c) / plain_c,
                })
                .sort_values("rel_diff", ascending=False)
                .reset_index(drop=True))

    # -- output ------------------------------------------------------------- #
    def export(self, directory: str, y_test=None) -> list[str]:
        """Write the tables as CSVs. Returns the written paths."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        frames = {"triage": self.triage(),
                  "triage_summary": self.triage_summary(),
                  "influence": self.influence()}
        if y_test is not None:
            frames["label_audit"] = self.audit_labels(y_test)

        written = []
        for name, frame in frames.items():
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
            written.append(str(path))
        return written

    def report(self, n_cases: int = 3) -> str:
        """Caseload summary plus worked examples. Returns a string -- print it."""
        summary = self.triage_summary()
        lines = [
            "KernelICL screening report", "=" * 66, "",
            f"{self.m:,} cases scored against {self.n:,} training records",
            f"kernel: {self.kernel}  scale: {self.gamma}",
            f"novelty basis: {self.novelty_basis}", "",
            "Caseload split", "-" * 66,
        ]
        for _, row in summary.iterrows():
            lines.append(f"  {row['action']:<22} {row['cases']:>6,} cases  "
                         f"({row['share']:.1%})  mean agreement {row['mean_agreement']:.0%}")

        # triage() preserves the positional index through sorting, so the frame's
        # index is the case index.
        top = self.triage().index[:n_cases]
        lines += ["", f"Top {len(top)} cases by review priority", "-" * 66]
        lines += [self.case(int(pos)) for pos in top]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
# ex = fit_explainer(
#     X_train, y_train, X_test,
#     train_ids=patient_ids_train,          # optional; defaults to row numbers
#     test_ids=patient_ids_test,
#     finetuned="kernelicl_finetuned.pt",   # optional; see kernelicl_finetune.py
# )
#
# # Text -- print it.
# cards = ex.with_kernel("knn", gamma=5)    # sparse kernel reads best on a card
# print(cards.case(0))
# print(ex.report(n_cases=3))
#
# # Tables -- assign, then display. A bare call renders only as a cell's last line.
# triage = ex.triage()
# summary = ex.triage_summary()
# audit = ex.audit_labels(y_test)
# equity = ex.equity(site_train, site_test)
# emphasis = ex.feature_emphasis()
#
# from IPython.display import display
# display(summary, audit.head(10), equity, emphasis.head(10))
#
# print(ex.export("report", y_test=y_test))
