"""Kernel regression head for interpretable in-context learning (KernelICL).

Replaces the MLP decoder of :class:`~tabicl._model.learning.ICLearning` with an
explicit kernel function, so that every prediction is a transparent weighted
average of training labels:

.. math::

    \\hat{y}(x) = \\sum_{i=1}^{n} w_i y_i,
    \\qquad
    w_i = \\frac{\\kappa_D(x, x_i)}{\\sum_j \\kappa_D(x, x_j)}

with :math:`\\kappa_D(x, x_i) = K_\\gamma(h_D(x), h_D(x_i))` and
:math:`h_D = W \\cdot E`, where :math:`E` are the in-context embeddings produced
by :meth:`~tabicl._model.learning.ICLearning.embed` and :math:`W` is a learnable
projection.

Reference: Miftachov, Charron & Valentin, "Interpretable Tabular Foundation
Models via In-Context Kernel Regression" (arXiv:2602.02162), Sections 3.2-3.4.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def squared_distances(q: Tensor, k: Tensor) -> Tensor:
    """Pairwise squared Euclidean distances between query and key embeddings.

    Uses the expansion :math:`\\|q - k\\|^2 = \\|q\\|^2 + \\|k\\|^2 - 2 q^\\top k`
    rather than :func:`torch.cdist`. ``cdist`` takes a square root whose gradient
    is undefined at zero distance, which is guaranteed to occur in symmetric mode
    where every training sample appears as both a query and a key.

    Parameters
    ----------
    q : Tensor
        Query embeddings of shape (B, m, d_k).

    k : Tensor
        Key embeddings of shape (B, n, d_k).

    Returns
    -------
    Tensor
        Squared distances of shape (B, m, n), clamped to be non-negative.
    """

    q_sq = q.pow(2).sum(-1).unsqueeze(-1)  # (B, m, 1)
    k_sq = k.pow(2).sum(-1).unsqueeze(-2)  # (B, 1, n)
    return (q_sq + k_sq - 2.0 * (q @ k.transpose(-1, -2))).clamp_min(0)


def relative_perplexity(w: Tensor, eps: float = 1e-12) -> Tensor:
    """Inspectability of a weight distribution, as perplexity relative to context size.

    Implements :math:`\\mathrm{PPL}(w) = \\exp(-\\sum_i w_i \\log w_i)` divided by
    the number of training samples ``n``. Lower values mean sparser, more
    inspectable weights: ``PPL(w) = 5`` corresponds to a sparsity level similar
    to using 5 nearest neighbors.

    Parameters
    ----------
    w : Tensor
        Normalized weights of shape (..., n), summing to 1 along the last axis.

    eps : float, default=1e-12
        Guard for the logarithm, needed because kNN weights contain exact zeros.

    Returns
    -------
    Tensor
        Relative perplexity in (0, 1], of shape (...,).
    """

    entropy = -(w * (w + eps).log()).sum(-1)
    return entropy.exp() / w.shape[-1]


class KernelHead(nn.Module):
    """Learnable projection followed by a single-parameter kernel.

    The projection concentrates representational complexity in the embedding
    while keeping the kernel operation transparent and inspectable.

    Parameters
    ----------
    d_model : int, default=512
        Dimension of the in-context embeddings ``E``. For TabICL this is
        ``embed_dim * row_num_cls``.

    d_k : int, default=512
        Dimension of the projected embedding space in which the kernel operates.

    kernel : {"gaussian", "dot", "knn"}, default="gaussian"
        Kernel function :math:`K_\\gamma`:

        - ``"dot"``: :math:`\\exp(\\gamma\\, q^\\top k)`, the kernel underlying
          standard scaled dot-product attention. Measures similarity through
          alignment rather than distance, so it is only interpretable
          geometrically when the embeddings are symmetric.
        - ``"gaussian"``: :math:`\\exp(-\\gamma \\|q - k\\|^2)`, isotropic and
          therefore distance-based.
        - ``"knn"``: uniform weight over the :math:`\\gamma` nearest neighbors.
          Maximally inspectable but **not differentiable** — train with the
          Gaussian kernel and swap the kernel at evaluation time.

    gamma : float, optional
        Kernel scale. Defaults follow the paper: ``1 / sqrt(d_k)`` for the
        dot-product kernel, ``1 / (2 * sqrt(d_k))`` for the Gaussian kernel
        (which makes the two equivalent under unit-norm embeddings), and ``5``
        neighbors for kNN. Typically re-calibrated per dataset by
        cross-validation; pass an override to :meth:`weights` or :meth:`forward`
        to sweep a grid without rebuilding the head.

    identity_init : bool, default=False
        If True and ``d_k == d_model``, initialize ``W`` to the identity so that
        the head starts as a kernel applied directly to the raw in-context
        embeddings. Useful as a no-fine-tuning baseline.
    """

    KERNELS = ("gaussian", "dot", "knn")

    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 512,
        kernel: Literal["gaussian", "dot", "knn"] = "gaussian",
        gamma: Optional[float] = None,
        identity_init: bool = False,
    ):
        super().__init__()

        if kernel not in self.KERNELS:
            raise ValueError(f"kernel must be one of {self.KERNELS}, got '{kernel}'")

        self.d_model = d_model
        self.d_k = d_k
        self.kernel = kernel

        self.proj = nn.Linear(d_model, d_k, bias=False)
        if identity_init:
            if d_k != d_model:
                raise ValueError(f"identity_init requires d_k == d_model, got {d_k} != {d_model}")
            nn.init.eye_(self.proj.weight)

        if gamma is None:
            gamma = {"dot": d_k**-0.5, "gaussian": 1.0 / (2 * d_k**0.5), "knn": 5}[kernel]
        self.gamma = gamma

    def embed(self, E: Tensor) -> Tensor:
        """Project in-context embeddings into the kernel space: :math:`h_D = W \\cdot E`."""

        return self.proj(E)

    def weights(
        self,
        E_train: Tensor,
        E_test: Tensor,
        gamma: Optional[Union[float, int]] = None,
        already_projected: bool = False,
    ) -> Tensor:
        """Kernel weights of each training sample for each test sample.

        Parameters
        ----------
        E_train : Tensor
            Training embeddings of shape (B, n, d_model), or (B, n, d_k) when
            ``already_projected=True``.

        E_test : Tensor
            Test embeddings of shape (B, m, d_model), or (B, m, d_k) when
            ``already_projected=True``.

        gamma : float or int, optional
            Overrides the kernel scale for this call. Kernel scale calibration
            re-runs only this method, never the embedding, so a whole grid can be
            swept over cached embeddings.

        already_projected : bool, default=False
            If True, skip the projection ``W``. Lets a calibration loop project
            once and reuse the result across the grid.

        Returns
        -------
        Tensor
            Weights of shape (B, m, n), non-negative and summing to 1 along the
            last axis.
        """

        g = self.gamma if gamma is None else gamma
        k = E_train if already_projected else self.proj(E_train)
        q = E_test if already_projected else self.proj(E_test)

        if self.kernel == "dot":
            return torch.softmax(g * (q @ k.transpose(-1, -2)), dim=-1)

        d2 = squared_distances(q, k)

        if self.kernel == "gaussian":
            # softmax(-g * d2) is exactly exp(-g d2) / sum_j exp(-g d2_j),
            # computed in a numerically stable way.
            return torch.softmax(-g * d2, dim=-1)

        # kNN: uniform weight over the g nearest neighbors, zero elsewhere.
        n_neighbors = min(int(g), d2.shape[-1])
        idx = d2.topk(n_neighbors, dim=-1, largest=False).indices
        return torch.zeros_like(d2).scatter_(-1, idx, 1.0 / n_neighbors)

    def forward(
        self,
        E_train: Tensor,
        E_test: Tensor,
        y_train: Tensor,
        num_classes: Optional[int] = None,
        gamma: Optional[Union[float, int]] = None,
        already_projected: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Predict as a weighted average of training labels.

        Parameters
        ----------
        E_train : Tensor
            Training embeddings of shape (B, n, d_model).

        E_test : Tensor
            Test embeddings of shape (B, m, d_model).

        y_train : Tensor
            Training targets of shape (B, n). Class indices for classification
            (``num_classes`` given), numeric values for regression.

        num_classes : int, optional
            Number of classes. If None, the head performs regression and returns
            a weighted average of the numeric targets.

        gamma : float or int, optional
            Overrides the kernel scale for this call.

        already_projected : bool, default=False
            If True, skip the projection ``W``.

        Returns
        -------
        pred : Tensor
            For classification: class **probabilities** of shape (B, m, C).
            These are already normalized — they are not logits, so train with
            ``F.nll_loss(pred.log(), target)`` rather than a cross-entropy that
            would apply a second softmax.
            For regression: values of shape (B, m).

        w : Tensor
            The weights of shape (B, m, n) that produced the prediction. This is
            the object to inspect: ``w[b, j, i]`` is the contribution of training
            sample ``i`` to the prediction for test sample ``j``.
        """

        w = self.weights(E_train, E_test, gamma=gamma, already_projected=already_projected)

        if num_classes is None:
            return (w @ y_train.unsqueeze(-1).to(w.dtype)).squeeze(-1), w

        Y = F.one_hot(y_train.long(), num_classes).to(w.dtype)  # (B, n, C)
        return w @ Y, w

    def extra_repr(self) -> str:
        return f"kernel={self.kernel}, gamma={self.gamma}, d_model={self.d_model}, d_k={self.d_k}"
