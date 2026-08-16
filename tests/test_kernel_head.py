"""Tests for the KernelICL symmetric embeddings and kernel regression head."""

import pytest
import torch

from src.tabicl._model.kernel_head import KernelHead, relative_perplexity, squared_distances
from src.tabicl._model.learning import ICLearning

D_MODEL = 32
N_TRAIN = 16
N_TEST = 5
MAX_CLASSES = 4


def make_icl(max_classes=MAX_CLASSES, d_model=D_MODEL, seed=0):
    """A small ICL transformer with non-degenerate weights.

    ``zero_init=True`` (the default) zeroes every residual branch, which makes
    the whole stack an identity map. Symmetry would then hold trivially and the
    tests below would pass on a broken implementation, so it must be disabled.
    """

    torch.manual_seed(seed)
    icl = ICLearning(
        max_classes=max_classes,
        out_dim=max_classes if max_classes > 0 else 999,
        d_model=d_model,
        num_blocks=2,
        nhead=4,
        dim_feedforward=2 * d_model,
        zero_init=False,
    )
    icl.eval()
    return icl


def make_batch(n_train=N_TRAIN, n_test=N_TEST, d_model=D_MODEL, max_classes=MAX_CLASSES, seed=1):
    generator = torch.Generator().manual_seed(seed)
    R = torch.randn(1, n_train + n_test, d_model, generator=generator)
    y_train = torch.randint(0, max_classes, (1, n_train), generator=generator).float()
    return R, y_train


# --------------------------------------------------------------------------- #
# Symmetric embeddings
# --------------------------------------------------------------------------- #


@torch.no_grad()
def test_duplicated_row_gets_identical_embedding():
    """The defining property: q_D = k_D = h_D.

    A training row and a test row with identical representations must receive
    identical embeddings. This is also the label-leak test — reading the training
    embeddings off the context positions instead would add g(y_i) to the training
    copy and break the equality.
    """

    icl = make_icl()
    R, y_train = make_batch()
    R[:, N_TRAIN] = R[:, 3]  # first test row duplicates training row 3

    E_train, E_test = icl.embed(R, y_train, symmetric=True)

    assert E_train.shape == (1, N_TRAIN, D_MODEL)
    assert E_test.shape == (1, N_TEST, D_MODEL)
    torch.testing.assert_close(E_train[0, 3], E_test[0, 0], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_context_positions_would_leak_labels():
    """Guards the reason `embed` exists rather than reusing the standard forward.

    Reading training embeddings off the context stream is the obvious shortcut
    and it is wrong: those positions carry their own label. If this ever starts
    passing, the label embedding has stopped reaching the context.
    """

    icl = make_icl()
    R, y_train = make_batch()
    R[:, N_TRAIN] = R[:, 3]

    _, E_test = icl.embed(R, y_train, symmetric=True)

    # The leaky alternative: the standard forward's train-position outputs.
    Ry = icl.y_encoder(y_train.float())
    leaky = icl.tf_icl(torch.cat([R[:, :N_TRAIN] + Ry, R], dim=1), train_size=N_TRAIN)[:, :N_TRAIN]
    leaky = icl.ln(leaky) if icl.norm_first else leaky

    assert not torch.allclose(leaky[0, 3], E_test[0, 0], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_symmetric_mode_leaves_test_embeddings_unchanged():
    """Duplicating the training rows into the query stream must not perturb the
    test embeddings: the ICL transformer uses no positional encoding, and its
    scalable-softmax scaling depends on the number of keys, which is unchanged.
    """

    icl = make_icl()
    R, y_train = make_batch()

    _, E_test_sym = icl.embed(R, y_train, symmetric=True)
    E_train_asym, E_test_asym = icl.embed(R, y_train, symmetric=False)

    assert E_train_asym is None
    torch.testing.assert_close(E_test_sym, E_test_asym, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_embed_does_not_mutate_input():
    """`_icl_predictions` adds the label embedding in place; `embed` must not,
    since the same representations feed both the context and the query stream."""

    icl = make_icl()
    R, y_train = make_batch()
    R_before = R.clone()

    icl.embed(R, y_train, symmetric=True)

    torch.testing.assert_close(R, R_before)


@torch.no_grad()
def test_embeddings_depend_on_context_labels():
    """Sanity check that the embeddings are actually in-context: changing the
    training labels must change the test embeddings."""

    icl = make_icl()
    R, y_train = make_batch()

    _, E_test = icl.embed(R, y_train, symmetric=True)
    _, E_test_shuffled = icl.embed(R, y_train.flip(-1), symmetric=True)

    assert not torch.allclose(E_test, E_test_shuffled, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_embed_supports_regression():
    icl = make_icl(max_classes=0)
    R, _ = make_batch()
    y_train = torch.randn(1, N_TRAIN)

    E_train, E_test = icl.embed(R, y_train, symmetric=True)

    assert E_train.shape == (1, N_TRAIN, D_MODEL)
    assert E_test.shape == (1, N_TEST, D_MODEL)


# --------------------------------------------------------------------------- #
# Kernel head
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", ["gaussian", "dot", "knn"])
@torch.no_grad()
def test_weights_are_a_probability_distribution(kernel):
    head = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel=kernel)
    E_train, E_test = torch.randn(2, N_TRAIN, D_MODEL), torch.randn(2, N_TEST, D_MODEL)

    w = head.weights(E_train, E_test)

    assert w.shape == (2, N_TEST, N_TRAIN)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(-1), torch.ones(2, N_TEST))


@torch.no_grad()
def test_knn_weights_are_uniform_over_k_neighbors():
    k = 5
    head = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="knn", gamma=k)
    E_train, E_test = torch.randn(1, N_TRAIN, D_MODEL), torch.randn(1, N_TEST, D_MODEL)

    w = head.weights(E_train, E_test)

    assert (w.count_nonzero(dim=-1) == k).all()
    nonzero = w[w > 0]
    torch.testing.assert_close(nonzero, torch.full_like(nonzero, 1.0 / k))


@torch.no_grad()
def test_knn_selects_the_nearest_neighbors():
    head = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="knn", gamma=3, identity_init=True)
    E_train = torch.randn(1, N_TRAIN, D_MODEL)
    E_test = E_train[:, [7]] + 1e-3 * torch.randn(1, 1, D_MODEL)

    w = head.weights(E_train, E_test)

    assert w[0, 0, 7] > 0  # the near-duplicate must be selected


@torch.no_grad()
def test_gaussian_matches_dot_product_on_the_unit_sphere():
    """With unit-norm embeddings, ||q - k||^2 = 2 - 2 q'k, so the Gaussian kernel
    at gamma = 1/(2 sqrt(d_k)) and the dot-product kernel at gamma = 1/sqrt(d_k)
    induce the same weights (the constant cancels under normalization)."""

    gaussian = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="gaussian", identity_init=True)
    dot = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="dot", identity_init=True)
    assert dot.gamma == pytest.approx(2 * gaussian.gamma)

    E_train = torch.nn.functional.normalize(torch.randn(1, N_TRAIN, D_MODEL), dim=-1)
    E_test = torch.nn.functional.normalize(torch.randn(1, N_TEST, D_MODEL), dim=-1)

    torch.testing.assert_close(gaussian.weights(E_train, E_test), dot.weights(E_train, E_test))


@torch.no_grad()
def test_predictions_are_weighted_averages_of_labels():
    head = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="gaussian")
    E_train, E_test = torch.randn(1, N_TRAIN, D_MODEL), torch.randn(1, N_TEST, D_MODEL)
    y_train = torch.randint(0, MAX_CLASSES, (1, N_TRAIN))

    probs, w = head(E_train, E_test, y_train, num_classes=MAX_CLASSES)

    assert probs.shape == (1, N_TEST, MAX_CLASSES)
    torch.testing.assert_close(probs.sum(-1), torch.ones(1, N_TEST))
    # Every class probability is the total weight assigned to that class.
    expected = w[0, :, y_train[0] == 0].sum(-1)
    torch.testing.assert_close(probs[0, :, 0], expected)


@torch.no_grad()
def test_regression_predictions_average_numeric_targets():
    head = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="knn", gamma=N_TRAIN)
    E_train, E_test = torch.randn(1, N_TRAIN, D_MODEL), torch.randn(1, N_TEST, D_MODEL)
    y_train = torch.randn(1, N_TRAIN)

    pred, _ = head(E_train, E_test, y_train, num_classes=None)

    # k = n neighbors means uniform weights, i.e. the unconditional mean.
    assert pred.shape == (1, N_TEST)
    torch.testing.assert_close(pred, y_train.mean().expand(1, N_TEST))


@torch.no_grad()
def test_gamma_override_bypasses_the_stored_scale():
    """Scale calibration sweeps a grid over cached embeddings, so the override
    must reach the kernel without touching the projection."""

    head = KernelHead(d_model=D_MODEL, d_k=D_MODEL, kernel="gaussian", gamma=0.01)
    E_train, E_test = torch.randn(1, N_TRAIN, D_MODEL), torch.randn(1, N_TEST, D_MODEL)

    diffuse = head.weights(E_train, E_test, gamma=0.001)
    peaked = head.weights(E_train, E_test, gamma=10.0)

    assert relative_perplexity(peaked).mean() < relative_perplexity(diffuse).mean()


@torch.no_grad()
def test_projecting_once_matches_projecting_per_call():
    head = KernelHead(d_model=D_MODEL, d_k=8, kernel="gaussian")
    E_train, E_test = torch.randn(1, N_TRAIN, D_MODEL), torch.randn(1, N_TEST, D_MODEL)

    direct = head.weights(E_train, E_test)
    cached = head.weights(head.embed(E_train), head.embed(E_test), already_projected=True)

    torch.testing.assert_close(direct, cached)


# --------------------------------------------------------------------------- #
# Inspectability
# --------------------------------------------------------------------------- #


def test_relative_perplexity_of_uniform_weights_is_one():
    w = torch.full((1, N_TEST, N_TRAIN), 1.0 / N_TRAIN)

    torch.testing.assert_close(relative_perplexity(w), torch.ones(1, N_TEST))


def test_relative_perplexity_of_knn_weights_is_k_over_n():
    k = 5
    w = torch.zeros(1, N_TEST, N_TRAIN)
    w[..., :k] = 1.0 / k

    torch.testing.assert_close(relative_perplexity(w), torch.full((1, N_TEST), k / N_TRAIN))


def test_relative_perplexity_of_a_point_mass_is_minimal():
    w = torch.zeros(1, 1, N_TRAIN)
    w[..., 0] = 1.0

    torch.testing.assert_close(relative_perplexity(w), torch.full((1, 1), 1.0 / N_TRAIN))


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #


def test_squared_distances_have_finite_gradients_at_zero():
    """Symmetric mode puts every training sample at distance exactly zero from
    itself. `torch.cdist` would return NaN gradients there; the expansion used
    here must not."""

    E = torch.randn(1, N_TRAIN, D_MODEL, requires_grad=True)

    squared_distances(E, E).sum().backward()

    assert torch.isfinite(E.grad).all()


def test_gaussian_head_is_trainable_end_to_end():
    head = KernelHead(d_model=D_MODEL, d_k=8, kernel="gaussian")
    E_train = torch.randn(1, N_TRAIN, D_MODEL, requires_grad=True)
    E_test = torch.randn(1, N_TEST, D_MODEL, requires_grad=True)
    y_train = torch.randint(0, MAX_CLASSES, (1, N_TRAIN))
    y_test = torch.randint(0, MAX_CLASSES, (1, N_TEST))

    probs, _ = head(E_train, E_test, y_train, num_classes=MAX_CLASSES)
    # Predictions are probabilities, so the loss takes their log directly.
    loss = torch.nn.functional.nll_loss((probs + 1e-8).log().reshape(-1, MAX_CLASSES), y_test.reshape(-1))
    loss.backward()

    for tensor in (E_train.grad, E_test.grad, head.proj.weight.grad):
        assert tensor is not None and torch.isfinite(tensor).all()
    assert head.proj.weight.grad.abs().sum() > 0


def test_knn_head_is_not_differentiable():
    """Documents why the paper trains kNN embeddings with the Gaussian kernel."""

    head = KernelHead(d_model=D_MODEL, d_k=8, kernel="knn", gamma=3)
    E_train = torch.randn(1, N_TRAIN, D_MODEL, requires_grad=True)
    E_test = torch.randn(1, N_TEST, D_MODEL, requires_grad=True)

    w = head.weights(E_train, E_test)

    assert not w.requires_grad


def test_symmetric_embeddings_are_trainable_end_to_end():
    icl = make_icl()
    head = KernelHead(d_model=D_MODEL, d_k=8, kernel="gaussian")
    R, y_train = make_batch()
    y_test = torch.randint(0, MAX_CLASSES, (1, N_TEST))

    E_train, E_test = icl.embed(R, y_train, symmetric=True)
    probs, _ = head(E_train, E_test, y_train, num_classes=MAX_CLASSES)
    loss = torch.nn.functional.nll_loss((probs + 1e-8).log().reshape(-1, MAX_CLASSES), y_test.reshape(-1))
    loss.backward()

    grads = [p.grad for p in icl.tf_icl.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert icl.y_encoder.weight.grad is not None  # the label encoder still trains
