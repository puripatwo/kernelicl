"""Fine-tune TabICL end-to-end with a kernel head — Path B of step D.

This is the step that makes KernelICL *KernelICL* rather than a kernel bolted onto
a frozen backbone. Section 4.1 of the paper: fine-tune the TabICL embedding module
together with the projection ``W`` on synthetic data from TabICL's own prior, with
cross-entropy on the kernel predictions.

Why it matters, in one sentence: with ``W = I`` you are reading a geometry that was
optimised for an MLP head and has no reason to make Euclidean distance meaningful,
which is why the default kernel scale was an order of magnitude off and why the
accuracy-versus-inspectability curve bends where it does. Fine-tuning reshapes the
representation so distance in it *means* similarity.

Paper's configuration (Appendix A), and what it costs
-----------------------------------------------------
5,000 batches, 64 synthetic datasets per batch, 5-100 features, up to 1,024 rows
per dataset, 60-80% of each used as context. Approximately **40 GB of GPU memory**,
plus 32 held-out batches (2,048 datasets) for validation-loss checkpoint selection.

That is an A100 job. ``PRESET`` below picks between the paper's settings and two
scaled-down configurations that fit smaller cards; the small one runs on a T4 and
will not reproduce the paper's numbers, but it will tell you whether fine-tuning
moves your own metrics in the right direction before you pay for an A100.

Usage
-----
::

    !pip install -e ".[pretrain]"          # the prior needs xgboost
    %run kernelicl_finetune.py             # or paste into a cell

    # then, with the rest of the tooling:
    model, head = load_finetuned("kernelicl_finetuned.pt", device="cuda")

Run ``smoke_test()`` first. It executes two optimiser steps on tiny batches and
takes about a minute on CPU, which is much cheaper than discovering a shape error
forty minutes into a GPU run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tabicl._model.kernel_head import KernelHead, relative_perplexity
from tabicl._model.tabicl import TabICL

__all__ = ["FinetuneConfig", "PRESETS", "finetune", "load_finetuned", "smoke_test"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FinetuneConfig:
    """Everything that decides what gets trained and how long for."""

    # --- what to train ---
    checkpoint: str = "tabicl-classifier-v1-20250208.ckpt"
    """Which pretrained TabICL to start from. v1 is the checkpoint the paper built
    on; the repo default is now v2, which has a different prior and will not
    reproduce the paper's numbers even after fine-tuning."""

    kernel: str = "gaussian"
    """Trained with the Gaussian kernel. kNN is non-differentiable (§4.1), so the
    paper trains the embedding with Gaussian and swaps the kernel at evaluation --
    they share the same distance-based structure. The dot-product variant is a
    separate training run."""

    d_k: int = 512
    """Projection dimension. 512 in the paper's main results (Table 9)."""

    gamma: Optional[float] = None
    """Kernel scale during *training*. None uses the head's default. This is not
    the scale you deploy with -- that one is cross-validated per dataset
    afterwards."""

    # --- prior sampling (Appendix A) ---
    batch_size: int = 64
    min_features: int = 5
    max_features: int = 100
    max_seq_len: int = 1024
    min_train_size: float = 0.6
    max_train_size: float = 0.8
    prior_n_jobs: int = -1
    """Synthetic data is generated on CPU while the GPU trains, so this wants most
    of your cores or the GPU will sit idle waiting for batches."""

    # --- optimisation ---
    steps: int = 5000
    micro_batch: int = 8
    """Datasets per forward pass. ``batch_size`` is reached by accumulating
    gradients over ``batch_size / micro_batch`` of these, so the effective batch is
    unchanged while peak memory tracks the micro-batch."""
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    """The head is randomly initialised (or identity) while the backbone is
    pretrained, so they want different rates -- a backbone rate that trains the
    head sensibly would destroy the pretrained representation."""
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # --- memory / speed ---
    amp: bool = True
    recompute: bool = False
    """Gradient checkpointing. Roughly halves activation memory for extra compute.
    Symmetric mode already runs 2n+m query positions through the ICL transformer
    and keeps them all for the backward pass, so this is the first dial to turn
    when you hit OOM."""

    # --- validation and output ---
    val_batches: int = 32
    val_every: int = 250
    out_path: str = "kernelicl_finetuned.pt"
    seed: int = 0
    device: Optional[str] = None
    log_every: int = 25


PRESETS = {
    # Appendix A verbatim. ~40 GB.
    "paper": FinetuneConfig(),
    # Fits comfortably on a 24 GB card; a real run, smaller.
    "medium": FinetuneConfig(steps=2000, micro_batch=4, max_seq_len=768,
                             max_features=60, val_batches=8, val_every=200,
                             recompute=True),
    # Fits a 16 GB T4. Enough to see whether the loss moves, not enough to trust.
    "small": FinetuneConfig(steps=500, batch_size=16, micro_batch=2,
                            max_seq_len=512, max_features=40, val_batches=4,
                            val_every=100, warmup_steps=50, recompute=True),
}


# --------------------------------------------------------------------------- #
# Model assembly
# --------------------------------------------------------------------------- #
def load_pretrained(checkpoint: str, device: str) -> tuple[TabICL, dict]:
    """Fetch a pretrained TabICL and rebuild it, mirroring TabICLClassifier."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="jingang/TabICL", filename=checkpoint)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt["config"]


def trainable_parameters(model: TabICL, head: nn.Module) -> tuple[list, list]:
    """(backbone, head) parameter groups.

    Everything that shapes the embedding is trained: column embedding, row
    interaction, the ICL transformer, its final norm, and the label encoder ``g``.
    The MLP decoder is deliberately excluded -- the kernel head replaces it, so its
    gradients would be wasted work on an output nobody reads.
    """
    backbone = []
    for module in (model.col_embedder, model.row_interactor,
                   model.icl_predictor.tf_icl, model.icl_predictor.y_encoder):
        backbone += list(module.parameters())
    if getattr(model.icl_predictor, "norm_first", False):
        backbone += list(model.icl_predictor.ln.parameters())

    decoder_ids = {id(p) for p in model.icl_predictor.decoder.parameters()}
    backbone = [p for p in backbone if id(p) not in decoder_ids]
    for p in model.icl_predictor.decoder.parameters():
        p.requires_grad_(False)

    return backbone, list(head.parameters())


def set_recompute(model: TabICL, enabled: bool) -> None:
    """Toggle gradient checkpointing on a model rebuilt from a checkpoint."""
    model.col_embedder.tf_col.recompute = enabled
    model.row_interactor.recompute = enabled
    model.row_interactor.tf_row.recompute = enabled
    model.icl_predictor.tf_icl.recompute = enabled


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def kernel_loss(probs: torch.Tensor, y_test: torch.Tensor, eps: float = 1e-8):
    """Negative log-likelihood of kernel predictions.

    The head returns **probabilities**, not logits -- they are already normalised
    by the kernel. Passing them to ``F.cross_entropy`` would apply a second softmax
    and flatten the gradients almost to nothing, so take the log directly.

    Test samples whose class never appears in their dataset's context are dropped.
    Such a sample has probability exactly zero on the truth by construction, so it
    contributes ``-log(eps)`` and a gradient that cannot be acted on. The prior
    occasionally produces these, and the package's own fine-tuning module skips
    them for the same reason.
    """
    B, m, C = probs.shape
    flat_probs = probs.reshape(-1, C)
    flat_true = y_test.reshape(-1).long()

    reachable = flat_probs.gather(1, flat_true[:, None]).squeeze(1) > 0
    if not reachable.any():
        return None, 0

    loss = F.nll_loss((flat_probs[reachable] + eps).log(), flat_true[reachable])
    return loss, int(reachable.sum())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _forward_batch(model, head, batch, cfg, device, num_classes):
    """One micro-batch forward. Returns (loss, n_used, accuracy, mean_perplexity)."""
    X, y, d, seq_lens, train_sizes = batch
    # All datasets in a prior batch share a sequence length and split position when
    # seq_len_per_gp is off, which it is by default.
    seq_len = int(seq_lens[0])
    train_size = int(train_sizes[0])
    X = X[:, :seq_len].to(device)
    y = y[:, :seq_len].to(device)

    y_train, y_test = y[:, :train_size], y[:, train_size:]
    if y_test.shape[1] == 0:
        return None, 0, 0.0, 0.0

    probs, w = model.forward_kernel(
        X, y_train, kernel_head=head, num_classes=num_classes, gamma=cfg.gamma,
    )
    loss, n_used = kernel_loss(probs, y_test)
    if loss is None:
        return None, 0, 0.0, 0.0

    with torch.no_grad():
        acc = (probs.argmax(-1) == y_test.long()).float().mean().item()
        ppl = relative_perplexity(w).mean().item()
    return loss, n_used, acc, ppl


def _make_prior(cfg, device):
    """A prior dataset, with the parallel worker count probed rather than assumed.

    Multi-process generation is a large speedup -- synthetic batches are built on
    CPU while the GPU trains, so a single worker can starve the GPU -- but some
    hyperparameter samplers build local closures that cannot be pickled, and the
    failure surfaces from inside a worker pool with an opaque traceback. Probing
    once here turns that into a warning at setup instead of a crash mid-run.
    """
    from tabicl.prior import PriorDataset

    def build(n_jobs):
        return PriorDataset(
            regression=False,
            batch_size=cfg.micro_batch,
            min_features=cfg.min_features,
            max_features=cfg.max_features,
            max_seq_len=cfg.max_seq_len,
            min_train_size=cfg.min_train_size,
            max_train_size=cfg.max_train_size,
            n_jobs=n_jobs,
            device="cpu",
        )

    if cfg.prior_n_jobs in (1, 0):
        return build(1)

    prior = build(cfg.prior_n_jobs)
    try:
        prior.get_batch()
        return prior
    except Exception as exc:  # pickling, worker startup, platform quirks
        print(f"! parallel prior generation failed ({type(exc).__name__}: "
              f"{str(exc)[:80]}); falling back to a single worker.")
        print("  Batch generation is now serial and may bottleneck the GPU.")
        return build(1)


@torch.no_grad()
def evaluate(model, head, val_batches, cfg, device, num_classes):
    """Mean loss / accuracy / perplexity over a fixed set of prior batches.

    The batches are generated once and reused, so validation loss is comparable
    across steps. Regenerating them each time would measure the prior's variance
    as much as the model's progress.
    """
    model.eval()
    head.eval()
    losses, accs, ppls, total = [], [], [], 0
    for batch in val_batches:
        loss, n_used, acc, ppl = _forward_batch(model, head, batch, cfg, device, num_classes)
        if loss is None:
            continue
        losses.append(loss.item() * n_used)
        accs.append(acc)
        ppls.append(ppl)
        total += n_used
    model.train()
    head.train()
    if not total:
        return dict(loss=float("nan"), accuracy=float("nan"), perplexity=float("nan"))
    return dict(loss=sum(losses) / total, accuracy=float(np.mean(accs)),
                perplexity=float(np.mean(ppls)))


def finetune(cfg: FinetuneConfig = None, preset: str = "medium") -> dict:
    """Fine-tune the embedding module and projection on synthetic prior data.

    Returns a history dict and writes the best checkpoint to ``cfg.out_path``,
    selected by validation loss as in Appendix A.
    """
    cfg = cfg or PRESETS[preset]
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model, model_config = load_pretrained(cfg.checkpoint, device)
    num_classes = model.max_classes
    d_model = model.embed_dim * model.row_num_cls
    head = KernelHead(d_model=d_model, d_k=cfg.d_k, kernel=cfg.kernel,
                      identity_init=(cfg.d_k == d_model)).to(device)
    model.kernel_head = head
    set_recompute(model, cfg.recompute)
    model.train()

    backbone_params, head_params = trainable_parameters(model, head)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.lr_backbone},
         {"params": head_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )

    def lr_scale(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    amp_enabled = cfg.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)

    prior = _make_prior(cfg, device)
    print(f"generating {cfg.val_batches} validation batches...")
    val_batches = [prior.get_batch() for _ in range(cfg.val_batches)]

    accum = max(cfg.batch_size // cfg.micro_batch, 1)
    n_backbone = sum(p.numel() for p in backbone_params)
    print(f"device={device} | amp={amp_enabled}({amp_dtype if amp_enabled else '-'}) | "
          f"recompute={cfg.recompute}")
    print(f"training {n_backbone / 1e6:.1f}M backbone params + "
          f"{sum(p.numel() for p in head_params) / 1e6:.2f}M head params")
    print(f"{cfg.steps} steps x {accum} x {cfg.micro_batch} datasets "
          f"= effective batch {accum * cfg.micro_batch}\n")

    history = {"step": [], "loss": [], "val": []}
    best_val, best_state = float("inf"), None
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        step_loss, step_acc, step_ppl, n_micro = 0.0, 0.0, 0.0, 0

        for _ in range(accum):
            batch = prior.get_batch()
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                loss, n_used, acc, ppl = _forward_batch(
                    model, head, batch, cfg, device, num_classes)
            if loss is None:
                continue
            scaler.scale(loss / accum).backward()
            step_loss += loss.item()
            step_acc += acc
            step_ppl += ppl
            n_micro += 1

        if n_micro == 0:
            continue

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(backbone_params + head_params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        history["step"].append(step)
        history["loss"].append(step_loss / n_micro)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            rate = (step + 1) / (time.perf_counter() - t0)
            print(f"step {step:>5}/{cfg.steps}  loss {step_loss / n_micro:.4f}  "
                  f"acc {step_acc / n_micro:.3f}  rel.PPL {100 * step_ppl / n_micro:.1f}%  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} step/s")

        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            metrics = evaluate(model, head, val_batches, cfg, device, num_classes)
            history["val"].append({"step": step, **metrics})
            flag = ""
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                # Appendix A selects the parameters at the lowest validation loss,
                # not the last ones. Kept on CPU so a long run does not hold two
                # copies of the model in GPU memory.
                # Exclude the head's own parameters from the backbone snapshot.
                # Attaching the head registers it as a submodule, so its weights
                # appear under "kernel_head.*" in model.state_dict() and would make
                # the saved dict incompatible with a plain TabICL(**config) --
                # including TabICLClassifier's own loader. Kept separate instead.
                best_state = {
                    "model": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()
                              if not k.startswith("kernel_head.")},
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                }
                flag = "  <- best"
            print(f"  [val] step {step:>5}  loss {metrics['loss']:.4f}  "
                  f"acc {metrics['accuracy']:.3f}  rel.PPL {100 * metrics['perplexity']:.1f}%{flag}")

    if best_state is not None:
        payload = {
            "config": model_config,
            "state_dict": best_state["model"],
            "kernel_head": best_state["head"],
            "head_config": {"d_model": d_model, "d_k": cfg.d_k, "kernel": cfg.kernel},
            "val_loss": best_val,
            "finetune_config": cfg.__dict__,
        }
        Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cfg.out_path)
        print(f"\nsaved {cfg.out_path} (validation loss {best_val:.4f})")
    else:
        print("\nno validation checkpoint was taken -- nothing saved")

    return history


def load_finetuned(path: str, device: Optional[str] = None) -> tuple[TabICL, KernelHead]:
    """Reload a fine-tuned model and its head, ready for the rest of the tooling.

    The returned model has ``kernel_head`` attached, so ``forward_kernel`` works
    directly. To use it with ``kernelicl_clinical``, hand both to
    :class:`ClinicalExplainer` rather than calling ``fit_explainer``, which always
    loads a pretrained checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TabICL(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    head = KernelHead(**payload["head_config"])
    head.load_state_dict(payload["kernel_head"])
    model.kernel_head = head
    return model.to(device).eval(), head.to(device).eval()


def smoke_test(device: str = "cpu") -> None:
    """Two optimiser steps on tiny batches, to catch shape and dtype errors cheaply.

    Runs in about a minute on CPU. Worth doing before any GPU run: the failures
    this catches otherwise surface forty minutes in.
    """
    cfg = FinetuneConfig(
        steps=2, batch_size=2, micro_batch=1, max_seq_len=128, max_features=8,
        min_features=4, val_batches=1, val_every=1, warmup_steps=1, log_every=1,
        amp=False, prior_n_jobs=1, device=device,
        out_path="/tmp/kicl_smoke.pt",
    )
    print("=== smoke test ===")
    history = finetune(cfg)
    assert history["loss"], "no optimiser step completed"
    assert history["val"], "validation never ran"
    model, head = load_finetuned(cfg.out_path, device=device)
    assert model.kernel_head is not None
    print("=== smoke test passed: forward, backward, save and reload all work ===")


if __name__ == "__main__":
    smoke_test()
