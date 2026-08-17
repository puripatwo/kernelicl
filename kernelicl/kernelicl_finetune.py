"""Fine-tune TabICL end-to-end with a kernel head.

Trains the embedding module together with the projection W on synthetic data from
TabICL's own prior, with the loss flowing through the kernel rather than an MLP.
Section 4.1 and Appendix A of the paper.

Requires the pretrain extra for the prior: pip install -e ".[pretrain]"

See README.md for what the presets trade away and how to tell a run is working.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
    """What gets trained, on what, and for how long."""

    # Starting point. v1 is the checkpoint the paper built on; the repo default is
    # now v2, which has a different prior.
    checkpoint: str = "tabicl-classifier-v1-20250208.ckpt"

    # Trained with the Gaussian kernel: kNN is non-differentiable, so the paper
    # trains Gaussian and swaps the kernel at evaluation. Dot-product is a separate run.
    kernel: str = "gaussian"
    d_k: int = 512
    gamma: Optional[float] = None
    """Scale during training. None uses the head default; the deployment scale is
    cross-validated separately afterwards."""

    # Prior sampling (Appendix A).
    batch_size: int = 64
    min_features: int = 5
    max_features: int = 100
    max_seq_len: int = 1024
    min_train_size: float = 0.6
    max_train_size: float = 0.8
    prior_n_jobs: int = -1

    # Optimisation.
    steps: int = 5000
    micro_batch: int = 8
    """Datasets per forward pass; gradients accumulate to reach batch_size, so peak
    memory tracks this rather than the effective batch."""
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    """Separate rates: the backbone is pretrained and needs nudging, the head starts
    at identity and needs training."""
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # Memory.
    amp: bool = True
    recompute: bool = False
    """Gradient checkpointing. The first dial to turn on OOM: symmetric mode runs
    2n+m query positions through the ICL transformer and keeps them all for backward."""

    # Validation and output.
    val_batches: int = 32
    val_every: int = 250
    out_path: str = "kernelicl_finetuned.pt"
    seed: int = 0
    device: Optional[str] = None
    log_every: int = 25


PRESETS = {
    "paper": FinetuneConfig(),
    "medium": FinetuneConfig(steps=2000, micro_batch=4, max_seq_len=768, max_features=60,
                             val_batches=8, val_every=200, recompute=True),
    "small": FinetuneConfig(steps=500, batch_size=16, micro_batch=2, max_seq_len=512,
                            max_features=40, val_batches=4, val_every=100,
                            warmup_steps=50, recompute=True),
}


# --------------------------------------------------------------------------- #
# Model assembly
# --------------------------------------------------------------------------- #
def load_pretrained(checkpoint: str, device: str) -> tuple[TabICL, dict]:
    """Fetch and rebuild a pretrained TabICL, mirroring TabICLClassifier."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="jingang/TabICL", filename=checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICL(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload["config"]


def trainable_parameters(model: TabICL, head: nn.Module) -> tuple[list, list]:
    """(backbone, head) parameter groups.

    Everything shaping the embedding trains: column embedding, row interaction, the
    ICL transformer, its norm, and the label encoder. The MLP decoder is frozen --
    the kernel replaces it, so its gradients would be work on an unread output.
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
# Loss and forward
# --------------------------------------------------------------------------- #
def kernel_loss(probs: torch.Tensor, y_test: torch.Tensor, eps: float = 1e-8):
    """Negative log-likelihood of kernel predictions.

    The head returns probabilities, already normalized by the kernel, so take the
    log directly. F.cross_entropy would apply a second softmax and flatten the
    gradients.

    Test samples whose class never appears in their context are dropped: their
    probability on the truth is exactly zero by construction, contributing
    -log(eps) and a gradient that cannot be acted on.
    """
    n_classes = probs.shape[-1]
    flat_probs = probs.reshape(-1, n_classes)
    flat_true = y_test.reshape(-1).long()

    reachable = flat_probs.gather(1, flat_true[:, None]).squeeze(1) > 0
    if not reachable.any():
        return None, 0
    return F.nll_loss((flat_probs[reachable] + eps).log(), flat_true[reachable]), int(reachable.sum())


def _forward_batch(model, head, batch, cfg, device, num_classes):
    """One micro-batch. Returns (loss, n_used, accuracy, mean_perplexity)."""
    X, y, _d, seq_lens, train_sizes = batch
    # Datasets in a prior batch share a sequence length and split position.
    seq_len, train_size = int(seq_lens[0]), int(train_sizes[0])
    X = X[:, :seq_len].to(device)
    y = y[:, :seq_len].to(device)
    y_train, y_test = y[:, :train_size], y[:, train_size:]
    if y_test.shape[1] == 0:
        return None, 0, 0.0, 0.0

    probs, w = model.forward_kernel(X, y_train, kernel_head=head,
                                    num_classes=num_classes, gamma=cfg.gamma)
    loss, n_used = kernel_loss(probs, y_test)
    if loss is None:
        return None, 0, 0.0, 0.0

    with torch.no_grad():
        accuracy = (probs.argmax(-1) == y_test.long()).float().mean().item()
        perplexity = relative_perplexity(w).mean().item()
    return loss, n_used, accuracy, perplexity


def _make_prior(cfg):
    """A prior dataset, with the worker count probed rather than assumed.

    Some hyperparameter samplers build unpicklable closures, and the failure
    surfaces from inside a worker pool. Probing turns that into a setup warning.
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

    if cfg.prior_n_jobs in (0, 1):
        return build(1)

    prior = build(cfg.prior_n_jobs)
    try:
        prior.get_batch()
        return prior
    except Exception as exc:
        print(f"! parallel prior generation failed ({type(exc).__name__}: {str(exc)[:80]}); "
              f"falling back to one worker. Generation may now bottleneck the GPU.")
        return build(1)


@torch.no_grad()
def evaluate(model, head, val_batches, cfg, device, num_classes) -> dict:
    """Mean loss, accuracy and perplexity over a fixed set of prior batches.

    The batches are generated once and reused, so validation loss is comparable
    across steps.
    """
    model.eval()
    head.eval()
    weighted_loss, accuracies, perplexities, total = 0.0, [], [], 0
    for batch in val_batches:
        loss, n_used, accuracy, perplexity = _forward_batch(
            model, head, batch, cfg, device, num_classes)
        if loss is None:
            continue
        weighted_loss += loss.item() * n_used
        accuracies.append(accuracy)
        perplexities.append(perplexity)
        total += n_used
    model.train()
    head.train()

    if not total:
        return dict(loss=float("nan"), accuracy=float("nan"), perplexity=float("nan"))
    return dict(loss=weighted_loss / total,
                accuracy=float(np.mean(accuracies)),
                perplexity=float(np.mean(perplexities)))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def finetune(cfg: Optional[FinetuneConfig] = None, preset: str = "medium") -> dict:
    """Fine-tune on synthetic prior data.

    Writes the best checkpoint by validation loss to ``cfg.out_path``, as Appendix A
    specifies, and returns a history dict.
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
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.lr_backbone},
         {"params": head_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )

    def lr_scale(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    amp_on = cfg.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if amp_on and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and amp_dtype is torch.float16)

    prior = _make_prior(cfg)
    print(f"generating {cfg.val_batches} validation batches...")
    val_batches = [prior.get_batch() for _ in range(cfg.val_batches)]

    accum = max(cfg.batch_size // cfg.micro_batch, 1)
    print(f"device={device} | amp={amp_on} | recompute={cfg.recompute}")
    print(f"training {sum(p.numel() for p in backbone_params) / 1e6:.1f}M backbone + "
          f"{sum(p.numel() for p in head_params) / 1e6:.2f}M head params")
    print(f"{cfg.steps} steps x {accum} x {cfg.micro_batch} datasets "
          f"= effective batch {accum * cfg.micro_batch}\n")

    history = {"step": [], "loss": [], "val": []}
    best_val, best_state = float("inf"), None
    started = time.perf_counter()

    for step in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3)  # loss, accuracy, perplexity
        n_micro = 0

        for _ in range(accum):
            batch = prior.get_batch()
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_on):
                loss, _n_used, accuracy, perplexity = _forward_batch(
                    model, head, batch, cfg, device, num_classes)
            if loss is None:
                continue
            scaler.scale(loss / accum).backward()
            totals += (loss.item(), accuracy, perplexity)
            n_micro += 1

        if n_micro == 0:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(backbone_params + head_params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        mean_loss, mean_acc, mean_ppl = totals / n_micro
        history["step"].append(step)
        history["loss"].append(mean_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            rate = (step + 1) / (time.perf_counter() - started)
            print(f"step {step:>5}/{cfg.steps}  loss {mean_loss:.4f}  acc {mean_acc:.3f}  "
                  f"rel.PPL {100 * mean_ppl:.1f}%  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"{rate:.2f} step/s")

        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            metrics = evaluate(model, head, val_batches, cfg, device, num_classes)
            history["val"].append({"step": step, **metrics})
            improved = metrics["loss"] < best_val
            if improved:
                best_val = metrics["loss"]
                # Excluding kernel_head.* keeps the backbone dict loadable by a plain
                # TabICL(**config): attaching the head registers it as a submodule.
                best_state = {
                    "model": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()
                              if not k.startswith("kernel_head.")},
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                }
            print(f"  [val] step {step:>5}  loss {metrics['loss']:.4f}  "
                  f"acc {metrics['accuracy']:.3f}  rel.PPL {100 * metrics['perplexity']:.1f}%"
                  f"{'  <- best' if improved else ''}")

    if best_state is None:
        print("\nno validation checkpoint was taken; nothing saved")
        return history

    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model_config,
        "state_dict": best_state["model"],
        "kernel_head": best_state["head"],
        "head_config": {"d_model": d_model, "d_k": cfg.d_k, "kernel": cfg.kernel},
        "val_loss": best_val,
        "finetune_config": cfg.__dict__,
    }, cfg.out_path)
    print(f"\nsaved {cfg.out_path} (validation loss {best_val:.4f})")
    return history


def load_finetuned(path: str, device: Optional[str] = None) -> tuple[TabICL, KernelHead]:
    """Reload a fine-tuned model with its head attached, ready for forward_kernel.

    To use it with kernelicl_clinical, pass the path to ``fit_explainer(finetuned=...)``
    instead -- it handles the preprocessing and recalibrates the scale.
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
    """Two optimiser steps on tiny batches. About a minute on CPU.

    Catches shape and dtype errors that would otherwise surface an hour into a
    GPU run.
    """
    cfg = FinetuneConfig(
        steps=2, batch_size=2, micro_batch=1, max_seq_len=128, max_features=8,
        min_features=4, val_batches=1, val_every=1, warmup_steps=1, log_every=1,
        amp=False, prior_n_jobs=1, device=device, out_path="/tmp/kicl_smoke.pt",
    )
    history = finetune(cfg)
    assert history["loss"], "no optimiser step completed"
    assert history["val"], "validation never ran"
    model, _head = load_finetuned(cfg.out_path, device=device)
    assert model.kernel_head is not None
    print("\nsmoke test passed: forward, backward, save and reload all work")


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
# smoke_test()                              # ~1 min, do this first
#
# history = finetune(preset="small")        # T4 / 16 GB    ~30 min, a sanity run
# history = finetune(preset="medium")       # 24 GB         a few hours
# history = finetune(preset="paper")        # A100 / 40 GB  Appendix A verbatim
#
# # Watch the [val] lines, not the step lines: every step draws a fresh random
# # synthetic problem, so train loss reflects that draw rather than progress.
#
# # Then everything downstream uses it, with the scale recalibrated:
# ex = fit_explainer(X_train, y_train, X_test, finetuned="kernelicl_finetuned.pt")
#
# # Save somewhere that survives a disconnect:
# from google.colab import drive; drive.mount("/content/drive")
# cfg = FinetuneConfig(**{**PRESETS["medium"].__dict__,
#                         "out_path": "/content/drive/MyDrive/kicl_ft.pt"})
# finetune(cfg)
