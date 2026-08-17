"""Fine-tune TabICL end-to-end with a kernel head.

Trains the embedding module together with the projection W on synthetic data from
TabICL's own prior, with the loss flowing through the kernel rather than an MLP.
Section 4.1 and Appendix A of the paper.

Requires the pretrain extra for the prior: pip install -e ".[pretrain]"

See README.md for what the presets trade away and how to tell a run is working.
"""

from __future__ import annotations

import math
import os
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

__all__ = ["FinetuneConfig", "PRESETS", "benchmark_prior", "finetune",
           "load_finetuned", "smoke_test"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FinetuneConfig:
    """What gets trained, on what, and for how long."""

    # Starting point. v1 is the checkpoint the paper built on; v2 is stronger but was
    # pretrained on a different prior, so match prior_type below to whichever you use.
    checkpoint: str = "tabicl-classifier-v1-20250208.ckpt"

    prior_type: str = "mlp_scm"
    """Which synthetic prior to sample. "mlp_scm" is Appendix A ("random MLP
    functions") and suits the v1 checkpoint; "graph_scm" is what v2 was pretrained on.
    Also accepts "tree_scm" and "mix_scm". Fine-tuning against a prior the checkpoint
    never saw asks it to adapt to two things at once."""

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
    prior_threads: int = 0
    """Background threads generating synthetic batches; 0 picks a conservative default
    from the CPU count. More is not better: producer threads hold the GIL to build
    each dataset, and the main thread needs it to dispatch CUDA kernels, so too many
    can starve the GPU and end up slower than generating inline. The log reports the
    share of each step spent waiting on data -- tune from that, or from
    benchmark_prior(). 1 reproduces plain inline generation."""

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
    """Written atomically every time validation improves, not only at the end, so an
    interrupted run keeps its best checkpoint. Point this at mounted Drive on Colab."""
    resume_from: Optional[str] = None
    """Continue from a local checkpoint instead of a pretrained one. A warm restart:
    optimiser state and the schedule are not saved, so both begin again."""
    seed: int = 0
    device: Optional[str] = None
    log_every: int = 25


V1_CHECKPOINT = "tabicl-classifier-v1-20250208.ckpt"
V2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"

# Coherent pairings: the prior has to match what the checkpoint was pretrained on.
# Everything else here defaults to v2, matching TabICLClassifier, so V2 keeps the
# whole toolkit on one lineage; V1 reproduces the paper.
V1 = {"checkpoint": V1_CHECKPOINT, "prior_type": "mlp_scm"}
V2 = {"checkpoint": V2_CHECKPOINT, "prior_type": "graph_scm"}

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


def _starting_point(cfg, device):
    """(model, model_config, head_state) to begin from.

    ``resume_from`` continues from a local checkpoint instead of a pretrained one.
    It is a warm restart, not an exact resume: optimiser state and the learning-rate
    schedule are not saved, so both begin again. Useful when a long run is cut short.
    """
    if not cfg.resume_from:
        model, model_config = load_pretrained(cfg.checkpoint, device)
        return model, model_config, None

    payload = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
    model = TabICL(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    print(f"resuming from {cfg.resume_from} (validation loss "
          f"{payload.get('val_loss', float('nan')):.4f}); optimiser state restarts")
    return model.to(device), payload["config"], payload.get("kernel_head")


def _save_checkpoint(path: str, model, head, model_config: dict, d_model: int,
                     cfg, val_loss: float) -> None:
    """Write the checkpoint atomically.

    Written to a temporary file and renamed, so an interrupted write cannot leave a
    truncated checkpoint where a good one used to be -- the difference between losing
    the last validation interval and losing the whole run, especially on a network
    filesystem like a mounted Drive.

    ``kernel_head.*`` is excluded from the backbone dict: attaching the head registers
    it as a submodule, and including it would make the dict unloadable by a plain
    ``TabICL(**config)``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({
        "config": model_config,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                       if not k.startswith("kernel_head.")},
        "kernel_head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "head_config": {"d_model": d_model, "d_k": cfg.d_k, "kernel": cfg.kernel},
        "val_loss": val_loss,
        "finetune_config": cfg.__dict__,
    }, temporary)
    os.replace(temporary, destination)


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


def _build_prior(cfg):
    """One prior dataset. ``n_jobs=1``: its own process pool cannot be used here.

    ``PriorDataset`` dispatches datasets within a batch through
    ``multiprocessing.Pool.map``, which must pickle the sampled hyperparameters. Those
    values are closures returned by ``HpSampler``, so the pool dies with an
    ``AttributeError`` about a local object. Parallelism comes from threads instead --
    see :class:`_BatchPrefetcher`.
    """
    from tabicl.prior import PriorDataset

    return PriorDataset(
        regression=False,
        prior_type=cfg.prior_type,
        batch_size=cfg.micro_batch,
        min_features=cfg.min_features,
        max_features=cfg.max_features,
        max_seq_len=cfg.max_seq_len,
        min_train_size=cfg.min_train_size,
        max_train_size=cfg.max_train_size,
        n_jobs=1,
        device="cpu",
    )


class _BatchPrefetcher:
    """Generates prior batches on background threads, overlapping them with training.

    Synthetic data is built on the CPU while the GPU trains, so generation throughput
    sets the floor on step time. At Appendix A settings one batch of 8 datasets takes
    about 0.7s single-threaded, which is ~6s of generation per step and over eight
    hours across 5000 steps -- with the GPU idle for nearly all of it.

    Threads rather than processes, because the prior's hyperparameters are closures
    that ``multiprocessing`` cannot pickle. Most of the work is numpy and torch, which
    release the GIL, so threading still recovers roughly 4x with 8 threads.

    Each thread owns its own ``PriorDataset``: ``HpSampler`` stores sampled
    hyperparameters on the instance with ``setattr``, so sharing one across threads
    would race.
    """

    def __init__(self, cfg, n_threads: int):
        import queue
        import threading

        # Each producer runs torch CPU ops, and torch defaults to one intra-op thread
        # per core, so N producers would ask for N x cores of them. The prior's own
        # process pool sets this to 1 in its workers for the same reason. Restored in
        # close(); the main thread is on the GPU, so its CPU intra-op count is moot.
        self._torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        self._queue = queue.Queue(maxsize=max(2 * n_threads, 4))
        self._stop = threading.Event()
        self._error = None
        self._queue_module = queue
        self._threads = [
            threading.Thread(target=self._work, args=(_build_prior(cfg),), daemon=True)
            for _ in range(n_threads)
        ]
        for thread in self._threads:
            thread.start()

    def _work(self, prior):
        while not self._stop.is_set():
            try:
                batch = prior.get_batch()
            except BaseException as exc:  # surface it on the consumer's next call
                self._error = exc
                self._stop.set()
                return
            while not self._stop.is_set():
                try:
                    self._queue.put(batch, timeout=0.5)
                    break
                except self._queue_module.Full:
                    continue

    def get_batch(self):
        while True:
            try:
                return self._queue.get(timeout=1.0)
            except self._queue_module.Empty:
                if self._error is not None:
                    raise RuntimeError("prior generation failed") from self._error
                if not any(t.is_alive() for t in self._threads):
                    raise RuntimeError("all prior generation threads have stopped")

    def close(self):
        self._stop.set()
        torch.set_num_threads(self._torch_threads)


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

    model, model_config, head_state = _starting_point(cfg, device)
    num_classes = model.max_classes
    d_model = model.embed_dim * model.row_num_cls
    head = KernelHead(d_model=d_model, d_k=cfg.d_k, kernel=cfg.kernel,
                      identity_init=(cfg.d_k == d_model)).to(device)
    if head_state is not None:
        head.load_state_dict(head_state)
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

    # Conservative by default: past a handful of producers the GIL contention with
    # the main thread's CUDA dispatch outweighs the extra generation throughput.
    n_threads = cfg.prior_threads or max(1, min(4, (os.cpu_count() or 2) // 2))
    prior = _BatchPrefetcher(cfg, n_threads)
    print(f"generating {cfg.val_batches} validation batches "
          f"on {n_threads} thread{'s' if n_threads > 1 else ''}...")
    val_batches = [prior.get_batch() for _ in range(cfg.val_batches)]

    accum = max(cfg.batch_size // cfg.micro_batch, 1)
    print(f"device={device} | amp={amp_on} | recompute={cfg.recompute}")
    print(f"training {sum(p.numel() for p in backbone_params) / 1e6:.1f}M backbone + "
          f"{sum(p.numel() for p in head_params) / 1e6:.2f}M head params")
    print(f"{cfg.steps} steps x {accum} x {cfg.micro_batch} datasets "
          f"= effective batch {accum * cfg.micro_batch}\n")

    history = {"step": [], "loss": [], "val": []}
    params = backbone_params + head_params
    best_val, saved_any = float("inf"), False
    started = time.perf_counter()

    try:
        for step in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            totals = np.zeros(3)  # loss, accuracy, perplexity
            n_micro = 0
            wait_seconds = 0.0
            step_started = time.perf_counter()

            for _ in range(accum):
                waiting = time.perf_counter()
                batch = prior.get_batch()
                wait_seconds += time.perf_counter() - waiting
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
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            mean_loss, mean_acc, mean_ppl = totals / n_micro
            history["step"].append(step)
            history["loss"].append(mean_loss)

            step_seconds = time.perf_counter() - step_started
            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                rate = (step + 1) / (time.perf_counter() - started)
                # "wait" is the share of the step spent blocked on synthetic data. High
                # means data-bound: raise prior_threads. Near zero means the GPU is the
                # limit and more threads will only take the GIL from CUDA dispatch.
                print(f"step {step:>5}/{cfg.steps}  loss {mean_loss:.4f}  acc {mean_acc:.3f}  "
                      f"rel.PPL {100 * mean_ppl:.1f}%  lr {scheduler.get_last_lr()[0]:.2e}  "
                      f"{rate:.2f} step/s  wait {100 * wait_seconds / max(step_seconds, 1e-9):.0f}%")

            if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
                metrics = evaluate(model, head, val_batches, cfg, device, num_classes)
                history["val"].append({"step": step, **metrics})
                improved = metrics["loss"] < best_val
                if improved:
                    best_val = metrics["loss"]
                    _save_checkpoint(cfg.out_path, model, head, model_config, d_model,
                                     cfg, best_val)
                    saved_any = True
                print(f"  [val] step {step:>5}  loss {metrics['loss']:.4f}  "
                      f"acc {metrics['accuracy']:.3f}  rel.PPL {100 * metrics['perplexity']:.1f}%"
                      f"{'  <- best, saved' if improved else ''}")

    finally:
        # Background generation threads must not outlive the run.
        prior.close()

    if not saved_any:
        print("\nno validation checkpoint was taken; nothing saved")
    else:
        print(f"\nbest checkpoint at {cfg.out_path} (validation loss {best_val:.4f})")
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


def benchmark_prior(cfg: Optional[FinetuneConfig] = None, preset: str = "paper",
                    thread_counts=(1, 2, 4, 8), batches: int = 6) -> dict:
    """Measure generation throughput at several thread counts, on this machine.

    Generation speed depends on core count, and the best thread count is not the
    largest -- producers hold the GIL that the main thread needs to dispatch CUDA
    work. This measures generation alone, so treat the winner as an upper bound and
    confirm with the ``wait`` percentage in the training log.
    """
    cfg = cfg or PRESETS[preset]
    print(f"cpu_count={os.cpu_count()}  micro_batch={cfg.micro_batch}  "
          f"max_seq_len={cfg.max_seq_len}  max_features={cfg.max_features}\n")
    print(f"{'threads':>8} {'s/batch':>9} {'s/step':>8}   {'5000 steps':>11}")
    results = {}
    accum = max(cfg.batch_size // cfg.micro_batch, 1)
    for n in thread_counts:
        prefetcher = _BatchPrefetcher(cfg, n)
        try:
            prefetcher.get_batch()
            begun = time.perf_counter()
            for _ in range(batches):
                prefetcher.get_batch()
            per_batch = (time.perf_counter() - begun) / batches
        finally:
            prefetcher.close()
        results[n] = per_batch
        print(f"{n:>8} {per_batch:>9.2f} {per_batch * accum:>8.1f}   "
              f"{per_batch * accum * cfg.steps / 3600:>10.1f}h")
    best = min(results, key=results.get)
    print(f"\nfastest at prior_threads={best}; the training log's wait% is the real test")
    return results


def smoke_test(device: Optional[str] = None) -> None:
    """Two optimiser steps on tiny batches, on the GPU if there is one.

    Catches shape, dtype and device errors that would otherwise surface an hour into
    a real run. About a minute on CPU, seconds on a GPU.
    """
    cfg = FinetuneConfig(
        steps=2, batch_size=2, micro_batch=1, max_seq_len=128, max_features=8,
        min_features=4, val_batches=1, val_every=1, warmup_steps=1, log_every=1,
        amp=False, prior_threads=1, device=device, out_path="/tmp/kicl_smoke.pt",
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
# smoke_test()                              # do this first; seconds on a GPU
#
# # The full Appendix A run, saved where a disconnect cannot take it. The best
# # checkpoint is rewritten atomically every time validation improves, so an
# # interrupted run keeps whatever it had reached.
# from google.colab import drive
# drive.mount("/content/drive")
#
# cfg = FinetuneConfig(**{**PRESETS["paper"].__dict__,
#                         "out_path": "/content/drive/MyDrive/kernelicl/paper.pt"})
# history = finetune(cfg)
#
# # If the session dies, continue from what was saved. A warm restart: optimiser
# # state and the schedule are not stored, so both begin again.
# cfg = FinetuneConfig(**{**PRESETS["paper"].__dict__, "steps": 2000,
#                         "resume_from": "/content/drive/MyDrive/kernelicl/paper.pt",
#                         "out_path": "/content/drive/MyDrive/kernelicl/paper2.pt"})
# finetune(cfg)
#
# # Smaller runs, if you want to check the loss moves before committing hours:
# history = finetune(preset="small")        # T4 / 16 GB    ~30 min
# history = finetune(preset="medium")       # 24 GB         a few hours
#
# # Lineage. The default here is v1 + mlp_scm, which is Appendix A. Every other file
# # defaults to v2, so if you want one lineage throughout, use V2 -- and then leave
# # CHECKPOINT = None in the analysis files, which is already v2.
# cfg = FinetuneConfig(**{**PRESETS["paper"].__dict__, **V2})
#
# # Staying on v1 instead means telling the analysis files so their stock-TabICL
# # baselines match: set CHECKPOINT = V1_CHECKPOINT there.
#
# # Watch the [val] lines, not the step lines: every step draws a fresh random
# # synthetic problem, so train loss reflects that draw rather than progress.
# # history is only for plotting that curve; the checkpoint is already on disk.
# import matplotlib.pyplot as plt
# v = history["val"]
# plt.plot([d["step"] for d in v], [d["loss"] for d in v], marker="o")
# plt.xlabel("step"); plt.ylabel("validation loss"); plt.show()
#
# # Then everything downstream uses it, with the scale recalibrated:
# ex = fit_explainer(X_train, y_train, X_test,
#                    finetuned="/content/drive/MyDrive/kernelicl/paper.pt")
