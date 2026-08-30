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

__all__ = ["FinetuneConfig", "PRESETS", "benchmark_micro_batch",
           "describe_checkpoint", "finetune", "load_finetuned", "smoke_test"]


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
    prior_n_jobs: int = -1
    """Dataloader workers generating synthetic batches in the background. -1 uses every
    core. Generation is the usual bottleneck, not the GPU."""
    prefetch_factor: int = 2
    """Batches each worker keeps ready. Raise if the GPU still waits on data."""
    tf32: bool = True
    """TF32 matmuls. A large free speedup on Ampere and later; no effect elsewhere."""

    # Optimisation.
    steps: int = 5000
    micro_batch: int = 8
    """Datasets per forward pass; gradients accumulate to reach batch_size, so peak
    memory tracks this rather than the effective batch. Once data generation is no
    longer the bottleneck this is the main speed lever: raising it means fewer, larger
    GPU launches for the same effective batch. Raise until it OOMs, then step back."""
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    """Separate rates: the backbone is pretrained and needs nudging, the head starts
    at identity and needs training.

    The backbone rate matches TabICLv1's own pretraining. Measured over 80 steps with
    everything else fixed, the rate does not change *whether* validation plateaus --
    it does, within about twenty steps -- but it does change the level it plateaus at:
    1.073 at 1e-5, 1.022 at 1e-4, 0.924 at 3e-4. Higher is tempting, but this is a
    pretrained model and a rate high enough to reshape it quickly is also high enough
    to damage what it already knows, which prior loss will not show. Check T5/T6 on
    your own data before trusting a larger value."""
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
    patience: int = 8
    """Stop after this many validations without improvement. Training on the prior
    saturates early -- the loss floor belongs to the random problems being generated,
    not to the model -- so a full run often spends hours after the last gain. 0
    disables it."""
    out_path: str = "kernelicl_finetuned.pt"
    """Written atomically every time validation improves, not only at the end, so an
    interrupted run keeps its best checkpoint. Point this at mounted Drive on Colab."""
    resume_from: Optional[str] = None
    """Continue from a local checkpoint instead of a pretrained one. A warm restart:
    optimiser state and the schedule are not saved, so both begin again."""
    seed: int = 0
    device: Optional[str] = None
    log_every: int = 25


V2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"

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
    X = X[:, :seq_len].to(device, non_blocking=True)
    y = y[:, :seq_len].to(device, non_blocking=True)
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


def _seed_worker(worker_id: int) -> None:
    """Give each dataloader worker its own seed, or they all generate the same data."""
    import random

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _make_batches(cfg, device):
    """An iterator of full batches, generated in background workers.

    Two things matter for throughput, both mirroring TabICL's own trainer:

    * ``n_jobs=1`` inside the dataset. ``run_parallel`` builds and tears down a fresh
      process pool on *every* call, so leaving it above 1 forks a pool per batch and
      nests that inside the dataloader's own workers. Parallelism belongs to the
      dataloader.
    * One batch of ``batch_size`` per step, split into micro-batches afterwards --
      not one generation per micro-batch. Combined with ``prefetch_factor``, the CPU
      builds the next batches while the GPU works on this one, instead of the GPU
      idling through every generation.

    Falls back to synchronous generation if workers cannot start.
    """
    from torch.utils.data import DataLoader

    from tabicl.prior import PriorDataset

    dataset = PriorDataset(
        regression=False,
        prior_type=cfg.prior_type,
        batch_size=cfg.batch_size,
        min_features=cfg.min_features,
        max_features=cfg.max_features,
        max_seq_len=cfg.max_seq_len,
        min_train_size=cfg.min_train_size,
        max_train_size=cfg.max_train_size,
        n_jobs=1,
        device="cpu",
    )

    n_workers = cfg.prior_n_jobs if cfg.prior_n_jobs > 0 else (os.cpu_count() or 1)
    if n_workers <= 1:
        return iter(dataset)

    try:
        loader = DataLoader(
            dataset,
            batch_size=None,        # PriorDataset already yields whole batches
            num_workers=n_workers,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=True,
            pin_memory=(device == "cuda"),
            worker_init_fn=_seed_worker,
        )
        batches = iter(loader)
        next(batches)
        return batches
    except Exception as exc:
        print(f"! background batch generation failed ({type(exc).__name__}: "
              f"{str(exc)[:80]}); generating synchronously, which will bottleneck the GPU.")
        return iter(dataset)


def _split_micro_batches(batch, micro_batch: int):
    """One prior batch -> a list of micro-batches, as the reference trainer does."""
    parts = [torch.split(t, micro_batch, dim=0) for t in batch]
    return list(zip(*parts))


@torch.no_grad()
def reference_loss(model, val_batches, cfg, device) -> float:
    """Loss of TabICL's own MLP decoder on the same batches: the headroom baseline.

    Printed once at the start so the training loss can be read against something. The
    prior generates random problems, many of which are close to unlearnable, so a large
    part of the loss is irreducible and the number that matters is the gap to this
    reference rather than the absolute value.

    Measured at init on synthetic batches: the MLP decoder reached 0.8722 and the
    untrained kernel head 0.8839. There is not much room between them, which is why
    training plateaus early -- the kernel head starts near what this architecture can
    do on this data, and the floor belongs to the prior, not to the head.
    """
    model.train()   # prior batches mix class counts, which the eval path forbids
    total, weight = 0.0, 0
    for batch in val_batches:
        X, y, _d, seq_lens, train_sizes = batch
        seq_len, train_size = int(seq_lens[0]), int(train_sizes[0])
        X = X[:, :seq_len].to(device, non_blocking=True)
        y = y[:, :seq_len].to(device, non_blocking=True)
        y_test = y[:, train_size:]
        if y_test.shape[1] == 0:
            continue
        logits = model(X, y[:, :train_size])
        n = y_test.numel()
        total += F.cross_entropy(logits.flatten(end_dim=-2), y_test.long().flatten()).item() * n
        weight += n
    return total / weight if weight else float("nan")


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

    if cfg.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batches = _make_batches(cfg, device)
    print(f"generating {cfg.val_batches} validation batches...")
    val_batches = [mb for _ in range(cfg.val_batches)
                   for mb in _split_micro_batches(next(batches), cfg.micro_batch)]

    # Two different reference points, and they answer different questions. This one is
    # the architecture's floor: what TabICL's own head manages on the same problems.
    # The `baseline` below is the starting point: what the untrained kernel head
    # manages. Progress is measured against the second, headroom against the first.
    mlp_floor = reference_loss(model, val_batches, cfg, device)
    accum = max(cfg.batch_size // cfg.micro_batch, 1)
    print(f"device={device} | amp={amp_on} | recompute={cfg.recompute}")
    print(f"stock MLP decoder on these batches: {mlp_floor:.4f} -- roughly the floor for "
          f"this architecture on this data,\n  so read the kernel loss as a gap to that, "
          f"not as an absolute")
    print(f"training {sum(p.numel() for p in backbone_params) / 1e6:.1f}M backbone + "
          f"{sum(p.numel() for p in head_params) / 1e6:.2f}M head params")
    print(f"{cfg.steps} steps x {accum} x {cfg.micro_batch} datasets "
          f"= effective batch {accum * cfg.micro_batch}\n")

    # Validate before training so every later number has something to beat. Without
    # this baseline a falling validation loss is only relative to the first checkpoint
    # taken, and there is no way to tell whether fine-tuning helped at all.
    baseline = evaluate(model, head, val_batches, cfg, device, num_classes)
    print(f"  [val] baseline    loss {baseline['loss']:.4f}  acc {baseline['accuracy']:.3f}  "
          f"rel.PPL {100 * baseline['perplexity']:.1f}%  (pretrained, untrained head)\n")

    history = {"baseline": baseline, "step": [], "loss": [], "val": []}
    best_val, saved_any, since_best = float("inf"), False, 0
    started = time.perf_counter()
    data_seconds = 0.0
    last_log_time, last_log_step, last_log_data = started, -1, 0.0

    for step in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3)  # loss, accuracy, perplexity
        n_micro = 0

        # One generation per step, split afterwards, so the workers can run ahead.
        waiting = time.perf_counter()
        micro_batches = _split_micro_batches(next(batches), cfg.micro_batch)
        data_seconds += time.perf_counter() - waiting

        for batch in micro_batches:
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
            now = time.perf_counter()
            # Rates over the window since the last log, not since the start: workers
            # spin up over the first steps, so a cumulative average understates
            # current throughput and inflates the projection.
            window_seconds = now - last_log_time
            window_steps = step - last_log_step
            window_data = data_seconds - last_log_data
            rate = window_steps / window_seconds if window_steps else 0.0
            remaining = (cfg.steps - step - 1) / rate / 3600 if rate else float("inf")
            print(f"step {step:>5}/{cfg.steps}  loss {mean_loss:.4f}  acc {mean_acc:.3f}  "
                  f"rel.PPL {100 * mean_ppl:.1f}%  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"{rate:.2f} step/s  "
                  f"({100 * window_data / window_seconds if window_seconds else 0:.0f}% "
                  f"waiting on data, {remaining:.1f}h left)")
            last_log_time, last_log_step, last_log_data = now, step, data_seconds

        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            metrics = evaluate(model, head, val_batches, cfg, device, num_classes)
            history["val"].append({"step": step, **metrics})
            improved = metrics["loss"] < best_val
            if improved:
                best_val = metrics["loss"]
                _save_checkpoint(cfg.out_path, model, head, model_config, d_model,
                                 cfg, best_val)
                saved_any = True
            since_best = 0 if improved else since_best + 1
            gain = baseline["loss"] - metrics["loss"]
            print(f"  [val] step {step:>5}  loss {metrics['loss']:.4f}  "
                  f"acc {metrics['accuracy']:.3f}  rel.PPL {100 * metrics['perplexity']:.1f}%  "
                  f"({gain:+.4f} vs baseline)"
                  f"{'  <- best, saved' if improved else f'  [{since_best}/{cfg.patience}]'}")
            if cfg.patience and since_best >= cfg.patience:
                print(f"\nstopping at step {step}: {cfg.patience} validations without "
                      f"improvement. The best checkpoint is already saved.")
                break

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


def describe_checkpoint(path: str) -> dict:
    """What produced a saved checkpoint, without loading it onto a device.

    A fine-tuned checkpoint is dataset-independent and reusable indefinitely, so it
    tends to outlive the memory of how it was made. Everything needed to answer that
    is stored alongside the weights.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("finetune_config", {})
    summary = {
        "val_loss": payload.get("val_loss"),
        "started_from": cfg.get("resume_from") or cfg.get("checkpoint"),
        "prior_type": cfg.get("prior_type"),
        "kernel": cfg.get("kernel"),
        "d_k": cfg.get("d_k"),
        "steps": cfg.get("steps"),
        "max_features": cfg.get("max_features"),
        "max_seq_len": cfg.get("max_seq_len"),
        "size_mb": round(os.path.getsize(path) / 1e6),
    }
    width = max(len(k) for k in summary)
    print(f"{path}")
    for key, value in summary.items():
        print(f"  {key:<{width}}  {value}")
    return summary


def benchmark_micro_batch(preset: str = "paper", candidates=(8, 16, 32),
                          steps: int = 12, cfg: Optional[FinetuneConfig] = None) -> dict:
    """Time a few steps at each micro_batch to find the fastest that fits.

    Once generation is no longer the bottleneck, the effective batch is fixed and the
    only remaining lever is how many datasets go through the GPU at once: larger
    micro-batches mean fewer, bigger launches for the same work. How large fits
    depends on the card, so measure rather than guess. Reports peak memory too, so a
    setting can be rejected as too close to the limit before a long run relies on it.
    """
    base = cfg or PRESETS[preset]
    device = base.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    print(f"{'micro_batch':>12} {'accum':>6} {'step/s':>8} {'peak GB':>9}")
    for micro in candidates:
        if base.batch_size % micro:
            continue
        trial = FinetuneConfig(**{**base.__dict__, "micro_batch": micro, "steps": steps})
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            rate = _time_steps(trial, device, steps)
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
            results[micro] = rate
            print(f"{micro:>12} {base.batch_size // micro:>6} {rate:>8.2f} {peak:>9.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{micro:>12} {base.batch_size // micro:>6} {'OOM':>8}")
            break
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    if results:
        best = max(results, key=results.get)
        print(f"\nfastest: micro_batch={best} at {results[best]:.2f} step/s "
              f"({base.steps / results[best] / 3600:.1f}h for {base.steps} steps)")
    return results


def _time_steps(cfg: FinetuneConfig, device: str, steps: int) -> float:
    """Steps per second, timing only the second half so warm-up is excluded."""
    model, _config, _head_state = _starting_point(cfg, device)
    head = KernelHead(d_model=model.embed_dim * model.row_num_cls, d_k=cfg.d_k,
                      kernel=cfg.kernel, identity_init=True).to(device)
    model.kernel_head = head
    set_recompute(model, cfg.recompute)
    model.train()

    backbone_params, head_params = trainable_parameters(model, head)
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": cfg.lr_backbone},
                                   {"params": head_params, "lr": cfg.lr_head}])
    amp_on = cfg.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if amp_on and torch.cuda.is_bf16_supported() else torch.float16
    batches = _make_batches(cfg, device)
    accum = max(cfg.batch_size // cfg.micro_batch, 1)

    started = None
    for step in range(steps):
        if step == steps // 2:
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch in _split_micro_batches(next(batches), cfg.micro_batch):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_on):
                loss, _n, _a, _p = _forward_batch(model, head, batch, cfg, device,
                                                  model.max_classes)
            if loss is not None:
                (loss / accum).backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    return (steps - steps // 2) / (time.perf_counter() - started)


def smoke_test(device: Optional[str] = None) -> None:
    """Two optimiser steps on tiny batches, on the GPU if there is one.

    Catches shape, dtype and device errors that would otherwise surface an hour into
    a real run. About a minute on CPU, seconds on a GPU.
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
# smoke_test()                              # do this first; seconds on a GPU
#
# # Speed, in order. The log reports "% waiting on data" over the last window.
# # If it is high, generation is the bottleneck -- add workers and prefetch depth:
# cfg = FinetuneConfig(**{**PRESETS["paper"].__dict__,
#                         "prior_n_jobs": 12, "prefetch_factor": 4})
#
# # If it is near zero, the GPU is the bottleneck and micro_batch is the lever.
# # Measure which size is fastest on your card rather than guessing:
# benchmark_micro_batch(preset="paper", candidates=(8, 16, 32))
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
# # To start from TabICLv2, switch the prior with it -- everything else (d_model,
# # class count, module paths) is read from the checkpoint:
# cfg = FinetuneConfig(**{**PRESETS["paper"].__dict__,
#                         "checkpoint": V2_CHECKPOINT, "prior_type": "graph_scm"})
#
# # Watch the [val] lines, not the step lines: every step draws a fresh random
# # synthetic problem, so train loss reflects that draw rather than progress.
# # history is only for plotting that curve; the checkpoint is already on disk.
# import matplotlib.pyplot as plt
# v = history["val"]
# plt.plot([d["step"] for d in v], [d["loss"] for d in v], marker="o")
# plt.xlabel("step"); plt.ylabel("validation loss"); plt.show()
#
# # Fine-tuning is dataset-independent -- it trains on synthetic prior data and never
# # sees yours -- so this is a one-time cost. Keep the checkpoint and reuse it for any
# # dataset, in any later session, indefinitely:
# describe_checkpoint("/content/drive/MyDrive/kernelicl/paper.pt")
#
# # Then everything downstream uses it, with the scale recalibrated:
# ex = fit_explainer(X_train, y_train, X_test,
#                    finetuned="/content/drive/MyDrive/kernelicl/paper.pt")
