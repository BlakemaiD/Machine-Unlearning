"""Reproducible, multi-seed evaluation against retain-only retrained models."""

import copy
import hashlib
import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18

from .device import DEVICE
from .metrics import accuracy_eval, compute_losses, get_forget_confs, simple_mia
from .official_metric import forget_score


@dataclass(frozen=True)
class TrainingRecipe:
    """Chosen CIFAR-10 scratch-training recipe for the retrained reference models.

    The starter-kit notebook documents the optimizer and augmentation, but not
    the checkpoint's training duration or schedule. The defaults therefore use
    those documented settings with an explicitly chosen 30-epoch cosine run.
    """

    epochs: int = 30
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    num_workers: int = 2
    scheduler: str = "cosine"
    augment: bool = True


@dataclass
class EvaluationResults:
    """Per-seed model outputs needed to calculate competition metrics."""

    name: str
    seeds: tuple
    forget_confs: np.ndarray
    retain_accs: np.ndarray
    test_accs: np.ndarray
    forget_accs: np.ndarray
    runtimes: np.ndarray
    forget_indices_hash: str
    forget_sample_seed: int
    mia_accs: Optional[np.ndarray] = None

    @property
    def n_models(self):
        return len(self.seeds)


def _validate_seeds(n, seeds):
    if not isinstance(n, (int, np.integer)):
        raise TypeError("N must be an integer.")
    n = int(n)
    seeds = tuple(range(n)) if seeds is None else tuple(int(seed) for seed in seeds)
    if n < 2:
        raise ValueError("At least two model seeds are required by the official F metric.")
    if len(seeds) != n or len(set(seeds)) != n:
        raise ValueError("Provide exactly N distinct seeds.")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Seeds must be in the range [0, 2**32).")
    return seeds


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    elif DEVICE.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _loader_kwargs(loader, *, shuffle, generator=None, batch_size=None):
    kwargs = {
        "batch_size": loader.batch_size if batch_size is None else batch_size,
        "shuffle": shuffle,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last if shuffle else False,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "generator": generator,
        "persistent_workers": loader.persistent_workers,
    }
    if loader.num_workers > 0 and loader.prefetch_factor is not None:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    if getattr(loader, "pin_memory_device", ""):
        kwargs["pin_memory_device"] = loader.pin_memory_device
    return kwargs


def make_forget_loader(forget_loader, max_examples=256, sample_seed=0):
    """Make a stable-order view of a forget loader, optionally subsampled once."""
    dataset = forget_loader.dataset
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("max_examples must be positive or None.")
        if max_examples < len(dataset):
            indices = np.random.default_rng(sample_seed).choice(
                len(dataset), size=max_examples, replace=False
            )
            dataset = Subset(dataset, np.sort(indices).tolist())
    return DataLoader(dataset, **_loader_kwargs(forget_loader, shuffle=False))


def _effective_indices(dataset):
    indices = np.arange(len(dataset), dtype=np.int64)
    current = dataset
    while isinstance(current, Subset):
        parent_indices = np.asarray(current.indices, dtype=np.int64)
        indices = parent_indices[indices]
        current = current.dataset
    return current, indices


def _dataset_signature(loader):
    base, indices = _effective_indices(loader.dataset)
    digest = hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()
    return {
        "base_dataset": f"{type(base).__module__}.{type(base).__qualname__}",
        "base_length": len(base),
        "indices_hash": digest,
        "loader_length": len(loader.dataset),
    }


def _forget_hash(loader):
    return _dataset_signature(loader)["indices_hash"]


def _new_loader(dataset, template, *, shuffle, seed=None, batch_size=None, workers=None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = _loader_kwargs(
        template,
        shuffle=shuffle,
        generator=generator,
        batch_size=batch_size,
    )
    kwargs["num_workers"] = template.num_workers if workers is None else workers
    kwargs["persistent_workers"] = kwargs["persistent_workers"] and kwargs["num_workers"] > 0
    if kwargs["num_workers"] == 0:
        kwargs.pop("prefetch_factor", None)
    return DataLoader(dataset, **kwargs)


def _training_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )


def _copy_with_transform(dataset, transform):
    if isinstance(dataset, Subset):
        return Subset(
            _copy_with_transform(dataset.dataset, transform),
            dataset.indices,
        )
    if not hasattr(dataset, "transform"):
        raise TypeError(
            "The selected recipe requests augmentation, but the training "
            "dataset does not expose a transform attribute. Use augment=False "
            "or a dataset with a transform."
        )
    cloned = copy.copy(dataset)
    cloned.transform = transform
    return cloned


def _normalize_recipe(recipe):
    if isinstance(recipe, TrainingRecipe):
        normalized = recipe
    elif isinstance(recipe, Mapping):
        normalized = TrainingRecipe(**recipe)
    else:
        raise TypeError("recipe must be a TrainingRecipe or mapping.")
    if normalized.epochs <= 0 or normalized.batch_size <= 0:
        raise ValueError("Recipe epochs and batch_size must be positive.")
    if normalized.num_workers < 0:
        raise ValueError("Recipe num_workers cannot be negative.")
    if normalized.scheduler not in ("cosine", "none"):
        raise ValueError("Recipe scheduler must be 'cosine' or 'none'.")
    return normalized


def _metadata(
    n,
    seeds,
    recipe,
    retain_loader,
    test_loader,
    forget_loader,
    sample_seed,
    model_identifier,
):
    return {
        "N": n,
        "seeds": list(seeds),
        "recipe": asdict(recipe),
        "model": model_identifier,
        "forget_indices_hash": _forget_hash(forget_loader),
        "forget_sample_seed": sample_seed,
        "retain_dataset": _dataset_signature(retain_loader),
        "test_dataset": _dataset_signature(test_loader),
        "forget_dataset": _dataset_signature(forget_loader),
    }


def _read_cache(path, expected_metadata):
    try:
        with np.load(path, allow_pickle=False) as cache:
            actual_metadata = json.loads(str(cache["metadata"].item()))
            if actual_metadata != expected_metadata:
                raise ValueError(
                    "Retrained-model cache metadata does not match this run. "
                    f"Expected {expected_metadata}; found {actual_metadata}. "
                    "Choose a new cache path to train a separate reference set."
                )
            results = EvaluationResults(
                name="retrained",
                seeds=tuple(actual_metadata["seeds"]),
                forget_confs=cache["forget_confs"].copy(),
                retain_accs=cache["retain_accs"].copy(),
                test_accs=cache["test_accs"].copy(),
                forget_accs=cache["forget_accs"].copy(),
                runtimes=cache["runtimes"].copy(),
                forget_indices_hash=actual_metadata["forget_indices_hash"],
                forget_sample_seed=actual_metadata["forget_sample_seed"],
            )
            n = actual_metadata["N"]
            n_forget = actual_metadata["forget_dataset"]["loader_length"]
            if results.forget_confs.shape != (n, n_forget):
                raise ValueError(f"Retrained-model cache {path} has invalid confidence dimensions.")
            if any(
                values.shape != (n,)
                for values in (
                    results.retain_accs,
                    results.test_accs,
                    results.forget_accs,
                    results.runtimes,
                )
            ):
                raise ValueError(f"Retrained-model cache {path} has invalid per-model arrays.")
            return results
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read retrained-model cache {path}: {exc}") from exc


def _write_cache(path, metadata, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                forget_confs=results.forget_confs,
                retain_accs=results.retain_accs,
                test_accs=results.test_accs,
                forget_accs=results.forget_accs,
                runtimes=results.runtimes,
            )
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _train_retrained_model(model, loader, recipe):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=recipe.lr,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
        if recipe.scheduler == "cosine"
        else None
    )
    model.train()
    for _ in range(recipe.epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    return model


def train_or_load_retrained(
    N,
    recipe,
    *,
    retain_loader,
    forget_loader,
    test_loader,
    seeds=None,
    cache_path="outputs/retrain_confs.npz",
    max_forget_examples=256,
    forget_sample_seed=0,
    model_factory: Optional[Callable[[], nn.Module]] = None,
):
    """Train retain-only reference models, or load a strictly matching cache."""
    recipe = _normalize_recipe(recipe)
    seeds = _validate_seeds(N, seeds)
    N = int(N)
    forget_eval_loader = make_forget_loader(
        forget_loader, max_forget_examples, forget_sample_seed
    )
    model_identifier = (
        "torchvision.models.resnet18(weights=None,num_classes=10)"
        if model_factory is None
        else (
            f"{getattr(model_factory, '__module__', type(model_factory).__module__)}."
            f"{getattr(model_factory, '__qualname__', type(model_factory).__qualname__)}"
        )
    )
    metadata = _metadata(
        N,
        seeds,
        recipe,
        retain_loader,
        test_loader,
        forget_eval_loader,
        forget_sample_seed,
        model_identifier,
    )
    path = Path(cache_path)
    if path.exists():
        return _read_cache(path, metadata)

    make_model = model_factory or (lambda: resnet18(weights=None, num_classes=10))
    train_dataset = retain_loader.dataset
    if recipe.augment:
        train_dataset = _copy_with_transform(train_dataset, _training_transform())

    forget_confs, retain_accs, test_accs, forget_accs, runtimes = [], [], [], [], []
    for seed in seeds:
        _seed_everything(seed)
        model = make_model().to(DEVICE)
        seeded_retain = _new_loader(
            train_dataset,
            retain_loader,
            shuffle=True,
            seed=seed,
            batch_size=recipe.batch_size,
            workers=recipe.num_workers,
        )
        _synchronize_device()
        started = time.perf_counter()
        model = _train_retrained_model(model, seeded_retain, recipe)
        _synchronize_device()
        runtimes.append(time.perf_counter() - started)
        retain_accs.append(accuracy_eval(model, retain_loader))
        test_accs.append(accuracy_eval(model, test_loader))
        forget_accs.append(accuracy_eval(model, forget_eval_loader))
        forget_confs.append(get_forget_confs(model, forget_eval_loader))

    results = EvaluationResults(
        name="retrained",
        seeds=seeds,
        forget_confs=np.asarray(forget_confs, dtype=np.float64),
        retain_accs=np.asarray(retain_accs, dtype=np.float64),
        test_accs=np.asarray(test_accs, dtype=np.float64),
        forget_accs=np.asarray(forget_accs, dtype=np.float64),
        runtimes=np.asarray(runtimes, dtype=np.float64),
        forget_indices_hash=metadata["forget_indices_hash"],
        forget_sample_seed=forget_sample_seed,
    )
    _write_cache(path, metadata, results)
    return results


def adapt_method(method_fn):
    """Wrap a legacy (net, retain, forget, validation) method in a seeded API."""
    signature = inspect.signature(method_fn)
    accepts_seed = "seed" in signature.parameters

    def adapted(net_copy, retain_loader, forget_loader, val_loader, seed):
        if accepts_seed:
            return method_fn(
                net_copy,
                retain_loader,
                forget_loader,
                val_loader,
                seed=seed,
            )
        return method_fn(net_copy, retain_loader, forget_loader, val_loader)

    return adapted


def _balanced_test_loader(test_loader, n_examples, seed):
    n = min(n_examples, len(test_loader.dataset))
    indices = np.random.default_rng(seed).choice(
        len(test_loader.dataset), size=n, replace=False
    )
    dataset = Subset(test_loader.dataset, np.sort(indices).tolist())
    return _new_loader(dataset, test_loader, shuffle=False)


def _loss_mia(net, forget_loader, test_loader, seed):
    was_training = net.training
    net.eval()
    try:
        forget_losses = compute_losses(net, forget_loader)
        balanced_test_loader = _balanced_test_loader(
            test_loader, len(forget_losses), seed
        )
        test_losses = compute_losses(net, balanced_test_loader)
    finally:
        net.train(was_training)
    features = np.concatenate((test_losses, forget_losses)).reshape(-1, 1)
    labels = np.concatenate(
        (np.zeros(len(test_losses), dtype=np.int64), np.ones(len(forget_losses), dtype=np.int64))
    )
    return float(simple_mia(features, labels, random_state=seed).mean())


def run_method(
    method_fn,
    N,
    seeds,
    *,
    original_model,
    retain_loader,
    forget_loader,
    val_loader,
    test_loader,
    max_forget_examples=256,
    forget_sample_seed=0,
):
    """Run a method for N paired seeds from fresh copies of the original model."""
    seeds = _validate_seeds(N, seeds)
    N = int(N)
    forget_eval_loader = make_forget_loader(
        forget_loader, max_forget_examples, forget_sample_seed
    )
    call_method = adapt_method(method_fn)
    forget_confs, retain_accs, test_accs, forget_accs, runtimes, mia_accs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for seed in seeds:
        _seed_everything(seed)
        net_copy = copy.deepcopy(original_model).to(DEVICE)
        seeded_retain = _new_loader(
            retain_loader.dataset,
            retain_loader,
            shuffle=True,
            seed=seed,
        )
        seeded_validation = _new_loader(
            val_loader.dataset,
            val_loader,
            shuffle=False,
        )
        _synchronize_device()
        started = time.perf_counter()
        model = call_method(
            net_copy,
            seeded_retain,
            forget_loader,
            seeded_validation,
            seed,
        )
        _synchronize_device()
        runtimes.append(time.perf_counter() - started)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"{method_fn!r} must return a torch.nn.Module; got {type(model)!r}."
            )
        retain_accs.append(accuracy_eval(model, retain_loader))
        test_accs.append(accuracy_eval(model, test_loader))
        forget_accs.append(accuracy_eval(model, forget_eval_loader))
        forget_confs.append(get_forget_confs(model, forget_eval_loader))
        mia_accs.append(_loss_mia(model, forget_eval_loader, test_loader, seed))

    return EvaluationResults(
        name=getattr(method_fn, "__name__", type(method_fn).__name__),
        seeds=seeds,
        forget_confs=np.asarray(forget_confs, dtype=np.float64),
        retain_accs=np.asarray(retain_accs, dtype=np.float64),
        test_accs=np.asarray(test_accs, dtype=np.float64),
        forget_accs=np.asarray(forget_accs, dtype=np.float64),
        runtimes=np.asarray(runtimes, dtype=np.float64),
        forget_indices_hash=_forget_hash(forget_eval_loader),
        forget_sample_seed=forget_sample_seed,
        mia_accs=np.asarray(mia_accs, dtype=np.float64),
    )


def score(unlearned, retrained):
    """Calculate official F, utility ratios, final score, and runtime ratio."""
    if unlearned.seeds != retrained.seeds:
        raise ValueError("Unlearned and retrained results must use identical seeds.")
    if unlearned.forget_indices_hash != retrained.forget_indices_hash:
        raise ValueError("Unlearned and retrained results must use the same forget examples.")
    if unlearned.forget_sample_seed != retrained.forget_sample_seed:
        raise ValueError("Unlearned and retrained results must use the same forget sample seed.")
    if unlearned.n_models != retrained.n_models:
        raise ValueError("Unlearned and retrained results must use the same N.")
    if (
        unlearned.forget_confs.shape[0] != unlearned.n_models
        or retrained.forget_confs.shape[0] != retrained.n_models
    ):
        raise ValueError("Confidence arrays must have one row per model seed.")

    mean_retrained_retain = float(np.mean(retrained.retain_accs))
    mean_retrained_test = float(np.mean(retrained.test_accs))
    if mean_retrained_retain == 0.0 or mean_retrained_test == 0.0:
        raise ZeroDivisionError("Retrained reference accuracy cannot be zero.")
    f_score = forget_score(unlearned.forget_confs, retrained.forget_confs)
    retain_ratio = float(np.mean(unlearned.retain_accs) / mean_retrained_retain)
    test_ratio = float(np.mean(unlearned.test_accs) / mean_retrained_test)
    mean_unlearning_time = float(np.mean(unlearned.runtimes))
    mean_retraining_time = float(np.mean(retrained.runtimes))
    runtime_ratio = (
        mean_unlearning_time / mean_retraining_time
        if mean_retraining_time > 0.0
        else float("inf")
    )
    return {
        "N": unlearned.n_models,
        "forget_examples": int(unlearned.forget_confs.shape[1]),
        "forget_sample_seed": unlearned.forget_sample_seed,
        "F": f_score,
        "retain_acc": float(np.mean(unlearned.retain_accs)),
        "test_acc": float(np.mean(unlearned.test_accs)),
        "forget_acc": float(np.mean(unlearned.forget_accs)),
        "retain_utility_ratio": retain_ratio,
        "test_utility_ratio": test_ratio,
        "final_score": f_score * retain_ratio * test_ratio,
        "runtime_sec": mean_unlearning_time,
        "retraining_runtime_sec": mean_retraining_time,
        "runtime_ratio": runtime_ratio,
        "within_20_percent_runtime": runtime_ratio <= 0.2,
        "loss_mia_acc": (
            float(np.mean(unlearned.mia_accs))
            if unlearned.mia_accs is not None
            else float("nan")
        ),
    }
