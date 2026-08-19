import random
from typing import Literal, Optional, Tuple, List, Any
import shutil
import numpy as np
import pandas as pd
import torch
import logging
import warnings
from argparse import Namespace
import time
import os
from scipy import sparse
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset

#-----------Logging---------------------
def get_run_info(argv: List[str], args: Namespace=None, **kwargs) -> str:
    s = list()
    s.append("")
    s.append("##time: {}".format(time.asctime()))
    s.append("##cwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(argv)))
    if args is not None:
        s.append("##args: {}".format(args))
    for k, v in kwargs.items():
        s.append("##{}: {}".format(k, v))
    return '\n'.join(s)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir


def make_logger(
        title: Optional[str]="",
        filename: Optional[str]=None,
        level: Literal["INFO", "DEBUG"]="INFO",
        mode: Literal['w', 'a']='w',
        trace: bool=True,
        **kwargs):
    if isinstance(level, str):
        level = getattr(logging, level)
    logger = logging.getLogger(title)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    if trace is True or ("show_line" in kwargs and kwargs["show_line"] is True):
        formatter = logging.Formatter(
                '%(levelname)s(%(asctime)s) [%(filename)s:%(lineno)d]:%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s(%(asctime)s):%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    # formatter = logging.Formatter(
    #     '%(message)s\t%(levelname)s(%(asctime)s)', datefmt='%Y%m%d %H:%M:%S'
    # )

    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename is not None:
        if os.path.exists(filename):
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(filename)))
            while os.path.exists("{}.conflict_{}".format(filename, suffix)):
                suffix = "{}_1".format(suffix)
            shutil.move(filename, "{}.conflict_{}".format(filename, suffix))
            warnings.warn("log {} exists, moved to to {}.conflict_{}.log".format(filename, filename, suffix))
        fh = logging.FileHandler(filename=filename, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# ---------- Random seed ----------

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Feature extraction & preprocessing ----------

def to_numpy_matrix(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")
    return x.astype(np.float32, copy=False)

# ---------- DataLoader ----------

def make_loader(x, y=None, batch_size=512, shuffle=False, extra=None) -> DataLoader:
    tensors = [torch.from_numpy(x).float()]
    if y is not None:
        tensors.append(torch.from_numpy(y).long())
    if extra:
        for a in extra:
            tensors.append(
                torch.from_numpy(a).float() if a.dtype.kind == "f" else torch.from_numpy(a)
            )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

# ---------- Class weights ----------
def compute_class_weights(
    y: np.ndarray, num_classes: int, device: torch.device, mode: str
) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32).clip(min=1.0)
    if mode == "none":
        w = np.ones_like(counts)
    elif mode == "inv":
        w = 1.0 / counts
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    w /= w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)

# ---------- Similarity / scoring ----------

def softmax_confidence(logits: np.ndarray):
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()
    pred = probs.argmax(1)
    conf = probs[np.arange(len(pred)), pred]
    return pred.astype(np.int64), conf.astype(np.float32), probs.astype(np.float32)

def compute_source_prototypes(feats, labels, n_cls):
    protos = np.zeros((n_cls, feats.shape[1]), dtype=np.float32)
    for c in range(n_cls):
        m = labels == c
        if m.any():
            p = feats[m].mean(0)
            norm = np.linalg.norm(p)
            protos[c] = p / norm if norm > 1e-12 else p
    return protos

def prototype_similarity(feats, protos):
    # Normalize the feature matrix (feats is not normalized and must be processed)
    feats_norm = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    sim = feats_norm @ protos.T
    # Get predicted classes and similarity values
    pred = sim.argmax(1)
    pred_sim = sim[np.arange(len(pred)), pred]
    return (pred.astype(np.int64),pred_sim.astype(np.float32),)

def compute_classifier_margin(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] <= 1:
        return scores[:, 0].astype(np.float32)
    part = np.partition(scores, scores.shape[1] - 2, axis=1)[:, -2:]
    return (part.max(1) - part.min(1)).astype(np.float32)


# ---------- GMM ----------
def safe_GMM(values, higher_is_better=True, reg_covar=5e-4, seed=0):
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size <= 1 or (values.max() - values.min()) < 1e-12:
        return (
            np.ones_like(values),
            np.ones(len(values), dtype=bool),
            np.array([values.mean(), values.mean()], dtype=np.float32),
        )
    scaled = (
        (values - values.min()) / max(values.max() - values.min(), 1e-12)
    ).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=200,tol=1e-3, reg_covar=reg_covar, n_init=8, random_state=seed,)
    gmm.fit(scaled)
    means = gmm.means_.ravel().astype(np.float32)
    high_comp = int(np.argmax(means) if higher_is_better else np.argmin(means))
    prob_high = gmm.predict_proba(scaled)[:, high_comp].astype(np.float32)
    return prob_high, (gmm.predict(scaled) == high_comp).astype(bool)

def log_pseudo_selection(logger, candidate, selected, agreement,
                         clf_conf, proto_sim, margin, reliability,
                         needed, x_remain):
    """Log key information about pseudo-label selection"""
    n_total = len(x_remain)
    n_cand = int(candidate.sum())
    n_sel = int(selected.sum())
    n_agree = int(agreement.sum())

    # 1. Count overview
    logger.info(
        f"  [Pseudo-label selection] "
        f"Remaining samples={n_total}, "
        f"Candidates={n_cand}({n_cand / n_total * 100:.1f}%), "
        f"Needed={needed}, "
        f"Selected={n_sel}, "
        f"Prediction agreement={n_agree}({n_agree / n_total * 100:.1f}%)"
    )

    # 2. Quality metrics for all samples
    logger.info(
        f"  [Overall quality] "
        f"Confidence={clf_conf.mean():.4f}, "
        f"Prototype similarity={proto_sim.mean():.4f}, "
        f"Margin={margin.mean():.4f}"
    )

    # 3. Quality metrics for selected samples (more informative)
    if n_sel > 0:
        logger.info(
            f"  [Selected quality] "
            f"Confidence={clf_conf[selected].mean():.4f}, "
            f"Prototype similarity={proto_sim[selected].mean():.4f}, "
            f"Margin={margin[selected].mean():.4f}, "
            f"Reliability={reliability[selected].mean():.4f}"
            f"±{reliability[selected].std():.4f}"
        )

    # 4. Insufficient-candidate warning
    if n_sel < needed:
        logger.warning(
            f"  Insufficient candidates! Needed {needed} but selected only {n_sel}; "
            f"shortfall of {needed - n_sel} will be filled in subsequent rounds"
        )

def percentile_rank(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).ravel()
    if v.size <= 1:
        return np.ones_like(v)
    return pd.Series(v).rank(method="average", pct=True).to_numpy(dtype=np.float32)


# ---------- Schedule ----------

def parse_ratio_schedule(raw: str, rounds: int) -> List[float]:
    if rounds <= 0:
        return []
    ratios = []
    for p in str(raw).split(","):
        p = p.strip()
        if not p:
            continue
        v = float(p)
        if v > 1.0:
            v /= 100.0
        if v <= 0 or v > 1.0:
            raise ValueError(f"Invalid ratio: {p}")
        ratios.append(v)
    if not ratios:
        raise ValueError("pseudo_ratio_schedule must contain at least one ratio")
    ratios.extend([ratios[-1]] * max(0, rounds - len(ratios)))
    ratios = ratios[:rounds]
    return np.maximum.accumulate(np.array(ratios, dtype=np.float32)).tolist()


def cumulative_target_count(total, rnd, schedule, args):
    ratio = schedule[min(rnd, len(schedule) - 1)]
    count = max(1, int(round(total * ratio))) if total > 0 else 0
    if rnd == 0:
        if args.first_round_min_count > 0:
            count = max(count, args.first_round_min_count)
        if args.first_round_max_count > 0:
            count = min(count, args.first_round_max_count)
    if args.max_total_pseudo_count > 0:
        count = min(count, args.max_total_pseudo_count)
    return min(count, total)


# ---------- Serialization ----------

def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
