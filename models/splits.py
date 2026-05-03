"""
Sequential train/val/test splits for time-series data.

All splits are strictly by time order — no random shuffling ever.
Two modes:
  - sequential : single train/val/test cut for a quick baseline
  - walk_forward: rolling windows for robust out-of-sample validation
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Split:
    train: np.ndarray
    val:   np.ndarray
    test:  np.ndarray


@dataclass
class Fold:
    fold_idx: int
    train:    np.ndarray
    test:     np.ndarray


def sequential(n: int, train_frac: float = 0.50, val_frac: float = 0.25) -> Split:
    """
    Split n rows into train/val/test by position (earliest → latest).
    Default: 50% train, 25% val, 25% test.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    return Split(
        train=np.arange(0, train_end),
        val=  np.arange(train_end, val_end),
        test= np.arange(val_end, n),
    )


def walk_forward(
    timestamps:  np.ndarray,
    train_days:  int = 90,
    test_days:   int = 30,
    step_days:   int = 30,
) -> list[Fold]:
    """
    Generate walk-forward folds over a timestamp array.

    Each fold:
      train — train_days of data immediately before the test window
      test  — test_days of data following the training window

    Folds advance by step_days. Stops when a full test window no longer fits.

    With 295 days and defaults (train=90, test=30, step=30) → 6 folds.
    """
    train_s = train_days * 86400
    test_s  = test_days  * 86400
    step_s  = step_days  * 86400

    t_min = timestamps[0]
    t_max = timestamps[-1]

    folds = []
    window_start = t_min

    while True:
        train_end = window_start + train_s
        test_end  = train_end   + test_s

        if test_end > t_max:
            break

        train_idx = np.where((timestamps >= window_start) & (timestamps < train_end))[0]
        test_idx  = np.where((timestamps >= train_end)    & (timestamps < test_end))[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append(Fold(fold_idx=len(folds), train=train_idx, test=test_idx))

        window_start += step_s

    return folds


def describe_split(split: Split, timestamps: np.ndarray) -> None:
    total = len(split.train) + len(split.val) + len(split.test)
    _fmt = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    print(f"{'Set':<6}  {'Rows':>7}  {'Share':>6}  {'Start':>12}  {'End':>12}")
    print("-" * 52)
    for name, idx in [("train", split.train), ("val", split.val), ("test", split.test)]:
        print(f"{name:<6}  {len(idx):>7,}  {len(idx)/total:>5.1%}  "
              f"  {_fmt(timestamps[idx[0]]):>12}  {_fmt(timestamps[idx[-1]]):>12}")


def describe_folds(folds: list[Fold], timestamps: np.ndarray) -> None:
    _fmt = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    print(f"{'Fold':>4}  {'Train rows':>10}  {'Train start':>12}  {'Train end':>12}  "
          f"{'Test rows':>9}  {'Test start':>12}  {'Test end':>12}")
    print("-" * 85)
    for f in folds:
        print(f"{f.fold_idx:>4}  {len(f.train):>10,}  "
              f"{_fmt(timestamps[f.train[0]]):>12}  {_fmt(timestamps[f.train[-1]]):>12}  "
              f"{len(f.test):>9,}  "
              f"{_fmt(timestamps[f.test[0]]):>12}  {_fmt(timestamps[f.test[-1]]):>12}")
