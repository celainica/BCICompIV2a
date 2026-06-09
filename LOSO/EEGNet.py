#!/usr/bin/env python3
"""Standalone EEGNet on BCI2a zero-target LOSO.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import mne
import moabb
import numpy as np
import pandas as pd
import torch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


WORK_DIR = Path("/root/autodl-tmp/EEG") if Path("/root/autodl-tmp/EEG").exists() else Path(".")
MNE_DATA = Path("/root/autodl-tmp/mne_data") if Path("/root/autodl-tmp").exists() else WORK_DIR / "mne_data"
OUT_DIR = WORK_DIR / "results" / "eegnet_bci2a_modern_loso"
CACHE_DIR = WORK_DIR / "cache" / "eegnet_bci2a_modern_loso_t0-4_rs250"

EVENTS = ("left_hand", "right_hand", "feet", "tongue")
SEED = 2026
MAX_EPOCHS = 500
PATIENCE = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FMIN, FMAX = 4.0, 40.0
TMIN, TMAX = 0.0, 4.0
RESAMPLE = 250.0


def setup() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for path in (MNE_DATA, OUT_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(MNE_DATA)
    os.environ["MOABB_DATA"] = str(MNE_DATA)
    mne.set_config("MNE_DATA", str(MNE_DATA), set_env=True)
    mne.set_config("MNE_DATASETS_BNCI_PATH", str(MNE_DATA), set_env=True)
    moabb.set_download_dir(str(MNE_DATA))


def ordered_unique(values) -> list:
    out, seen = [], set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def load_subject(subject: int):
    cache_file = CACHE_DIR / f"subject_{subject:03d}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X"].astype(np.float32),
            data["y"].astype(np.int64),
            data["subjects"].astype(np.int64),
            data["sessions"].astype(str),
        )

    dataset = BNCI2014_001(subjects=[subject])
    paradigm = MotorImagery(
        n_classes=4,
        events=list(EVENTS),
        fmin=FMIN,
        fmax=FMAX,
        tmin=TMIN,
        tmax=TMAX,
        resample=RESAMPLE,
        baseline=None,
    )
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    label_to_idx = {name: idx for idx, name in enumerate(EVENTS)}
    y = np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)
    subjects = metadata["subject"].to_numpy(dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    X = X.astype(np.float32, copy=False)
    np.savez_compressed(cache_file, X=X, y=y, subjects=subjects, sessions=sessions)
    return X, y, subjects, sessions


def load_data():
    xs, ys, ss, sess = [], [], [], []
    for subject in tqdm(range(1, 10), desc="load/cache BCI2a"):
        X, y, subjects, sessions = load_subject(subject)
        xs.append(X)
        ys.append(y)
        ss.append(subjects)
        sess.append(sessions)
    n_times = min(x.shape[-1] for x in xs)
    X = np.concatenate([x[..., :n_times] for x in xs]).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    subjects = np.concatenate(ss).astype(np.int64)
    sessions = np.concatenate(sess).astype(str)
    return X, y, subjects, sessions


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_classes: int):
        super().__init__()
        f1, d, f2 = 8, 2, 16
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1, momentum=0.01, eps=1e-3),
            Conv2dWithConstraint(f1, f1 * d, kernel_size=(n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            out = self.features(torch.zeros(1, 1, n_chans, n_times))
        self.classifier = nn.Conv2d(f2, n_classes, kernel_size=(out.shape[2], out.shape[3]))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x.unsqueeze(1))
        return self.classifier(x).squeeze(-1).squeeze(-1)


def split_masks(subjects, sessions, test_subject: int):
    source_subjects = np.asarray([s for s in sorted(np.unique(subjects)) if s != test_subject])
    rng = np.random.default_rng(SEED + test_subject)
    shuffled = source_subjects.copy()
    rng.shuffle(shuffled)
    val_subjects = np.sort(shuffled[:1])
    train_subjects = np.sort(shuffled[1:])

    train_session = {
        int(s): ordered_unique(sessions[subjects == int(s)].tolist())[0]
        for s in source_subjects
    }
    target_sessions = ordered_unique(sessions[subjects == test_subject].tolist())
    test_session = target_sessions[1]

    train_set, val_set = set(train_subjects.tolist()), set(val_subjects.tolist())
    train_mask = np.asarray(
        [int(s) in train_set and session == train_session[int(s)] for s, session in zip(subjects, sessions)]
    )
    val_mask = np.asarray(
        [int(s) in val_set and session == train_session[int(s)] for s, session in zip(subjects, sessions)]
    )
    test_mask = (subjects == test_subject) & (sessions == test_session)
    return train_mask, val_mask, test_mask, train_subjects, val_subjects, test_session


def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), 1e-6)
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )


def loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=DEVICE.startswith("cuda"))


@torch.no_grad()
def evaluate(model, data_loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    preds, trues = [], []
    for xb, yb in data_loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss_sum += float(criterion(logits, yb).item()) * len(yb)
        n += len(yb)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())
    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    return loss_sum / max(1, n), float((y_true == y_pred).mean()), y_true, y_pred


def balanced_acc(y_true, y_pred):
    accs = []
    for cls in range(4):
        mask = y_true == cls
        if mask.any():
            accs.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(accs))


def train_fold(X, y, subjects, sessions, test_subject: int):
    train_mask, val_mask, test_mask, train_subjects, val_subjects, test_session = split_masks(
        subjects, sessions, test_subject
    )
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    train_loader, val_loader, test_loader = loader(X_train, y_train, True), loader(X_val, y_val, False), loader(X_test, y_test, False)
    model = EEGNet(X.shape[1], X.shape[2], 4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state, best_acc, best_loss, best_epoch = None, -1.0, float("inf"), 0
    bad_epochs, history = 0, []
    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"s{test_subject:03d}", leave=False):
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(yb)
            n += len(yb)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss / max(1, n), "val_loss": val_loss, "val_acc": val_acc})
        improved = val_acc > best_acc or (math.isclose(val_acc, best_acc) and val_loss < best_loss)
        if improved:
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

    artifact_dir = OUT_DIR / "fold_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifact_dir / f"modern_s{test_subject:03d}_history.csv"
    model_path = artifact_dir / f"modern_s{test_subject:03d}_best.pt"
    pd.DataFrame(history).to_csv(history_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "test_subject": test_subject}, model_path)

    return {
        "fold_id": f"modern_s{test_subject:03d}",
        "test_subject": int(test_subject),
        "train_subjects": " ".join(map(str, train_subjects.tolist())),
        "val_subjects": " ".join(map(str, val_subjects.tolist())),
        "test_session": test_session,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_balanced_acc": balanced_acc(y_true, y_pred),
        "history_path": str(history_path),
        "model_path": str(model_path),
    }


def main():
    setup()
    print(f"MNE_DATA={MNE_DATA}")
    print(f"out_dir={OUT_DIR}")
    print(f"device={DEVICE}")
    if DEVICE.startswith("cuda"):
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    X, y, subjects, sessions = load_data()
    print(f"Loaded X={X.shape}, subjects={len(np.unique(subjects))}, sessions={ordered_unique(sessions.tolist())}")

    rows, folds_path = [], OUT_DIR / "folds.csv"
    for test_subject in sorted(int(s) for s in np.unique(subjects)):
        row = train_fold(X, y, subjects, sessions, test_subject)
        rows.append(row)
        pd.DataFrame(rows).to_csv(folds_path, index=False)
        print(f"{row['fold_id']}: acc={row['test_acc']:.4f}, bal={row['test_balanced_acc']:.4f}")

    df = pd.DataFrame(rows)
    summary = {
        "dataset": "bci2a",
        "protocol": "source_train_target_test",
        "n_folds": int(len(df)),
        "mean_test_acc": float(df["test_acc"].mean()),
        "std_test_acc": float(df["test_acc"].std(ddof=1)),
        "mean_test_balanced_acc": float(df["test_balanced_acc"].mean()),
        "std_test_balanced_acc": float(df["test_balanced_acc"].std(ddof=1)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDone.")
    print(df[["test_subject", "test_acc", "test_balanced_acc"]].to_string(index=False))
    print(f"\nmean acc={summary['mean_test_acc']:.4f} +/- {summary['std_test_acc']:.4f}")

if __name__ == "__main__":
    main()
