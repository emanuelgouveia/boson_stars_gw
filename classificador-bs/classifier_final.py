from pathlib import Path
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from fastai.vision.all import *

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ===============================
# Configurações globais
# ===============================
SEED = 42
NUM_WORKERS = 0   # seguro para HPC/WSL
PRETRAINED = True

# ===============================
# Paths
# ===============================
BASE = Path("/projects/F202509140CPCAA1")

SIG_PATH = BASE / "bbh_gen/data/outputs/sig"
BG_PATH  = BASE / "bbh_gen/data/outputs/bg"

PLOT_PATH = BASE / "plots/classifier_plots"
PLOT_PATH.mkdir(exist_ok=True)

LOG_PATH = BASE / "models/modelos_criados"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# ===============================
# Device
# ===============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ===============================
# Dataset
# ===============================
class GWDataset(Dataset):
    def __init__(self, sig_path, bg_path, ext="*.npz", max_files=None):
        self.sig_files = sorted(sig_path.glob(ext))
        self.bg_files  = sorted(bg_path.glob(ext))

        if max_files is not None:
            self.sig_files = self.sig_files[:max_files]
            self.bg_files  = self.bg_files[:max_files]

        self.all_files = self.sig_files + self.bg_files
        self.labels = np.array(
            [1]*len(self.sig_files) + [0]*len(self.bg_files),
            dtype=np.int64
        )

        if len(self.all_files) == 0:
            raise RuntimeError("Dataset vazio — verifica os paths!")

        print(f"Dataset: {len(self.sig_files)} sig | {len(self.bg_files)} bg")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        data = np.load(self.all_files[idx])
        arr = data['qgraph'].astype(np.float32)

        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

        return torch.from_numpy(arr), int(self.labels[idx])

# ===============================
# Splitter
# ===============================
def RandomThreeSplitter(valid_pct=0.15, test_pct=0.15, seed=None):
    def _inner(o):
        if seed is not None:
            torch.manual_seed(seed)

        idxs = L(range_of(o)).shuffle()
        n = len(idxs)

        n_val = int(valid_pct * n)
        n_test = int(test_pct * n)

        val_idx  = idxs[:n_val]
        test_idx = idxs[n_val:n_val+n_test]
        train_idx = idxs[n_val+n_test:]

        return train_idx, val_idx, test_idx
    return _inner

def create_dls(dataset, batch_size=128, seed=None):
    tr_idx, val_idx, test_idx = RandomThreeSplitter(seed=seed)(dataset)

    train_ds = Subset(dataset, tr_idx)
    valid_ds = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    print(f"Train: {len(train_ds)} | Val: {len(valid_ds)} | Test: {len(test_ds)}")

    dls = DataLoaders.from_dsets(
        train_ds, valid_ds,
        bs=batch_size,
        num_workers=NUM_WORKERS
    )

    dls.c = 2
    return dls, test_ds

# ===============================
# Modelos
# ===============================
def get_model(name):
    models = {
        'resnet34': resnet34,
        'resnet50': resnet50,
        'efficientnet_b0': efficientnet_b0,
        'densenet121': densenet121,
        'convnext_tiny': convnext_tiny
    }
    if name not in models:
        raise ValueError(f"Modelo desconhecido: {name}")
    return models[name]

# ===============================
# Plots
# ===============================
def plot_metrics(learn, test_ds):
    test_dl = learn.dls.test_dl(test_ds)
    preds, targs = learn.get_preds(dl=test_dl)

    y_true = targs.numpy()
    probs = preds.softmax(dim=1).numpy()

    y_pred = np.argmax(probs, axis=1)
    y_proba = probs[:,1]

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.plot([0,1],[0,1],'k--')
    plt.legend()
    plt.savefig(PLOT_PATH / "roc_curve.png")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(PLOT_PATH / "confusion_matrix.png")
    plt.close()

# ===============================
# Learner
# ===============================
METRICS = [accuracy, Precision(), Recall(), F1Score()]

def create_learner(dls, model_name, resume=False):
    learn = vision_learner(
        dls,
        get_model(model_name),
        metrics=METRICS,
        loss_func=nn.CrossEntropyLoss(),
        pretrained=PRETRAINED
    )

    ckpt = LOG_PATH / f"{model_name}_checkpoint"

    if resume and ckpt.with_suffix(".pth").exists():
        learn.load(ckpt)
        print("Checkpoint carregado")

    return learn

# ===============================
# Treino
# ===============================
def train_and_eval(args, dataset):
    dls, test_ds = create_dls(dataset, args.batch_size, SEED)

    learn = create_learner(dls, args.model, args.resume)

    learn.fine_tune(args.epochs)

    learn.save(LOG_PATH / f"{args.model}_checkpoint")

    results = learn.validate(dl=dls.test_dl(test_ds))
    print("Test:", results)

    plot_metrics(learn, test_ds)

# ===============================
# Cross-validation
# ===============================
def cross_validate(args, dataset):
    accs = []

    for fold in range(args.n_folds):
        print(f"\nFold {fold+1}/{args.n_folds}")

        dls, test_ds = create_dls(dataset, args.batch_size, seed=fold)

        learn = create_learner(dls, args.model)
        learn.fine_tune(args.epochs)

        acc = float(learn.validate(dl=dls.test_dl(test_ds))[1])
        accs.append(acc)

    print(f"\nCV accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# ===============================
# Args
# ===============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--model', type=str, default='resnet34')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--cross-validate', action='store_true')
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-files', type=int, default=None)
    return p.parse_args()

# ===============================
# Main
# ===============================
def main():
    args = parse_args()

    dataset = GWDataset(
        SIG_PATH,
        BG_PATH,
        max_files=args.max_files
    )

    if args.cross_validate:
        cross_validate(args, dataset)
    else:
        train_and_eval(args, dataset)

if __name__ == "__main__":
    main()