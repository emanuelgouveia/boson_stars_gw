from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt

from fastai.vision.all import *

# ===============================
# CONFIG
# ===============================
BASE = Path("/projects/F202509140CPCAA1")

SIG_PATH = BASE / "bbh_gen/data/outputs/sig"
BG_PATH  = BASE / "bbh_gen/data/outputs/bg"
CONFIG_PATH = BASE / "bbh_gen/data/outputs/config"

PLOT_PATH = BASE / "plots/classifier_bins"
PLOT_PATH.mkdir(exist_ok=True)

SEED = 42
NUM_WORKERS = 0
PRETRAINED = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ===============================
# LOAD SNR
# ===============================
def build_snr_dict(config_dir):
    snr_dict = {}

    for f in Path(config_dir).glob("*.json"):
        with open(f) as jf:
            data = json.load(jf)

        key = Path(f).stem
        if "snr_matchfilter" in data:
            snr_dict[key] = data["snr_matchfilter"]

    return snr_dict

# ===============================
# DATASET
# ===============================
class GWDataset(Dataset):
    def __init__(self, sig_path, bg_path,
                 snr_dict=None, snr_range=None,
                 ext="*.npz"):

        sig_files = list(sig_path.glob(ext))
        bg_files  = list(bg_path.glob(ext))

        all_files = sig_files + bg_files
        labels = [1]*len(sig_files) + [0]*len(bg_files)

        if snr_dict is not None and snr_range is not None:
            filtered_files = []
            filtered_labels = []

            for f, lab in zip(all_files, labels):
                key = Path(f).stem

                if key in snr_dict:
                    snr = snr_dict[key]

                    if snr_range[0] <= snr < snr_range[1]:
                        filtered_files.append(f)
                        filtered_labels.append(lab)

            self.all_files = filtered_files
            self.labels = np.array(filtered_labels)

        else:
            self.all_files = all_files
            self.labels = np.array(labels)

        print(f"Dataset size: {len(self.all_files)}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        data = np.load(self.all_files[idx])
        arr = data['qgraph'].astype(np.float32)

        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

        return torch.from_numpy(arr), int(self.labels[idx])

# ===============================
# SPLIT
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

    dls = DataLoaders.from_dsets(
        train_ds, valid_ds,
        bs=batch_size,
        num_workers=NUM_WORKERS
    )

    dls.c = 2
    return dls, test_ds

# ===============================
# MODEL
# ===============================
def get_model(name):
    models = {
        'resnet34': resnet34,
        'resnet50': resnet50,
        'efficientnet_b0': efficientnet_b0,
        'densenet121': densenet121,
        'convnext_tiny': convnext_tiny
    }
    return models[name]

METRICS = [accuracy]

def create_learner(dls, model_name, checkpoint_path):
    learn = vision_learner(
        dls,
        get_model(model_name),
        metrics=METRICS,
        loss_func=nn.CrossEntropyLoss(),
        pretrained=PRETRAINED
    )

    if checkpoint_path.exists():
        learn.load(checkpoint_path)
        print("Checkpoint loaded")

    return learn

# ===============================
# MAIN ANALYSIS
# ===============================
def evaluate_by_bins():

    snr_dict = build_snr_dict(CONFIG_PATH)

    bins = [
        (0, 10),
        (10, 50),
        (50, 100),
        (100, 200),
        (200, 500),
    ]

    checkpoint = BASE / "modelos_criados/resnet34_checkpoint"

    bin_centers = []
    accs = []

    for low, high in bins:
        print(f"\n=== SNR {low}–{high} ===")

        dataset = GWDataset(
            SIG_PATH,
            BG_PATH,
            snr_dict=snr_dict,
            snr_range=(low, high)
        )

        if len(dataset) < 50:
            print("Poucos dados, skip")
            continue

        dls, test_ds = create_dls(dataset, batch_size=128)

        learn = create_learner(dls, "resnet34", checkpoint)

        results = learn.validate(dl=dls.test_dl(test_ds))
        acc = float(results[1])

        print(f"Accuracy: {acc:.4f}")

        bin_centers.append((low + high) / 2)
        accs.append(acc)

    # ===============================
    # PLOT FINAL
    # ===============================
    plt.figure()
    plt.plot(bin_centers, accs, marker='o')
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title("Classifier Performance vs SNR")
    plt.grid()

    plt.savefig(PLOT_PATH / "accuracy_vs_snr.png")
    plt.close()

    print("\nSaved plot: accuracy_vs_snr.png")

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    evaluate_by_bins()