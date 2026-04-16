import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import datetime
import pandas as pd
from pathlib import Path

# ===================== PATHS =====================
BASE = Path("/projects/F202509140CPCAA1")
PLOT_PATH = BASE / "classifier_plots"
CONFIG_PATH = BASE / "bbh_gen/data/outputs/config"


# ===================== LOAD DATA =====================
def load_snr_values(config_dir=CONFIG_PATH):
    """Load BOTH SNR values (matchfilter + correlate) from JSON files."""
    
    config_files = glob(os.path.join(config_dir, "*.json"))
    
    snr_match = []
    snr_corr = []
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
            if "snr_matchfilter" in config and "snr_correlate" in config:
                snr_match.append(config["snr_matchfilter"])
                snr_corr.append(config["snr_correlate"])
            else:
                print(f"⚠️ Missing keys in {config_file}")
    
    return np.array(snr_match), np.array(snr_corr)


# ===================== HISTOGRAM =====================
def plot_snr_histogram(snr_1, snr_2, output_dir=PLOT_PATH):
    os.makedirs(output_dir, exist_ok=True)

    # Remover valores inválidos
    snr_1 = snr_1[np.isfinite(snr_1)]
    snr_2 = snr_2[np.isfinite(snr_2)]

    
    bins = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, np.inf]

    plt.figure(figsize=(12, 8))

    
    sns.histplot(snr_1, bins=bins, label="Matchfilter", alpha=0.5)
    sns.histplot(snr_2, bins=bins, label="Correlate", alpha=0.5)

    plt.xscale("log")
    plt.yscale("log")

    plt.ylim(bottom=0.5)

    plt.title("SNR Distribution (log bins)")
    plt.xlabel("SNR")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    # Estatísticas
    stats = (
        f"Matchfilter:\n"
        f"Mean: {np.mean(snr_1):.2f}\n"
        f"Std: {np.std(snr_1):.2f}\n\n"
        f"Correlate:\n"
        f"Mean: {np.mean(snr_2):.2f}\n"
        f"Std: {np.std(snr_2):.2f}"
    )

    plt.text(0.98, 0.98, stats,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_plot("snr_histogram_logbins", output_dir)


# ===================== SCATTER =====================
def plot_snr_scatter(snr_1, snr_2, output_dir=PLOT_PATH):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        "matchfilter": snr_1,
        "correlate": snr_2
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="matchfilter", y="correlate", alpha=0.5)

    # Linha y = x
    min_val = min(df.min())
    max_val = max(df.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--')

    plt.title("SNR Matchfilter vs Correlate")
    plt.xlabel("Matchfilter")
    plt.ylabel("Correlate")
    plt.grid(True, alpha=0.3)

    # Correlação
    corr = np.corrcoef(snr_1, snr_2)[0, 1]
    plt.text(0.05, 0.95, f"Correlation: {corr:.3f}",
             transform=plt.gca().transAxes,
             verticalalignment='top')

    save_plot("snr_scatter", output_dir)


# ===================== HELPER =====================
def save_plot(name, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{name}_{timestamp}.png")
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    
    plt.close()


# ===================== MAIN =====================
def main():
    snr_match, snr_corr = load_snr_values()

    if len(snr_match) == 0:
        print("❌ No valid data found!")
        return

    print(f"Loaded {len(snr_match)} samples")

    # Plots
    plot_snr_histogram(snr_match, snr_corr)
    plot_snr_scatter(snr_match, snr_corr)

    # Stats
    print("\n📊 Summary:")
    print(f"Matchfilter Mean: {np.mean(snr_match):.2f}")
    print(f"Correlate Mean: {np.mean(snr_corr):.2f}")

    corr = np.corrcoef(snr_match, snr_corr)[0, 1]
    print(f"Correlation: {corr:.3f}")


if __name__ == "__main__":
    main()