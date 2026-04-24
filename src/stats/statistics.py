"""
Module de statistiques descriptives pour l'indice FWI.
Une fonction = une tâche. À importer depuis un script principal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

path = "/home/onyxia/work/gen-ai-fwi"

# =====================================================================
# 0. CONFIGURATION
# =====================================================================


def truncate_cmap(cmap_name, minval=0.3, maxval=1.0, n=256):
    """Renvoie une version tronquée d'une colormap (enlève le début clair)."""
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc", colors
    )


def setup_style(n_colors=6, start=0.5):
    """Style global avec palette YlOrRd tronquée (démarre à l'orange)."""
    cmap = truncate_cmap("YlOrRd", minval=start, maxval=1.0)
    palette = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    sns.set_theme(style="darkgrid")
    sns.set_palette(palette)
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


# =====================================================================
# 1. PRÉPARATION DES DONNÉES
# =====================================================================

def prepare_data(df, value_col="fwi-daily-proj", drop_zeros=True):
    """Convertit 'time' en datetime, trie, supprime NaN (et zéros optionnel)."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").dropna(subset=[value_col])
    if drop_zeros:
        df = df[df[value_col] != 0]
    df["year"] = df["time"].dt.year
    return df


def filter_years(df, years):
    """Filtre le DataFrame sur une liste d'années."""
    return df[df["year"].isin(years)].copy()


# =====================================================================
# 2. STATISTIQUES DESCRIPTIVES CHIFFRÉES
# =====================================================================

def describe_variable(df, col="fwi-daily-proj",
                      percentiles=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    """Retourne describe() + skewness + kurtosis dans un dict."""
    s = df[col]
    return {
        "describe": s.describe(percentiles=list(percentiles)),
        "skewness": s.skew(),
        "kurtosis": s.kurtosis(),
    }


def print_stats(stats, col="fwi-daily-proj"):
    """Affichage propre du dict renvoyé par describe_variable."""
    print(f"=== Statistiques descriptives de '{col}' ===\n")
    print(stats["describe"])
    print(f"\nSkewness (asymétrie)     : {stats['skewness']:.3f}")
    print(f"Kurtosis (aplatissement) : {stats['kurtosis']:.3f}")


# =====================================================================
# 3. VISUALISATIONS UNIVARIÉES
# =====================================================================

def plot_histogram(df, col="fwi-daily-proj", bins=30, log_bool=False, color="steelblue"):
    """Histogramme + KDE."""
    plt.figure()
    if log_bool:
        sns.histplot(np.sign(df[col])*np.log1p(np.abs(df[col])), bins=bins, kde=True)  # color=color
        plt.title(f"Distribution de la transformée logarithmique de l'indice {col}", fontsize=14)
    else:
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f"Distribution de l'indice {col}", fontsize=14)
    plt.xlabel(col); plt.ylabel("Fréquence")
    plt.savefig(f"{path}/output/histogram_fwi.png")
    plt.tight_layout(); plt.show()


def plot_violin(df, col="fwi-daily-proj"):
    """Violin plot univarié."""
    plt.figure()
    sns.violinplot(y=df[col])
    plt.title(f"Violin plot de {col}"); plt.ylabel("FWI")
    plt.savefig(f"{path}/output/violin_fwi.png")
    plt.tight_layout(); plt.show()


# =====================================================================
# 4. AGRÉGATION TEMPORELLE + SÉRIES
# =====================================================================

def daily_agg(df, col="fwi-daily-proj"):
    """Agrégation journalière : min / mean / max sur l'ensemble des pixels."""
    out = df.copy()
    out["date"] = out["time"].dt.date
    return out.groupby("date")[col].agg(["min", "mean", "max"]).reset_index()


def plot_daily_range(df_daily, title="Évolution temporelle du FWI (min / mean / max)", num=1):
    """Trace min, mean, max journaliers sur un même graphique."""
    plt.figure()
    plt.plot(df_daily["date"], df_daily["min"],  label="Min",  linewidth=1,   color="blue")
    plt.plot(df_daily["date"], df_daily["mean"], label="Mean", linewidth=1.5, color="black")
    plt.plot(df_daily["date"], df_daily["max"],  label="Max",  linewidth=1,   color="red")
    plt.title(title); plt.xlabel("Date"); plt.ylabel("FWI")
    plt.xticks(rotation=45); plt.legend()
    plt.savefig(f"{path}/output/daily_range_fwi_{num}.png")
    plt.tight_layout(); plt.show()

# =====================================================================
# 5. DISTRIBUTIONS PAR GROUPE
# =====================================================================


def plot_boxplot_by(df, by="year", col="fwi-daily-proj"):
    """Boxplot groupé (par année, mois, saison...)."""
    plt.figure()
    sns.boxplot(data=df, x=by, y=col, legend=False)
    plt.title(f"Distribution de {col} par {by}")
    plt.xticks(rotation=45)
    plt.savefig(f"{path}/output/boxplot_fwi.png")
    plt.tight_layout(); plt.show()