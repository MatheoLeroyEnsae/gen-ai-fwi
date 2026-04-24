"""
Module de statistiques descriptives pour l'indice FWI.
Une fonction = une tâche. À importer depuis un script principal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

def setup_style():
    """Applique le style graphique global (seaborn + matplotlib)."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)


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

def plot_histogram(df, col="fwi-daily-proj", bins=30):
    """Histogramme + KDE."""
    plt.figure()
    sns.histplot(df[col], bins=bins, kde=True, color="steelblue")
    plt.title(f"Distribution de l'indice {col}", fontsize=14)
    plt.xlabel(col); plt.ylabel("Fréquence")
    plt.tight_layout(); plt.show()


def plot_boxplot(df, col="fwi-daily-proj"):
    """Boxplot univarié."""
    plt.figure()
    sns.boxplot(y=df[col], color="lightgreen")
    plt.title(f"Boxplot de {col}"); plt.ylabel("FWI")
    plt.tight_layout(); plt.show()


def plot_violin(df, col="fwi-daily-proj"):
    """Violin plot univarié."""
    plt.figure()
    sns.violinplot(y=df[col], color="orange")
    plt.title(f"Violin plot de {col}"); plt.ylabel("FWI")
    plt.tight_layout(); plt.show()


# =====================================================================
# 4. AGRÉGATION TEMPORELLE + SÉRIES
# =====================================================================

def daily_agg(df, col="fwi-daily-proj"):
    """Agrégation journalière : min / mean / max sur l'ensemble des pixels."""
    out = df.copy()
    out["date"] = out["time"].dt.date
    return out.groupby("date")[col].agg(["min", "mean", "max"]).reset_index()


def plot_daily_range(df_daily, title="Évolution temporelle du FWI (min / mean / max)"):
    """Trace min, mean, max journaliers sur un même graphique."""
    plt.figure(figsize=(12, 5))
    plt.plot(df_daily["date"], df_daily["min"],  label="Min",  linewidth=1,   color="blue")
    plt.plot(df_daily["date"], df_daily["mean"], label="Mean", linewidth=1.5, color="black")
    plt.plot(df_daily["date"], df_daily["max"],  label="Max",  linewidth=1,   color="red")
    plt.title(title); plt.xlabel("Date"); plt.ylabel("FWI")
    plt.xticks(rotation=45); plt.legend()
    plt.tight_layout(); plt.show()


def plot_daily_band(df_daily, title="FWI journalier : variabilité min-max + moyenne"):
    """Bande min-max en fond + moyenne par-dessus."""
    plt.figure(figsize=(12, 5))
    plt.fill_between(df_daily["date"], df_daily["min"], df_daily["max"],
                     alpha=0.2, color="orange", label="Min-Max range")
    plt.plot(df_daily["date"], df_daily["mean"], color="black",
             linewidth=1.5, label="Mean")
    plt.title(title); plt.xlabel("Date"); plt.ylabel("FWI")
    plt.xticks(rotation=45); plt.legend()
    plt.tight_layout(); plt.show()


def plot_raw_timeseries(df, col="fwi-daily-proj"):
    """Série brute (tous points) — utile pour voir la dispersion."""
    plt.figure()
    plt.plot(df["time"], df[col], linewidth=0.8, color="darkred")
    plt.title(f"Évolution temporelle de {col}")
    plt.xlabel("Date"); plt.ylabel("FWI")
    plt.xticks(rotation=90)
    plt.tight_layout(); plt.show()


# =====================================================================
# 5. DISTRIBUTIONS PAR GROUPE
# =====================================================================

def plot_boxplot_by(df, by="year", col="fwi-daily-proj", palette="Set2"):
    """Boxplot groupé (par année, mois, saison...)."""
    plt.figure()
    sns.boxplot(data=df, x=by, y=col, hue=by, palette=palette, legend=False)
    plt.title(f"Distribution de {col} par {by}")
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.show()


# =====================================================================
# 6. CORRÉLATIONS
# =====================================================================

def plot_correlation_matrix(df):
    """Heatmap de corrélation sur toutes les colonnes numériques."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) <= 1:
        print("Pas assez de variables numériques pour une matrice de corrélation.")
        return None
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice de corrélation (variables numériques)")
    plt.tight_layout(); plt.show()
    return corr
