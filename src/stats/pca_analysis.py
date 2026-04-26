"""
PCA / EOF analysis for spatio-temporal FWI data.

Ce module fournit les outils pour effectuer une analyse en composantes
principales (PCA / EOF) sur des champs spatio-temporels du Fire Weather
Index (FWI), tels que stockés dans `df_small` (grille 28x28 sur le sud
de l'Espagne, 1970-2005, fréquence journalière).

Deux orientations de la matrice sont supportées :

- **S-mode** : lignes = temps, colonnes = pixels.
  Les composantes principales (EOF) sont des **patterns spatiaux** ;
  les scores (PC) sont des **séries temporelles** qui donnent le poids
  de chaque pattern à chaque date. 

- **T-mode** : lignes = pixels, colonnes = temps.
  Les composantes sont alors des **profils temporels** types ; les
  scores donnent, pour chaque pixel, le poids de chaque profil. Utile
  pour regrouper des pixels au comportement temporel similaire.

Convention de nommage des colonnes pivot :
    - colonnes spatiales : `pixel_id` = "lat_lon" (string)
    - index temporel : `time` (datetime64)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# 1. Mise en forme de la matrice
# ---------------------------------------------------------------------

def build_pixel_id(df: pd.DataFrame,
                   lat_col: str = "lat",
                   lon_col: str = "lon",
                   round_decimals: int = 4) -> pd.DataFrame:
    """Ajoute une colonne `pixel_id` = "lat_lon" (chaîne).

    Arrondir les coordonnées évite les soucis d'égalité flottante quand
    on pivote.
    """
    out = df.copy()
    out["pixel_id"] = (
        out[lat_col].round(round_decimals).astype(str)
        + "_"
        + out[lon_col].round(round_decimals).astype(str)
    )
    return out


def aggregate_temporal(df: pd.DataFrame,
                       freq: str = "M",
                       value_col: str = "fwi-daily-proj",
                       time_col: str = "time",
                       lat_col: str = "lat",
                       lon_col: str = "lon",
                       agg: str = "mean") -> pd.DataFrame:
    """Agrège le DataFrame long sur une fréquence temporelle donnée.

    Indispensable pour le T-mode : avec ~13 000 jours, on aurait plus
    de "variables temporelles" que d'observations (pixels)

    Parameters
    ----------
    freq : str
        Fréquence pandas. Exemples :
        - "D"  : journalier (pas d'agrégation)
        - "W"  : hebdomadaire
        - "MS" : début de mois (équivalent moderne de "M")
        - "QS" : début de trimestre (saisonnier)
        - "YS" : annuel
    agg : str
        Fonction d'agrégation : "mean", "max", "median", ou un percentile
        sous la forme "q95" (95e percentile), "q05" (5e percentile)...

    Returns
    -------
    pd.DataFrame
        Mêmes colonnes que df, mais avec `time` arrondi à la fréquence.
    """
    if "pixel_id" not in df.columns:
        df = build_pixel_id(df, lat_col=lat_col, lon_col=lon_col)

    # remap "M" (deprecated) vers "MS" pour rester silencieux
    freq_map = {"M": "MS", "Q": "QS", "Y": "YS", "A": "YS"}
    pandas_freq = freq_map.get(freq, freq)

    # règle d'agrégation
    if agg.startswith("q") and len(agg) > 1:
        try:
            q = int(agg[1:]) / 100
        except ValueError as exc:
            raise ValueError(f"agg='{agg}' invalide. Attendu 'q95', 'q05'...") from exc
        agg_func = lambda s: s.quantile(q)  # noqa: E731
    elif agg in {"mean", "median", "max", "min", "std", "sum"}:
        agg_func = agg
    else:
        raise ValueError(f"agg='{agg}' non supporté.")

    out = (df
           .groupby([pd.Grouper(key=time_col, freq=pandas_freq),
                     "pixel_id", lat_col, lon_col])[value_col]
           .agg(agg_func)
           .reset_index())
    return out


def to_space_time_matrix(df: pd.DataFrame,
                         value_col: str = "fwi-daily-proj",
                         time_col: str = "time",
                         pixel_col: str = "pixel_id") -> pd.DataFrame:
    """Pivote en matrice S-mode : lignes = temps, colonnes = pixels.

    Returns
    -------
    pd.DataFrame
        index = dates, colonnes = pixel_id, valeurs = FWI.
    """
    if pixel_col not in df.columns:
        df = build_pixel_id(df)

    mat = (df
           .pivot_table(index=time_col,
                        columns=pixel_col,
                        values=value_col,
                        aggfunc="mean")
           .sort_index())
    return mat


def to_time_space_matrix(df: pd.DataFrame,
                         value_col: str = "fwi-daily-proj",
                         time_col: str = "time",
                         pixel_col: str = "pixel_id") -> pd.DataFrame:
    """Pivote en matrice T-mode : lignes = pixels, colonnes = temps."""
    return to_space_time_matrix(df, value_col, time_col, pixel_col).T


def clean_matrix(mat: pd.DataFrame,
                 max_nan_frac_col: float = 0.05) -> pd.DataFrame:
    """Nettoie la matrice avant PCA.

    - supprime les colonnes (pixels) qui ont trop de NaN
    - remplit les NaN restants par la moyenne de la colonne
    """
    nan_frac = mat.isna().mean(axis=0)
    keep = nan_frac[nan_frac <= max_nan_frac_col].index
    mat_clean = mat[keep].copy()
    # impute par la moyenne de chaque colonne
    mat_clean = mat_clean.fillna(mat_clean.mean(axis=0))
    return mat_clean


# ---------------------------------------------------------------------
# 2. Exécution de la PCA
# ---------------------------------------------------------------------

@dataclass
class PCAResult:
    """Résultats d'une PCA sur un champ FWI.

    Attributes
    ----------
    pca : sklearn.decomposition.PCA
        L'objet sklearn ajusté.
    scaler : StandardScaler | None
        Le scaler utilisé (None si pas de standardisation).
    scores : pd.DataFrame
        Projections des observations sur les composantes principales.
        - S-mode : index = dates, colonnes = PC1..PCn
        - T-mode : index = pixels, colonnes = PC1..PCn
    components : pd.DataFrame
        Matrice des composantes (loadings) :
        - S-mode : index = PCn, colonnes = pixels  -> patterns spatiaux
        - T-mode : index = PCn, colonnes = dates   -> profils temporels
    explained_variance_ratio : np.ndarray
        Part de variance expliquée par chaque composante.
    mode : str
        'S' ou 'T'.
    """
    pca: PCA
    scaler: Optional[StandardScaler]
    scores: pd.DataFrame
    components: pd.DataFrame
    explained_variance_ratio: np.ndarray
    mode: str

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def n_components_for(self, threshold: float = 0.9) -> int:
        """Nb de composantes nécessaires pour atteindre `threshold` de variance."""
        cum = self.cumulative_variance
        return int(np.searchsorted(cum, threshold) + 1)


def run_pca(mat: pd.DataFrame,
            n_components: Optional[int] = None,
            standardize: bool = True,
            mode: str = "S") -> PCAResult:
    """Exécute une PCA sur la matrice donnée.

    Parameters
    ----------
    mat : pd.DataFrame
        Matrice déjà pivotée et nettoyée (cf. `to_space_time_matrix`
        ou `to_time_space_matrix` + `clean_matrix`).
    n_components : int, optional
        Nombre de composantes à conserver. Si None, garde
        min(n_samples, n_features).
    standardize : bool
        Si True, standardise les colonnes (moyenne 0, var 1) avant la PCA.
        Recommandé en S-mode pour donner le même poids à chaque pixel.
    mode : {'S', 'T'}
        Orientation utilisée (purement informatif, sert au labelling).
    """
    X = mat.values.astype(float)

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        scaler = None
        X_proc = X - X.mean(axis=0)  # PCA centre toujours

    if n_components is None:
        n_components = min(X_proc.shape)

    pca = PCA(n_components=n_components)
    scores_arr = pca.fit_transform(X_proc)

    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    scores = pd.DataFrame(scores_arr, index=mat.index, columns=pc_labels)
    components = pd.DataFrame(pca.components_,
                              index=pc_labels,
                              columns=mat.columns)

    return PCAResult(
        pca=pca,
        scaler=scaler,
        scores=scores,
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        mode=mode,
    )


# ---------------------------------------------------------------------
# 3. Visualisations
# ---------------------------------------------------------------------

def setup_dark_style() -> None:
    """Style sombre cohérent avec le reste du notebook."""
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor":   "#1e1e1e",
        "axes.edgecolor":   "white",
        "axes.labelcolor":  "white",
        "axes.titlecolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
        "text.color":       "white",
        "legend.facecolor": "#1e1e1e",
        "legend.edgecolor": "white",
        "grid.color":       "#444444",
    })


def plot_explained_variance(result: PCAResult,
                            n_show: int = 30,
                            ax: Optional[Axes] = None) -> Axes:
    """Bar plot + courbe cumulative de la variance expliquée."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    n = min(n_show, len(result.explained_variance_ratio))
    x = np.arange(1, n + 1)
    ax.bar(x, result.explained_variance_ratio[:n] * 100,
           color="#ff8c42", alpha=0.85, label="Individuelle")

    ax2 = ax.twinx()
    ax2.plot(x, result.cumulative_variance[:n] * 100,
             color="#4cc9f0", marker="o", linewidth=2, label="Cumulée")
    ax2.axhline(90, color="#aaaaaa", linestyle="--", linewidth=0.8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Variance cumulée (%)", color="#4cc9f0")
    ax2.tick_params(axis="y", colors="#4cc9f0")
    ax2.spines["right"].set_color("#4cc9f0")

    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Variance expliquée (%)", color="#ff8c42")
    ax.tick_params(axis="y", colors="#ff8c42")
    ax.set_title(f"Variance expliquée - PCA mode {result.mode}")
    return ax


def _pixels_to_grid(values: pd.Series,
                    pixel_index: pd.Index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruit une grille 2D (lat, lon) à partir des valeurs par pixel.

    Le `pixel_index` doit contenir des chaînes "lat_lon".
    """
    lats = np.array([float(p.split("_")[0]) for p in pixel_index])
    lons = np.array([float(p.split("_")[1]) for p in pixel_index])

    uniq_lat = np.sort(np.unique(lats))[::-1]   # nord en haut
    uniq_lon = np.sort(np.unique(lons))

    grid = np.full((len(uniq_lat), len(uniq_lon)), np.nan)
    lat_idx = {v: i for i, v in enumerate(uniq_lat)}
    lon_idx = {v: i for i, v in enumerate(uniq_lon)}

    for p, v in zip(pixel_index, values.values):
        la, lo = p.split("_")
        grid[lat_idx[float(la)], lon_idx[float(lo)]] = v

    return grid, uniq_lat, uniq_lon


def plot_spatial_eof(result: PCAResult,
                     n_components: int = 6,
                     ncols: int = 3,
                     cmap: str = "RdBu_r",
                     symmetric: bool = True) -> Figure:
    """Trace les premiers patterns spatiaux (S-mode uniquement).

    Chaque sous-figure montre la composante en tant que carte 2D :
    - rouge = corrélation positive avec le score PCk
    - bleu  = corrélation négative
    """
    if result.mode != "S":
        raise ValueError("plot_spatial_eof : seulement valable en S-mode.")

    n = min(n_components, len(result.components))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             facecolor="#1e1e1e")
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        comp = result.components.iloc[i]
        grid, lat, lon = _pixels_to_grid(comp, result.components.columns)

        if symmetric:
            vmax = np.nanmax(np.abs(grid))
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(grid), np.nanmax(grid)

        ax = axes[i]
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       aspect="auto", origin="upper")
        var = result.explained_variance_ratio[i] * 100
        ax.set_title(f"EOF{i+1} - {var:.1f}% var.", color="white")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Patterns spatiaux (EOF) du FWI", color="white", fontsize=14)
    fig.tight_layout()
    return fig


def plot_pc_timeseries(result: PCAResult,
                       n_components: int = 4,
                       rolling: Optional[int] = 365) -> Figure:
    """Trace les séries temporelles des premières PC (S-mode).

    `rolling` ajoute une moyenne glissante (en jours par défaut) pour
    lisser le signal journalier — utile pour faire ressortir le cycle
    annuel et les tendances de plus basse fréquence.
    """
    if result.mode != "S":
        raise ValueError("plot_pc_timeseries : seulement valable en S-mode.")

    n = min(n_components, result.scores.shape[1])
    fig, axes = plt.subplots(n, 1,
                             figsize=(12, 2.5 * n),
                             sharex=True,
                             facecolor="#1e1e1e")
    axes = np.atleast_1d(axes)

    for i in range(n):
        ax = axes[i]
        s = result.scores[f"PC{i+1}"]
        ax.plot(s.index, s.values, color="#888888", linewidth=0.5,
                alpha=0.6, label="journalier")
        if rolling and rolling > 1:
            ax.plot(s.index, s.rolling(rolling, center=True).mean(),
                    color="#ff8c42", linewidth=1.6,
                    label=f"moyenne glissante {rolling} j")
        var = result.explained_variance_ratio[i] * 100
        ax.set_ylabel(f"PC{i+1}\n({var:.1f}%)", color="white")
        ax.axhline(0, color="white", linewidth=0.4, alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Séries temporelles des composantes principales",
                 color="white", fontsize=13)
    fig.tight_layout()
    return fig


def plot_temporal_modes(result: PCAResult,
                        n_components: int = 4,
                        rolling: Optional[int] = 365) -> Figure:
    """Trace les profils temporels (T-mode) : composantes vues comme séries."""
    if result.mode != "T":
        raise ValueError("plot_temporal_modes : seulement valable en T-mode.")

    n = min(n_components, result.components.shape[0])
    fig, axes = plt.subplots(n, 1,
                             figsize=(12, 2.5 * n),
                             sharex=True,
                             facecolor="#1e1e1e")
    axes = np.atleast_1d(axes)

    # en T-mode, les colonnes des components sont les dates
    dates = pd.to_datetime(result.components.columns)

    for i in range(n):
        ax = axes[i]
        s = pd.Series(result.components.iloc[i].values, index=dates)
        ax.plot(s.index, s.values, color="#888888", linewidth=0.5, alpha=0.6)
        if rolling and rolling > 1:
            ax.plot(s.index, s.rolling(rolling, center=True).mean(),
                    color="#4cc9f0", linewidth=1.6)
        var = result.explained_variance_ratio[i] * 100
        ax.set_ylabel(f"Mode {i+1}\n({var:.1f}%)", color="white")
        ax.axhline(0, color="white", linewidth=0.4, alpha=0.4)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Profils temporels (T-mode)", color="white", fontsize=13)
    fig.tight_layout()
    return fig


def plot_pixel_loadings(result: PCAResult,
                        n_components: int = 4,
                        ncols: int = 2,
                        cmap: str = "RdBu_r") -> Figure:
    """En T-mode, les *scores* sont indexés par pixel et représentent
    le poids de chaque pixel sur chaque profil temporel.
    On peut donc les recartographier en 2D.
    """
    if result.mode != "T":
        raise ValueError("plot_pixel_loadings : seulement valable en T-mode.")

    n = min(n_components, result.scores.shape[1])
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             facecolor="#1e1e1e")
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        loads = result.scores[f"PC{i+1}"]
        grid, lat, lon = _pixels_to_grid(loads, result.scores.index)
        vmax = np.nanmax(np.abs(grid))
        ax = axes[i]
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax,
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       aspect="auto", origin="upper")
        var = result.explained_variance_ratio[i] * 100
        ax.set_title(f"Mode {i+1} - {var:.1f}% var.", color="white")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cartes des poids par pixel (T-mode)",
                 color="white", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 4. Comparaison S-mode / T-mode
# ---------------------------------------------------------------------

def compare_modes_variance(result_s: PCAResult,
                           result_t: PCAResult,
                           n_show: int = 20) -> Figure:
    """Compare les courbes de variance cumulée S-mode vs T-mode."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#1e1e1e")
    n_s = min(n_show, len(result_s.cumulative_variance))
    n_t = min(n_show, len(result_t.cumulative_variance))
    ax.plot(np.arange(1, n_s + 1), result_s.cumulative_variance[:n_s] * 100,
            "o-", color="#ff8c42", label="S-mode (EOF spatiales)")
    ax.plot(np.arange(1, n_t + 1), result_t.cumulative_variance[:n_t] * 100,
            "s-", color="#4cc9f0", label="T-mode (modes temporels)")
    ax.axhline(90, color="#aaaaaa", linestyle="--", linewidth=0.8,
               label="90 %")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance cumulée (%)")
    ax.set_title("Variance cumulée : S-mode vs T-mode")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def summary_table(result: PCAResult, n: int = 10) -> pd.DataFrame:
    """Petit tableau récapitulatif des n premières composantes."""
    n = min(n, len(result.explained_variance_ratio))
    return pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n)],
        "Variance expliquée (%)": (result.explained_variance_ratio[:n] * 100).round(2),
        "Variance cumulée (%)":   (result.cumulative_variance[:n] * 100).round(2),
    })
