"""
Module d'analyses statistiques spatiales pour données FWI.

Dépendances : geopandas, libpysal, esda, scipy, numpy, pandas, matplotlib
    pip install geopandas libpysal esda
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats


# =====================================================================
# 1. PRÉPARATION DES DONNÉES
# =====================================================================

def load_and_clean(df, value_col="fwi-daily-proj"):
    """Convertit 'time' en datetime et supprime les NaN de l'indicateur."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=[value_col])
    return df


def add_time_features(df):
    """Ajoute année, mois, saison."""
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["season"] = df["month"].map({12: "DJF", 1: "DJF", 2: "DJF",
                                     3: "MAM", 4: "MAM", 5: "MAM",
                                     6: "JJA", 7: "JJA", 8: "JJA",
                                     9: "SON", 10: "SON", 11: "SON"})
    return df


def to_geodataframe(df, lon_col="lon", lat_col="lat", crs="EPSG:4326"):
    """Transforme un DataFrame en GeoDataFrame (points lon/lat)."""
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )


# =====================================================================
# 2. STATISTIQUES PAR PIXEL (agrégation temporelle -> GeoDataFrame)
# =====================================================================

def pixel_stats(df, value_col="fwi-daily-proj", group_cols=("rlat", "rlon")):
    """
    Agrège les statistiques par pixel (grille rotée) sur toute la période.
    Retourne un GeoDataFrame avec une ligne par pixel et les colonnes stat.
    """
    agg = (df.groupby(list(group_cols))
             .agg(mean=(value_col, "mean"),
                  median=(value_col, "median"),
                  std=(value_col, "std"),
                  q95=(value_col, lambda s: s.quantile(0.95)),
                  q05=(value_col, lambda s: s.quantile(0.05)),
                  max=(value_col, "max"),
                  lon=("lon", "first"),
                  lat=("lat", "first"))
             .reset_index())
    return to_geodataframe(agg)


def annual_pixel_mean(df, value_col="fwi-daily-proj", group_cols=("rlat", "rlon")):
    """Moyenne annuelle par pixel -> DataFrame long (pixel, year, mean)."""
    df = df.copy()
    df["year"] = df["time"].dt.year
    return (df.groupby(list(group_cols) + ["year"])
              .agg(mean=(value_col, "mean"),
                   lon=("lon", "first"),
                   lat=("lat", "first"))
              .reset_index())


def seasonal_pixel_mean(df, value_col="fwi-daily-proj", group_cols=("rlat", "rlon")):
    """Moyenne saisonnière par pixel -> GeoDataFrame large (1 colonne / saison)."""
    df = df.copy()
    df["season"] = df["time"].dt.month.map(
        {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    wide = (df.groupby(list(group_cols) + ["season"])[value_col].mean()
              .unstack("season").reset_index())
    coords = df.groupby(list(group_cols))[["lon", "lat"]].first().reset_index()
    wide = wide.merge(coords, on=list(group_cols))
    return to_geodataframe(wide)


def spatial_mean_timeseries(df, value_col="fwi-daily-proj"):
    """Moyenne spatiale par date -> DataFrame (time, value)."""
    return df.groupby("time")[value_col].mean().reset_index()


# =====================================================================
# 3. TENDANCES (Mann-Kendall + pente de Sen par pixel)
# =====================================================================

def _mk_sen_1d(y):
    """Mann-Kendall + Sen sur une série 1D -> (pente, p-value)."""
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    if len(y) < 4:
        return np.nan, np.nan
    n = len(y)
    i, j = np.triu_indices(n, k=1)
    slopes = (y[j] - y[i]) / (j - i)
    sen = np.median(slopes)
    s = np.sum(np.sign(y[j] - y[i]))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return sen, p


def trend_map(df_annual, group_cols=("rlat", "rlon")):
    """
    Applique Mann-Kendall + Sen sur la série annuelle de chaque pixel.
    Entrée : DataFrame long issu de annual_pixel_mean.
    Sortie : GeoDataFrame (1 ligne/pixel) avec sen_slope et p_value.
    """
    results = []
    for key, grp in df_annual.groupby(list(group_cols)):
        sen, p = _mk_sen_1d(grp.sort_values("year")["mean"].values)
        row = dict(zip(group_cols, key))
        row.update({"sen_slope": sen, "p_value": p,
                    "lon": grp["lon"].iloc[0], "lat": grp["lat"].iloc[0]})
        results.append(row)
    return to_geodataframe(pd.DataFrame(results))


# =====================================================================
# 4. AUTOCORRÉLATION SPATIALE (Moran's I) via libpysal + esda
# =====================================================================

def build_weights(gdf, k=8):
    """Matrice de poids spatiaux par k plus proches voisins (KNN)."""
    from libpysal.weights import KNN
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"   # standardisation en ligne
    return w


def morans_i(gdf, col="mean", k=8):
    """Moran's I global -> (I, p_value_sim)."""
    from esda.moran import Moran
    w = build_weights(gdf, k=k)
    mi = Moran(gdf[col].values, w, permutations=999)
    return mi.I, mi.p_sim


def local_moran(gdf, col="mean", k=8):
    """
    LISA / Moran local : ajoute les colonnes Ii, p_sim, quadrant au GeoDataFrame.
    quadrant : 1=HH (hot), 2=LH, 3=LL (cold), 4=HL.
    """
    from esda.moran import Moran_Local
    w = build_weights(gdf, k=k)
    lm = Moran_Local(gdf[col].values, w, permutations=499, seed=0)
    out = gdf.copy()
    out["Ii"] = lm.Is
    out["p_sim"] = lm.p_sim
    out["quadrant"] = lm.q
    out["lisa_sig"] = out["quadrant"].where(out["p_sim"] < 0.05, 0)
    return out


# =====================================================================
# 5. HOT SPOTS (Getis-Ord Gi*) via esda
# =====================================================================

def getis_ord_gi(gdf, col="mean", k=8):
    """Gi* par pixel -> colonne 'gi_z' (z-score) et 'gi_p' (p-value)."""
    from esda.getisord import G_Local
    from libpysal.weights import fill_diagonal
    w = build_weights(gdf, k=k)
    w = fill_diagonal(w, 1.0)   # inclure le pixel lui-même (Gi*)
    w.transform = "r"
    g = G_Local(gdf[col].values, w, star=True, permutations=499, seed=0)
    out = gdf.copy()
    out["gi_z"] = g.Zs
    out["gi_p"] = g.p_sim
    return out


# =====================================================================
# 6. VISUALISATIONS
# =====================================================================

def plot_map(gdf, col, title, ax=None, cmap="YlOrRd",
             vmin=None, vmax=None, markersize=7,
             facecolor="#1e1e1e", fg="white"):
    """Carte d'un GeoDataFrame avec fond sombre par défaut."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=facecolor)
    ax.set_facecolor(facecolor)
    gdf.plot(ax=ax, column=col, cmap=cmap, markersize=markersize,
             vmin=vmin, vmax=vmax, legend=True,
             legend_kwds={"shrink": 0.7})
    ax.set_title(title, color=fg, fontsize=12)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg)
        spine.set_alpha(0.15)
        spine.set_linewidth(0.5)
    # stylise la colorbar éventuellement créée
    for cax in ax.get_figure().axes:
        if cax is not ax:
            cax.tick_params(colors=fg)
            cax.yaxis.label.set_color(fg)
    return ax


def classify_pvalue(pvalues):
    """
    Transforme une série de p-values en 4 classes de significativité.

    Classes
    -------
    1 : p >= 0.10   -> H0 non rejeté (test rejeté au seuil 10%)
    2 : 0.05 <= p < 0.10 -> significatif au seuil 10% seulement
    3 : 0.01 <= p < 0.05 -> significatif au seuil 5% mais pas 1%
    4 : p < 0.01    -> significatif au seuil 1% (très significatif)
    """
    p = np.asarray(pvalues, dtype=float)
    classes = np.full_like(p, np.nan, dtype=float)
    classes[p >= 0.10]                  = 1
    classes[(p >= 0.05) & (p < 0.10)]   = 2
    classes[(p >= 0.01) & (p < 0.05)]   = 3
    classes[p < 0.01]                   = 4
    return classes


# =====================================================================
# Visualisation dédiée (colormap discrète + légende explicite)
# =====================================================================

SIG_COLORS = {
    1: "#3b3b3b",   # gris foncé  - non significatif (H0 non rejeté)
    2: "#fcae91",   # rouge clair - seuil 10%
    3: "#de2d26",   # rouge moyen - seuil 5%
    4: "#67000d",   # rouge foncé - seuil 1% (très significatif)
}

SIG_LABELS = {
    1: "p ≥ 0.10  (non significatif)",
    2: "0.05 ≤ p < 0.10  (10%)",
    3: "0.01 ≤ p < 0.05  (5%)",
    4: "p < 0.01  (1%)",
}


def plot_significance_map(gdf, col="sig_class", ax=None, title="Significativité",
                          facecolor="#1e1e1e", fg="white", markersize=3):
    """Carte avec 4 couleurs discrètes selon la classe de significativité."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=facecolor)
    ax.set_facecolor(facecolor)

    # colormap discrète à partir du dictionnaire
    cmap = mcolors.ListedColormap([SIG_COLORS[k] for k in sorted(SIG_COLORS)])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    gdf.plot(ax=ax, column=col, cmap=cmap, norm=norm,
             markersize=markersize, legend=False)

    # Légende manuelle (plus lisible qu'une colorbar discrète)
    handles = [plt.Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=SIG_COLORS[k],
                          markersize=10, label=SIG_LABELS[k])
               for k in sorted(SIG_COLORS)]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=9,
                    facecolor=facecolor, edgecolor=fg, labelcolor=fg,
                    framealpha=0.8)

    ax.set_title(title, color=fg, fontsize=12)
    ax.tick_params(colors=fg, length=3, width=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg); spine.set_alpha(0.25); spine.set_linewidth(0.5)
    ax.grid(True, color=fg, alpha=0.10, linewidth=0.3, linestyle="--")
    ax.set_axisbelow(True)
    return ax
