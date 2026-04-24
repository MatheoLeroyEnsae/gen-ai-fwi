"""
Module de cartographie du FWI sur l'Espagne.
Une fonction = une tâche.
"""

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd


# =====================================================================
# 1. GÉNÉRATION DES DATES
# =====================================================================

def monthly_dates(year, day=1):
    """Renvoie les 12 premiers jours de chaque mois pour une année donnée."""
    return pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS").strftime("%Y-%m-%d").tolist()


# =====================================================================
# 2. CONSTRUCTION DU GEODATAFRAME
# =====================================================================

def to_geodataframe(df, lon_col="lon", lat_col="lat", crs="EPSG:4326"):
    """Transforme un DataFrame en GeoDataFrame à partir de lon/lat."""
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )


def filter_spain_bbox(df, lon_min=-10, lon_max=4, lat_min=35, lat_max=44):
    """Filtre sur la bounding box de l'Espagne péninsulaire (optionnel)."""
    return df[(df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
              (df["lat"] >= lat_min) & (df["lat"] <= lat_max)].copy()


# =====================================================================
# 3. TRACÉ D'UNE CARTE
# =====================================================================

def plot_fwi_map(df_date, ax, col="fwi-daily-proj",
                 cmap="YlOrRd", vmin=None, vmax=None, markersize=1.5):
    """Trace une seule carte FWI sur un axe donné."""
    gdf = to_geodataframe(df_date)
    gdf.plot(
        ax=ax,
        column=col,
        cmap=cmap,
        markersize=markersize,
        vmin=vmin, vmax=vmax,
        legend=True,
    )
    ax.axis("off")
    return ax


# =====================================================================
# 4. GRILLE MULTI-DATES
# =====================================================================

def plot_monthly_grid(df, dates, col="fwi-daily-proj",
                      cmap="YlOrRd", nrows=3, ncols=4,
                      figsize=(18, 12), shared_scale=True, title=None, facecolor="#f5f5f5", 
                      axes_facecolor=None):
    """
    Trace une grille de cartes (une par date).
    shared_scale=True -> même échelle de couleur pour toutes les cartes
                         (essentiel pour comparer visuellement les mois).
    """
    if axes_facecolor is None:
        axes_facecolor = facecolor
    # bornes communes éventuelles
    vmin = vmax = None
    if shared_scale:
        subset = df[df["time"].isin(pd.to_datetime(dates))][col]
        vmin, vmax = subset.min(), subset.max()

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=facecolor)
    axes = axes.flatten()

    for i, date in enumerate(dates):
        axes[i].set_facecolor(axes_facecolor)
        temp = df[df["time"] == pd.to_datetime(date)]
        plot_fwi_map(temp, ax=axes[i], col=col,
                     cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(str(date)[:10])

    # cacher les axes vides si la grille est plus grande que le nb de dates
    for j in range(len(dates), len(axes)):
        axes[j].set_facecolor(axes_facecolor)
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    return fig
