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

def _auto_text_color(bg_color):
    """Renvoie 'white' si le fond est sombre, 'black' sinon."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(bg_color)
    # luminance perçue (formule standard)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def _style_axis_text(ax, color):
    """Met tout le texte de l'axe (ticks, labels, titre) dans une couleur donnée."""
    ax.tick_params(colors=color, which="both")
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.title.set_color(color)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)


def plot_monthly_grid(df, dates, col="fwi-daily-proj",
                      cmap="YlOrRd", nrows=3, ncols=4,
                      figsize=(18, 12), shared_scale=True, title=None,
                      facecolor="#f5f5f5", axes_facecolor=None,
                      fg=None, markersize=1.5):
    """
    Trace une grille de cartes (une par date).

    Paramètres
    ----------
    facecolor : str
        Couleur de fond de la figure.
    axes_facecolor : str or None
        Couleur de fond de chaque carte. Si None, hérite de facecolor.
    fg : str or None
        Couleur du texte (titres, ticks, échelles, colorbars).
        Si None, choisie automatiquement selon la luminance du fond :
        blanc sur fond sombre, noir sur fond clair.
    shared_scale : bool
        True -> même échelle de couleur pour toutes les cartes.
    """
    if axes_facecolor is None:
        axes_facecolor = facecolor
    if fg is None:
        fg = _auto_text_color(facecolor)

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
                     cmap=cmap, vmin=vmin, vmax=vmax, markersize=markersize)
        axes[i].set_title(str(date)[:10], color=fg)
        _style_axis_text(axes[i], fg)          

    # cacher les axes vides
    for j in range(len(dates), len(axes)):
        axes[j].set_facecolor(axes_facecolor)
        axes[j].axis("off")

    map_axes = set(axes)
    for cax in fig.axes:
        if cax not in map_axes:                
            cax.tick_params(colors=fg)
            cax.yaxis.label.set_color(fg)
            for spine in cax.spines.values():
                spine.set_edgecolor(fg)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02, color=fg)

    plt.tight_layout()
    plt.show()
    return fig
