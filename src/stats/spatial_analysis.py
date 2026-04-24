"""
Module d'analyses statistiques spatiales pour données FWI sur grille rotée (Espagne).
Chaque fonction fait UNE chose. À appeler depuis un script principal.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
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


def to_xarray(df, value_col="fwi-daily-proj"):
    """
    Transforme le DataFrame long en DataArray (time, rlat, rlon).
    Conserve lon/lat comme coordonnées auxiliaires 2D.
    """
    ds = (df.set_index(["time", "rlat", "rlon"])[[value_col, "lon", "lat"]]
            .to_xarray())
    # lon/lat ne dépendent pas du temps -> on prend le 1er pas de temps
    ds["lon"] = ds["lon"].isel(time=0)
    ds["lat"] = ds["lat"].isel(time=0)
    return ds[value_col], ds[["lon", "lat"]]


def add_time_features(df):
    """Ajoute année, mois, saison pour les agrégations temporelles."""
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["season"] = df["month"].map({12: "DJF", 1: "DJF", 2: "DJF",
                                     3: "MAM", 4: "MAM", 5: "MAM",
                                     6: "JJA", 7: "JJA", 8: "JJA",
                                     9: "SON", 10: "SON", 11: "SON"})
    return df


# =====================================================================
# 2. STATISTIQUES DESCRIPTIVES
# =====================================================================

def pixel_stats(da):
    """Statistiques par pixel sur toute la période (moyenne, médiane, std, q95)."""
    return xr.Dataset({
        "mean":   da.mean("time"),
        "median": da.median("time"),
        "std":    da.std("time"),
        "q95":    da.quantile(0.95, "time").drop_vars("quantile"),
        "max":    da.max("time"),
    })


def temporal_stats(da):
    """Moyenne spatiale par pas de temps -> série temporelle scalaire."""
    return da.mean(["rlat", "rlon"])


def annual_mean(da):
    """Moyenne annuelle par pixel."""
    return da.groupby("time.year").mean("time")


def seasonal_mean(da):
    """Moyenne saisonnière par pixel (DJF, MAM, JJA, SON)."""
    return da.groupby("time.season").mean("time")


# =====================================================================
# 3. TENDANCES TEMPORELLES (Mann-Kendall + pente de Sen par pixel)
# =====================================================================

def _mk_sen_1d(y):
    """Mann-Kendall + Sen sur une série 1D. Renvoie (pente, p-value)."""
    y = y[~np.isnan(y)]
    if len(y) < 4:
        return np.nan, np.nan
    # pente de Sen
    n = len(y)
    i, j = np.triu_indices(n, k=1)
    slopes = (y[j] - y[i]) / (j - i)
    sen = np.median(slopes)
    # Mann-Kendall (approx normale)
    s = np.sum(np.sign(y[j] - y[i]))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return sen, p


def trend_map(da_annual):
    """Applique MK+Sen sur chaque pixel à partir d'une série annuelle."""
    slope, pval = xr.apply_ufunc(
        _mk_sen_1d, da_annual,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True, dask="parallelized",
        output_dtypes=[float, float],
    )
    return xr.Dataset({"sen_slope": slope, "p_value": pval})


# =====================================================================
# 4. AUTOCORRÉLATION SPATIALE (Moran's I global)
# =====================================================================

def morans_i(values_2d):
    """
    Moran's I global avec voisinage rook (4 voisins) sur grille régulière.
    Retourne I et p-value (permutation simple).
    """
    v = values_2d.astype(float)
    mask = ~np.isnan(v)
    vm = v - np.nanmean(v)
    vm = np.where(mask, vm, 0)

    # somme des produits entre voisins (haut/bas/gauche/droite)
    num = (
        np.sum(vm[:-1, :] * vm[1:, :] * mask[:-1, :] * mask[1:, :]) +
        np.sum(vm[:, :-1] * vm[:, 1:] * mask[:, :-1] * mask[:, 1:])
    ) * 2  # symétrie
    w_sum = 2 * (np.sum(mask[:-1, :] * mask[1:, :]) +
                 np.sum(mask[:, :-1] * mask[:, 1:]))
    n = mask.sum()
    denom = np.nansum(vm ** 2)
    if denom == 0 or w_sum == 0:
        return np.nan, np.nan
    I = (n / w_sum) * (num / denom)
    # p-value par permutation
    rng = np.random.default_rng(0)
    flat = v[mask]
    perms = []
    for _ in range(99):
        shuffled = v.copy()
        shuffled[mask] = rng.permutation(flat)
        perms.append(morans_i_raw(shuffled))
    p = (np.sum(np.array(perms) >= I) + 1) / 100
    return I, p


def morans_i_raw(v):
    """Version sans p-value pour les permutations."""
    mask = ~np.isnan(v)
    vm = np.where(mask, v - np.nanmean(v), 0)
    num = (np.sum(vm[:-1, :] * vm[1:, :] * mask[:-1, :] * mask[1:, :]) +
           np.sum(vm[:, :-1] * vm[:, 1:] * mask[:, :-1] * mask[:, 1:])) * 2
    w_sum = 2 * (np.sum(mask[:-1, :] * mask[1:, :]) +
                 np.sum(mask[:, :-1] * mask[:, 1:]))
    denom = np.nansum(vm ** 2)
    n = mask.sum()
    return (n / w_sum) * (num / denom) if denom > 0 and w_sum > 0 else np.nan


# =====================================================================
# 5. HOT SPOTS (Getis-Ord Gi* simplifié, grille régulière)
# =====================================================================

def getis_ord_gi(values_2d, window=3):
    """
    Gi* avec fenêtre carrée (window x window). Retourne un z-score par pixel.
    Z > 1.96 : hot spot ; Z < -1.96 : cold spot (seuil 5%).
    """
    from scipy.ndimage import uniform_filter
    v = values_2d.astype(float)
    mask = (~np.isnan(v)).astype(float)
    v_filled = np.where(mask == 1, v, 0)

    # somme locale et nombre de voisins valides
    local_sum = uniform_filter(v_filled, size=window) * (window ** 2)
    local_n = uniform_filter(mask, size=window) * (window ** 2)

    mean_global = np.nanmean(v)
    std_global = np.nanstd(v)
    N = mask.sum()

    numer = local_sum - mean_global * local_n
    denom = std_global * np.sqrt((N * local_n - local_n ** 2) / (N - 1))
    z = np.where(denom > 0, numer / denom, np.nan)
    return np.where(mask == 1, z, np.nan)


# =====================================================================
# 6. VISUALISATIONS
# =====================================================================

def plot_map(da, title, cmap="YlOrRd", ax=None):
    """Carte simple d'un DataArray 2D en coordonnées (lat, lon) réelles."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    lon = da["lon"].values if "lon" in da.coords else None
    lat = da["lat"].values if "lat" in da.coords else None
    if lon is not None and lat is not None:
        sc = ax.pcolormesh(lon, lat, da.values, cmap=cmap, shading="auto")
    else:
        sc = ax.imshow(da.values, cmap=cmap, origin="lower")
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    return ax


def plot_timeseries(ts, title="Moyenne spatiale FWI"):
    """Trace une série temporelle scalaire."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ts.plot(ax=ax)
    ax.set_title(title); ax.grid(alpha=0.3)
    return ax