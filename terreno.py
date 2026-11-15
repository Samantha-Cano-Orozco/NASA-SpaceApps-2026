"""
terreno.py  —  Space Apps 2026 · Análogos de Marte (suelos)
Elevación, pendiente, rugosidad y humedad del suelo de una o muchas regiones de la Tierra.

USO
    python terreno.py --carpeta resources --region pinacate     # una región: DEM y SMAP sueltos en la carpeta
    python terreno.py --carpeta regiones                         # muchas: una subcarpeta por región
    python terreno.py --carpeta regiones --sin-filtro-calidad    # usar también la humedad SMAP de calidad baja

ARCHIVOS QUE BUSCA (en la carpeta de cada región)
    *.hgt (+ *.swb)   SRTM / NASADEM. El .swb (máscara de agua) es opcional; el agua se excluye de todo.
    *.tif, *.tiff     cualquier otro DEM en GeoTIFF (Copernicus, ALOS, etc.), en grados o en metros.
    SMAP*.h5          humedad del suelo SMAP L3 (SPL3SMP 36 km, SPL3SMP_E 9 km; pases AM y PM) o L4 (sm_surface).
                      Varias fechas = serie de tiempo.
    Cuadros DEM de 1° o mosaicos: se procesan todos y se unen. El área de estudio = extensión de los DEM.

SALIDAS (resultados_terreno/<region>/)
    terreno_celdas_<region>.csv    una fila por celda de 0.01° (~1 km), con las mismas lat_centro/lon_centro que
                                   minerologia2.py (se pueden unir): elevación, pendiente, rugosidad, agua, humedad
    terreno_resumen_<region>.csv   toda la región en una columna
    humedad_smap_<region>.csv      cada píxel SMAP dentro de la región, por fecha y pase
    mapa_terreno_<region>.png      elevación, pendiente, rugosidad y humedad
    histogramas_<region>.npz       histogramas por PÍXEL de pendiente y rugosidad (toda la región y planicie
                                   con pendiente < 15°), para comparar con Marte en la misma base
    Con varias regiones, además: resultados_terreno/resumen_terreno_regiones.csv (una columna por región)

DEFINICIONES
    elevación       metros sobre el nivel del mar (DEM), sin píxeles de agua
    pendiente       grados, desde el gradiente del DEM con el tamaño real del píxel en metros
    rugosidad       desviación estándar de la elevación en una ventana de 3×3 píxeles (~90 m con SRTM), metros
    relieve         elevación máxima − mínima dentro de la celda, metros
    humedad         humedad volumétrica del suelo SMAP (m³/m³; 0.10 = 10 % del volumen es agua), capa de ~5 cm.
                    Por defecto solo píxeles con calidad recomendada (bit 0 de retrieval_qual_flag = 0).
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

# =====================================================
# CONFIGURACIÓN
# =====================================================
CELDA_DEG = 0.01            # tamaño de celda (~1 km), igual que minerologia2.py
VENTANA_RUGOSIDAD = 3       # ventana de rugosidad en píxeles del DEM (3×3 ≈ 90 m con SRTM de 30 m)
SWB_AGUA = 255              # valor de agua en el .swb de SRTM
VACIO_DEM = -32768          # nulo de SRTM si el archivo no declara otro
SMAP_VACIO = -9999.0
FILAS_POR_BLOQUE = 600      # filas del DEM que se agregan a la vez (controla la memoria)
M_POR_GRADO = 111_320.0     # metros por grado (latitud; longitud × cos(lat))
PATRONES_DEM = ("*.hgt", "*.tif", "*.tiff")

# Histogramas para percentiles de toda la región sin guardar millones de valores
BORDES_ELEV = np.arange(-500, 9001, 1.0)
BORDES_PEND = np.round(np.arange(0, 90.001, 0.01), 2)      # 0.01° (también se guardan en histogramas_<region>.npz)
BORDES_RUG = np.round(np.arange(0, 500.001, 0.01), 2)      # 0.01 m
PENDIENTE_PLANICIE = 15.0   # "planicie" = píxeles con pendiente < este valor (para comparar con Marte)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)


# =====================================================
# UTILIDADES
# =====================================================
def archivos_dem(carpeta):
    return sorted({p for patron in PATRONES_DEM for p in Path(carpeta).glob(patron)})


def en_cajas(lat, lon, cajas):
    m = np.zeros(np.shape(lat), dtype=bool)
    for la0, la1, lo0, lo1 in cajas:
        m |= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    return m


def percentil_hist(hist, bordes, q):
    if hist.sum() == 0:
        return np.nan
    acum = np.cumsum(hist) / hist.sum()
    i = min(int(np.searchsorted(acum, q / 100)), len(bordes) - 2)
    return float((bordes[i] + bordes[i + 1]) / 2)          # centro del intervalo del histograma


def media_hist(hist, bordes):
    return float(np.sum(hist * (bordes[:-1] + bordes[1:]) / 2) / hist.sum()) if hist.sum() else np.nan


def max_hist(hist, bordes):
    nz = np.nonzero(hist)[0]
    return float(bordes[nz[-1] + 1]) if nz.size else np.nan


def rugosidad(dem, ventana):
    """Desviación estándar local de la elevación (ventana × ventana), ignorando NaN."""
    valido = np.isfinite(dem)
    d = np.where(valido, dem - np.nanmean(dem), 0.0)
    w = uniform_filter(valido.astype("float64"), ventana)
    w[w == 0] = np.nan
    m1 = uniform_filter(d, ventana) / w
    m2 = uniform_filter(d ** 2, ventana) / w
    r = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    r[~valido] = np.nan
    return r


class Georef:
    """Coordenadas geográficas de los píxeles de un DEM (en grados o proyectado en metros)."""
    def __init__(self, src):
        self.tr, self.crs = src.transform, src.crs
        self.geografico = self.crs is None or self.crs.is_geographic
        self.ancho, self.alto = src.width, src.height
        if not self.geografico:
            from pyproj import CRS, Transformer
            self.a_ll = Transformer.from_crs(CRS.from_wkt(self.crs.to_wkt()), "EPSG:4326", always_xy=True)

    def lonlat(self, filas, cols):
        x = self.tr.c + self.tr.a * (cols + 0.5)
        y = self.tr.f + self.tr.e * (filas + 0.5)
        if self.geografico:
            return x, y
        return self.a_ll.transform(x, y)

    def caja(self):
        f = np.array([0, 0, self.alto - 1, self.alto - 1, self.alto // 2])
        c = np.array([0, self.ancho - 1, 0, self.ancho - 1, self.ancho // 2])
        lon, lat = self.lonlat(f, c)
        return float(np.min(lat)), float(np.max(lat)), float(np.min(lon)), float(np.max(lon))

    def tam_pixel_m(self, lat_filas):
        """(dx por fila, dy) en metros."""
        if self.geografico:
            return (abs(self.tr.a) * M_POR_GRADO * np.cos(np.radians(lat_filas)))[:, None], \
                   abs(self.tr.e) * M_POR_GRADO
        return np.full((len(lat_filas), 1), abs(self.tr.a)), abs(self.tr.e)


# =====================================================
# 1) TERRENO
# =====================================================
def procesar_dem(ruta, H):
    """Agrega un DEM por celda. Devuelve (DataFrame por celda, caja geográfica). Acumula histogramas en H."""
    with rasterio.open(ruta) as src:
        dem = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else VACIO_DEM
        g = Georef(src)
    dem[(dem == nodata) | ~np.isfinite(dem) | (dem < -1000)] = np.nan

    agua = np.zeros(dem.shape, dtype=bool)
    swb = ruta.with_suffix(".swb")
    if ruta.suffix.lower() == ".hgt" and swb.exists():
        a = np.fromfile(swb, dtype=np.uint8)
        if a.size == dem.size:
            agua = a.reshape(dem.shape) == SWB_AGUA
        else:
            print(f"  aviso: {swb.name} no tiene el tamaño del DEM; se ignora")
    dem_tierra = np.where(agua, np.nan, dem)

    filas = np.arange(dem.shape[0])
    _, lat_filas = g.lonlat(filas, np.full(filas.shape, dem.shape[1] // 2))
    dx, dy = g.tam_pixel_m(np.asarray(lat_filas))
    pend = np.degrees(np.arctan(np.hypot(np.gradient(dem_tierra, axis=1) / dx,
                                         np.gradient(dem_tierra, axis=0) / dy)))
    rug = rugosidad(dem_tierra, VENTANA_RUGOSIDAD)

    for arr, clave, bordes in ((dem_tierra, "elev", BORDES_ELEV), (pend, "pend", BORDES_PEND),
                               (rug, "rug", BORDES_RUG)):
        v = arr[np.isfinite(arr)]
        H[clave] += np.histogram(np.clip(v, bordes[0], bordes[-1]), bordes)[0]
    plan = np.isfinite(pend) & (pend < PENDIENTE_PLANICIE)
    H["pend_plan"] += np.histogram(pend[plan], BORDES_PEND)[0]
    rp = rug[plan & np.isfinite(rug)]
    H["rug_plan"] += np.histogram(np.clip(rp, BORDES_RUG[0], BORDES_RUG[-1]), BORDES_RUG)[0]
    H["agua"] += int(agua.sum())
    H["px"] += int(np.isfinite(dem).sum())

    partes = []
    cols = np.arange(dem.shape[1])
    for f0 in range(0, dem.shape[0], FILAS_POR_BLOQUE):
        ff, cc = np.meshgrid(filas[f0:f0 + FILAS_POR_BLOQUE], cols, indexing="ij")
        lon, lat = g.lonlat(ff.ravel(), cc.ravel())
        sl = slice(f0, f0 + FILAS_POR_BLOQUE)
        e, p, r = dem_tierra[sl].ravel(), pend[sl].ravel(), rug[sl].ravel()
        d = pd.DataFrame({
            "ci": np.floor(np.asarray(lat) / CELDA_DEG).astype(np.int64),
            "cj": np.floor(np.asarray(lon) / CELDA_DEG).astype(np.int64),
            "n_px": np.isfinite(dem[sl]).ravel().astype(np.int32),
            "n_agua": agua[sl].ravel().astype(np.int32),
            "e_n": np.isfinite(e).astype(np.int32), "e_s": np.nan_to_num(e), "e_ss": np.nan_to_num(e) ** 2,
            "e_min": e, "e_max": e,
            "p_n": np.isfinite(p).astype(np.int32), "p_s": np.nan_to_num(p), "p_max": p,
            "r_n": np.isfinite(r).astype(np.int32), "r_s": np.nan_to_num(r),
        })
        partes.append(d.groupby(["ci", "cj"]).agg(AGREGA))
    return pd.concat(partes), g.caja()


AGREGA = {"n_px": "sum", "n_agua": "sum", "e_n": "sum", "e_s": "sum", "e_ss": "sum", "e_min": "min",
          "e_max": "max", "p_n": "sum", "p_s": "sum", "p_max": "max", "r_n": "sum", "r_s": "sum"}


# =====================================================
# 2) HUMEDAD DEL SUELO (SMAP)
# =====================================================
def leer_smap(ruta):
    """Lista de (pase, dict de arreglos) de un archivo SMAP L3 (AM/PM) o L4."""
    capas = []
    with h5py.File(ruta, "r") as f:
        for grupo, suf, pase in (("Soil_Moisture_Retrieval_Data_AM", "", "AM"),
                                 ("Soil_Moisture_Retrieval_Data_PM", "_pm", "PM")):
            if grupo not in f or f"soil_moisture{suf}" not in f[grupo]:
                continue
            g = f[grupo]
            leer = lambda n: g[f"{n}{suf}"][()].astype("float64") if f"{n}{suf}" in g else None
            q = g[f"retrieval_qual_flag{suf}"][()].astype("int64") if f"retrieval_qual_flag{suf}" in g else None
            capas.append((pase, {"sm": leer("soil_moisture"), "lat": leer("latitude"), "lon": leer("longitude"),
                                 "error": leer("soil_moisture_error"),
                                 "agua_vegetacion_kgm2": leer("vegetation_water_content"),
                                 "temp_superficie_K": leer("surface_temperature"),
                                 "calidad_ok": None if q is None else ((q & 1) == 0) & (q != 65534),
                                 "fila_ease": leer("EASE_row_index"), "col_ease": leer("EASE_column_index")}))
        if not capas and "Geophysical_Data" in f and "cell_lat" in f:
            capas.append(("L4", {"sm": f["Geophysical_Data/sm_surface"][()].astype("float64"),
                                 "lat": f["cell_lat"][()].astype("float64"),
                                 "lon": f["cell_lon"][()].astype("float64")}))
    return capas


def procesar_smap(archivos, cajas, filtro_calidad):
    filas = []
    for ruta in archivos:
        m = re.search(r"_(\d{4})(\d{2})(\d{2})", ruta.name)
        fecha = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ruta.stem
        capas = leer_smap(ruta)
        if not capas:
            print(f"  aviso: no reconocí la estructura de {ruta.name}; se omite")
            continue
        for pase, a in capas:
            sm, lat, lon = a["sm"], a["lat"], a["lon"]
            dentro = np.isfinite(sm) & (sm != SMAP_VACIO) & (sm >= 0) & en_cajas(lat, lon, cajas)
            if not dentro.any():
                continue
            d = {"fecha": fecha, "pase": pase, "archivo": ruta.name,
                 "lat": lat[dentro], "lon": lon[dentro], "humedad_m3m3": sm[dentro]}
            for k in ("error", "agua_vegetacion_kgm2", "temp_superficie_K", "fila_ease", "col_ease"):
                if a.get(k) is not None:
                    v = a[k][dentro]
                    d[k] = np.where(v == SMAP_VACIO, np.nan, v)
            d["calidad_ok"] = a["calidad_ok"][dentro] if a.get("calidad_ok") is not None else True
            filas.append(pd.DataFrame(d))
    if not filas:
        return pd.DataFrame()
    px = pd.concat(filas, ignore_index=True)
    px["usado"] = px["calidad_ok"] if filtro_calidad else True
    return px


def humedad_por_celda(celdas, px):
    """Asigna a cada celda de ~1 km el píxel SMAP más cercano (media de todas las fechas/pases usados)."""
    u = px[px["usado"]]
    if u.empty:
        return pd.DataFrame({"humedad_media_m3m3": np.nan, "humedad_n_obs": 0, "humedad_dist_px_km": np.nan},
                            index=celdas.index), np.nan
    centros = u.groupby([u["lat"].round(4), u["lon"].round(4)])["humedad_m3m3"].agg(["mean", "count"]).reset_index()
    lat0 = np.radians(centros["lat"].mean())
    xy = lambda la, lo: np.c_[np.asarray(lo) * np.cos(lat0) * M_POR_GRADO / 1000, np.asarray(la) * M_POR_GRADO / 1000]
    arbol = cKDTree(xy(centros["lat"], centros["lon"]))
    if len(centros) > 1:
        espaciado = float(np.median(arbol.query(xy(centros["lat"], centros["lon"]), k=2)[0][:, 1]))
    else:
        espaciado = 36.0 if not any("_E_" in a for a in u["archivo"].unique()) else 9.0
    dist, i = arbol.query(xy(celdas["lat_centro"], celdas["lon_centro"]))
    ok = dist <= 0.75 * espaciado
    return pd.DataFrame({"humedad_media_m3m3": np.where(ok, centros["mean"].to_numpy()[i], np.nan),
                         "humedad_n_obs": np.where(ok, centros["count"].to_numpy()[i], 0),
                         "humedad_dist_px_km": np.where(ok, dist, np.nan)}, index=celdas.index), espaciado


# =====================================================
# 3) UNA REGIÓN
# =====================================================
def procesar_region(nombre, carpeta, salida, filtro_calidad):
    carpeta = Path(carpeta)
    out = Path(salida) / nombre
    out.mkdir(parents=True, exist_ok=True)
    dems = archivos_dem(carpeta)
    smaps = sorted(carpeta.glob("SMAP*.h5"))
    if not dems:
        raise SystemExit(f"No hay DEM (*.hgt, *.tif) en {carpeta.resolve()}")
    print(f"\n=== {nombre}: {len(dems)} DEM, {len(smaps)} SMAP ===")

    H = {"elev": np.zeros(len(BORDES_ELEV) - 1), "pend": np.zeros(len(BORDES_PEND) - 1),
         "rug": np.zeros(len(BORDES_RUG) - 1), "pend_plan": np.zeros(len(BORDES_PEND) - 1),
         "rug_plan": np.zeros(len(BORDES_RUG) - 1), "agua": 0, "px": 0}
    partes, cajas = [], []
    for ruta in dems:
        t, caja = procesar_dem(ruta, H)
        partes.append(t); cajas.append(caja)
        print(f"  {ruta.name}: lat {caja[0]:.2f}…{caja[1]:.2f}, lon {caja[2]:.2f}…{caja[3]:.2f}")
    a = pd.concat(partes).groupby(level=[0, 1]).agg(AGREGA)

    c = pd.DataFrame(index=a.index)
    c["lat_centro"] = (a.index.get_level_values(0) + 0.5) * CELDA_DEG
    c["lon_centro"] = (a.index.get_level_values(1) + 0.5) * CELDA_DEG
    en = a["e_n"].replace(0, np.nan)
    c["elev_media_m"] = a["e_s"] / en
    c["elev_min_m"], c["elev_max_m"] = a["e_min"], a["e_max"]
    c["elev_desv_m"] = np.sqrt(np.maximum(a["e_ss"] / en - c["elev_media_m"] ** 2, 0))
    c["relieve_m"] = a["e_max"] - a["e_min"]
    c["pendiente_media_deg"] = a["p_s"] / a["p_n"].replace(0, np.nan)
    c["pendiente_max_deg"] = a["p_max"]
    c["rugosidad_media_m"] = a["r_s"] / a["r_n"].replace(0, np.nan)
    c["pct_agua"] = 100 * a["n_agua"] / a["n_px"].replace(0, np.nan)
    c["n_px_dem"] = a["n_px"]
    c = c[c["n_px_dem"] > 0].reset_index(drop=True)

    px = procesar_smap(smaps, cajas, filtro_calidad) if smaps else pd.DataFrame()
    espaciado = np.nan
    if not px.empty:
        h, espaciado = humedad_por_celda(c, px)
        c = c.join(h)
        px.round(5).to_csv(out / f"humedad_smap_{nombre}.csv", index=False)
    else:
        c["humedad_media_m3m3"], c["humedad_n_obs"] = np.nan, 0
    c.round(4).to_csv(out / f"terreno_celdas_{nombre}.csv", index=False)

    # ---- resumen de la región ----
    tot_e = H["elev"].sum()
    centros_e = (BORDES_ELEV[:-1] + BORDES_ELEV[1:]) / 2
    r = {"lat_min": min(k[0] for k in cajas), "lat_max": max(k[1] for k in cajas),
         "lon_min": min(k[2] for k in cajas), "lon_max": max(k[3] for k in cajas),
         "archivos_dem": len(dems), "celdas_1km": len(c),
         "elev_media_m": float(np.nansum(c["elev_media_m"] * c["n_px_dem"] * (1 - c["pct_agua"].fillna(0) / 100))
                               / np.nansum(c["n_px_dem"] * (1 - c["pct_agua"].fillna(0) / 100))),
         "elev_min_m": float(c["elev_min_m"].min()), "elev_max_m": float(c["elev_max_m"].max()),
         "elev_p5_m": percentil_hist(H["elev"], BORDES_ELEV, 5),
         "elev_mediana_m": percentil_hist(H["elev"], BORDES_ELEV, 50),
         "elev_p95_m": percentil_hist(H["elev"], BORDES_ELEV, 95),
         "relieve_desv_est_m": float(np.sqrt(np.sum(H["elev"] * centros_e ** 2) / tot_e
                                             - (np.sum(H["elev"] * centros_e) / tot_e) ** 2)) if tot_e else np.nan,
         "pendiente_media_deg": media_hist(H["pend"], BORDES_PEND),
         "pendiente_mediana_deg": percentil_hist(H["pend"], BORDES_PEND, 50),
         "pendiente_p95_deg": percentil_hist(H["pend"], BORDES_PEND, 95),
         "pendiente_max_deg": max_hist(H["pend"], BORDES_PEND),
         "pct_terreno_pendiente_lt_5deg": 100 * H["pend"][BORDES_PEND[:-1] < 5].sum() / H["pend"].sum(),
         "rugosidad_media_m": media_hist(H["rug"], BORDES_RUG),
         "rugosidad_mediana_m": percentil_hist(H["rug"], BORDES_RUG, 50),
         "rugosidad_p95_m": percentil_hist(H["rug"], BORDES_RUG, 95),
         "rugosidad_max_m": max_hist(H["rug"], BORDES_RUG),
         "pct_terreno_planicie": 100 * H["pend_plan"].sum() / H["pend"].sum(),
         "planicie_pendiente_mediana_deg": percentil_hist(H["pend_plan"], BORDES_PEND, 50),
         "planicie_pendiente_p95_deg": percentil_hist(H["pend_plan"], BORDES_PEND, 95),
         "planicie_pct_terreno_pendiente_lt_5deg": 100 * H["pend_plan"][BORDES_PEND[:-1] < 5].sum()
                                                   / H["pend_plan"].sum(),
         "planicie_rugosidad_mediana_m": percentil_hist(H["rug_plan"], BORDES_RUG, 50),
         "planicie_rugosidad_p95_m": percentil_hist(H["rug_plan"], BORDES_RUG, 95),
         "pct_agua": 100 * H["agua"] / H["px"] if H["px"] else np.nan,
         "ventana_rugosidad_px": VENTANA_RUGOSIDAD,
         "smap_archivos": len(smaps),
         "smap_filtro_calidad": "solo calidad recomendada" if filtro_calidad else "todas las calidades"}
    if not px.empty:
        u = px[px["usado"]]
        r["smap_espaciado_km"] = round(espaciado, 1)
        r["humedad_media_m3m3"] = float(u["humedad_m3m3"].mean()) if len(u) else np.nan
        r["humedad_mediana_m3m3"] = float(u["humedad_m3m3"].median()) if len(u) else np.nan
        r["humedad_min_m3m3"] = float(u["humedad_m3m3"].min()) if len(u) else np.nan
        r["humedad_max_m3m3"] = float(u["humedad_m3m3"].max()) if len(u) else np.nan
        r["humedad_obs_usadas"] = int(len(u))
        r["humedad_obs_descartadas_calidad"] = int((~px["usado"]).sum())
        r["pct_celdas_con_humedad"] = 100 * (c["humedad_n_obs"] > 0).mean()
        for (fecha, pase), g in px.groupby(["fecha", "pase"]):
            gu = g[g["usado"]]
            r[f"humedad_{fecha}_{pase}_media_m3m3"] = float(gu["humedad_m3m3"].mean()) if len(gu) else np.nan
            r[f"humedad_{fecha}_{pase}_pixeles_usados"] = int(len(gu))
            r[f"humedad_{fecha}_{pase}_pixeles_total"] = int(len(g))
            if "error" in g and gu["error"].notna().any():
                r[f"humedad_{fecha}_{pase}_error_medio_m3m3"] = float(gu["error"].mean())
    else:
        r["humedad_media_m3m3"] = np.nan
        print("  sin datos SMAP dentro de la región")
    resumen = pd.Series(r, name=nombre)
    resumen.to_frame("region_completa").to_csv(out / f"terreno_resumen_{nombre}.csv")
    np.savez_compressed(out / f"histogramas_{nombre}.npz", bordes_pend=BORDES_PEND, bordes_rug=BORDES_RUG,
                        hist_pend=H["pend"], hist_rug=H["rug"], hist_pend_planicie=H["pend_plan"],
                        hist_rug_planicie=H["rug_plan"], pendiente_planicie_deg=PENDIENTE_PLANICIE)
    figura(c, nombre, out)

    print(resumen.to_string())
    print(f"Archivos en {out.resolve()}")
    return resumen


def figura(c, nombre, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator

    def malla(col):
        t = c.pivot_table(index="lat_centro", columns="lon_centro", values=col)
        return t.columns.values, t.index.values, t.values

    paneles = [("elev_media_m", "cividis", "Elevación media (m)", None),
               ("pendiente_media_deg", "magma", "Pendiente media (°)", None),
               ("rugosidad_media_m", "viridis", "Rugosidad media (m, escala log)", "log"),
               ("humedad_media_m3m3", "Blues", "Humedad del suelo SMAP (m³/m³)", None)]
    fig, axs = plt.subplots(2, 2, figsize=(13, 11), facecolor="#fcfcfb")
    for ax, (col, cmap, et, escala) in zip(axs.ravel(), paneles):
        ax.set_facecolor("#fcfcfb")
        if c[col].notna().any():
            x, y, z = malla(col)
            norm = LogNorm(vmin=max(np.nanpercentile(z, 1), 0.01), vmax=np.nanpercentile(z, 99.5)) \
                if escala == "log" else None
            m = ax.pcolormesh(x, y, z, cmap=cmap, shading="nearest", norm=norm)
            fig.colorbar(m, ax=ax, label=et)
        else:
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes, color="#898781")
        ax.set_title(et, loc="left", fontsize=11)
        ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
        ax.set_xlim(c["lon_centro"].min() - CELDA_DEG, c["lon_centro"].max() + CELDA_DEG)
        ax.set_ylim(c["lat_centro"].min() - CELDA_DEG, c["lat_centro"].max() + CELDA_DEG)
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.set_aspect(1 / np.cos(np.radians(c["lat_centro"].mean())))
    fig.suptitle(f"{nombre.replace('_', ' ').title()}: terreno y humedad del suelo (celdas de ~1 km)",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / f"mapa_terreno_{nombre}.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# =====================================================
# PRINCIPAL
# =====================================================
def main():
    ap = argparse.ArgumentParser(description="Elevación, pendiente, rugosidad y humedad del suelo por región")
    ap.add_argument("--carpeta", default="./resources",
                    help="carpeta de UNA región (DEM sueltos) o carpeta con una subcarpeta por región")
    ap.add_argument("--region", default=None, help="nombre de la región si la carpeta es de una sola región")
    ap.add_argument("--salida", default="resultados_terreno")
    ap.add_argument("--sin-filtro-calidad", action="store_true",
                    help="usar también los píxeles SMAP sin calidad recomendada")
    a = ap.parse_args()
    carpeta = Path(a.carpeta)

    if archivos_dem(carpeta):
        regiones = {a.region or carpeta.resolve().name: carpeta}
    else:
        regiones = {p.name: p for p in sorted(carpeta.iterdir()) if p.is_dir() and archivos_dem(p)}
        if not regiones:
            raise SystemExit(f"No encontré DEM (*.hgt, *.tif) en {carpeta.resolve()} ni en sus subcarpetas")
    resumenes = [procesar_region(n, p, a.salida, not a.sin_filtro_calidad) for n, p in regiones.items()]
    if len(resumenes) > 1:
        tabla = pd.concat(resumenes, axis=1)
        tabla.to_csv(Path(a.salida) / "resumen_terreno_regiones.csv")
        print(f"\nResumen de {len(resumenes)} regiones: {(Path(a.salida) / 'resumen_terreno_regiones.csv').resolve()}")


if __name__ == "__main__":
    main()
