import re
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import h5py
from pathlib import Path
from scipy.ndimage import uniform_filter

# =====================================================
# CONFIGURACIÓN
# =====================================================
CARPETA = Path("./resources")
MIN_BAND_DEPTH = 0.0      # sube a 0.02-0.05 para quedarte solo con detecciones fuertes
VENTANA = 3               # ventana de rugosidad (3x3 = ~90 m)
LWMAP_AGUA = 1            # 1 = agua en AG100
SWB_AGUA = 255            # 255 = agua en el .swb
EXCLUIR = {"Cyanide", "Plastic", "Wtc", "Coated"}
COL = "region_completa"   # nombre de la única columna de resultados

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

def limites_zona(nombre):
    lat0 = int(nombre[1:3]) * (1 if nombre[0] == "n" else -1)
    lon0 = int(nombre[4:7]) * (1 if nombre[3] == "e" else -1)
    return lat0, lat0 + 1, lon0, lon0 + 1

# Los mosaicos .hgt definen el área de estudio (cuadros de 1° x 1°).
# Se usan solo para delimitar qué píxeles de EMIT, AG100 y SMAP cuentan.
zonas = [limites_zona(p.stem.lower()) + (p.stem.lower(),)
         for p in sorted(CARPETA.glob("*.hgt"))]
if not zonas:
    print("No encontré archivos .hgt en:", CARPETA.resolve())
    raise SystemExit

def a_tabla(dic):
    """Convierte {variable: valor} en tabla de una columna."""
    return pd.DataFrame({COL: dic})

# =====================================================
# PARTE 1: TERRENO (.hgt + .swb), todos los mosaicos juntos
# =====================================================
# elev_media_m          Elevación media de toda la región, sin agua (metros).
# elev_min_m / max_m    Punto más bajo / más alto.
# relieve_desv_est_m    Desviación estándar de la elevación: qué tan variado es el relieve.
# pendiente_media_deg   Inclinación media del terreno (grados).
# pendiente_p95_deg     El 95% del terreno tiene pendiente menor a este valor.
# rugosidad_media_m     Variación local de elevación en ventana 3x3 (~90 m), en metros.
# pct_agua_swb          % de la región que es agua (excluida de todo lo anterior).

def rugosidad(dem, ventana):
    valido = np.isfinite(dem)
    d = np.where(valido, dem - np.nanmean(dem), 0.0)
    w = uniform_filter(valido.astype("float64"), ventana)
    w[w == 0] = np.nan
    m1 = uniform_filter(d, ventana) / w
    m2 = uniform_filter(d**2, ventana) / w
    r = np.sqrt(np.maximum(m2 - m1**2, 0))
    r[~valido] = np.nan
    return r

T = dict(n=0, s=0.0, ss=0.0, mn=np.inf, mx=-np.inf,
         ps=0.0, pn=0, rs=0.0, rn=0, agua=0, px=0)
muestras_pend = []

for hgt in sorted(CARPETA.glob("*.hgt")):
    with rasterio.open(hgt) as src:
        dem = src.read(1).astype("float64")
        top, bottom = src.bounds.top, src.bounds.bottom
    dem[dem == -32768] = np.nan

    ruta_swb = hgt.with_suffix(".swb")
    if ruta_swb.exists():
        swb = np.fromfile(ruta_swb, dtype=np.uint8).reshape(dem.shape)
        agua = swb == SWB_AGUA
        T["agua"] += int(agua.sum())
        dem[agua] = np.nan
    T["px"] += dem.size

    lat = np.linspace(top, bottom, dem.shape[0])
    dy = 30.87
    dx = 30.87 * np.cos(np.radians(lat))[:, None]
    pend = np.degrees(np.arctan(np.hypot(np.gradient(dem, axis=1) / dx,
                                         np.gradient(dem, dy, axis=0))))
    rug = rugosidad(dem, VENTANA)

    v = dem[np.isfinite(dem)]
    T["n"] += v.size; T["s"] += v.sum(); T["ss"] += (v ** 2).sum()
    T["mn"] = min(T["mn"], v.min()); T["mx"] = max(T["mx"], v.max())
    p = pend[np.isfinite(pend)]
    T["ps"] += p.sum(); T["pn"] += p.size
    muestras_pend.append(p[::50])          # muestra para el percentil 95
    r = rug[np.isfinite(rug)]
    T["rs"] += r.sum(); T["rn"] += r.size

media_elev = T["s"] / T["n"]
terreno = a_tabla({
    "elev_media_m": media_elev,
    "elev_min_m": T["mn"],
    "elev_max_m": T["mx"],
    "relieve_desv_est_m": np.sqrt(T["ss"] / T["n"] - media_elev ** 2),
    "pendiente_media_deg": T["ps"] / T["pn"],
    "pendiente_p95_deg": np.percentile(np.concatenate(muestras_pend), 95),
    "rugosidad_media_m": T["rs"] / T["rn"],
    "pct_agua_swb": 100 * T["agua"] / T["px"],
}).round(2)

# =====================================================
# PARTE 2: MINERALES (EMIT .nc), región completa
# =====================================================
# pct_<mineral>   % de los píxeles EMIT válidos de la región donde ese mineral fue
#                 la mejor coincidencia. Valores pequeños son normales en desierto.
#                 No es abundancia real, es detección por biblioteca espectral.
# pct_Otros       Todo lo que no entró en el top 15.
# emit_pixeles_validos        Píxeles EMIT usados en total.
# emit_pct_con_deteccion      % de esos píxeles con algún mineral detectado.
conteos = {}
validos_total = 0

for nc in sorted(CARPETA.glob("EMIT_L2B_MIN*.nc")):
    ds = xr.open_dataset(nc)
    loc = xr.open_dataset(nc, group="location")
    meta = xr.open_dataset(nc, group="mineral_metadata")

    gt = ds.attrs["geotransform"]
    glt_x = np.nan_to_num(loc["glt_x"].values, nan=0).astype(int)
    glt_y = np.nan_to_num(loc["glt_y"].values, nan=0).astype(int)
    ok = (glt_x > 0) & (glt_y > 0)

    ids = np.zeros(glt_x.shape, dtype=float)
    depth = np.zeros(glt_x.shape, dtype=float)
    ids[ok] = ds["group_1_mineral_id"].values[glt_y[ok] - 1, glt_x[ok] - 1]
    depth[ok] = ds["group_1_band_depth"].values[glt_y[ok] - 1, glt_x[ok] - 1]
    ids[~np.isfinite(ids) | (ids < 0)] = 0
    ids = ids.astype(int)

    filas_i, cols_i = np.indices(glt_x.shape)
    lon = gt[0] + gt[1] * (cols_i + 0.5)
    lat = gt[3] + gt[5] * (filas_i + 0.5)

    nombres_dict = dict(zip(meta["index"].values.astype(int),
                            meta["name"].values.astype(str)))

    # Píxeles dentro de CUALQUIERA de las zonas de estudio
    en_region = np.zeros(glt_x.shape, dtype=bool)
    for lat0, lat1, lon0, lon1, z in zonas:
        en_region |= (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
    dentro = ok & en_region
    if not dentro.any():
        continue
    validos_total += int(dentro.sum())
    det = dentro & (ids > 0) & (depth >= MIN_BAND_DEPTH)
    for v, c in zip(*np.unique(ids[det], return_counts=True)):
        nom = nombres_dict.get(int(v), f"id_{v}")
        conteos[nom] = conteos.get(nom, 0) + int(c)

def mineral_base(nombre):
    m = re.match(r"[A-Za-z_]+", nombre.strip())
    base = m.group(0).strip("_").capitalize() if m else nombre
    alias = {"Montmorill": "Montmorillonite", "Kaolin": "Kaolinite"}
    return alias.get(base, base)

agrupado = {}
for nombre, c in conteos.items():
    b = mineral_base(nombre)
    agrupado[b] = agrupado.get(b, 0) + c

cuentas = pd.Series(agrupado, dtype=float)
cuentas = cuentas.drop([m for m in cuentas.index if m in EXCLUIR])
top15 = cuentas.sort_values(ascending=False).head(15)
otros = cuentas.drop(top15.index).sum()
total_detectado = cuentas.sum()

pct_min = (100 * top15 / validos_total).round(4)
pct_min["Otros"] = round(100 * otros / validos_total, 4)
pct_min.index = "pct_" + pct_min.index.astype(str)
pct_min.to_csv("top15_minerales_region.csv")

# % relativo entre minerales detectados (cuál domina)
pct_rel = (100 * top15 / total_detectado).round(2)
pct_rel.to_csv("top15_minerales_relativo_region.csv")

minerales = a_tabla({**pct_min.to_dict(),
                     "emit_pixeles_validos": validos_total,
                     "emit_pct_con_deteccion": round(100 * total_detectado / validos_total, 2)})

# =====================================================
# PARTE 3: EMISIVIDAD INFRARROJA (AG100 .h5), sin agua
# =====================================================
# emis_bXX_<µm>       Emisividad media de la región en cada banda térmica (0 a 1).
#                     Suelo desértico: ~0.92-0.98. Un valor más bajo en 8.6 µm
#                     indica más sílice/cuarzo.
# indice_cuarzo       b11^2/(b10*b12). Más bajo = más señal de cuarzo (confirma la
#                     fórmula en la guía de ASTER GED antes de citarla).
# emis_n_pixeles_tierra   Píxeles de tierra usados.
# pct_agua_lwmap          % de la región que es agua (excluida).
ESCALA_EMIS = 0.001
VALOR_VACIO = -9999
BANDAS = ["emis_b10_8.3um", "emis_b11_8.6um", "emis_b12_9.1um",
          "emis_b13_10.6um", "emis_b14_11.3um"]

suma = np.zeros(5); cuenta = np.zeros(5)
suma_qi = 0.0; cuenta_qi = 0
agua_px = 0; total_px = 0

for h5 in sorted(CARPETA.glob("AG100*.h5")):
    with h5py.File(h5, "r") as f:
        em = f["Emissivity/Mean"][()].astype("float64")
        lat = f["Geolocation/Latitude"][()].astype("float64")
        lon = f["Geolocation/Longitude"][()].astype("float64")
        lw = f["Land Water Map/LWmap"][()]

    if em.shape[0] != 5 and em.shape[-1] == 5:
        em = np.moveaxis(em, -1, 0)
    em[em == VALOR_VACIO] = np.nan
    em = em * ESCALA_EMIS
    em[(em <= 0) | (em > 1.2)] = np.nan
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    es_agua = lw == LWMAP_AGUA
    qi = em[1] ** 2 / (em[0] * em[2])

    en_region = np.zeros(lat.shape, dtype=bool)
    for lat0, lat1, lon0, lon1, z in zonas:
        en_region |= (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
    total_px += int(en_region.sum())
    agua_px += int((en_region & es_agua).sum())
    dentro = en_region & ~es_agua
    for b in range(5):
        v = em[b][dentro]; v = v[np.isfinite(v)]
        suma[b] += v.sum(); cuenta[b] += v.size
    q = qi[dentro]; q = q[np.isfinite(q)]
    suma_qi += q.sum(); cuenta_qi += q.size

emis_dic = {BANDAS[b]: suma[b] / cuenta[b] if cuenta[b] else np.nan for b in range(5)}
emis_dic["indice_cuarzo"] = suma_qi / cuenta_qi if cuenta_qi else np.nan
emis_dic["emis_n_pixeles_tierra"] = cuenta[0]
emis_dic["pct_agua_lwmap"] = 100 * agua_px / total_px if total_px else np.nan
emis = a_tabla(emis_dic).round(4)

# =====================================================
# PARTE 4: HUMEDAD DEL SUELO (SMAP .h5)
# =====================================================
# humedad_suelo_m3m3_<fecha>_<pase>
#   Humedad volumétrica media de la región (m3/m3). 0.10 = 10% del volumen del
#   suelo es agua. Solo los primeros ~5 cm. Desierto seco: ~0.02-0.10.
# humedad_pixeles_SMAP_<fecha>_<pase>
#   Píxeles SMAP usados en el promedio. Más píxeles = más confiable.
# <pase> = "manana" (AM) o "tarde" (PM).
# Es una foto de un día (clima), no una propiedad fija del suelo: úsala como contexto.
SM_VACIO = -9999
APLICAR_FILTRO_CALIDAD = True
NOMBRE_PASE = {"AM": "manana", "PM": "tarde", "L4": "L4"}

def leer_smap(f):
    salida = []
    for grupo, suf, etiqueta in (("Soil_Moisture_Retrieval_Data_AM", "", "AM"),
                                 ("Soil_Moisture_Retrieval_Data_PM", "_pm", "PM")):
        if grupo not in f:
            continue
        g = f[grupo]
        if "soil_moisture" + suf not in g:
            continue
        sm = g["soil_moisture" + suf][()].astype("float64")
        lat = g["latitude" + suf][()].astype("float64")
        lon = g["longitude" + suf][()].astype("float64")
        sm[sm == SM_VACIO] = np.nan
        lat[lat == SM_VACIO] = np.nan
        lon[lon == SM_VACIO] = np.nan
        if APLICAR_FILTRO_CALIDAD and "retrieval_qual_flag" + suf in g:
            q = g["retrieval_qual_flag" + suf][()].astype("int64")
            sm[(q & 1) != 0] = np.nan
        salida.append((etiqueta, sm, lat, lon))
    if not salida and "Geophysical_Data" in f and "cell_lat" in f:
        sm = f["Geophysical_Data/sm_surface"][()].astype("float64")
        sm[sm < -9000] = np.nan
        salida.append(("L4", sm, f["cell_lat"][()].astype("float64"),
                       f["cell_lon"][()].astype("float64")))
    return salida

hum_dic = {}
for h5 in sorted(CARPETA.glob("SMAP*.h5")):
    m = re.search(r"(\d{4})(\d{2})(\d{2})", h5.name)
    fecha = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else h5.stem
    with h5py.File(h5, "r") as f:
        capas = leer_smap(f)
        if not capas:
            print("No reconocí la estructura de", h5.name)
            f.visititems(lambda n, o: print("  ", n, getattr(o, "shape", "")))
            continue
    for pase, sm, lat, lon in capas:
        en_region = np.zeros(sm.shape, dtype=bool)
        for lat0, lat1, lon0, lon1, z in zonas:
            en_region |= (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
        v = sm[en_region]; v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        sufijo = f"{fecha}_{NOMBRE_PASE.get(pase, pase)}"
        hum_dic[f"humedad_suelo_m3m3_{sufijo}"] = v.mean()
        hum_dic[f"humedad_pixeles_SMAP_{sufijo}"] = v.size

humedad = a_tabla(hum_dic).round(4) if hum_dic else pd.DataFrame()
if not hum_dic:
    print("No encontré datos SMAP utilizables en:", CARPETA.resolve())

# =====================================================
# PARTE 5: TABLA FINAL (una sola columna: región completa)
# =====================================================
final = pd.concat([terreno, minerales, emis, humedad])
print("\n=== TABLA FINAL: REGIÓN COMPLETA ===")
print(final)

try:
    final.to_csv("comparacion_final_Pincate.csv")
    print("\nGuardado: comparacion_final_Pinacate.csv")
except PermissionError:
    from datetime import datetime
    alterno = f"comparacion_final_{datetime.now():%H%M%S}.csv"
    final.to_csv(alterno)
    print(f"\ncomparacion_final.csv estaba abierto. Guardado como: {alterno}")