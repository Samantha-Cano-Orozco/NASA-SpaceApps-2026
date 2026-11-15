"""
minerologia2.py  —  Space Apps 2026 · Análogos de Marte (suelos)
Compara una región de la Tierra contra la huella mineral de Meridiani (Marte).

USO
    python minerologia2.py                                       # carpeta ./resources, región "pinacate"
    python minerologia2.py --carpeta ./resources/pinacate --region pinacate
    python minerologia2.py --carpeta ./resources/otra --region otra --min-elev no

    Una carpeta por región: el área de estudio son TODOS los .hgt de la carpeta.
    --min-elev  auto (por defecto: valor de MIN_ELEV_POR_REGION, o sin filtro si la región no está),
                no (sin filtro) o un número en metros.
    --salida    carpeta base de resultados (por defecto resultados_tierra); se crea <salida>/<region>/.
    Zonas de control: opcionales, por región, en ZONAS_CONTROL_POR_REGION.

ARCHIVOS QUE BUSCA EN LA CARPETA
    EMIT_L2B_MIN_*.nc        (obligatorio)  mineralogía EMIT, grupos 1 y 2
    EMIT_L2B_MINUNCERT_*.nc  (recomendado)  incertidumbre: detección = profundidad > 3 × incertidumbre
    AG100*.h5                (obligatorio)  ASTER GED, emisividad térmica (índice de cuarzo)
    *.hgt (+ *.swb)          (obligatorio)  SRTM/NASADEM: definen el área de estudio (cuadros de 1°)
    SMAP*.h5                 (opcional)     humedad del suelo, solo contexto

SALIDAS (carpeta resultados_tierra/<region>/)
    mapeo_emit_grupos_<region>.csv      nombre EMIT -> grupo de la huella (revisar los "revisar")
    resumen_region_<region>.csv         toda la región en una columna
    celdas_<region>.csv                 una fila por celda de ~1 km, con su puntaje
    top10_<region>.csv                  las 10 celdas más parecidas a Meridiani
    control_lava_dunas_<region>.csv     solo si la región tiene zonas en ZONAS_CONTROL_POR_REGION
    El resumen incluye una fila "avisos" (también se imprimen al final): revísala en cada región nueva.
    mapa_puntaje_<region>.png, mapa_qi_hematita_<region>.png, control_lava_dunas_<region>.png
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

# =====================================================
# CONFIGURACIÓN
# =====================================================
CELDA_DEG = 0.01            # tamaño de celda (~1 km)
FACTOR_RUIDO = 3.0          # detección EMIT = band_depth > FACTOR_RUIDO × incertidumbre
MIN_BAND_DEPTH = 0.02       # SOLO se usa si no hay archivo MINUNCERT (respaldo, queda marcado)
MIN_PIX_EMIT_CELDA = 50     # celdas con menos píxeles EMIT válidos se descartan
MAX_AGUA_CELDA = 10.0       # % máximo de agua por celda (antes 50; 10 saca esteros y franja costera)
# Altura mínima por celda (m), por región. Se cambia al correr con --min-elev.
# Pinacate: 20 m saca salitrales/esteros secos que SRTM marca como tierra (EMIT confunde sus sales con
# sulfatos de Fe y su espectro térmico plano imita al basalto).
# NO usar en regiones bajo el nivel del mar (Valle de la Muerte, Danakil/Dallol, Mar Muerto...).
MIN_ELEV_POR_REGION = {
    "pinacate": 20.0,
}
# Avisos automáticos (no cambian el puntaje; solo alertan para revisar la región)
AVISO_PCT_MAFICO = 50.0         # > este % de celdas con MI ≥ MI_BASALTO: vegetación/humedad imitan basalto
AVISO_PCT_DETECCION_EMIT = 40.0 # EMIT detecta minerales en < este % de píxeles: vegetación, nubes o suelo oscuro
AVISO_HUMEDAD_SMAP = 0.25       # m³/m³: suelo húmedo, la emisividad térmica se aplana
AVISO_PCT_DESCARTE_ELEV = 30.0  # el filtro de altura quita > este % de celdas: ¿región baja?
LWMAP_AGUA = 1              # 1 = agua en AG100
SWB_AGUA = 255              # 255 = agua en .swb
ESCALA_EMIS, VACIO_EMIS = 0.001, -9999

# Niveles (mismas reglas que en Marte)
DOMINANTE, TRAZA = 20.0, 0.5            # %: >20 dominante, 0.5–20 traza, <0.5 ausente

# Huella de Meridiani: nivel objetivo por grupo (2 = dominante, 1 = traza/presente, 0 = ausente)
OBJETIVO = {
    "oxidos_Fe": 2,
    "basalto": 1,          # "presente": cuenta como coincidencia si la celda tiene traza o más
    "carbonatos": 0,
    "arcillas_FeMg": 0,
    "arcillas_Al": 1,      # traza
    "micas_felsicas": 0,   # ausente (micas y feldespatos: no son parte de la huella de Meridiani)
}
PESOS = {                  # ajustables
    "oxidos_Fe": 3,
    "cuarzo": 3,           # se evalúa con el índice de cuarzo de ASTER, no con EMIT
    "basalto": 2,
    "carbonatos": 2,
    "arcillas_FeMg": 2,
    "arcillas_Al": 1,
    "micas_felsicas": 1,   # peso bajo: EMIT detecta poco feldespato y las micas se parecen a las arcillas
}
BONUS_SULFATOS_FE = 0.0    # desactivado: CRISM no ve sulfatos en ninguna de las 4 escenas de Meridiani
                           # (solo Opportunity in situ) y en la Tierra premiaba salitrales costeros.
                           # pct_sulfatos_Fe se sigue reportando como dato informativo.
# Cuarzo con ASTER: QI = b11²/(b10·b12). QI ≈ 1 = espectro sin el doblete del cuarzo (basalto);
# QI > 1 = doblete del cuarzo. Umbral físico (no depende de qué celdas haya en la región):
QI_SIN_CUARZO = 1.010      # QI ≤ este valor -> sin cuarzo (puntaje 1)
QI_CUARZO_PARCIAL = 1.020  # entre los dos -> 0.5 ; por encima -> 0
# Basalto con ASTER (EMIT casi no ve basalto: es oscuro y sin rasgos en el VSWIR).
# Índice máfico MI = b12·b14³/b13⁴ (Ninomiya et al. 2005, mismo trabajo que el QI). MI ≈ 1 = espectro
# máfico (basalto); MI bajo = cuarzo o carbonato. Umbral fijo, revisado con las zonas de control de Pinacate:
# lava mediana 0.986 (81 % ≥ 0.97), dunas mediana 0.844 (0 % ≥ 0.95).
MI_BASALTO = 0.970         # MI ≥ este valor -> basalto (puntaje 1)
MI_BASALTO_PARCIAL = 0.950 # entre los dos -> 0.5 ; por debajo -> 0
# Bonus de sulfatos de Fe (si se reactiva con BONUS_SULFATOS_FE > 0): solo cuenta si la celda es
# basáltica y sin cuarzo, para no premiar salitrales/evaporitas costeras.

# Zonas de control por región (lat_min, lat_max, lon_min, lon_max). Opcionales: si la región no está,
# se omite el control. Convención: primero la zona que DEBERÍA parecerse a Meridiani (basalto),
# después la de contraste (p. ej. arena de cuarzo). AJUSTAR viendo mapa_qi_hematita_<region>.png.
ZONAS_CONTROL_POR_REGION = {
    "pinacate": {
        "lava_pinacate": (31.60, 31.95, -113.62, -113.30),   # núcleo del campo volcánico (sin dunas)
        "dunas_altar":   (31.95, 32.30, -114.60, -114.10),
    },
}

# Mapeo de nombres de la biblioteca EMIT -> grupo de la huella (por palabras clave, en minúsculas).
# El orden importa: se usa la primera regla que coincide.
# micas_felsicas va ANTES de las arcillas, y sus minerales (biotita, flogopita, moscovita, sericita)
# ya no están repetidos en arcillas_FeMg ni arcillas_Al.
REGLAS_GRUPO = [
    ("excluir",        ["cyanide", "plastic", "wtc", "coated", "renyolds", "reynolds", "sludge", "paint"]),
    ("sulfatos_Fe",    ["jarosit", "copiapit", "butlerit", "schwertmann", "coquimbit", "fibroferrit",
                        "rozenit", "melanterit", "szomolnokit", "botryogen"]),
    ("yeso_sulfatos",  ["gypsum", "anhydrit", "bassanit", "alunit", "polyhalit", "epsomit", "kieserit"]),
    ("oxidos_Fe",      ["hematit", "nhematit", "goethit", "ferrihydrit", "lepidocrocit", "maghemit",
                        "magnetit"]),
    ("carbonatos",     ["calcit", "dolomit", "siderit", "magnesit", "ankerit", "aragonit", "rhodochrosit"]),
    ("micas_felsicas", ["muscovit", "sericit", "biotit", "phlogopit", "lepidolit",
                        "orthoclas", "microclin", "albit", "feldspar"]),
    ("arcillas_FeMg",  ["chlorit", "clinochlor", "chamosit", "vermiculit", "nontronit", "saponit",
                        "serpentin", "talc"]),
    ("arcillas_Al",    ["kaolin", "montmor", "illit", "halloysit", "dickit", "pyrophyllit",
                        "beidellit", "smectit"]),
    ("basalto",        ["basalt", "pyroxen", "augit", "olivin"]),
]
GRUPOS = ["oxidos_Fe", "sulfatos_Fe", "yeso_sulfatos", "carbonatos", "arcillas_FeMg",
          "arcillas_Al", "basalto", "micas_felsicas", "otros"]

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# =====================================================
# UTILIDADES
# =====================================================
def grupo_de(nombre):
    n = nombre.strip().lower()
    if n in {"fe", "fe_oxide", "fe-oxide"} or n.startswith("fe_") or n.startswith("fe "):
        return "oxidos_Fe"
    for grupo, claves in REGLAS_GRUPO:
        if any(k in n for k in claves):
            return grupo
    return "otros"


def limites_zona(nombre):
    """n31w114 -> (31, 32, -114, -113)"""
    m = re.match(r"([ns])(\d{2})([ew])(\d{3})", nombre.lower())
    lat0 = int(m.group(2)) * (1 if m.group(1) == "n" else -1)
    lon0 = int(m.group(4)) * (1 if m.group(3) == "e" else -1)
    return lat0, lat0 + 1, lon0, lon0 + 1


def en_zonas(lat, lon, zonas):
    m = np.zeros(np.shape(lat), dtype=bool)
    for lat0, lat1, lon0, lon1 in zonas:
        m |= (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
    return m


def grupos_de_cuadros(zonas):
    """Cuántos grupos separados forman los cuadros de 1° (vecinos incluyen diagonales)."""
    pendientes, n = {(z[0], z[2]) for z in zonas}, 0
    while pendientes:
        n += 1
        pila = [pendientes.pop()]
        while pila:
            la, lo = pila.pop()
            for d_la in (-1, 0, 1):
                for d_lo in (-1, 0, 1):
                    v = (la + d_la, lo + d_lo)
                    if v in pendientes:
                        pendientes.remove(v); pila.append(v)
    return n


def se_solapa(caja, zonas):
    """¿La caja (lat0, lat1, lon0, lon1) toca algún cuadro del área de estudio?"""
    la0, la1, lo0, lo1 = caja
    return any(la0 < z[1] and la1 > z[0] and lo0 < z[3] and lo1 > z[2] for z in zonas)


def clave_celda(lat, lon):
    """Índices enteros de celda (fila, col) para lat/lon."""
    return (np.floor(lat / CELDA_DEG).astype(np.int64),
            np.floor(lon / CELDA_DEG).astype(np.int64))


def acumular(df_list, filas, cols, **valores):
    """Agrega valores por celda (suma) y los apila en df_list."""
    d = pd.DataFrame({"ci": filas, "cj": cols, **valores})
    df_list.append(d.groupby(["ci", "cj"], sort=False).sum())


def nivel(pct):
    return np.where(pct > DOMINANTE, 2, np.where(pct >= TRAZA, 1, 0))


def buscar_var(ds, *partes):
    """Primera variable cuyo nombre contiene todas las partes (minúsculas)."""
    for v in ds.data_vars:
        n = v.lower()
        if all(p in n for p in partes):
            return v
    return None


# =====================================================
# PRINCIPAL
# =====================================================
def main(carpeta, region, min_elev="auto", salida="resultados_tierra"):
    carpeta = Path(carpeta)
    clave = region.lower()
    out = Path(salida) / region
    out.mkdir(parents=True, exist_ok=True)
    avisos = []

    hgts = sorted(carpeta.glob("*.hgt"))
    if not hgts:
        raise SystemExit(f"No encontré archivos .hgt en {carpeta.resolve()}")
    zonas = [limites_zona(p.stem) for p in hgts]
    print(f"Región '{region}': {len(zonas)} cuadros de 1° -> {[p.stem for p in hgts]}")
    n_grupos = grupos_de_cuadros(zonas)
    if n_grupos > 1:
        avisos.append(f"los .hgt forman {n_grupos} áreas separadas: ¿hay varias regiones en la misma "
                      f"carpeta? Usa una carpeta por región")

    # Altura mínima: 'auto' = la de la región (o sin filtro), 'no' = sin filtro, o un número
    if min_elev is None or str(min_elev).lower() in ("no", "none", "off", "sin"):
        min_elev = None
    elif str(min_elev).lower() == "auto":
        min_elev = MIN_ELEV_POR_REGION.get(clave)
    else:
        min_elev = float(min_elev)
    print(f"Altura mínima por celda: {'sin filtro' if min_elev is None else f'{min_elev:g} m'}")

    # Zonas de control: solo las de esta región que caen dentro del área
    zonas_control = {}
    for zona, caja in ZONAS_CONTROL_POR_REGION.get(clave, {}).items():
        if se_solapa(caja, zonas):
            zonas_control[zona] = caja
        else:
            avisos.append(f"la zona de control '{zona}' queda fuera de los .hgt; se omite")
    if not ZONAS_CONTROL_POR_REGION.get(clave):
        print(f"Sin zonas de control para '{region}' (opcional: agrégalas en ZONAS_CONTROL_POR_REGION)")

    # -------------------------------------------------
    # 1) TERRENO por celda
    # -------------------------------------------------
    terr = []
    agua_total = px_total = 0
    for hgt in hgts:
        with rasterio.open(hgt) as src:
            dem = src.read(1).astype("float64")
            tr = src.transform
        dem[dem == -32768] = np.nan
        agua = np.zeros(dem.shape, bool)
        swb = hgt.with_suffix(".swb")
        if swb.exists():
            a = np.fromfile(swb, dtype=np.uint8)
            if a.size == dem.size:
                agua = a.reshape(dem.shape) == SWB_AGUA
            else:
                print(f"  aviso: {swb.name} no tiene el tamaño del .hgt; se ignora")
        filas_i, cols_i = np.indices(dem.shape)
        lon = tr.c + tr.a * (cols_i + 0.5)
        lat = tr.f + tr.e * (filas_i + 0.5)
        dy = 30.87
        dx = 30.87 * np.cos(np.radians(lat))
        demt = np.where(agua, np.nan, dem)
        pend = np.degrees(np.arctan(np.hypot(np.gradient(demt, axis=1) / dx,
                                             np.gradient(demt, dy, axis=0))))
        agua_total += int(agua.sum()); px_total += dem.size
        ci, cj = clave_celda(lat.ravel(), lon.ravel())
        e = demt.ravel(); p = pend.ravel()
        acumular(terr, ci, cj,
                 n_dem=np.ones(e.size), n_agua_dem=agua.ravel().astype(float),
                 elev_s=np.nan_to_num(e), elev_n=np.isfinite(e).astype(float),
                 pend_s=np.nan_to_num(p), pend_n=np.isfinite(p).astype(float))
    terr = pd.concat(terr).groupby(level=[0, 1]).sum()

    # -------------------------------------------------
    # 2) EMIT por celda (grupos 1 y 2, filtro de ruido)
    # -------------------------------------------------
    nombres_vistos = {}           # nombre -> conteo de detecciones (para el mapeo)
    emit = []
    criterio_usado = set()
    for nc in sorted(carpeta.glob("EMIT_L2B_MIN_*.nc")):
        ds = xr.open_dataset(nc)
        loc = xr.open_dataset(nc, group="location")
        meta = xr.open_dataset(nc, group="mineral_metadata")
        nombres = dict(zip(meta["index"].values.astype(int), meta["name"].values.astype(str)))

        unc_path = nc.with_name(nc.name.replace("_MIN_", "_MINUNCERT_"))
        unc = xr.open_dataset(unc_path) if unc_path.exists() else None

        gt = ds.attrs["geotransform"]
        glt_x = np.nan_to_num(loc["glt_x"].values, nan=0).astype(int)
        glt_y = np.nan_to_num(loc["glt_y"].values, nan=0).astype(int)
        ok = (glt_x > 0) & (glt_y > 0)
        fi, co = np.indices(glt_x.shape)
        lon = gt[0] + gt[1] * (co + 0.5)
        lat = gt[3] + gt[5] * (fi + 0.5)
        dentro = ok & en_zonas(lat, lon, zonas)
        if not dentro.any():
            print(f"  {nc.name}: fuera del área, se omite")
            continue
        yy, xx = glt_y[dentro] - 1, glt_x[dentro] - 1
        ci, cj = clave_celda(lat[dentro], lon[dentro])
        conteos = {g: np.zeros(yy.size) for g in GRUPOS}
        conteos["excluir"] = np.zeros(yy.size)
        alguna = np.zeros(yy.size, dtype=bool)        # píxel con al menos una detección EMIT

        for g in (1, 2):
            ids = ds[f"group_{g}_mineral_id"].values[yy, xx].astype("float64")
            dep = ds[f"group_{g}_band_depth"].values[yy, xx].astype("float64")
            ids = np.where(np.isfinite(ids) & (ids > 0), ids, 0).astype(int)
            if unc is not None:
                vu = buscar_var(unc, f"group_{g}", "depth", "unc") or buscar_var(unc, f"group_{g}", "unc")
                if vu is None:
                    raise SystemExit(f"No encontré la incertidumbre del grupo {g} en {unc_path.name}. "
                                     f"Variables: {list(unc.data_vars)}")
                u = unc[vu].values[yy, xx].astype("float64")
                det = (ids > 0) & np.isfinite(dep) & np.isfinite(u) & (dep > 0) & (dep > FACTOR_RUIDO * u)
                criterio_usado.add(f"band_depth > {FACTOR_RUIDO} × incertidumbre")
            else:
                det = (ids > 0) & np.isfinite(dep) & (dep >= MIN_BAND_DEPTH)
                criterio_usado.add(f"RESPALDO: band_depth >= {MIN_BAND_DEPTH} (sin MINUNCERT)")
            for v in np.unique(ids[det]):
                nom = nombres.get(int(v), f"id_{v}")
                sel = det & (ids == v)
                nombres_vistos[nom] = nombres_vistos.get(nom, 0) + int(sel.sum())
                conteos[grupo_de(nom)] += sel
            alguna |= det
        acumular(emit, ci, cj, n_emit=np.ones(yy.size), n_det=alguna.astype(float),
                 **{f"n_{g}": conteos[g] for g in GRUPOS})
        print(f"  {nc.name}: {yy.size:,} píxeles en el área "
              f"({'con' if unc is not None else 'SIN'} MINUNCERT)")
    if not emit:
        raise SystemExit("Ningún archivo EMIT cae dentro del área de los .hgt")
    emit = pd.concat(emit).groupby(level=[0, 1]).sum()

    mapeo = (pd.DataFrame({"nombre_emit": list(nombres_vistos),
                           "detecciones": list(nombres_vistos.values())})
             .assign(grupo=lambda d: d["nombre_emit"].map(grupo_de))
             .sort_values("detecciones", ascending=False))
    mapeo["revisar"] = mapeo["grupo"].eq("otros")
    mapeo.to_csv(out / f"mapeo_emit_grupos_{region}.csv", index=False)

    # -------------------------------------------------
    # 3) ASTER AG100 por celda (índice de cuarzo y máfico)
    # -------------------------------------------------
    # QI = b11² / (b10 · b12). El cuarzo tiene mínimos de emisividad cerca de 8.3 µm (b10)
    # y 9.1 µm (b12), así que QI SUBE con más cuarzo. Ref.: Ninomiya et al. (2005).
    ast = []
    em_sum = np.zeros(5); em_n = np.zeros(5)
    for h5 in sorted(carpeta.glob("AG100*.h5")):
        with h5py.File(h5, "r") as f:
            em = f["Emissivity/Mean"][()].astype("float64")
            lat = f["Geolocation/Latitude"][()].astype("float64")
            lon = f["Geolocation/Longitude"][()].astype("float64")
            lw = f["Land Water Map/LWmap"][()]
        if em.shape[0] != 5 and em.shape[-1] == 5:
            em = np.moveaxis(em, -1, 0)
        em[em == VACIO_EMIS] = np.nan
        em = em * ESCALA_EMIS
        em[(em <= 0) | (em > 1.2)] = np.nan
        if lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        dentro = en_zonas(lat, lon, zonas) & (lw != LWMAP_AGUA)
        qi = em[1] ** 2 / (em[0] * em[2])
        mi = em[2] * em[4] ** 3 / em[3] ** 4          # índice máfico (Ninomiya et al. 2005)
        ci, cj = clave_celda(lat[dentro], lon[dentro])
        q, mf = qi[dentro], mi[dentro]
        acumular(ast, ci, cj, qi_s=np.nan_to_num(q), qi_n=np.isfinite(q).astype(float),
                 mi_s=np.nan_to_num(mf), mi_n=np.isfinite(mf).astype(float))
        for b in range(5):
            v = em[b][dentro]; v = v[np.isfinite(v)]
            em_sum[b] += v.sum(); em_n[b] += v.size
    if not ast:
        raise SystemExit("No encontré archivos AG100*.h5 dentro del área")
    ast = pd.concat(ast).groupby(level=[0, 1]).sum()

    # -------------------------------------------------
    # 4) TABLA POR CELDA + PUNTAJE
    # -------------------------------------------------
    c = emit.join(ast, how="outer").join(terr, how="left").fillna(0)
    c["lat_centro"] = (c.index.get_level_values(0) + 0.5) * CELDA_DEG
    c["lon_centro"] = (c.index.get_level_values(1) + 0.5) * CELDA_DEG
    c["pct_agua"] = 100 * c["n_agua_dem"] / c["n_dem"].replace(0, np.nan)
    c["elev_media_m"] = c["elev_s"] / c["elev_n"].replace(0, np.nan)
    c["pendiente_media_deg"] = c["pend_s"] / c["pend_n"].replace(0, np.nan)
    c["qi"] = c["qi_s"] / c["qi_n"].replace(0, np.nan)
    c["mi"] = c["mi_s"] / c["mi_n"].replace(0, np.nan)
    for g in GRUPOS:
        c[f"pct_{g}"] = 100 * c[f"n_{g}"] / c["n_emit"].replace(0, np.nan)

    validas = (c["n_emit"] >= MIN_PIX_EMIT_CELDA) & (c["pct_agua"].fillna(100) <= MAX_AGUA_CELDA) \
              & c["qi"].notna() & c["mi"].notna()
    texto_elev = ""
    if min_elev is not None:
        bajas = validas & ~(c["elev_media_m"] >= min_elev)
        pct_bajas = 100 * bajas.sum() / max(int(validas.sum()), 1)
        validas &= c["elev_media_m"] >= min_elev
        texto_elev = f", ≥{min_elev:g} m"
        if pct_bajas > AVISO_PCT_DESCARTE_ELEV:
            avisos.append(f"el filtro de altura (≥{min_elev:g} m) quitó {pct_bajas:.0f} % de las celdas; "
                          f"si la región es baja, corre con --min-elev no")
    print(f"\nCeldas: {len(c):,} en total, {int(validas.sum()):,} válidas "
          f"(≥{MIN_PIX_EMIT_CELDA} px EMIT, ≤{MAX_AGUA_CELDA}% agua{texto_elev}, con ASTER)")
    c = c[validas].copy()
    if c.empty:
        raise SystemExit("No quedó ninguna celda válida; revisa que EMIT y ASTER cubran el área"
                         + (" o prueba con --min-elev no" if min_elev is not None else ""))

    for g in OBJETIVO:
        c[f"nivel_{g}"] = nivel(c[f"pct_{g}"])
    c["nivel_sulfatos_Fe"] = nivel(c["pct_sulfatos_Fe"])

    umbral_qi = QI_SIN_CUARZO
    origen_qi = f"físico: QI ≤ {QI_SIN_CUARZO} sin cuarzo, ≤ {QI_CUARZO_PARCIAL} parcial"
    print(f"Umbral de cuarzo: {origen_qi}")
    print(f"Umbral máfico (basalto, ASTER): MI ≥ {MI_BASALTO} basalto, ≥ {MI_BASALTO_PARCIAL} parcial")
    print("  Sensibilidad (fracción de celdas 'basalto'):",
          {t: round(float((c['mi'] >= t).mean()), 3) for t in (0.94, 0.95, 0.96, 0.97, 0.98)})
    print("  Sensibilidad (fracción de celdas 'sin cuarzo'):",
          {t: round(float((c['qi'] <= t).mean()), 3) for t in (1.0, 1.005, 1.01, 1.015, 1.02)})

    def coincide(nivel_celda, objetivo, minimo=False):
        if minimo:   # "presente": basta con alcanzar el objetivo
            return np.where(nivel_celda >= objetivo, 1.0, 0.0)
        d = np.abs(nivel_celda - objetivo)
        return np.where(d == 0, 1.0, np.where(d == 1, 0.5, 0.0))

    c["s_oxidos_Fe"] = coincide(c["nivel_oxidos_Fe"], OBJETIVO["oxidos_Fe"])
    # Basalto: ASTER (índice máfico). pct_basalto de EMIT queda solo como dato informativo.
    c["s_basalto"] = np.where(c["mi"] >= MI_BASALTO, 1.0,
                              np.where(c["mi"] >= MI_BASALTO_PARCIAL, 0.5, 0.0))
    c["s_carbonatos"] = coincide(c["nivel_carbonatos"], OBJETIVO["carbonatos"])
    c["s_arcillas_FeMg"] = coincide(c["nivel_arcillas_FeMg"], OBJETIVO["arcillas_FeMg"])
    c["s_arcillas_Al"] = coincide(c["nivel_arcillas_Al"], OBJETIVO["arcillas_Al"])
    c["s_cuarzo"] = np.where(c["qi"] <= QI_SIN_CUARZO, 1.0,
                             np.where(c["qi"] <= QI_CUARZO_PARCIAL, 0.5, 0.0))
    c["s_micas_felsicas"] = coincide(c["nivel_micas_felsicas"], OBJETIVO["micas_felsicas"])
    c["bonus_sulfatos_Fe"] = np.where((c["nivel_sulfatos_Fe"] >= 1) & (c["s_basalto"] == 1)
                                      & (c["s_cuarzo"] == 1), BONUS_SULFATOS_FE, 0.0)

    suma_w = sum(PESOS.values()) + BONUS_SULFATOS_FE
    c["puntaje"] = (sum(PESOS[k] * c[f"s_{k}"] for k in PESOS) + c["bonus_sulfatos_Fe"]) / suma_w

    cols = (["lat_centro", "lon_centro", "puntaje"] + [f"s_{k}" for k in PESOS] + ["bonus_sulfatos_Fe"]
            + [f"pct_{g}" for g in GRUPOS] + ["qi", "mi", "n_emit", "pct_agua", "elev_media_m",
                                              "pendiente_media_deg"])
    celdas = (c[cols].reset_index(drop=True)
              # Desempate continuo: más máfico (MI), luego menos cuarzo (QI), luego más óxidos de Fe
              .sort_values(["puntaje", "mi", "qi", "pct_oxidos_Fe"], ascending=[False, False, True, False])
              .reset_index(drop=True))
    celdas.insert(3, "rango", np.arange(1, len(celdas) + 1))
    celdas.round(4).to_csv(out / f"celdas_{region}.csv", index=False)

    def explica(r):
        si = [k for k in PESOS if r[f"s_{k}"] == 1]
        no = [k for k in PESOS if r[f"s_{k}"] < 1]
        extra = " + sulfatos de Fe" if r["bonus_sulfatos_Fe"] > 0 else ""
        return f"coincide: {', '.join(si) or '—'}{extra} | no coincide: {', '.join(no) or '—'}"
    top = celdas.head(10).copy()
    top["explicacion"] = top.apply(explica, axis=1)
    top[["rango", "lat_centro", "lon_centro", "puntaje", "mi", "qi", "pct_oxidos_Fe", "pct_agua",
         "explicacion"]].round(4).to_csv(out / f"top10_{region}.csv", index=False)

    # -------------------------------------------------
    # 5) CONTROL lava vs dunas
    # -------------------------------------------------
    control = None
    if zonas_control:
        filas = []
        for zona, (la0, la1, lo0, lo1) in zonas_control.items():
            m = celdas["lat_centro"].between(la0, la1) & celdas["lon_centro"].between(lo0, lo1)
            z = celdas[m]
            if z.empty:
                avisos.append(f"la zona de control '{zona}' no tiene celdas válidas; ajústala")
                continue
            filas.append({"zona": zona, "n_celdas": len(z),
                          "puntaje_medio": z["puntaje"].mean(),
                          "pct_oxidos_Fe": z["pct_oxidos_Fe"].mean(),
                          "pct_basalto": z["pct_basalto"].mean(),
                          "pct_sulfatos_Fe": z["pct_sulfatos_Fe"].mean(),
                          "pct_carbonatos": z["pct_carbonatos"].mean(),
                          "pct_micas_felsicas": z["pct_micas_felsicas"].mean(),
                          "qi_medio": z["qi"].mean(),
                          "mi_medio": z["mi"].mean(),
                          "pct_celdas_basalto_aster": 100 * (z["s_basalto"] == 1).mean()})
        if filas:
            control = pd.DataFrame(filas).set_index("zona").round(4)
            control.to_csv(out / f"control_lava_dunas_{region}.csv")
            if len(control) >= 2 and control["puntaje_medio"].iloc[0] <= control["puntaje_medio"].iloc[1]:
                avisos.append(f"control al revés: '{control.index[0]}' no supera a '{control.index[1]}'")

    # -------------------------------------------------
    # 5b) AVISOS de calidad (no cambian el puntaje)
    # -------------------------------------------------
    tot = emit.sum()
    pct_det = 100 * tot["n_det"] / tot["n_emit"]
    if pct_det < AVISO_PCT_DETECCION_EMIT:
        avisos.append(f"EMIT detecta minerales solo en {pct_det:.0f} % de los píxeles: posible vegetación, "
                      f"nubes o suelo oscuro; los % por grupo son poco representativos")
    pct_maf = 100 * (celdas["s_basalto"] == 1).mean()
    if pct_maf > AVISO_PCT_MAFICO:
        avisos.append(f"{pct_maf:.0f} % de las celdas cuentan como basalto (MI ≥ {MI_BASALTO}): la vegetación "
                      f"o el suelo húmedo también dan MI alto; revisa el mapa de índice máfico")
    smap = leer_smap(carpeta, zonas)
    for k, v in smap.items():
        if k.startswith("humedad_m3m3") and v > AVISO_HUMEDAD_SMAP:
            avisos.append(f"suelo húmedo según SMAP ({k}: {v} m³/m³): el índice de cuarzo y el máfico "
                          f"pierden contraste")

    # -------------------------------------------------
    # 6) RESUMEN DE TODA LA REGIÓN (una columna)
    # -------------------------------------------------
    resumen = {"criterio_deteccion_emit": "; ".join(sorted(criterio_usado)),
               "emit_pixeles_validos": int(tot["n_emit"]),
               "emit_pct_pixeles_con_deteccion": round(pct_det, 2)}
    for g in GRUPOS:
        resumen[f"pct_{g}"] = round(100 * tot[f"n_{g}"] / tot["n_emit"], 4)
    bandas = ["b10_8.3um", "b11_8.6um", "b12_9.1um", "b13_10.6um", "b14_11.3um"]
    for b in range(5):
        resumen[f"emis_{bandas[b]}"] = round(em_sum[b] / em_n[b], 4) if em_n[b] else np.nan
    resumen["indice_cuarzo_medio"] = round(ast["qi_s"].sum() / ast["qi_n"].sum(), 4)
    resumen["umbral_cuarzo_usado"] = round(umbral_qi, 4)
    resumen["origen_umbral_cuarzo"] = origen_qi
    resumen["indice_mafico_medio"] = round(ast["mi_s"].sum() / ast["mi_n"].sum(), 4)
    resumen["umbral_basalto_mi"] = f"MI ≥ {MI_BASALTO} basalto, ≥ {MI_BASALTO_PARCIAL} parcial"
    resumen["max_agua_celda_pct"] = MAX_AGUA_CELDA
    resumen["min_elev_celda_m"] = "sin filtro" if min_elev is None else min_elev
    resumen["pct_celdas_basalto_aster"] = round(pct_maf, 2)
    resumen["zonas_control"] = ", ".join(zonas_control) or "ninguna"
    resumen["bonus_sulfatos_Fe"] = BONUS_SULFATOS_FE
    resumen["pct_agua_region"] = round(100 * agua_total / px_total, 2)
    resumen["elev_media_m"] = round(terr["elev_s"].sum() / terr["elev_n"].sum(), 1)
    resumen["pendiente_media_deg"] = round(terr["pend_s"].sum() / terr["pend_n"].sum(), 2)
    resumen["celdas_validas"] = len(celdas)
    resumen["puntaje_medio"] = round(celdas["puntaje"].mean(), 4)
    resumen["puntaje_max"] = round(celdas["puntaje"].max(), 4)
    resumen.update(smap)
    resumen["avisos"] = " | ".join(avisos) or "ninguno"
    pd.DataFrame({"region_completa": resumen}).to_csv(out / f"resumen_region_{region}.csv")

    # -------------------------------------------------
    # 7) FIGURAS
    # -------------------------------------------------
    figuras(celdas, control, region, out, zonas_control)

    # -------------------------------------------------
    # IMPRESIÓN FINAL
    # -------------------------------------------------
    print("\n=== MAPEO EMIT -> GRUPOS (revisar los marcados) ===")
    print(mapeo.to_string(index=False))
    print("\n=== RESUMEN DE LA REGIÓN ===")
    print(pd.Series(resumen).to_string())
    if control is not None:
        print("\n=== CONTROL LAVA vs DUNAS ===")
        print(control.to_string())
    print("\n=== TOP 10 CELDAS MÁS PARECIDAS A MERIDIANI ===")
    print(top[["rango", "lat_centro", "lon_centro", "puntaje", "mi", "qi", "pct_agua", "explicacion"]]
          .round(4).to_string(index=False))
    print("\n=== AVISOS ===")
    print("\n".join(f"  ! {a}" for a in avisos) if avisos else "  ninguno")
    print(f"\nArchivos guardados en {out.resolve()}")


def leer_smap(carpeta, zonas):
    """Humedad media de la región por archivo y pase (solo contexto)."""
    salida = {}
    for h5 in sorted(carpeta.glob("SMAP*.h5")):
        m = re.search(r"(\d{4})(\d{2})(\d{2})", h5.name)
        fecha = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else h5.stem
        with h5py.File(h5, "r") as f:
            for grupo, suf, pase in (("Soil_Moisture_Retrieval_Data_AM", "", "manana"),
                                     ("Soil_Moisture_Retrieval_Data_PM", "_pm", "tarde")):
                if grupo not in f or f"soil_moisture{suf}" not in f[grupo]:
                    continue
                g = f[grupo]
                sm = g[f"soil_moisture{suf}"][()].astype("float64")
                lat = g[f"latitude{suf}"][()].astype("float64")
                lon = g[f"longitude{suf}"][()].astype("float64")
                sm[sm == -9999] = np.nan
                q = f"retrieval_qual_flag{suf}"
                if q in g:
                    sm[(g[q][()].astype("int64") & 1) != 0] = np.nan
                v = sm[en_zonas(lat, lon, zonas)]
                v = v[np.isfinite(v)]
                if v.size:
                    salida[f"humedad_m3m3_{fecha}_{pase}"] = round(float(v.mean()), 4)
                    salida[f"humedad_pixeles_{fecha}_{pase}"] = int(v.size)
    return salida


def figuras(celdas, control, region, out, zonas_control):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def malla(col):
        t = celdas.pivot_table(index="lat_centro", columns="lon_centro", values=col)
        return t.columns.values, t.index.values, t.values

    x, y, z = malla("puntaje")
    fig, ax = plt.subplots(figsize=(8, 7))
    m = ax.pcolormesh(x, y, z, cmap="viridis", vmin=0, vmax=1, shading="nearest")
    fig.colorbar(m, ax=ax, label="Puntaje de similitud con Meridiani (0–1)")
    for zona, (la0, la1, lo0, lo1) in zonas_control.items():
        ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, ec="red", lw=1.2))
        ax.text(lo0, la1, zona, color="red", fontsize=8, va="bottom")
    t = celdas.head(10)
    ax.scatter(t["lon_centro"], t["lat_centro"], marker="*", s=90, c="white", ec="black",
               label="top 10")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.legend(loc="lower left")
    ax.set_title(f"{region}: similitud del suelo con Meridiani (celdas de ~1 km)")
    fig.tight_layout(); fig.savefig(out / f"mapa_puntaje_{region}.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, col, cmap, et in ((axs[0], "qi", "magma", "Índice de cuarzo (más alto = más cuarzo)"),
                              (axs[1], "mi", "viridis", "Índice máfico ASTER (más alto = más basáltico)"),
                              (axs[2], "pct_oxidos_Fe", "Reds", "% óxidos de Fe (EMIT)")):
        x, y, z = malla(col)
        m = ax.pcolormesh(x, y, z, cmap=cmap, shading="nearest")
        fig.colorbar(m, ax=ax, label=et)
        for zona, (la0, la1, lo0, lo1) in zonas_control.items():
            ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, ec="cyan", lw=1.2))
        ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_title(et)
    fig.tight_layout(); fig.savefig(out / f"mapa_qi_hematita_{region}.png", dpi=150); plt.close(fig)

    if control is not None and len(control) >= 2:
        cols = ["puntaje_medio", "pct_oxidos_Fe", "mi_medio", "qi_medio", "pct_micas_felsicas"]
        fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
        for ax, col in zip(axs, cols):
            ax.bar(control.index, control[col], color=["#8c2d04", "#e6ab02"][:len(control)])
            ax.set_title(col); ax.tick_params(axis="x", rotation=20)
        fig.suptitle(f"{region}: control campo de lava vs dunas")
        fig.tight_layout(); fig.savefig(out / f"control_lava_dunas_{region}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compara una región de la Tierra con la huella de Meridiani")
    ap.add_argument("--carpeta", default="./resources", help="carpeta con los datos de UNA región")
    ap.add_argument("--region", default="pinacate", help="nombre de la región (salidas y configuración)")
    ap.add_argument("--min-elev", default="auto",
                    help="altura mínima por celda: 'auto' (MIN_ELEV_POR_REGION), 'no' o metros")
    ap.add_argument("--salida", default="resultados_tierra", help="carpeta base de resultados")
    a = ap.parse_args()
    main(a.carpeta, a.region, a.min_elev, a.salida)