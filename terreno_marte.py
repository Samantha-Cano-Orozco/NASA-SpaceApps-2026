"""
terreno_marte.py  —  Space Apps 2026 · Análogos de Marte (suelos)
Pendiente, rugosidad y relieve del cráter Victoria (Meridiani Planum) a partir del DTM estéreo de HiRISE,
con EXACTAMENTE las mismas definiciones que terreno.py (Tierra), para poder compararlos.

USO
    python terreno_marte.py
    python terreno_marte.py --dtm DTEEC_021747_1780_022380_1780_U01.IMG --salida resultados_terreno

ENTRADA
    DTEEC_021747_1780_022380_1780_U01.IMG  DTM HiRISE (PDS3 con etiqueta adjunta; rasterio lo abre directo).
    Alturas en metros respecto al areoide, ~1 m por píxel. Nulos ≈ −3.4e38 → NaN.
    Fuente: https://www.uahirise.org/PDS/DTM/ESP/ORB_021700_021799/ESP_021747_1780_ESP_022380_1780/

QUÉ HACE
    1) Verifica que el DTM cubre Victoria (~2.05°S, 354.5°E); si no, se detiene.
    2) Promedia a 30 m y a 90 m (Resampling.average, sin contar nulos; una celda nueva necesita ≥ 50 % de
       píxeles válidos).
    3) Pendiente, rugosidad (ventana 3×3) y relieve con las mismas fórmulas que terreno.py.
       La elevación absoluta NO se compara con la Tierra (areoide vs nivel del mar).
    4) Tres conjuntos: escena completa · planicie (pendiente < 15°) · planicie fuera del cráter (sin un círculo
       de 600 m de radio alrededor del centro de Victoria).
    5) Métricas por conjunto y resolución.

CENTRO DEL CRÁTER (cómo se encuentra)
    En el DTM a 30 m: nivel de la planicie = mediana de la elevación de la escena; fondo = punto más bajo.
    Se toman los píxeles más bajos que la mitad de la profundidad ((planicie + fondo) / 2) y se queda la región
    conectada que contiene el punto más bajo: es el interior del cráter. Su centroide es el centro.
    Comprobaciones: diámetro equivalente de esa región y diámetro del borde (radio donde el perfil radial
    promedio de elevación alcanza su máximo), y distancia al centro nominal (2.05°S, 354.5°E).

SALIDAS (resultados_terreno/victoria/)
    terreno_victoria.csv           una fila por conjunto × resolución (mismos nombres de columna que
                                   terreno_resumen_<region>.csv cuando es la misma métrica)
    distribuciones_victoria.npz    pendiente y rugosidad por píxel de la planicie fuera del cráter a 30 m
    mapa_terreno_victoria.png      elevación, pendiente y rugosidad a 30 m con el cráter marcado
"""
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from scipy.ndimage import uniform_filter

# =====================================================
# CONFIGURACIÓN
# =====================================================
DTM = "DTEEC_021747_1780_022380_1780_U01.IMG"
REGION = "victoria"
CENTRO_NOMINAL = (-2.05, 354.5)      # lat, lon °E del cráter Victoria
TOLERANCIA_CENTRO_M = 500.0          # el centro encontrado debe estar a menos de esto del nominal
RESOLUCIONES_M = (30, 90)            # 30 m = misma escala que SRTM/NASADEM de 1"
FRACCION_VALIDA_MIN = 0.5            # fracción mínima de píxeles de 1 m válidos para aceptar un píxel nuevo
VENTANA_RUGOSIDAD = 3                # igual que terreno.py
PENDIENTE_PLANICIE = 15.0            # planicie = pendiente < 15°
RADIO_EXCLUSION_M = 600.0            # círculo excluido alrededor del centro (el cráter mide ~750 m)
BLOQUE_RELIEVE_M = 1000.0            # bloques de ~1 km para el relieve local (como las celdas de terreno.py)
NULO_MIN = -1e30                     # valores por debajo = nulo


# =====================================================
# FUNCIONES COPIADAS DE terreno.py (mismas definiciones)
# =====================================================
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


def pendiente(dem, dx, dy):
    """Igual que terreno.py: grados, desde el gradiente con el tamaño real del píxel en metros."""
    return np.degrees(np.arctan(np.hypot(np.gradient(dem, axis=1) / dx, np.gradient(dem, axis=0) / dy)))


# =====================================================
# UTILIDADES
# =====================================================
def remuestrear(arr, valido, src_tr, crs, bounds, res):
    """Promedio a 'res' metros, ignorando nulos; exige FRACCION_VALIDA_MIN de píxeles válidos."""
    ancho = int(math.ceil((bounds.right - bounds.left) / res))
    alto = int(math.ceil((bounds.top - bounds.bottom) / res))
    tr = from_origin(bounds.left, bounds.top, res, res)
    dst = np.full((alto, ancho), np.nan, dtype="float32")
    reproject(arr, dst, src_transform=src_tr, src_crs=crs, dst_transform=tr, dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    frac = np.zeros((alto, ancho), dtype="float32")
    reproject(valido, frac, src_transform=src_tr, src_crs=crs, dst_transform=tr, dst_crs=crs,
              resampling=Resampling.average)
    dst[frac < FRACCION_VALIDA_MIN] = np.nan
    return dst.astype("float64"), tr


def coords(tr, forma):
    filas, cols = np.indices(forma)
    return tr.c + tr.a * (cols + 0.5), tr.f + tr.e * (filas + 0.5)


def centro_crater(elev, tr):
    """Centro del cráter = centroide de la región conectada más baja que la mitad de la profundidad."""
    plan = float(np.nanmedian(elev))
    i_min = np.unravel_index(np.nanargmin(elev), elev.shape)
    fondo = float(elev[i_min])
    umbral = (plan + fondo) / 2
    etiquetas, _ = ndimage.label(np.nan_to_num(elev, nan=np.inf) < umbral)
    interior = etiquetas == etiquetas[i_min]
    x, y = coords(tr, elev.shape)
    cx, cy = float(x[interior].mean()), float(y[interior].mean())
    d_equiv = 2 * math.sqrt(interior.sum() * abs(tr.a * tr.e) / math.pi)
    # borde: máximo del perfil radial promedio de elevación entre 150 y 900 m
    r = np.hypot(x - cx, y - cy)
    bordes = np.arange(0, 1500 + 15, 15.0)
    idx = np.digitize(r.ravel(), bordes) - 1
    ok = np.isfinite(elev.ravel()) & (idx >= 0) & (idx < len(bordes) - 1)
    suma = np.bincount(idx[ok], elev.ravel()[ok], minlength=len(bordes) - 1)
    n = np.bincount(idx[ok], minlength=len(bordes) - 1)
    perfil = np.where(n > 0, suma / np.maximum(n, 1), np.nan)
    radios = (bordes[:-1] + bordes[1:]) / 2
    sel = (radios >= 150) & (radios <= 900)
    r_borde = float(radios[sel][np.nanargmax(perfil[sel])])
    return {"cx": cx, "cy": cy, "nivel_planicie_m": plan, "fondo_m": fondo, "profundidad_m": plan - fondo,
            "diametro_equiv_mitad_prof_m": d_equiv, "diametro_borde_m": 2 * r_borde}


def paralelo_estandar(crs):
    """Latitud de escala verdadera de la proyección equirectangular (0 si no la declara)."""
    for p in getattr(crs.coordinate_operation, "params", []) or []:
        if "standard parallel" in p.name.lower() or "latitude of 1st" in p.name.lower():
            return float(p.value)
    return 0.0


def relieve_bloques(elev, en_conjunto, res):
    """Mediana del relieve (máx − mín) en bloques de ~1 km que caen al menos a la mitad dentro del conjunto."""
    b = max(int(round(BLOQUE_RELIEVE_M / res)), 2)
    nf, nc = elev.shape[0] // b, elev.shape[1] // b
    e = elev[:nf * b, :nc * b].reshape(nf, b, nc, b)
    m = en_conjunto[:nf * b, :nc * b].reshape(nf, b, nc, b)
    validos = np.isfinite(e).sum(axis=(1, 3))
    usa = (m.mean(axis=(1, 3)) >= 0.5) & (validos >= 0.5 * b * b)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # bloques sin datos -> NaN
        rel = np.nanmax(e, axis=(1, 3)) - np.nanmin(e, axis=(1, 3))
    return float(np.median(rel[usa])) if usa.any() else np.nan, int(usa.sum())


def metricas(elev, pend, rug, conjunto, res):
    p = pend[conjunto & np.isfinite(pend)]
    r = rug[conjunto & np.isfinite(rug)]
    rel_1km, n_bloques = relieve_bloques(elev, conjunto, res)
    return {
        "n_pixeles": int(p.size), "area_km2": p.size * res * res / 1e6,
        "pendiente_media_deg": float(p.mean()), "pendiente_mediana_deg": float(np.median(p)),
        "pendiente_p95_deg": float(np.percentile(p, 95)), "pendiente_max_deg": float(p.max()),
        "pct_terreno_pendiente_lt_5deg": float(100 * (p < 5).mean()),
        "rugosidad_media_m": float(r.mean()), "rugosidad_mediana_m": float(np.median(r)),
        "rugosidad_p95_m": float(np.percentile(r, 95)), "rugosidad_max_m": float(r.max()),
        "relieve_m": float(np.nanmax(elev[conjunto]) - np.nanmin(elev[conjunto])),
        "relieve_1km_mediano_m": rel_1km, "bloques_1km": n_bloques,
    }


# =====================================================
# PRINCIPAL
# =====================================================
def main(ruta_dtm, salida):
    ruta_dtm = Path(ruta_dtm)
    if not ruta_dtm.exists():
        raise SystemExit(f"No encontré {ruta_dtm.resolve()}")
    out = Path(salida) / REGION
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ruta_dtm) as src:
        crs_r, tr1, b = src.crs, src.transform, src.bounds
        crs = CRS.from_wkt(crs_r.to_wkt())
        a_geo = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        a_map = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        lon_c, lat_c = a_geo.transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)
        esquinas = [a_geo.transform(x, y) for x in (b.left, b.right) for y in (b.bottom, b.top)]
        print(f"DTM: {ruta_dtm.resolve()}")
        print(f"  CRS: {crs.name} · {crs.geodetic_crs.name} (radio {crs.ellipsoid.semi_major_metre:.0f} m) · "
              f"unidades {crs.axis_info[0].unit_name}")
        print(f"  píxel: {src.res[0]:.4f} × {src.res[1]:.4f} m · dimensiones {src.width} × {src.height} "
              f"({(b.right - b.left) / 1000:.1f} × {(b.top - b.bottom) / 1000:.1f} km)")
        print(f"  centro aprox.: {lat_c:.4f}°, {lon_c % 360:.4f}°E · lat {min(e[1] for e in esquinas):.3f}…"
              f"{max(e[1] for e in esquinas):.3f}, lon {min(e[0] for e in esquinas) % 360:.3f}…"
              f"{max(e[0] for e in esquinas) % 360:.3f}°E")
        # ¿cubre Victoria?
        lon_n = CENTRO_NOMINAL[1] if CENTRO_NOMINAL[1] <= 180 else CENTRO_NOMINAL[1] - 360
        xn, yn = a_map.transform(lon_n, CENTRO_NOMINAL[0])
        if not (b.left <= xn <= b.right and b.bottom <= yn <= b.top):
            raise SystemExit(f"El DTM NO cubre el cráter Victoria ({CENTRO_NOMINAL[0]}°, {CENTRO_NOMINAL[1]}°E). "
                             f"Me detengo.")
        print(f"  cubre Victoria ({CENTRO_NOMINAL[0]}°, {CENTRO_NOMINAL[1]}°E): sí")
        arr = src.read(1)
    arr = arr.astype("float32")
    valido = (np.isfinite(arr) & (arr > NULO_MIN)).astype("uint8")
    arr[valido == 0] = np.nan
    print(f"  píxeles válidos: {100 * valido.mean():.1f} %")
    lat_ts = math.radians(paralelo_estandar(crs))

    filas_csv, crater, guardado = [], None, {}
    for res in RESOLUCIONES_M:
        elev, tr = remuestrear(arr, valido, tr1, crs_r, b, res)
        x, y = coords(tr, elev.shape)
        _, lat_filas = a_geo.transform(np.full(elev.shape[0], (b.left + b.right) / 2), y[:, 0])
        # equirectangular: la distancia E-O real = distancia en el mapa × cos(lat) / cos(lat_ts)
        dx = (res * np.cos(np.radians(lat_filas)) / math.cos(lat_ts))[:, None]
        dy = res
        pend = pendiente(elev, dx, dy)
        rug = rugosidad(elev, VENTANA_RUGOSIDAD)

        if crater is None:          # el centro se busca una vez, a 30 m
            crater = centro_crater(elev, tr)
            lon_k, lat_k = a_geo.transform(crater["cx"], crater["cy"])
            crater.update({"lat": lat_k, "lon_E": lon_k % 360,
                           "dist_al_nominal_m": math.hypot(crater["cx"] - xn, crater["cy"] - yn)})
            print(f"\nCráter: centro {lat_k:.5f}°, {lon_k % 360:.5f}°E · a {crater['dist_al_nominal_m']:.0f} m del "
                  f"nominal · profundidad {crater['profundidad_m']:.1f} m · diámetro del borde "
                  f"{crater['diametro_borde_m']:.0f} m · diámetro a media profundidad "
                  f"{crater['diametro_equiv_mitad_prof_m']:.0f} m")
            if crater["dist_al_nominal_m"] > TOLERANCIA_CENTRO_M:
                raise SystemExit("El centro encontrado está lejos del nominal: revisa el DTM. Me detengo.")

        dist = np.hypot(x - crater["cx"], y - crater["cy"])
        valid = np.isfinite(elev)
        conjuntos = {"escena_completa": valid,
                     "planicie": valid & (pend < PENDIENTE_PLANICIE),
                     "planicie_fuera_crater": valid & (pend < PENDIENTE_PLANICIE) & (dist > RADIO_EXCLUSION_M)}
        for nombre, m in conjuntos.items():
            filas_csv.append({"region": REGION, "conjunto": nombre, "resolucion_m": res,
                              **metricas(elev, pend, rug, m, res)})
        if res == 30:
            m = conjuntos["planicie_fuera_crater"]
            guardado = {"elev": elev, "pend": pend, "rug": rug, "x": x, "y": y,
                        "pend_plan": pend[m & np.isfinite(pend)].astype("float32"),
                        "rug_plan": rug[m & np.isfinite(rug)].astype("float32")}

    tabla = pd.DataFrame(filas_csv)
    for k in ("lat", "lon_E", "profundidad_m", "diametro_borde_m", "dist_al_nominal_m"):
        tabla[f"crater_{k}"] = crater[k]
    tabla["radio_exclusion_m"] = RADIO_EXCLUSION_M
    tabla["pendiente_planicie_deg"] = PENDIENTE_PLANICIE
    tabla.round(4).to_csv(out / f"terreno_{REGION}.csv", index=False)
    np.savez_compressed(out / f"distribuciones_{REGION}.npz", pendiente_deg=guardado["pend_plan"],
                        rugosidad_m=guardado["rug_plan"], conjunto="planicie_fuera_crater", resolucion_m=30,
                        radio_exclusion_m=RADIO_EXCLUSION_M, pendiente_planicie_deg=PENDIENTE_PLANICIE,
                        crater_lat=crater["lat"], crater_lon_E=crater["lon_E"])
    figura(guardado, crater, out)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cols = ["conjunto", "resolucion_m", "n_pixeles", "area_km2", "pendiente_mediana_deg", "pendiente_p95_deg",
            "pendiente_max_deg", "rugosidad_mediana_m", "rugosidad_p95_m", "rugosidad_max_m",
            "pct_terreno_pendiente_lt_5deg", "relieve_m", "relieve_1km_mediano_m"]
    print("\n" + tabla[cols].round(3).to_string(index=False))
    print(f"\nArchivos en {out.resolve()}")


def figura(g, crater, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.patheffects as pe

    xk, yk = (g["x"] - crater["cx"]) / 1000, (g["y"] - crater["cy"]) / 1000
    ext = [xk.min(), xk.max(), yk.min(), yk.max()]
    paneles = [(g["elev"], "cividis", "Elevación (m, respecto al areoide)", None),
               (g["pend"], "magma", "Pendiente (°)", None),
               (g["rug"], "viridis", "Rugosidad 3×3 (m, escala log)", "log")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 7.6), facecolor="#fcfcfb")
    for ax, (z, cmap, et, escala) in zip(axs, paneles):
        ax.set_facecolor("#fcfcfb")
        v = z[np.isfinite(z)]
        norm = LogNorm(vmin=max(np.percentile(v, 1), 1e-3), vmax=np.percentile(v, 99.5)) if escala else None
        kw = {} if escala else {"vmin": np.percentile(v, 0.5), "vmax": np.percentile(v, 99.5)}
        m = ax.imshow(z, extent=ext, cmap=cmap, norm=norm, origin="upper", interpolation="nearest", **kw)
        fig.colorbar(m, ax=ax, label=et, shrink=0.7)
        t = np.linspace(0, 2 * np.pi, 200)
        r_ex, r_b = RADIO_EXCLUSION_M / 1000, crater["diametro_borde_m"] / 2000
        ax.plot(r_ex * np.cos(t), r_ex * np.sin(t), color="white", lw=1.4, ls="--",
                path_effects=[pe.Stroke(linewidth=3, foreground="#0b0b0b"), pe.Normal()],
                label=f"excluido ({RADIO_EXCLUSION_M:.0f} m de radio)")
        ax.plot(r_b * np.cos(t), r_b * np.sin(t), color="#e34948", lw=1.2,
                label=f"borde del cráter ({crater['diametro_borde_m']:.0f} m de diámetro)")
        ax.set_title(et, loc="left", fontsize=11)
        ax.set_xlabel("km al este del centro del cráter"); ax.set_ylabel("km al norte")
    axs[0].legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.suptitle(f"Cráter Victoria (HiRISE DTM a 30 m) · centro {crater['lat']:.4f}°, {crater['lon_E']:.4f}°E",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / f"mapa_terreno_{REGION}.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Terreno del cráter Victoria (HiRISE DTM) con las definiciones de terreno.py")
    ap.add_argument("--dtm", default=DTM)
    ap.add_argument("--salida", default="resultados_terreno")
    a = ap.parse_args()
    main(a.dtm, a.salida)
