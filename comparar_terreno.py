"""
comparar_terreno.py  —  Space Apps 2026 · Análogos de Marte (suelos)
Compara el terreno del cráter Victoria (Marte) con TODAS las regiones de la Tierra procesadas con terreno.py.

USO
    python comparar_terreno.py
    python comparar_terreno.py --carpeta resultados_terreno

ENTRADAS (se detectan solas; no hay nombres de región escritos a mano)
    <carpeta>/victoria/terreno_victoria.csv, distribuciones_victoria.npz        (de terreno_marte.py)
    <carpeta>/<region>/terreno_resumen_<region>.csv, terreno_celdas_<region>.csv (de terreno.py)
    <carpeta>/<region>/histogramas_<region>.npz   opcional: histogramas por píxel (terreno.py nuevo)
    Zonas dentro de una región: diccionario ZONAS (mismo formato que minerologia2.py).

BASE DE COMPARACIÓN (ojo con las escalas)
    Victoria: píxeles de 30 m (DTM HiRISE promediado). En la Tierra se usa la MISMA base cuando existe:
      1. píxeles de ~30 m de la planicie (pendiente < 15°)  -> terreno.py nuevo (columnas planicie_* del resumen)
      2. píxeles de ~30 m de toda la región                 -> terreno.py anterior (incluye sierras y laderas)
      3. promedios por celda de 1 km (terreno_celdas)       -> zonas; suaviza los extremos (p95 y máximos bajan)
    La columna "base" dice cuál se usó y "nota" avisa cuando no es la misma base que Victoria.

SIMILITUD DE TERRENO (0 a 1)
    Para cada métrica k con peso: diferencia relativa simétrica  d_k = |Tierra − Victoria| / max(Tierra, Victoria)
    (0 = iguales; 1 = una de las dos es 0 o la otra es muchísimo mayor; nunca pasa de 1, así que una sola métrica
    no puede "romper" la escala aunque la Tierra sea 10 veces más rugosa).
    similitud = 1 − Σ(peso_k · d_k) / Σ peso_k, recortada a [0, 1].  Pesos en PESOS.

SALIDAS (<carpeta>/comparacion/)
    comparacion_terreno_victoria.csv   tabla completa, ordenada de más a menos parecida
    ranking_terreno.png                barras con la similitud de cada región/zona
    histogramas_terreno.png            distribución de pendiente y rugosidad: Victoria vs cada región/zona

    Y una carpeta por cada comparación Marte–lugar de la Tierra (resultados_comparacion/victoria_vs_<region>/):
    comparacion_terreno_victoria_vs_<region>.csv   métrica por métrica: Victoria, la región y sus zonas,
                                                   diferencia (Tierra − Victoria), diferencia relativa, similitud
    histogramas_victoria_vs_<region>.png           pendiente y rugosidad: Victoria vs ese lugar
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================
# CONFIGURACIÓN
# =====================================================
REFERENCIA = {"conjunto": "planicie_fuera_crater", "resolucion_m": 30}   # fila de terreno_victoria.csv
REGION_MARTE = "victoria"
PENDIENTE_PLANICIE = 15.0
PESOS = {                                    # métricas que entran en la similitud y su peso
    "pendiente_mediana_deg": 1.0,
    "pendiente_p95_deg": 1.0,
    "rugosidad_mediana_m": 1.0,
    "rugosidad_p95_m": 1.0,
}
DESCRIPCION = {
    "pendiente_mediana_deg": "pendiente mediana (°)", "pendiente_p95_deg": "pendiente p95 (°)",
    "rugosidad_mediana_m": "rugosidad mediana 3×3 (m)", "rugosidad_p95_m": "rugosidad p95 3×3 (m)",
    "pct_terreno_pendiente_lt_5deg": "% del terreno con pendiente < 5°",
    "relieve_1km_mediano_m": "relieve mediano en bloques/celdas de 1 km (m)",
    "relieve_m": "relieve total del área (m; depende del tamaño del área)",
}
METRICAS = ["pendiente_mediana_deg", "pendiente_p95_deg", "rugosidad_mediana_m", "rugosidad_p95_m",
            "pct_terreno_pendiente_lt_5deg", "relieve_1km_mediano_m", "relieve_m"]
ZONAS = {                                    # (lat_min, lat_max, lon_min, lon_max), igual que minerologia2.py
    "pinacate": {"lava_pinacate": (31.60, 31.95, -113.62, -113.30),
                 "dunas_altar": (31.95, 32.30, -114.60, -114.10)},
}
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"


# =====================================================
# MÉTRICAS DE LA TIERRA
# =====================================================
def usa_planicie():
    return REFERENCIA["conjunto"].startswith("planicie")


def metricas_celdas(c):
    """Métricas desde promedios por celda de 1 km (base 3)."""
    p, r = c["pendiente_media_deg"].dropna(), c["rugosidad_media_m"].dropna()
    return {"pendiente_mediana_deg": p.median(), "pendiente_p95_deg": p.quantile(0.95),
            "rugosidad_mediana_m": r.median(), "rugosidad_p95_m": r.quantile(0.95),
            "pct_terreno_pendiente_lt_5deg": 100 * (p < 5).mean(),
            "relieve_1km_mediano_m": c["relieve_m"].median(),
            "relieve_m": c["elev_max_m"].max() - c["elev_min_m"].min()}


def filas_region(region, d):
    res = pd.read_csv(d / f"terreno_resumen_{region}.csv", index_col=0)["region_completa"]
    res = res.apply(lambda v: pd.to_numeric(v, errors="coerce") if pd.notna(pd.to_numeric(v, errors="coerce")) else v)
    cel = pd.read_csv(d / f"terreno_celdas_{region}.csv")
    cel = cel[cel["pct_agua"].fillna(0) < 50]
    cel_plan = cel[cel["pendiente_media_deg"] < PENDIENTE_PLANICIE] if usa_planicie() else cel
    hist = d / f"histogramas_{region}.npz"
    filas = []

    # --- región completa: píxeles si existen ---
    f = {"lugar": region, "tipo": "región", "region": region, "zona": ""}
    if usa_planicie() and "planicie_pendiente_mediana_deg" in res:
        f.update({k: float(res[f"planicie_{k}"]) for k in ("pendiente_mediana_deg", "pendiente_p95_deg",
                                                          "rugosidad_mediana_m", "rugosidad_p95_m",
                                                          "pct_terreno_pendiente_lt_5deg")})
        f["base"], f["nota"] = "píxeles ~30 m · planicie (< 15°)", ""
        f["hist"] = ("npz_planicie", hist) if hist.exists() else ("celdas", cel_plan)
    else:
        f.update({k: float(res[k]) for k in ("pendiente_mediana_deg", "pendiente_p95_deg", "rugosidad_mediana_m",
                                             "rugosidad_p95_m", "pct_terreno_pendiente_lt_5deg")})
        f["base"] = "píxeles ~30 m · región completa"
        f["nota"] = ("incluye sierras y laderas: no hay planicie por píxel (vuelve a correr terreno.py)"
                     if usa_planicie() else "")
        f["hist"] = ("npz", hist) if hist.exists() else ("celdas", cel)
    f["relieve_1km_mediano_m"] = cel_plan["relieve_m"].median()
    f["relieve_m"] = float(res["elev_max_m"]) - float(res["elev_min_m"])
    f["nota"] = "; ".join(x for x in (f["nota"], "relieve_1km: mediana por celda de 1 km",
                                       f"relieve_m: de toda la región "
                                       f"({float(res['lat_max']) - float(res['lat_min']):.0f}°×"
                                       f"{float(res['lon_max']) - float(res['lon_min']):.0f}°), "
                                       f"no comparable con el área de Victoria") if x)
    f["rugosidad_piso_dem_m"] = cel["rugosidad_media_m"].quantile(0.01)
    filas.append(f)

    # --- zonas: solo existen como promedios por celda ---
    for zona, (la0, la1, lo0, lo1) in ZONAS.get(region, {}).items():
        z = cel[cel["lat_centro"].between(la0, la1) & cel["lon_centro"].between(lo0, lo1)]
        if usa_planicie():
            z = z[z["pendiente_media_deg"] < PENDIENTE_PLANICIE]
        if z.empty:
            print(f"  aviso: la zona {zona} no tiene celdas en {region}")
            continue
        g = {"lugar": f"{region} · {zona}", "tipo": "zona", "region": region, "zona": zona,
             **metricas_celdas(z),
             "base": "promedio por celda de 1 km" + (" · planicie (< 15°)" if usa_planicie() else ""),
             "nota": "promedios por celda de 1 km de píxeles de 30 m: suaviza la distribución "
                     "(p95 y rugosidad quedan subestimados); no es la misma base que Victoria",
             "rugosidad_piso_dem_m": cel["rugosidad_media_m"].quantile(0.01),
             "hist": ("celdas", z)}
        filas.append(g)
    return filas


def distribucion(fuente, es_pend):
    """(valores, pesos) para el histograma."""
    tipo, obj = fuente
    if tipo == "celdas":
        v = obj["pendiente_media_deg" if es_pend else "rugosidad_media_m"].dropna().to_numpy()
        return v, np.ones_like(v)
    h = np.load(obj)
    bordes = h["bordes_pend"] if es_pend else h["bordes_rug"]
    clave = ("hist_pend" if es_pend else "hist_rug") + ("_planicie" if tipo == "npz_planicie" else "")
    return (bordes[:-1] + bordes[1:]) / 2, h[clave]


# =====================================================
# PRINCIPAL
# =====================================================
def main(carpeta, salida_lugares):
    carpeta = Path(carpeta)
    out = carpeta / "comparacion"
    out.mkdir(parents=True, exist_ok=True)

    vic = pd.read_csv(carpeta / REGION_MARTE / f"terreno_{REGION_MARTE}.csv")
    ref = vic[(vic["conjunto"] == REFERENCIA["conjunto"]) & (vic["resolucion_m"] == REFERENCIA["resolucion_m"])]
    if len(ref) != 1:
        raise SystemExit(f"No encontré la referencia {REFERENCIA} en terreno_{REGION_MARTE}.csv")
    ref = ref.iloc[0]
    dist_v = np.load(carpeta / REGION_MARTE / f"distribuciones_{REGION_MARTE}.npz")

    regiones = sorted(p.name for p in carpeta.iterdir()
                      if p.is_dir() and p.name not in (REGION_MARTE, "comparacion")
                      and (p / f"terreno_resumen_{p.name}.csv").exists())
    if not regiones:
        raise SystemExit(f"No encontré regiones de la Tierra en {carpeta.resolve()}")
    print(f"Referencia: Victoria · {REFERENCIA['conjunto']} · {REFERENCIA['resolucion_m']} m")
    print(f"Regiones encontradas: {', '.join(regiones)}")

    filas = [f for r in regiones for f in filas_region(r, carpeta / r)]
    hists = {f["lugar"]: f.pop("hist") for f in filas}
    t = pd.DataFrame(filas)
    t.insert(1, "cuerpo", "Tierra")

    # diferencias y similitud
    for k in METRICAS:
        t[f"dif_{k}"] = t[k] - ref[k]
    for k in PESOS:
        a, b = t[k].to_numpy(float), float(ref[k])
        mx = np.maximum(np.abs(a), abs(b))
        t[f"difrel_{k}"] = np.where(mx > 0, np.abs(a - b) / np.where(mx > 0, mx, 1), 0.0)
    t["similitud_terreno"] = (1 - sum(PESOS[k] * t[f"difrel_{k}"] for k in PESOS) / sum(PESOS.values())).clip(0, 1)
    t = t.sort_values("similitud_terreno", ascending=False).reset_index(drop=True)
    t.insert(0, "rango", np.arange(1, len(t) + 1))

    fila_v = {"rango": 0, "lugar": f"Victoria (Marte) · {REFERENCIA['conjunto']}", "cuerpo": "Marte",
              "tipo": "referencia", "region": REGION_MARTE, "zona": REFERENCIA["conjunto"],
              **{k: ref[k] for k in METRICAS}, "similitud_terreno": 1.0,
              "base": f"píxeles de {REFERENCIA['resolucion_m']} m (DTM HiRISE promediado)",
              "nota": "referencia; relieve_m de una escena de "
                      f"{ref['area_km2']:.0f} km² sin el cráter", "rugosidad_piso_dem_m": np.nan}
    tabla = pd.concat([pd.DataFrame([fila_v]), t], ignore_index=True)
    orden = (["rango", "lugar", "cuerpo", "tipo", "region", "zona", "similitud_terreno"] + METRICAS
             + [f"dif_{k}" for k in METRICAS] + [f"difrel_{k}" for k in PESOS]
             + ["rugosidad_piso_dem_m", "base", "nota"])
    tabla = tabla[orden]
    tabla.round(4).to_csv(out / "comparacion_terreno_victoria.csv", index=False, encoding="utf-8-sig")

    figura_ranking(t, out)
    figura_histogramas(t, hists, dist_v, out / "histogramas_terreno.png")
    por_lugar(t, ref, hists, dist_v, Path(salida_lugares))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 45)
    print("\n" + tabla[["rango", "lugar", "similitud_terreno"] + METRICAS[:6] + ["rugosidad_piso_dem_m", "base"]]
          .round(3).to_string(index=False))
    print(f"\nArchivos en {out.resolve()}")


def por_lugar(t, ref, hists, dist_v, base):
    """Una carpeta por región de la Tierra: Victoria vs esa región (y sus zonas)."""
    for region, g in t.groupby("region", sort=False):
        out = base / f"{REGION_MARTE}_vs_{region}"
        out.mkdir(parents=True, exist_ok=True)
        g = g.sort_values("tipo", key=lambda s: s.map({"región": 0, "zona": 1}))
        nombres = [r["lugar"].replace(" · ", "_").replace(" ", "_") for _, r in g.iterrows()]
        filas = []
        for k in METRICAS:
            f = {"metrica": k, "descripcion": DESCRIPCION[k], "victoria": ref[k]}
            for n, (_, r) in zip(nombres, g.iterrows()):
                f[n] = r[k]
                f[f"diferencia_{n}"] = r[f"dif_{k}"]
                if k in PESOS:
                    f[f"diferencia_relativa_{n}"] = r[f"difrel_{k}"]
            f["peso_en_similitud"] = PESOS.get(k, 0.0)
            filas.append(f)
        filas.append({"metrica": "similitud_terreno", "descripcion": "similitud de terreno con Victoria (0 a 1)",
                      "victoria": 1.0, **{n: r["similitud_terreno"] for n, (_, r) in zip(nombres, g.iterrows())}})
        filas.append({"metrica": "rango_general", "descripcion": f"lugar en el ranking de {len(t)} regiones/zonas",
                      **{n: r["rango"] for n, (_, r) in zip(nombres, g.iterrows())}})
        filas.append({"metrica": "base", "descripcion": "de dónde salen los datos",
                      "victoria": f"píxeles de {REFERENCIA['resolucion_m']} m · {REFERENCIA['conjunto']}",
                      **{n: r["base"] for n, (_, r) in zip(nombres, g.iterrows())}})
        filas.append({"metrica": "nota", "descripcion": "advertencias",
                      **{n: r["nota"] for n, (_, r) in zip(nombres, g.iterrows())}})
        filas.append({"metrica": "rugosidad_piso_dem_m", "descripcion": "rugosidad del 1 % de celdas más planas "
                      "(piso de ruido aproximado del DEM)",
                      **{n: r["rugosidad_piso_dem_m"] for n, (_, r) in zip(nombres, g.iterrows())}})
        redondear = lambda v: round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v
        pd.DataFrame(filas).map(redondear).to_csv(out / f"comparacion_terreno_{REGION_MARTE}_vs_{region}.csv",
                                            index=False, encoding="utf-8-sig")
        figura_histogramas(g, hists, dist_v, out / f"histogramas_{REGION_MARTE}_vs_{region}.png",
                           titulo=f"Terreno: cráter Victoria (Marte) vs {region.replace('_', ' ').title()}")
        print(f"  {out}")


def figura_ranking(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(t) + 1.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    y = np.arange(len(t))[::-1]
    por_celda = t["base"].str.startswith("promedio")
    ax.barh(y, t["similitud_terreno"], height=0.6, color=np.where(por_celda, "#9ec5f4", "#2a78d6"),
            hatch=None)
    for yi, (_, r), pc in zip(y, t.iterrows(), por_celda):
        ax.text(r["similitud_terreno"] + 0.01, yi, f"{r['similitud_terreno']:.2f}" + ("  (por celda)" if pc else ""),
                va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [l.replace("_", " ") for l in t["lugar"]], color=INK)
    ax.set_xlim(0, 1.15); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Similitud de terreno con el cráter Victoria (0 = nada, 1 = igual)", color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=INK2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2a78d6"), plt.Rectangle((0, 0), 1, 1, color="#9ec5f4")],
              labels=["píxeles de ~30 m (misma base que Victoria)", "promedios por celda de 1 km"],
              loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False, fontsize=9)
    ax.set_title(f"Similitud de terreno con Victoria ({REFERENCIA['conjunto'].replace('_', ' ')}, "
                 f"{REFERENCIA['resolucion_m']} m)", loc="left", color=INK, fontsize=12, pad=28)
    fig.tight_layout()
    fig.savefig(out / "ranking_terreno.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def figura_histogramas(t, hists, dist_v, ruta, titulo=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bordes = {True: np.logspace(np.log10(0.05), np.log10(60), 55), False: np.logspace(np.log10(0.02), np.log10(60), 55)}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURF)
    for ax, es_pend, subtitulo, et in ((axs[0], True, "Pendiente", "pendiente (°, escala log)"),
                                     (axs[1], False, "Rugosidad 3×3", "rugosidad (m, escala log)")):
        ax.set_facecolor(SURF)
        b = bordes[es_pend]
        centros = np.sqrt(b[:-1] * b[1:])
        for i, (_, r) in enumerate(t.iterrows()):
            v, w = distribucion(hists[r["lugar"]], es_pend)
            h, _ = np.histogram(np.clip(v, b[0], b[-1]), b, weights=w)
            if h.sum() == 0:
                continue
            etiqueta = r["lugar"].replace("_", " ") + (" (promedio por celda 1 km)" if hists[r["lugar"]][0] == "celdas"
                                                       else " (píxeles)")
            ax.plot(centros, h / h.sum(), color=COLORES[i % len(COLORES)], lw=1.8,
                    ls="--" if hists[r["lugar"]][0] == "celdas" else "-", label=etiqueta)
        vv = dist_v["pendiente_deg" if es_pend else "rugosidad_m"]
        h, _ = np.histogram(np.clip(vv, b[0], b[-1]), b)
        ax.plot(centros, h / h.sum(), color=INK, lw=2.6, label="Victoria, Marte (píxeles de 30 m)")
        ax.set_xscale("log")
        ax.set_xlabel(et, color=INK2); ax.set_ylabel("fracción del terreno", color=INK2)
        ax.set_title(subtitulo, loc="left", color=INK, fontsize=12)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2)
    axs[1].legend(loc="upper right", fontsize=8, frameon=False)
    titulo = titulo or "Distribución del terreno: cráter Victoria vs regiones de la Tierra"
    fig.suptitle(titulo + f"  ·  Victoria: {REFERENCIA['conjunto'].replace('_', ' ')}, {REFERENCIA['resolucion_m']} m",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compara el terreno del cráter Victoria con las regiones de la Tierra")
    ap.add_argument("--carpeta", default="resultados_terreno")
    ap.add_argument("--salida-lugares", default="resultados_comparacion",
                    help="carpeta donde se crea una subcarpeta por cada comparación Marte–lugar")
    a = ap.parse_args()
    main(a.carpeta, a.salida_lugares)
