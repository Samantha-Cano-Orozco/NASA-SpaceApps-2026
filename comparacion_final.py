"""
comparacion_final.py  —  Space Apps 2026 · Análogos de Marte (suelos)
Comparación final: huella mineral de Meridiani (Marte) contra varias regiones de la Tierra
(resultados de minerologia2.py).

USO
    python comparacion_final.py --marte <carpeta con huella_marte_final.csv>

    Las regiones se definen en REGIONES (carpeta de resultados + zonas de control opcionales).

DOS PUNTAJES
    entregado   = columna 'puntaje' de celdas_<region>.csv, tal como la calculó minerologia2.py.
                  Se verifica que los pesos de este script la reproducen exactamente.
    armonizado  = mismo cálculo, pero con el umbral FIJO de cuarzo (QI ≤ 1.010 sin cuarzo, ≤ 1.020
                  parcial) y sin bonus de sulfatos. Sirve para comparar regiones entre sí: el umbral por
                  percentil de la versión vieja depende de cada región.

SALIDAS (resultados_comparacion/final/)
    comparacion_final_total.csv       similitud (%) por región y zona, entregada y armonizada
    comparacion_final_criterios.csv   coincidencia (%) y puntos perdidos por criterio, región y zona
    comparacion_final_composicion.csv composición EMIT/ASTER de cada región junto a la huella de Marte
    comparacion_final.png             figura resumen
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONES = {
    "pinacate": {"carpeta": "resultados_tierra_Pinacate",
                 "control": {"lava_pinacate": (31.60, 31.95, -113.62, -113.30),
                             "dunas_altar": (31.95, 32.30, -114.60, -114.10)}},
    "black_point": {"carpeta": "resultados_tierra_black_point", "control": {}},
}

# Configuraciones conocidas de minerologia2.py (se detecta cuál produjo cada CSV)
VERSIONES = {
    "vieja (sin micas, QI por percentil, bonus sulfatos)": {
        "pesos": {"oxidos_Fe": 3, "cuarzo": 3, "basalto": 2, "carbonatos": 2, "arcillas_FeMg": 2,
                  "arcillas_Al": 1},
        "bonus": 0.5},
    "nueva (micas, índice máfico ASTER, sin bonus)": {
        "pesos": {"oxidos_Fe": 3, "cuarzo": 3, "basalto": 2, "carbonatos": 2, "arcillas_FeMg": 2,
                  "arcillas_Al": 1, "micas_felsicas": 2},
        "bonus": 0.0},
}
QI_SIN_CUARZO, QI_CUARZO_PARCIAL = 1.010, 1.020
OBJETIVO_TXT = {"oxidos_Fe": "dominante", "cuarzo": "ausente", "basalto": "presente", "carbonatos": "ausente",
                "arcillas_FeMg": "ausente", "arcillas_Al": "traza", "micas_felsicas": "ausente"}
ETIQUETA = {"oxidos_Fe": "Óxidos de Fe", "cuarzo": "Cuarzo", "basalto": "Basalto", "carbonatos": "Carbonatos",
            "arcillas_FeMg": "Arcillas Fe/Mg", "arcillas_Al": "Arcillas Al", "micas_felsicas": "Micas félsicas"}


def detectar_version(c):
    for nombre, v in VERSIONES.items():
        if not all(f"s_{k}" in c for k in v["pesos"]):
            continue
        rec = (sum(w * c[f"s_{k}"] for k, w in v["pesos"].items()) + c["bonus_sulfatos_Fe"]) \
              / (sum(v["pesos"].values()) + v["bonus"])
        if (rec - c["puntaje"]).abs().max() < 1e-3:
            return nombre, v
    raise SystemExit("Ninguna configuración conocida reproduce el puntaje del CSV")


def cargar(region, cfg):
    d = Path(cfg["carpeta"])
    c = pd.read_csv(d / f"celdas_{region}.csv")
    resumen = pd.read_csv(d / f"resumen_region_{region}.csv", index_col=0)["region_completa"]
    nombre_v, v = detectar_version(c)
    pesos = v["pesos"]
    # puntaje armonizado: cuarzo con umbral fijo, sin bonus
    c["s_cuarzo_fijo"] = np.where(c["qi"] <= QI_SIN_CUARZO, 1.0, np.where(c["qi"] <= QI_CUARZO_PARCIAL, 0.5, 0.0))
    c["puntaje_arm"] = sum(w * c["s_cuarzo_fijo" if k == "cuarzo" else f"s_{k}"] for k, w in pesos.items()) \
                       / sum(pesos.values())
    orden_arm = c.sort_values(["puntaje_arm", "qi", "pct_oxidos_Fe"], ascending=[False, True, False])
    orden_ent = c.sort_values(["puntaje", "pct_oxidos_Fe", "qi"], ascending=[False, False, True])
    zonas = {"region_completa": (c, c)}
    for z, (la0, la1, lo0, lo1) in cfg["control"].items():
        m = c["lat_centro"].between(la0, la1) & c["lon_centro"].between(lo0, lo1)
        zonas[z] = (c[m], c[m])
    zonas["top10"] = (orden_ent.head(10), orden_arm.head(10))   # el top 10 depende del puntaje
    zonas["mejor_celda"] = (orden_ent.head(1), orden_arm.head(1))
    return c, resumen, nombre_v, pesos, zonas


def main(dir_marte, salida):
    out = Path(salida) / "final"
    out.mkdir(parents=True, exist_ok=True)
    huella = pd.read_csv(Path(dir_marte) / "huella_marte_final.csv")

    total, criterios, versiones, composicion = [], [], {}, {}
    for region, cfg in REGIONES.items():
        c, resumen, nombre_v, pesos, zonas = cargar(region, cfg)
        versiones[region] = nombre_v
        suma = sum(pesos.values())
        print(f"{region}: {len(c):,} celdas · versión {nombre_v} · puntaje verificado")
        for zona, (ze, za) in zonas.items():
            total.append({"region": region, "zona": zona, "n_celdas": len(za),
                          "similitud_entregada_pct": 100 * ze["puntaje"].mean(),
                          "similitud_armonizada_pct": 100 * za["puntaje_arm"].mean(),
                          "pct_celdas_arm_ge_80": 100 * (za["puntaje_arm"] >= 0.8).mean(),
                          "lat": za["lat_centro"].mean(), "lon": za["lon_centro"].mean()})
            for k, w in pesos.items():
                s = za["s_cuarzo_fijo" if k == "cuarzo" else f"s_{k}"].mean()
                criterios.append({"region": region, "zona": zona, "criterio": k, "peso": w,
                                  "objetivo_marte": OBJETIVO_TXT[k], "coincidencia_pct": 100 * s,
                                  "puntos_perdidos": 100 * w * (1 - s) / suma,
                                  "valor_tierra": (f"QI {za['qi'].mean():.4f}" if k == "cuarzo"
                                                   else f"{za[f'pct_{k}'].mean():.1f} % píxeles EMIT")})
        composicion[region] = {
            "oxidos_Fe": resumen.get("pct_oxidos_Fe"), "carbonatos": resumen.get("pct_carbonatos"),
            "arcillas_FeMg": resumen.get("pct_arcillas_FeMg"), "arcillas_Al": resumen.get("pct_arcillas_Al"),
            "basalto": resumen.get("pct_basalto"), "sulfatos_Fe": resumen.get("pct_sulfatos_Fe"),
            "yeso_sulfatos": resumen.get("pct_yeso_sulfatos"), "otros": resumen.get("pct_otros"),
            "indice_cuarzo_medio": resumen.get("indice_cuarzo_medio"),
            "umbral_cuarzo_original": resumen.get("umbral_cuarzo_usado"),
            "elev_media_m": resumen.get("elev_media_m"),
            "lat": f"{c['lat_centro'].min():.2f} a {c['lat_centro'].max():.2f}",
            "lon": f"{c['lon_centro'].min():.2f} a {c['lon_centro'].max():.2f}",
        }

    if len(set(versiones.values())) > 1:
        raise SystemExit(f"Las regiones vienen de versiones distintas de minerologia2.py: {versiones}. "
                         f"Vuelve a correrlas con la misma versión antes de compararlas.")

    total = pd.DataFrame(total).round(2)
    criterios = pd.DataFrame(criterios).round(2)
    for (r, z), g in criterios.groupby(["region", "zona"]):
        sim = total[(total["region"] == r) & (total["zona"] == z)]["similitud_armonizada_pct"].item()
        assert abs(g["puntos_perdidos"].sum() + sim - 100) < 0.1, (r, z)

    # Composición junto a la huella de Marte
    crism = huella[huella["bloque"] == "superficie_CRISM"].set_index("nombre")
    marte = {"oxidos_Fe": "dominante (BD530_2 0.181)", "carbonatos": "ausente (0 %)",
             "arcillas_FeMg": f"ausente ({float(crism.loc['arcillas_FeMg', 'pct_max_escena']):.3f} %)",
             "arcillas_Al": f"traza ({float(crism.loc['arcillas_Al', 'pct_max_escena']):.2f} %)",
             "basalto": "presente (TES plag. 8.8 %, piroxeno 5.2 %)",
             "sulfatos_Fe": f"ausente en CRISM ({float(crism.loc['sulfatos_polihidrat', 'pct_max_escena']):.2f} %); "
                            f"jarosita in situ (Opportunity)",
             "yeso_sulfatos": "ausente", "otros": "—",
             "indice_cuarzo_medio": "cuarzo no detectado (TES 0 %)", "umbral_cuarzo_original": "—",
             "elev_media_m": "—", "lat": "−2.05 (Victoria)", "lon": "354.5 °E"}
    comp = pd.DataFrame(composicion)
    comp.insert(0, "marte_meridiani", pd.Series(marte))
    comp.index.name = "dato"

    total.to_csv(out / "comparacion_final_total.csv", index=False)
    criterios.to_csv(out / "comparacion_final_criterios.csv", index=False)
    comp.to_csv(out / "comparacion_final_composicion.csv")
    figura(total, criterios, out)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print("\n=== SIMILITUD CON MERIDIANI (%) ===")
    print(total.drop(columns=["lat", "lon"]).to_string(index=False))
    print("\n=== COINCIDENCIA POR CRITERIO (%, puntaje armonizado) ===")
    print(criterios.pivot_table(index="criterio", columns=["region", "zona"], values="coincidencia_pct")
          .round(1).to_string())
    print("\n=== PUNTOS PERDIDOS POR CRITERIO (región completa, armonizado) ===")
    print(criterios[criterios["zona"] == "region_completa"]
          .pivot(index="criterio", columns="region", values="puntos_perdidos").round(1).to_string())
    print("\n=== COMPOSICIÓN (EMIT: % de píxeles con detección; ASTER: índice de cuarzo) ===")
    print(comp.to_string())
    print("\n=== TOP 10 (armonizado) ===")
    for region, cfg in REGIONES.items():
        _, _, _, _, zonas = cargar(region, cfg)
        t = zonas["top10"][1]
        print(f"  {region}: lat {t['lat_centro'].min():.2f}–{t['lat_centro'].max():.2f}, "
              f"lon {t['lon_centro'].min():.2f}–{t['lon_centro'].max():.2f}, elev mediana "
              f"{t['elev_media_m'].median():.0f} m, agua media {t['pct_agua'].mean():.1f} %")
    print(f"\nArchivos guardados en {out.resolve()}")


def figura(total, criterios, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"
    COLOR = {"pinacate": "#2a78d6", "black_point": "#eb6834"}     # categórico, orden fijo
    seq = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])

    zonas_fig = ["region_completa", "lava_pinacate", "dunas_altar", "top10", "mejor_celda"]
    t = total[total["zona"].isin(zonas_fig)].copy()
    t["orden"] = t["zona"].map({z: i for i, z in enumerate(zonas_fig)})
    t = t.sort_values(["region", "orden"])
    etiquetas = [f"{r.replace('_', ' ').title()} · {z.replace('_', ' ')}" for r, z in zip(t["region"], t["zona"])]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1, 1.25]},
                                 facecolor=SURF)
    for ax in (a1, a2):
        ax.set_facecolor(SURF)

    y = np.arange(len(t))[::-1]
    a1.barh(y, t["similitud_armonizada_pct"], color=[COLOR[r] for r in t["region"]], height=0.6)
    for yi, (_, r) in zip(y, t.iterrows()):
        a1.text(r["similitud_armonizada_pct"] + 1, yi,
                f"{r['similitud_armonizada_pct']:.1f} %  (entregado {r['similitud_entregada_pct']:.1f})",
                va="center", color=INK, fontsize=9)
    a1.set_yticks(y, etiquetas, color=INK, fontsize=9)
    a1.set_xlim(0, 125); a1.set_xticks(range(0, 101, 20))
    a1.set_xlabel("Similitud con Meridiani, armonizada (%)", color=INK2)
    a1.set_title("Similitud total", color=INK, loc="left", fontsize=12)
    a1.xaxis.grid(True, color=GRID, lw=0.8); a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.spines["left"].set_color(MUTED); a1.spines["bottom"].set_color(MUTED); a1.tick_params(colors=INK2)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[r]) for r in COLOR],
              labels=[r.replace("_", " ").title() for r in COLOR], loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)

    cols = [("pinacate", "region_completa"), ("pinacate", "lava_pinacate"), ("pinacate", "top10"),
            ("black_point", "region_completa"), ("black_point", "top10")]
    crits = list(dict.fromkeys(criterios["criterio"]))
    m = np.array([[criterios[(criterios["region"] == r) & (criterios["zona"] == z) & (criterios["criterio"] == k)]
                   ["coincidencia_pct"].item() for (r, z) in cols] for k in crits])
    im = a2.imshow(m, cmap=seq, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a2.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] >= 60 else INK)
    pesos = criterios.drop_duplicates("criterio").set_index("criterio")["peso"]
    a2.set_yticks(range(len(crits)), [f"{ETIQUETA[k]} · Marte: {OBJETIVO_TXT[k]} · peso {pesos[k]}"
                                      for k in crits], color=INK)
    a2.set_xticks(range(len(cols)), [f"{r.replace('_', ' ').title()}\n{z.replace('_', ' ')}" for r, z in cols],
                  color=INK2, fontsize=9)
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_xticks(np.arange(-.5, len(cols)), minor=True); a2.set_yticks(np.arange(-.5, len(crits)), minor=True)
    a2.grid(which="minor", color=SURF, lw=2); a2.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=a2, fraction=0.035, pad=0.02)
    cb.set_label("% de celdas que coinciden con Marte", color=INK2); cb.outline.set_visible(False)
    a2.set_title("Coincidencia por criterio (%)", color=INK, loc="left", fontsize=12)

    fig.suptitle("Comparación final: suelos de la Tierra vs Meridiani Planum (Marte)", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, "Armonizado = umbral fijo de cuarzo (QI ≤ 1.010) y sin bonus de sulfatos. "
             "Entregado = puntaje original de cada región.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "comparacion_final.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comparación final Marte (Meridiani) vs regiones de la Tierra")
    ap.add_argument("--marte", default="resultados", help="carpeta con huella_marte_final.csv")
    ap.add_argument("--salida", default="resultados_comparacion")
    a = ap.parse_args()
    main(a.marte, a.salida)
