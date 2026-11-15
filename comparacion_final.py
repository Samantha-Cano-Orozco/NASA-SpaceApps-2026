"""
comparacion_final.py  —  Space Apps 2026 · Análogos de Marte (suelos)
Compara la huella mineral de Meridiani (Marte) con CADA región de la Tierra por separado
(resultados de minerologia2.py).

USO
    python comparacion_final.py                          # todas las regiones de ./resultados_tierra
    python comparacion_final.py --region pinacate        # solo una
    python comparacion_final.py --marte <carpeta> --tierra <carpeta>

    --marte   carpeta con huella_marte_final.csv (sale de huella_marte_crism.ipynb, rama master)
    --tierra  carpeta con una subcarpeta por región: <tierra>/<region>/celdas_<region>.csv, resumen_region_...

QUÉ VERIFICA
    Que los pesos de este script reproducen EXACTAMENTE la columna 'puntaje' de la región.

CÓMO SE LEE EL %
    Similitud = puntaje de minerologia2.py × 100 (suma ponderada de coincidencias con Marte).
    % coincidencia de un criterio = promedio de su puntaje en la zona (1 coincide, 0.5 parcial, 0 no).
    Puntos perdidos = peso × (1 − coincidencia) / suma de pesos × 100. Suman 100 − similitud.

SALIDAS (una carpeta por región: resultados_comparacion/<region>/)
    similitud_<region>.csv      similitud (%) por zona: región completa, zonas de control, top 10, mejor celda
    criterios_<region>.csv      por criterio y zona: objetivo de Marte, valor en la Tierra, % de coincidencia,
                                puntos perdidos
    composicion_<region>.csv    composición EMIT/ASTER de la región junto a la huella de Marte
    top10_<region>.csv          las 10 celdas más parecidas, con sus datos
    comparacion_<region>.png    figura
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Configuraciones conocidas de minerologia2.py (se detecta cuál produjo cada CSV)
VERSIONES = {
    "v1 (sin micas, QI por percentil, bonus sulfatos)": {
        "pesos": {"oxidos_Fe": 3, "cuarzo": 3, "basalto": 2, "carbonatos": 2, "arcillas_FeMg": 2,
                  "arcillas_Al": 1}, "bonus": 0.5},
    "v2 (micas peso 2, índice máfico ASTER, sin bonus)": {
        "pesos": {"oxidos_Fe": 3, "cuarzo": 3, "basalto": 2, "carbonatos": 2, "arcillas_FeMg": 2,
                  "arcillas_Al": 1, "micas_felsicas": 2}, "bonus": 0.0},
    "v3 (micas peso 1, índice máfico ASTER, sin bonus)": {
        "pesos": {"oxidos_Fe": 3, "cuarzo": 3, "basalto": 2, "carbonatos": 2, "arcillas_FeMg": 2,
                  "arcillas_Al": 1, "micas_felsicas": 1}, "bonus": 0.0},
}
# Zonas de control conocidas (lat0, lat1, lon0, lon1)
ZONAS_CONTROL = {
    "pinacate": {"lava_pinacate": (31.60, 31.95, -113.62, -113.30),
                 "dunas_altar": (31.95, 32.30, -114.60, -114.10)},
}
OBJETIVO_TXT = {"oxidos_Fe": "dominante", "cuarzo": "ausente", "basalto": "presente", "carbonatos": "ausente",
                "arcillas_FeMg": "ausente", "arcillas_Al": "traza", "micas_felsicas": "ausente"}
ETIQUETA = {"oxidos_Fe": "Óxidos de Fe", "cuarzo": "Cuarzo", "basalto": "Basalto", "carbonatos": "Carbonatos",
            "arcillas_FeMg": "Arcillas Fe/Mg", "arcillas_Al": "Arcillas Al", "micas_felsicas": "Micas félsicas"}


def detectar_version(c):
    for nombre, v in VERSIONES.items():
        if not all(f"s_{k}" in c for k in v["pesos"]):
            continue
        if any(f"s_{k}" in c for k in OBJETIVO_TXT if k not in v["pesos"]):
            continue
        rec = (sum(w * c[f"s_{k}"] for k, w in v["pesos"].items()) + c["bonus_sulfatos_Fe"]) \
              / (sum(v["pesos"].values()) + v["bonus"])
        if (rec - c["puntaje"]).abs().max() < 1e-3:
            return nombre, v
    raise SystemExit("Ninguna configuración conocida reproduce el puntaje del CSV; agrega la nueva a VERSIONES")


def valor_tierra(z, k):
    if k == "cuarzo":
        return f"QI {z['qi'].mean():.4f} ({100 * (z['s_cuarzo'] == 1).mean():.0f} % celdas sin cuarzo)"
    if k == "basalto":
        return f"MI {z['mi'].mean():.3f} ({100 * (z['s_basalto'] == 1).mean():.0f} % celdas basalto)"
    return f"{z[f'pct_{k}'].mean():.1f} % píxeles EMIT"


def marte_txt(huella):
    crism = huella[huella["bloque"] == "superficie_CRISM"].set_index("nombre")
    tes = huella[huella["bloque"] == "fondo_TES"].set_index("nombre")
    f = lambda df, i, c: float(df.loc[i, c])
    return {
        "lat / lon": "−2.05 / 354.5 °E (cráter Victoria)",
        "elev_media_m": "—",
        "celdas_validas": "4 escenas CRISM + mapas TES",
        "pct_oxidos_Fe": f"dominante (BD530_2 {f(crism, 'oxidos_Fe', 'intensidad'):.3f}); "
                         f"TES hematita {f(tes, 'hematita', 'abundancia_pct'):.1f} %",
        "pct_oxidos_finos_de_los_oxidos": "CRISM ve óxidos finos; TES ve hematita gruesa (gris)",
        "pct_carbonatos": "ausente (0 %)",
        "pct_arcillas_FeMg": f"ausente ({f(crism, 'arcillas_FeMg', 'pct_max_escena'):.3f} %)",
        "pct_arcillas_Al": f"traza ({f(crism, 'arcillas_Al', 'pct_max_escena'):.2f} %, 1 escena)",
        "pct_micas_felsicas": "ausente (supuesto: Meridiani es basáltico)",
        "pct_basalto_EMIT": "CRISM piroxeno/olivino ausentes en superficie",
        "pct_sulfatos_Fe": f"CRISM ausente ({f(crism, 'sulfatos_polihidrat', 'pct_max_escena'):.2f} %); "
                           f"jarosita in situ (Opportunity)",
        "pct_yeso_sulfatos": "ausente",
        "indice_cuarzo_medio": f"TES cuarzo {f(tes, 'cuarzo', 'abundancia_pct'):.0f} % (no detectado)",
        "indice_mafico_medio": f"TES plagioclasa {f(tes, 'plagioclasa', 'abundancia_pct'):.1f} %, "
                               f"piroxeno {f(tes, 'piroxeno_altoCa', 'abundancia_pct'):.1f} %",
        "pct_celdas_basalto_aster": "basáltico",
    }


def comparar(region, dir_tierra, huella, salida):
    d = Path(dir_tierra) / region
    out = Path(salida) / region
    out.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(d / f"celdas_{region}.csv").sort_values("rango")
    res = pd.read_csv(d / f"resumen_region_{region}.csv", index_col=0)["region_completa"]
    nombre_v, v = detectar_version(c)
    pesos, suma = v["pesos"], sum(v["pesos"].values()) + v["bonus"]

    zonas = {"region_completa": c}
    for z, (la0, la1, lo0, lo1) in ZONAS_CONTROL.get(region, {}).items():
        zonas[z] = c[c["lat_centro"].between(la0, la1) & c["lon_centro"].between(lo0, lo1)]
    zonas["top10"] = c.head(10)
    zonas["mejor_celda"] = c.head(1)

    total, criterios = [], []
    for zona, z in zonas.items():
        total.append({"zona": zona, "n_celdas": len(z),
                      "similitud_pct": 100 * z["puntaje"].mean(),
                      "similitud_mediana_pct": 100 * z["puntaje"].median(),
                      "pct_celdas_sim_ge_80": 100 * (z["puntaje"] >= 0.8).mean(),
                      "lat": z["lat_centro"].mean(), "lon": z["lon_centro"].mean(),
                      "elev_mediana_m": z["elev_media_m"].median()})
        for k, w in pesos.items():
            s = z[f"s_{k}"].mean()
            criterios.append({"zona": zona, "criterio": k, "peso": w, "objetivo_marte": OBJETIVO_TXT[k],
                              "valor_tierra": valor_tierra(z, k), "coincidencia_pct": 100 * s,
                              "puntos_perdidos": 100 * w * (1 - s) / suma})
    total = pd.DataFrame(total).round(2)
    criterios = pd.DataFrame(criterios).round(2)
    for zona, g in criterios.groupby("zona"):
        sim = total.loc[total["zona"] == zona, "similitud_pct"].item()
        assert abs(g["puntos_perdidos"].sum() + sim - 100) < 0.1, zona

    ox = pd.read_csv(d / f"mapeo_emit_grupos_{region}.csv")
    ox = ox[ox["grupo"] == "oxidos_Fe"]
    finos = ox[ox["nombre_emit"].str.lower().str.contains("nano|nhemat|fe-hydrox|thin_film|ferrihyd")]
    tierra = {
        "lat / lon": f"{c['lat_centro'].min():.1f}…{c['lat_centro'].max():.1f} / "
                     f"{c['lon_centro'].min():.1f}…{c['lon_centro'].max():.1f}",
        "elev_media_m": res.get("elev_media_m"), "celdas_validas": res.get("celdas_validas"),
        "pct_oxidos_Fe": res.get("pct_oxidos_Fe"),
        "pct_oxidos_finos_de_los_oxidos": round(100 * finos["detecciones"].sum() / ox["detecciones"].sum(), 1),
        "pct_carbonatos": res.get("pct_carbonatos"), "pct_arcillas_FeMg": res.get("pct_arcillas_FeMg"),
        "pct_arcillas_Al": res.get("pct_arcillas_Al"), "pct_micas_felsicas": res.get("pct_micas_felsicas"),
        "pct_basalto_EMIT": res.get("pct_basalto"), "pct_sulfatos_Fe": res.get("pct_sulfatos_Fe"),
        "pct_yeso_sulfatos": res.get("pct_yeso_sulfatos"),
        "indice_cuarzo_medio": res.get("indice_cuarzo_medio"),
        "indice_mafico_medio": res.get("indice_mafico_medio"),
        "pct_celdas_basalto_aster": res.get("pct_celdas_basalto_aster"),
    }
    comp = pd.DataFrame({"marte_meridiani": pd.Series(marte_txt(huella)), region: pd.Series(tierra)})
    comp.index.name = "dato"
    top = c.head(10)[["rango", "lat_centro", "lon_centro", "puntaje", "pct_oxidos_Fe", "qi", "mi",
                      "pct_carbonatos", "pct_arcillas_FeMg", "pct_arcillas_Al", "pct_micas_felsicas",
                      "pct_sulfatos_Fe", "pct_agua", "elev_media_m", "pendiente_media_deg"]].round(3)

    total.to_csv(out / f"similitud_{region}.csv", index=False)
    criterios.to_csv(out / f"criterios_{region}.csv", index=False)
    comp.to_csv(out / f"composicion_{region}.csv")
    top.to_csv(out / f"top10_{region}.csv", index=False)
    figura(region, total, criterios, out)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(f"\n{'#' * 90}\n{region.upper()} vs MERIDIANI · {len(c):,} celdas · {nombre_v} · puntaje verificado")
    print("\n--- Similitud (%) ---")
    print(total.to_string(index=False))
    print("\n--- Por criterio: coincidencia % / puntos perdidos ---")
    for zona in zonas:
        g = criterios[criterios["zona"] == zona].set_index("criterio")
        print(f"  {zona}: " + " | ".join(f"{k} {g.loc[k, 'coincidencia_pct']:.0f}%/−{g.loc[k, 'puntos_perdidos']:.1f}"
                                         for k in pesos))
    print("\n--- Valor en la Tierra (región completa) vs objetivo de Marte ---")
    g = criterios[criterios["zona"] == "region_completa"]
    print(g[["criterio", "peso", "objetivo_marte", "valor_tierra", "coincidencia_pct", "puntos_perdidos"]]
          .to_string(index=False))
    print("\n--- Composición ---")
    print(comp.to_string())
    print("\n--- Top 10 ---")
    print(top.to_string(index=False))
    print(f"\nArchivos en {out.resolve()}")


def figura(region, total, criterios, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"
    AZUL, AZUL_SUAVE = "#2a78d6", "#9ec5f4"
    seq = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
    nombre = lambda s: s.replace("_", " ")
    titulo = nombre(region).title()

    zonas = list(total["zona"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.4]}, facecolor=SURF)
    for ax in (a1, a2):
        ax.set_facecolor(SURF)

    y = np.arange(len(zonas))[::-1]
    a1.barh(y, total["similitud_pct"], height=0.55,
            color=[AZUL_SUAVE if z.startswith("dunas") else AZUL for z in zonas])
    for yi, (_, r) in zip(y, total.iterrows()):
        a1.text(r["similitud_pct"] + 1, yi, f"{r['similitud_pct']:.1f} %", va="center", color=INK, fontsize=10)
    a1.set_yticks(y, [f"{nombre(z)}\n{int(n):,} celda{'s' if n != 1 else ''}"
                      for z, n in zip(zonas, total["n_celdas"])], color=INK)
    a1.set_xlim(0, 108); a1.set_xticks(range(0, 101, 20))
    a1.set_xlabel("Similitud con Meridiani (%)", color=INK2)
    a1.set_title("Similitud total por zona", color=INK, loc="left", fontsize=12)
    a1.xaxis.grid(True, color=GRID, lw=0.8); a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.spines["left"].set_color(MUTED); a1.spines["bottom"].set_color(MUTED); a1.tick_params(colors=INK2)

    crits = list(dict.fromkeys(criterios["criterio"]))
    m = criterios.pivot(index="criterio", columns="zona", values="coincidencia_pct").loc[crits, zonas]
    im = a2.imshow(m.values, cmap=seq, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            a2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10, color="white" if v >= 60 else INK)
    pesos = criterios.drop_duplicates("criterio").set_index("criterio")["peso"]
    a2.set_yticks(range(len(crits)), [f"{ETIQUETA[k]} · Marte: {OBJETIVO_TXT[k]} · peso {pesos[k]:g}"
                                      for k in crits], color=INK)
    a2.set_xticks(range(len(zonas)), [nombre(z).replace(" ", "\n") for z in zonas], color=INK2)
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_xticks(np.arange(-.5, len(zonas)), minor=True); a2.set_yticks(np.arange(-.5, len(crits)), minor=True)
    a2.grid(which="minor", color=SURF, lw=2); a2.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=a2, fraction=0.035, pad=0.02)
    cb.set_label("% de celdas que coinciden con Marte", color=INK2); cb.outline.set_visible(False)
    a2.set_title("Coincidencia por criterio (%)", color=INK, loc="left", fontsize=12)

    fig.suptitle(f"{titulo} vs Meridiani Planum (Marte): comparación mineralógica", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"comparacion_{region}.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compara Meridiani (Marte) con cada región de la Tierra por separado")
    ap.add_argument("--marte", default="resultados", help="carpeta con huella_marte_final.csv")
    ap.add_argument("--tierra", default="resultados_tierra", help="carpeta con una subcarpeta por región")
    ap.add_argument("--region", default=None, help="solo esta región (por defecto, todas)")
    ap.add_argument("--salida", default="resultados_comparacion")
    a = ap.parse_args()

    huella = pd.read_csv(Path(a.marte) / "huella_marte_final.csv")
    regiones = [a.region] if a.region else sorted(
        p.name for p in Path(a.tierra).iterdir() if p.is_dir() and (p / f"celdas_{p.name}.csv").exists())
    if not regiones:
        raise SystemExit(f"No encontré regiones en {Path(a.tierra).resolve()} (<region>/celdas_<region>.csv)")
    for r in regiones:
        comparar(r, a.tierra, huella, a.salida)
