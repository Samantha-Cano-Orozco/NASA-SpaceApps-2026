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
    top10_<region>.csv          las 10 celdas más parecidas, con sus datos y avisos automáticos
    comparacion_<region>.png    figura
    Y un libro de Excel con todo junto: resultados_comparacion/comparacion_final.xlsx
        Resumen   las regiones lado a lado (similitud, criterios, composición)
        <Región>  una hoja por región con sus datos individuales
        Marte     la huella de Meridiani usada
        Notas     cómo leer los datos y de dónde salen
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
# Observaciones de la revisión manual de cada región (se copian a su hoja del Excel)
NOTAS_MANUALES = {
    "pinacate": [
        "Dato principal para presentar: campo de lava (lava_pinacate). Las dunas son el contraste.",
        "7 de las 10 mejores celdas están entre San Luis Río Colorado y Yuma (~30 m, terreno plano, 66–100 % "
        "micas): zona de cultivos de riego del valle del río Colorado. Verificar en imagen satelital.",
    ],
    "black_point": [
        "Los datos cubren el cuadro n36w115 (lago Mead / Las Vegas, Nevada), NO el derrame de lava Black Point "
        "de Arizona (~35.7°N, 111.4°O). Confirmar con el equipo qué región se quería.",
        "Carbonatos en ~78 % de los píxeles: calizas de la zona. Es la mayor diferencia con Marte.",
        "La mitad del top 10 cae en la ciudad de Henderson (terreno plano); la otra mitad en las River Mountains "
        "(volcánicas, con pendiente). Verificar en imagen satelital.",
    ],
    "desierto_diablo": [
        "9 de las 10 mejores celdas son una sola planicie a ~3 471 m con índice máfico > 1 (inusual para "
        "basalto): podría ser costra salina o sedimento fino. Verificar en imagen satelital.",
    ],
}


def avisos_celda(r):
    """Señales automáticas para revisar una celda del top 10."""
    a = []
    if r["pct_micas_felsicas"] > 50:
        a.append("micas > 50 %")
    if r["mi"] > 1.0:
        a.append("MI > 1 (inusual para basalto)")
    if r["elev_media_m"] < 50 and r["pendiente_media_deg"] < 3:
        a.append("planicie baja (< 50 m)")
    if r["pct_sulfatos_Fe"] > 10:
        a.append("sulfatos de Fe > 10 %")
    if r["pct_oxidos_Fe"] < 40:
        a.append("óxidos de Fe < 40 %")
    return "; ".join(a)


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
    top["avisos"] = top.apply(avisos_celda, axis=1)

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
    return {"total": total, "criterios": criterios, "comp": comp, "top": top, "version": nombre_v,
            "n_celdas": len(c), "pesos": pesos}


DATO_TXT = {
    "lat / lon": "Ubicación (lat / lon)", "elev_media_m": "Elevación media (m)", "celdas_validas": "Celdas válidas",
    "pct_oxidos_Fe": "Óxidos de Fe (% píxeles)", "pct_oxidos_finos_de_los_oxidos": "Óxidos finos (% de los óxidos)",
    "pct_carbonatos": "Carbonatos (% píxeles)", "pct_arcillas_FeMg": "Arcillas Fe/Mg (% píxeles)",
    "pct_arcillas_Al": "Arcillas Al (% píxeles)", "pct_micas_felsicas": "Micas félsicas (% píxeles)",
    "pct_basalto_EMIT": "Basalto según EMIT (% píxeles)", "pct_sulfatos_Fe": "Sulfatos de Fe (% píxeles)",
    "pct_yeso_sulfatos": "Yeso y sulfatos (% píxeles)", "indice_cuarzo_medio": "Índice de cuarzo medio (ASTER)",
    "indice_mafico_medio": "Índice máfico medio (ASTER)",
    "pct_celdas_basalto_aster": "Celdas basálticas según ASTER (%)",
}


def numero(v):
    """'63.1052' -> 63.1052 ; texto -> texto."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def escribir_excel(res, huella, ruta):
    """Libro con una hoja de resumen, una por región, la huella de Marte y notas."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter as L

    FUENTE = "Arial"
    f_tit = Font(name=FUENTE, size=14, bold=True)
    f_sub = Font(name=FUENTE, size=11, bold=True)
    f_head = Font(name=FUENTE, size=10, bold=True, color="FFFFFF")
    f_txt = Font(name=FUENTE, size=10)
    f_dato = Font(name=FUENTE, size=10, color="0000FF")      # azul: dato calculado en Python desde los CSV
    f_form = Font(name=FUENTE, size=10, color="000000")      # negro: fórmula de Excel
    f_nota = Font(name=FUENTE, size=9, italic=True, color="52514E")
    relleno_head = PatternFill("solid", fgColor="1C5CAB")
    relleno_total = PatternFill("solid", fgColor="E6EEF8")
    borde = Border(bottom=Side(style="thin", color="D0CFCA"))
    PCT, NUM1, NUM3, NUM4 = "0.0%", "0.0", "0.000", "0.0000"
    nombre = lambda s: s.replace("_", " ")

    def encabezado(ws, fila, col, textos):
        for j, t in enumerate(textos):
            cel = ws.cell(fila, col + j, t)
            cel.font, cel.fill = f_head, relleno_head
            cel.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.row_dimensions[fila].height = 30

    def celda(ws, fila, col, valor, fmt=None, fuente=None):
        v = numero(valor) if not (isinstance(valor, str) and valor.startswith("=")) else valor
        cel = ws.cell(fila, col, v)
        es_form = isinstance(v, str) and v.startswith("=")
        cel.font = fuente or (f_form if es_form else (f_dato if isinstance(v, (int, float)) else f_txt))
        if fmt and (es_form or isinstance(v, (int, float))):
            cel.number_format = fmt
        cel.border = borde
        return cel

    def titulo(ws, fila, texto, fuente=f_sub):
        ws.cell(fila, 1, texto).font = fuente
        return fila + 1

    def tabla(ws, fila, df, formatos=None):
        formatos = formatos or {}
        encabezado(ws, fila, 1, list(df.columns))
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            for j, col in enumerate(df.columns, start=1):
                celda(ws, fila + i, j, r[col], formatos.get(col))
        return fila + len(df) + 2

    def anchos(ws, anchos_col):
        for j, a in enumerate(anchos_col, start=1):
            ws.column_dimensions[L(j)].width = a

    wb = Workbook()
    regiones = list(res)

    # ---------------- RESUMEN ----------------
    ws = wb.active
    ws.title = "Resumen"
    f = titulo(ws, 1, "Comparación mineralógica: regiones de la Tierra vs Meridiani Planum (Marte)", f_tit)
    ws.cell(f, 1, "Cada región se compara con Marte por separado. Detalle en la hoja de cada región; "
                  "cómo leer los datos en 'Notas'.").font = f_nota
    f += 2

    f = titulo(ws, f, "1. Similitud con Meridiani por región y zona")
    sim = pd.concat([r["total"].assign(region=nombre(k).title()) for k, r in res.items()])
    sim = sim[["region", "zona", "n_celdas", "similitud_pct", "similitud_mediana_pct", "pct_celdas_sim_ge_80",
               "lat", "lon", "elev_mediana_m"]].copy()
    for col in ("similitud_pct", "similitud_mediana_pct", "pct_celdas_sim_ge_80"):
        sim[col] = sim[col] / 100
    sim["zona"] = sim["zona"].map(nombre)
    sim.columns = ["Región", "Zona", "Celdas", "Similitud media", "Similitud mediana", "Celdas con similitud ≥ 80 %",
                   "Lat. media", "Lon. media", "Elevación mediana (m)"]
    f = tabla(ws, f, sim, {"Celdas": "#,##0", "Similitud media": PCT, "Similitud mediana": PCT,
                           "Celdas con similitud ≥ 80 %": PCT, "Lat. media": "0.00", "Lon. media": "0.00",
                           "Elevación mediana (m)": "#,##0"})

    f = titulo(ws, f, "2. Coincidencia por criterio (región completa) y puntos de similitud que se pierden")
    crits = list(res[regiones[0]]["pesos"])
    cab = ["Criterio", "Peso", "Objetivo en Marte"] + [f"{nombre(r).title()}\ncoincidencia" for r in regiones] \
          + [f"{nombre(r).title()}\npuntos perdidos" for r in regiones]
    encabezado(ws, f, 1, cab)
    f0 = f + 1
    nreg = len(regiones)
    for i, k in enumerate(crits):
        fila = f0 + i
        celda(ws, fila, 1, ETIQUETA[k]); celda(ws, fila, 2, res[regiones[0]]["pesos"][k], "0")
        celda(ws, fila, 3, OBJETIVO_TXT[k])
        for j, r in enumerate(regiones):
            g = res[r]["criterios"]
            v = g[(g["zona"] == "region_completa") & (g["criterio"] == k)]["coincidencia_pct"].item() / 100
            celda(ws, fila, 4 + j, v, PCT)
            cc = f"{L(4 + j)}{fila}"
            celda(ws, fila, 4 + nreg + j, f"=$B{fila}*(1-{cc})/SUM($B${f0}:$B${f0 + len(crits) - 1})", PCT)
    ft = f0 + len(crits)
    celda(ws, ft, 1, "Similitud = 100 % − Σ puntos perdidos", fuente=Font(name=FUENTE, size=10, bold=True))
    for j in range(nreg):
        col = L(4 + nreg + j)
        c = celda(ws, ft, 4 + nreg + j, f"=1-SUM({col}{f0}:{col}{ft - 1})", PCT)
        c.fill = relleno_total
    f = ft + 3

    f = titulo(ws, f, "3. Composición de cada región junto a la huella de Marte")
    comp = pd.concat([res[regiones[0]]["comp"][["marte_meridiani"]]]
                     + [res[r]["comp"][[r]] for r in regiones], axis=1).reset_index()
    comp.columns = ["Dato", "Marte (Meridiani)"] + [nombre(r).title() for r in regiones]
    comp["Dato"] = comp["Dato"].map(lambda x: DATO_TXT.get(x, x))
    f = tabla(ws, f, comp, {nombre(r).title(): "0.00##" for r in regiones})
    ws.cell(f - 1, 1, "Porcentajes de la Tierra = % de píxeles EMIT con detección del grupo; índices = ASTER.").font = f_nota
    anchos(ws, [38, 20, 44] + [18] * (2 * nreg))
    ws.freeze_panes = "A4"

    # ---------------- UNA HOJA POR REGIÓN ----------------
    for r in regiones:
        d = res[r]
        ws = wb.create_sheet(nombre(r).title()[:31])
        f = titulo(ws, 1, f"{nombre(r).title()} vs Meridiani Planum (Marte)", f_tit)
        for txt in (f"Celdas válidas de ~1 km: {d['n_celdas']:,}",
                    f"Versión de minerologia2.py: {d['version']} (puntaje verificado)",
                    f"Ubicación (lat / lon): {d['comp'].loc['lat / lon', r]}  ·  "
                    f"elevación media: {numero(d['comp'].loc['elev_media_m', r]):,.0f} m"):
            ws.cell(f, 1, txt).font = f_txt; f += 1
        f += 1

        f = titulo(ws, f, "1. Similitud por zona")
        t = d["total"].copy()
        for col in ("similitud_pct", "similitud_mediana_pct", "pct_celdas_sim_ge_80"):
            t[col] = t[col] / 100
        t["zona"] = t["zona"].map(nombre)
        t.columns = ["Zona", "Celdas", "Similitud media", "Similitud mediana", "Celdas con similitud ≥ 80 %",
                     "Lat. media", "Lon. media", "Elevación mediana (m)"]
        f = tabla(ws, f, t, {"Celdas": "#,##0", "Similitud media": PCT, "Similitud mediana": PCT,
                             "Celdas con similitud ≥ 80 %": PCT, "Lat. media": "0.00", "Lon. media": "0.00",
                             "Elevación mediana (m)": "#,##0"})

        f = titulo(ws, f, "2. Criterio por criterio: coincidencia con Marte y puntos de similitud que se pierden")
        zonas = list(d["total"]["zona"])
        g = d["criterios"]
        cab = ["Criterio", "Peso", "Objetivo en Marte", "Valor en la Tierra (región completa)"]
        for z in zonas:
            cab += [f"{nombre(z)}\ncoincidencia", f"{nombre(z)}\npuntos perdidos"]
        encabezado(ws, f, 1, cab)
        f0 = f + 1
        for i, k in enumerate(d["pesos"]):
            fila = f0 + i
            celda(ws, fila, 1, ETIQUETA[k]); celda(ws, fila, 2, d["pesos"][k], "0")
            celda(ws, fila, 3, OBJETIVO_TXT[k])
            celda(ws, fila, 4, g[(g["zona"] == "region_completa") & (g["criterio"] == k)]["valor_tierra"].item())
            for j, z in enumerate(zonas):
                cc = 5 + 2 * j
                v = g[(g["zona"] == z) & (g["criterio"] == k)]["coincidencia_pct"].item() / 100
                celda(ws, fila, cc, v, PCT)
                celda(ws, fila, cc + 1,
                      f"=$B{fila}*(1-{L(cc)}{fila})/SUM($B${f0}:$B${f0 + len(d['pesos']) - 1})", PCT)
        ft = f0 + len(d["pesos"])
        celda(ws, ft, 1, "Similitud = 100 % − Σ puntos perdidos", fuente=Font(name=FUENTE, size=10, bold=True))
        celda(ws, ft + 1, 1, "Similitud media (del CSV, para comparar)", fuente=f_nota)
        for j, z in enumerate(zonas):
            col = L(6 + 2 * j)
            c = celda(ws, ft, 6 + 2 * j, f"=1-SUM({col}{f0}:{col}{ft - 1})", PCT)
            c.fill = relleno_total
            celda(ws, ft + 1, 6 + 2 * j, d["total"].loc[d["total"]["zona"] == z, "similitud_pct"].item() / 100, PCT)
        f = ft + 3

        f = titulo(ws, f, "3. Composición: Marte vs región")
        cp = d["comp"].reset_index()
        cp.columns = ["Dato", "Marte (Meridiani)", nombre(r).title()]
        cp["Dato"] = cp["Dato"].map(lambda x: DATO_TXT.get(x, x))
        f = tabla(ws, f, cp, {nombre(r).title(): "0.00##"})

        f = titulo(ws, f, "4. Las 10 celdas más parecidas a Meridiani")
        tp = d["top"].copy()
        tp["puntaje"] = tp["puntaje"]
        tp.columns = ["Rango", "Lat.", "Lon.", "Similitud", "Óxidos de Fe (%)", "Índice de cuarzo (QI)",
                      "Índice máfico (MI)", "Carbonatos (%)", "Arcillas Fe/Mg (%)", "Arcillas Al (%)",
                      "Micas félsicas (%)", "Sulfatos de Fe (%)", "Agua (%)", "Elevación (m)", "Pendiente (°)",
                      "Avisos automáticos"]
        f_top = f
        f = tabla(ws, f, tp, {"Lat.": "0.000", "Lon.": "0.000", "Similitud": PCT, "Óxidos de Fe (%)": NUM1,
                              "Índice de cuarzo (QI)": NUM4, "Índice máfico (MI)": NUM3, "Carbonatos (%)": NUM1,
                              "Arcillas Fe/Mg (%)": NUM1, "Arcillas Al (%)": NUM1, "Micas félsicas (%)": NUM1,
                              "Sulfatos de Fe (%)": NUM1, "Agua (%)": NUM1, "Elevación (m)": "#,##0",
                              "Pendiente (°)": NUM1})
        ws.auto_filter.ref = f"A{f_top}:{L(len(tp.columns))}{f_top + len(tp)}"

        f = titulo(ws, f, "5. Observaciones de la revisión")
        for txt in NOTAS_MANUALES.get(r, ["Sin observaciones manuales; revisar el top 10 en imagen satelital."]):
            c = ws.cell(f, 1, f"• {txt}"); c.font = f_txt; f += 1
        anchos(ws, [40, 10, 18, 40] + [14] * 12)
        ws.freeze_panes = "B2"

    # ---------------- MARTE ----------------
    ws = wb.create_sheet("Marte")
    f = titulo(ws, 1, "Huella mineral de Meridiani Planum (huella_marte_final.csv)", f_tit)
    ws.cell(f, 1, "superficie_CRISM: % del área sobre el umbral (4 escenas). fondo_TES: abundancia en el "
                  "cráter Victoria (ventana 3×3).").font = f_nota
    f += 2
    hm = huella.copy()
    f = tabla(ws, f, hm, {"pct_max_escena": "0.000", "intensidad": "0.000", "abundancia": "0.000",
                          "abundancia_pct": NUM1, "std_3x3": "0.000", "percentil_global": NUM1,
                          "rms_error": "0.0000"})
    anchos(ws, [18, 20, 14, 26, 12, 14, 34, 12, 12, 14, 10, 16, 12, 14])

    # ---------------- NOTAS ----------------
    ws = wb.create_sheet("Notas")
    f = titulo(ws, 1, "Cómo leer este libro", f_tit)
    notas = [
        "Similitud = puntaje de minerologia2.py × 100: suma ponderada de 7 criterios que comparan cada celda "
        "de ~1 km con la huella de Meridiani (1 = coincide, 0.5 = parcial, 0 = no).",
        "Coincidencia de un criterio = % promedio de coincidencia de las celdas de la zona en ese criterio.",
        "Puntos perdidos = peso × (1 − coincidencia) / suma de pesos. Suman 100 % − similitud "
        "(fila 'Similitud = 100 % − Σ puntos perdidos', calculada con fórmulas).",
        "Porcentajes de la Tierra (óxidos, carbonatos, arcillas, micas...) = % de píxeles EMIT donde se detecta "
        "el grupo. Marte (CRISM) = % de píxeles sobre el umbral. TES = abundancia. Se comparan NIVELES "
        "(dominante / traza / ausente), no los porcentajes directamente.",
        "Índice de cuarzo QI (ASTER) ≤ 1.010 = sin cuarzo. Índice máfico MI (ASTER) ≥ 0.970 = basalto.",
        "Colores: azul = dato calculado en Python desde los CSV de minerologia2.py; negro = fórmula de Excel.",
        "Fuentes: resultados_tierra/<region>/celdas_<region>.csv y resumen_region_<region>.csv "
        "(minerologia2.py); huella_marte_final.csv (huella_marte_crism.ipynb). Generado con comparacion_final.py.",
        "Las fórmulas se calculan al abrir el archivo en Excel (o LibreOffice / Google Sheets).",
    ]
    for t in notas:
        c = ws.cell(f, 1, f"• {t}"); c.font = f_txt; c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[f].height = 42; f += 1
    ws.column_dimensions["A"].width = 130

    wb.calculation.fullCalcOnLoad = True
    wb.save(ruta)


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
    resultados = {r: comparar(r, a.tierra, huella, a.salida) for r in regiones}
    ruta_xlsx = Path(a.salida) / "comparacion_final.xlsx"
    escribir_excel(resultados, huella, ruta_xlsx)
    print(f"\nExcel con los datos de cada región: {ruta_xlsx.resolve()}")
