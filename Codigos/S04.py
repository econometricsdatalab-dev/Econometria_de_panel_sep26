# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Econometría de datos panel con Python
Tema: Modelo de efectos fijos (Datos de bank_panel_data)
Sesión: 04
Fecha: 10/10/2026
Asesor: Alexis Adonai Morales Alberto
"""

# Modulos a cargar 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Clases directamente desde el modulo 

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from itertools import combinations

# Configuración inicial 

ALPHA = 0.05
EXPORTAR_GRAFICOS = True
carpeta = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Datos de panel\PANEL_SEP26"
    )
Path = Path.cwd()

# Lectura de los datos 

if "data_bank" in globals():
    Panel_original = data_bank.copy()
else:
    candidatos = [carpeta / "Datos/bank_panel_data.csv", carpeta / "bank_panel_data.csv"]
    ruta = next((p for p in candidatos if p.is_file()), None)
    if ruta is None:
        raise FileNotFoundError("Coloca bank_panel_data.csv junto al script o en Datos.")
    Panel_original = pd.read_csv(ruta)

Panel_original = Panel_original.drop(columns=["index"], errors="ignore")
print(Panel_original.head())
print(Panel_original.dtypes)

# Muestra común para la estimación 

explicativas = ["Revenue", "Assets", "Interest_Rate", "GDP_Growth"]
variables = ["Profit"] + explicativas
necesarias = ["Bank_ID", "Year"] + variables
if not set(necesarias).issubset(Panel_original.columns):
    raise ValueError(f"Faltan columnas: {set(necesarias) - set(Panel_original.columns)}")

Panel = Panel_original.copy().replace(r"^\s*$", np.nan, regex=True)
for variable in ["Year"] + variables:
    Panel[variable] = pd.to_numeric(Panel[variable], errors="raise")
Panel[["Year"] + variables] = Panel[["Year"] + variables].replace([np.inf, -np.inf], np.nan)
print("\nValores ausentes en variables del modelo:")
print(Panel[necesarias].isna().sum())
filas_excluidas = Panel.loc[Panel[necesarias].isna().any(axis=1)].copy()
Panel = (Panel.dropna(subset=necesarias).sort_values(["Bank_ID", "Year"])
         .reset_index(drop=True))
if Panel.empty or Panel.duplicated(["Bank_ID", "Year"]).any():
    raise ValueError("Muestra vacía o duplicados banco-año; revisar los datos.")
if not np.equal(Panel.Year, np.floor(Panel.Year)).all():
    raise ValueError("Year debe contener años enteros.")
Panel["Year"] = Panel["Year"].astype(int)
periodos = Panel.groupby("Bank_ID").size()
if periodos.lt(2).any():
    raise ValueError("Hay bancos con menos de dos años completos; no aportan información within.")
N, T, NT = Panel.Bank_ID.nunique(), Panel.Year.nunique(), len(Panel)
K = len(explicativas)
if N <= K or NT <= N + K:
    raise ValueError("Se requieren más bancos/grados de libertad para esta especificación.")
print(f"\nBancos: {N}; años distintos: {T}; observaciones: {NT}")
print(f"Filas excluidas por datos ausentes: {len(filas_excluidas)}")
print("Periodos por banco:\n", periodos)
print("Se excluyen NA sólo en variables necesarias; no en columnas ajenas al modelo.")

# Heterogeneidad individual no observable

## GDP_Growth debería coincidir por año si se mide una misma economía nacional

print("\nValores distintos de GDP_Growth por año:")
print(Panel.groupby("Year").GDP_Growth.nunique())
if Panel.groupby("Year").GDP_Growth.nunique().gt(1).any():
    print("GDP_Growth varía entre bancos dentro del año. Revisar metadatos; "
          "no interpretarlo automáticamente como PIB nacional.")

## Promedios individuales 

Medias_banco = Panel.groupby("Bank_ID")[variables].mean()
print("\nPromedios temporales de cada banco:")
print(Medias_banco.round(4).to_string())

## Promedios alineados con cada observación 

Medias_alineadas = Panel.groupby("Bank_ID")[variables].transform("mean")
print("\nPromedios alineados con las observaciones:")
print(Medias_alineadas.head(12))

## Transformación Within

Within = Panel[variables] - Medias_alineadas
Within.columns = [v + "_within" for v in variables]
Panel = pd.concat([Panel, Within], axis=1)

Ejemplo_within = Panel.loc[
    Panel.Bank_ID == Panel.Bank_ID.iloc[0],
    ["Bank_ID", "Year", "Profit", "Profit_within", "Revenue", "Revenue_within"]
].copy()
Ejemplo_within["Media_Profit"] = Medias_alineadas.loc[Ejemplo_within.index, "Profit"]
Ejemplo_within["Media_Revenue"] = Medias_alineadas.loc[Ejemplo_within.index, "Revenue"]
print("\nEjemplo del centrado para el primer banco:")
print(Ejemplo_within.round(4).to_string(index=False))
print("\nMedia de variables centradas por banco (debe ser prácticamente cero):")
print(Panel.groupby("Bank_ID")[Within.columns].mean().round(10))
assert np.allclose(Panel.groupby("Bank_ID")[Within.columns].mean(), 0, atol=1e-9)

# Identificación: Las variables constantes en el tiempo desaparecen 

variacion_within = Within.std().rename("DE_within")
print("\nVariación temporal disponible:")
print(variacion_within)
y_within = Within["Profit_within"]
X_within = Within[[v + "_within" for v in explicativas]]
if np.linalg.matrix_rank(X_within.to_numpy()) < K:
    raise ValueError("X within no tiene rango completo: variable invariante o colinealidad.")
if np.isclose(np.sum(y_within ** 2), 0):
    raise ValueError("Profit no tiene variación dentro de los bancos.")

# Estimador Within manual ----

beta_within = np.linalg.lstsq(X_within.to_numpy(), y_within.to_numpy(), rcond=None)[0]
Coeficientes_within = pd.Series(beta_within, index=explicativas, name="Within_manual")
print("\nCoeficientes within:")
print(Coeficientes_within)

# Estimación equivalente LSDV (indicadores por banco)

formula = "Profit ~ Revenue + Assets + Interest_Rate + GDP_Growth"
Modelo_pooled = smf.ols(formula, data=Panel).fit()
Modelo_FE_clasico = smf.ols(formula + " + C(Bank_ID)", data=Panel).fit()
Modelo_FE = smf.ols(formula + " + C(Bank_ID)", data=Panel).fit(
    cov_type="cluster",
    cov_kwds={"groups": Panel.Bank_ID, "use_correction": True, "df_correction": True},
    use_t=True
)

Comparacion = pd.DataFrame({
    "Pooled": Modelo_pooled.params[explicativas],
    "Within_manual": Coeficientes_within,
    "LSDV_FE": Modelo_FE.params[explicativas]
})
print("\nComparación de coeficientes:")
print(Comparacion.round(6).to_string())
assert np.allclose(Coeficientes_within, Modelo_FE.params[explicativas], atol=1e-9)
assert int(Modelo_FE_clasico.df_resid) == NT - N - K
print(f"Grados de libertad residuales clásicos: NT - N - K = {NT - N - K}")
print(f"Inferencia cluster con t: {N - 1} grados de libertad entre bancos.")
print("Con 20 bancos la aproximación cluster es limitada. "
      "Considerar wild cluster bootstrap para inferencia formal.")


# Tabla de pendientes 

IC = Modelo_FE.conf_int(alpha=ALPHA).loc[explicativas]
Resultados_FE = pd.DataFrame({
    "Coeficiente": Modelo_FE.params[explicativas],
    "EE_clasico": Modelo_FE_clasico.bse[explicativas],
    "EE_cluster": Modelo_FE.bse[explicativas],
    "t_cluster": Modelo_FE.tvalues[explicativas],
    "p_cluster": Modelo_FE.pvalues[explicativas],
    "IC95_inf": IC[0], "IC95_sup": IC[1]
})
print("\nPendientes de efectos fijos: inferencia agrupada por banco")
print(Resultados_FE.round(6).to_string())

for variable in explicativas:
    b, p = Modelo_FE.params[variable], 
    print(f"\n{variable}: dentro del MISMO banco, una unidad adicionModelo_FE.pvalues[variable]al se asocia "
          f"con {b:.5f} unidades de cambio en Profit, ceteris paribus.")
    print(f"H0: beta_{variable}=0; H1: beta_{variable}!=0. p={p:.6g}.")
    print("Rechazar H0 al 5 %." if p < ALPHA else "No rechazar H0 al 5 %.")
print("No rechazar no demuestra ausencia de asociación. "
      "Si las tasas son porcentajes, una unidad equivale a un punto porcentual.")

# Interceptos individuales y ajuste 

Interceptos = Medias_banco["Profit"] - Medias_banco[explicativas].dot(Coeficientes_within)
Efectos_individuales = Interceptos.to_frame("Intercepto_alpha_i")
beta_0 = Interceptos.mean()
Efectos_individuales["Efecto_centrado_a_i"] = Interceptos - beta_0
print(f"\nIntercepto común bajo normalización promedio(a_i)=0: {beta_0:.5f}")
print(Efectos_individuales.round(4).to_string())
print("Un a_i positivo indica un intercepto mayor que el promedio, controlando las X. "
      "No revela cuál característica lo causa ni mide eficiencia por sí solo.")

Panel["Profit_estimado_FE"] = (
    Panel.Bank_ID.map(Interceptos) + Panel[explicativas].dot(Coeficientes_within)
)
Panel["Residuo_FE"] = Panel.Profit - Panel.Profit_estimado_FE
assert np.allclose(Panel.Residuo_FE, Modelo_FE.resid, atol=1e-8)
assert np.allclose(Panel.groupby("Bank_ID").Residuo_FE.mean(), 0, atol=1e-8)
# Los residuos medios por banco son cero POR CONSTRUCCIÓN, no validación.
SSE = float(np.sum(Panel.Residuo_FE ** 2))
SST_within = float(np.sum(y_within ** 2))
R2_within = 1 - SSE / SST_within
R2_LSDV = Modelo_FE.rsquared
print(f"\nR² within = {R2_within:.4f}")
print(f"R² LSDV (con interceptos individuales) = {R2_LSDV:.4f}")
print("R² within: variación temporal alrededor de la media de cada banco explicada por X.")
print("R² LSDV: ajuste total, incluidos los interceptos. No son la misma medida.")


# Prueba conjunta de las pendientes ----
# H0: beta_Revenue = beta_Assets = beta_Interest_Rate = beta_GDP_Growth = 0.
# H1: al menos una pendiente es distinta de cero.
Prueba_conjunta = Modelo_FE.f_test(
    "Revenue=0, Assets=0, Interest_Rate=0, GDP_Growth=0"
)
print("\nWald/F para las pendientes, con covarianza cluster:")
print(Prueba_conjunta)

# F clásica: pooled frente a efectos individuales ----
# H0: alpha_1 = ... = alpha_N; H1: algún intercepto difiere.
# Comparar modelos con idéntica muestra y variables explicativas.
F_efectos, p_efectos, gl_efectos = Modelo_FE_clasico.compare_f_test(Modelo_pooled)
print(f"\nF de efectos individuales: F={F_efectos:.5f}; p={p_efectos:.6g}")
print(f"Restricciones: {int(gl_efectos)}; gl residuales: {NT - N - K}")
print("Es una F CLÁSICA: requiere homocedasticidad y ausencia de correlación; "
      "la distribución F exacta también requiere normalidad.")
print("No rechazar NO demuestra que alpha_i no se correlacione con las X.")

# Heterocedasticidad entre bancos: Wald modificado (Baum, 2001) ----
# H0: sigma_i² = sigma² para todos los bancos.
# H1: la varianza difiere entre bancos.
# s_i² = promedio(e_it²); V_i = SUMA((e_it²-s_i²)²) / [T_i*(T_i-1)].
# W = SUMA((s_i²-s²)²/V_i), referencia asintótica chi²(N).
# Se define s² como promedio global de cuadrados residuales (momento común).
# No detecta toda forma de heterocedasticidad temporal. Con T corto, cautela.
# La fórmula V_i no es robusta a autocorrelación de los cuadrados residuales.
Panel["Residuo2_FE"] = Panel.Residuo_FE ** 2
Varianzas_banco = Panel.groupby("Bank_ID").agg(
    T_i=("Residuo_FE", "size"), s_i2=("Residuo2_FE", "mean"),
    var_e2=("Residuo2_FE", "var")
)
Varianzas_banco["V_i"] = Varianzas_banco.var_e2 / Varianzas_banco.T_i
s2_comun = Panel.Residuo2_FE.mean()
if Varianzas_banco.V_i.gt(0).all():
    W_hetero = float(np.sum((Varianzas_banco.s_i2 - s2_comun) ** 2 / Varianzas_banco.V_i))
    p_hetero = float(stats.chi2.sf(W_hetero, df=N))
else:
    W_hetero, p_hetero = np.nan, np.nan
print("\nVarianzas estimadas por banco:")
print(Varianzas_banco.round(4).to_string())
print(f"Wald modificado: W={W_hetero:.5f}; p={p_hetero:.6g}; gl={N}")
print("Referencia aproximada; con nueve años y posible autocorrelación "
      "no tratar el p-value como prueba definitiva.")

# Autocorrelación: Wooldridge-Drukker (modelo estático) ----
# H0: ausencia de correlación serial en los errores idiosincráticos en niveles.
# La implementación contrasta rho=-0.5 en residuos de PRIMERAS DIFERENCIAS.
# Si epsilon_t no se correlaciona y su varianza es estable dentro del banco,
# Corr(epsilon_t-epsilon_t-1, epsilon_t-1-epsilon_t-2) = -0.5.
# H1: rho != -0.5. Es un diagnóstico de correlación serial, no identifica su orden.
# No usar residuos within en lugar de los residuos del modelo en diferencias.
Diferencias = Panel.groupby("Bank_ID")[variables].diff()
Diferencias.columns = [v + "_diff" for v in variables]
Diferencias["Bank_ID"] = Panel.Bank_ID
Diferencias["Year"] = Panel.Year
salto_anual = Panel.groupby("Bank_ID").Year.diff()
# No confundir diferencias entre años separados por huecos con diferencias anuales.
Diferencias.loc[salto_anual.ne(1), [v + "_diff" for v in variables]] = np.nan
Muestra_FD = Diferencias.dropna(subset=[v + "_diff" for v in variables]).copy()
rho_serial = F_serial = p_serial = np.nan
if len(Muestra_FD) > K and np.linalg.matrix_rank(
    Muestra_FD[[v + "_diff" for v in explicativas]].to_numpy()
) == K:
    Modelo_FD = sm.OLS(Muestra_FD.Profit_diff,
                       Muestra_FD[[v + "_diff" for v in explicativas]]).fit()
    Diferencias["Residuo_FD"] = np.nan
    Diferencias.loc[Muestra_FD.index, "Residuo_FD"] = Modelo_FD.resid
    Diferencias["Residuo_FD_lag"] = Diferencias.groupby("Bank_ID").Residuo_FD.shift(1)
    Diferencias.loc[salto_anual.ne(1), "Residuo_FD_lag"] = np.nan
    Auxiliar_serial = Diferencias.dropna(subset=["Residuo_FD", "Residuo_FD_lag"]).copy()
    if Auxiliar_serial.Bank_ID.nunique() >= 2 and len(Auxiliar_serial) > 2:
        Regresion_serial = sm.OLS(Auxiliar_serial.Residuo_FD,
                                 Auxiliar_serial[["Residuo_FD_lag"]]).fit(
            cov_type="cluster",
            cov_kwds={"groups": Auxiliar_serial.Bank_ID,
                      "use_correction": True, "df_correction": True}, use_t=True
        )
        rho_serial = float(Regresion_serial.params.iloc[0])
        Prueba_serial = Regresion_serial.f_test("Residuo_FD_lag = -0.5")
        F_serial = float(np.asarray(Prueba_serial.fvalue).item())
        p_serial = float(Prueba_serial.pvalue)
print(f"\nWooldridge-Drukker: rho_FD={rho_serial:.5f}; F={F_serial:.5f}; p={p_serial:.6g}")
print("Un rho_FD próximo a -0.5 es compatible con H0; no se contrasta rho=0.")
print("Requiere exogeneidad estricta y regularidad; volatilidad cambiante en el tiempo "
      "puede afectar la restricción -0.5. Inferencia aproximada con pocos bancos.")

# Dependencia transversal: CD de Pesaran (versión por pares) ----
# H0: errores no correlacionados entre bancos; H1: dependencia transversal.
# CD = sqrt(2/[N(N-1)]) * SUMA_i<j sqrt(T_ij)*r_ij.
# Correlaciones sobre años comunes. Referencia normal estándar aproximada.
# Con T=9, correlaciones imprecisas. Correlaciones positivas y negativas pueden
# cancelarse; no rechazar NO demuestra independencia.
Residuos_ancho = Panel.pivot(index="Year", columns="Bank_ID", values="Residuo_FE")
pares = []
for banco_i, banco_j in combinations(Residuos_ancho.columns, 2):
    comun = Residuos_ancho[[banco_i, banco_j]].dropna()
    if len(comun) >= 3 and comun.std().gt(0).all():
        r_ij = comun.corr().iloc[0, 1]
        pares.append((banco_i, banco_j, len(comun), r_ij))
Correlaciones_pares = pd.DataFrame(pares, columns=["Banco_i", "Banco_j", "T_ij", "r_ij"])
CD = p_CD = np.nan
if len(pares) == N * (N - 1) // 2:
    CD = float(np.sqrt(2 / (N * (N - 1))) * np.sum(
        np.sqrt(Correlaciones_pares.T_ij) * Correlaciones_pares.r_ij
    ))
    p_CD = float(2 * stats.norm.sf(abs(CD)))
print(f"\nPesaran CD: CD={CD:.5f}; p={p_CD:.6g}")
if len(pares):
    print(f"Correlación promedio: {Correlaciones_pares.r_ij.mean():.4f}; "
          f"promedio absoluto: {Correlaciones_pares.r_ij.abs().mean():.4f}")
print("EE agrupados por banco NO corrigen dependencia entre bancos. "
      "La referencia normal requiere regularidad adicional, incluida ausencia "
      "de fuerte dependencia temporal. CD es orientativo en esta muestra corta.")

# Normalidad y multicolinealidad ----
# JB: H0 normalidad marginal; H1 asimetría y/o curtosis incompatibles con normal.
# El p-value chi²(2) usual supone condiciones que no necesariamente se cumplen
# para residuos FE dependientes. Diagnóstico descriptivo, no condición para FE.
JB, p_JB = stats.jarque_bera(Panel.Residuo_FE)
print(f"\nJarque-Bera (referencial): JB={JB:.5f}; p={p_JB:.6g}")
print("Normalidad no es necesaria para consistencia FE ni para inferencia cluster asintótica.")
X_vif = sm.add_constant(X_within, has_constant="add")
VIF_within = pd.DataFrame({
    "Variable": explicativas,
    "VIF_within": [variance_inflation_factor(X_vif.to_numpy(), j)
                   for j in range(1, K + 1)]
})
print("\nVIF calculado DESPUÉS del centrado:")
print(VIF_within.round(4).to_string(index=False))
print("VIF es diagnóstico sin H0/H1; >5 o >10 son reglas orientativas.")

# Anticipo: efectos de año además de banco 
# H0: todos los indicadores de año adicionales tienen coeficiente cero.
# H1: algún efecto de año adicional es distinto de cero.
Modelo_FE_dos_vias = smf.ols(formula + " + C(Bank_ID) + C(Year)", data=Panel).fit(
    cov_type="cluster",
    cov_kwds={"groups": Panel.Bank_ID, "use_correction": True, "df_correction": True},
    use_t=True
)
p_tiempo = F_tiempo = np.nan
X_dos_vias = Modelo_FE_dos_vias.model.exog
if np.linalg.matrix_rank(X_dos_vias) == X_dos_vias.shape[1]:
    indicadores_anio = [s for s in Modelo_FE_dos_vias.params.index if s.startswith("C(Year)")]
    if indicadores_anio and len(indicadores_anio) < N:
        Prueba_tiempo = Modelo_FE_dos_vias.f_test(
            ", ".join(s + " = 0" for s in indicadores_anio)
        )
        F_tiempo = float(np.asarray(Prueba_tiempo.fvalue).item())
        p_tiempo = float(Prueba_tiempo.pvalue)
        print("\nEfectos de año adicionales (Wald/F agrupada):")
        print(Prueba_tiempo)
else:
    print("Modelo de dos vías no identificado: explicativas comunes por año "
          "pueden ser colineales con los indicadores de año.")
print("Agregar años absorbe shocks comunes aditivos; no garantiza eliminar toda "
      "dependencia transversal. No reaplicar CD ingenuamente a residuos de dos vías.")

# Resumen de hipótesis y decisiones ----
Diagnosticos = pd.DataFrame({
    "Prueba": ["Pendientes conjuntas (cluster)", "F individual clásica",
               "Wald heterocedasticidad", "Wooldridge-Drukker",
               "Pesaran CD", "Jarque-Bera referencial", "Efectos de año (cluster)"],
    "H0": ["Todas las pendientes son cero", "Interceptos iguales",
           "Varianzas iguales entre bancos", "Sin correlación serial en niveles",
           "Sin correlación entre bancos", "Normalidad marginal", "Sin efectos de año adicionales"],
    "Estadistico": [float(np.asarray(Prueba_conjunta.fvalue).item()), F_efectos,
                    W_hetero, F_serial, CD, float(JB), F_tiempo],
    "p_value": [float(Prueba_conjunta.pvalue), p_efectos, p_hetero, p_serial,
                p_CD, float(p_JB), p_tiempo],
    "Alcance": ["Aproximada, 20 clusters", "Errores esféricos",
                "Asintótica; T corto", "Modelo estático; pocos clusters",
                "Referencia normal; T corto", "Descriptiva si hay dependencia",
                "Aproximada, pocos clusters"]
})
Diagnosticos["Decision_5pct"] = np.where(
    Diagnosticos.p_value.isna(), "No calculable",
    np.where(Diagnosticos.p_value < ALPHA, "Rechazar H0", "No rechazar H0")
)
print("\nRESUMEN DE DIAGNÓSTICOS:")
print(Diagnosticos.to_string(index=False))

# Gráficos con el sistema visual
NARANJA, AZUL, GRIS, FONDO = "#F04A1D", "#003057", "#4D565E", "#F7F7F5"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "axes.labelcolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18
})

# Gráfico 1: transformación within ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
for banco, temporal in Panel.groupby("Bank_ID"):
    axes[0].plot(temporal.Year, temporal.Profit, color=AZUL, alpha=0.20, lw=0.8)
    axes[1].plot(temporal.Year, temporal.Profit_within, color=AZUL, alpha=0.20, lw=0.8)
primer_banco = Panel.loc[Panel.Bank_ID == Panel.Bank_ID.iloc[0]]
axes[0].plot(primer_banco.Year, primer_banco.Profit, color=NARANJA, lw=2, marker="o", ms=4)
axes[1].plot(primer_banco.Year, primer_banco.Profit_within, color=NARANJA, lw=2, marker="o", ms=4)
axes[1].axhline(0, color=GRIS, ls="--", lw=1)
axes[0].set(xlabel="Año", ylabel="Beneficio original", title="Datos en niveles")
axes[1].set(xlabel="Año", ylabel="Beneficio menos su media individual", title="Datos centrados por banco")
fig.text(0.07, 0.95, "Transformación within: retirar el nivel de cada banco", fontsize=17, weight="bold", color=AZUL)
fig.text(0.07, 0.89, "Naranja: primer banco | Azul: resto de los bancos", fontsize=10, color=GRIS)
fig.text(0.07, 0.025, "Nota: El centrado conserva las fluctuaciones temporales y elimina la media individual.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.subplots_adjust(top=0.79, bottom=0.17, left=0.08, right=0.97, wspace=0.30)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S04_01_within.png", dpi=300)
plt.show()
plt.close(fig)

# Gráfico 2: efectos individuales bajo una normalización explícita ----
fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
efectos_ordenados = Efectos_individuales.sort_values("Efecto_centrado_a_i")
posiciones = np.arange(N)
ax.barh(posiciones, efectos_ordenados.Efecto_centrado_a_i,
        color=np.where(efectos_ordenados.Efecto_centrado_a_i >= 0, NARANJA, AZUL))
ax.set_yticks(posiciones, [f"Banco {i}" for i in efectos_ordenados.index], fontsize=8)
ax.axvline(0, color=GRIS, lw=1)
ax.set_xlabel("Efecto individual respecto al promedio de interceptos")
ax.grid(axis="y", visible=False)
fig.text(0.07, 0.95, "Heterogeneidad individual estimada", fontsize=18, weight="bold", color=AZUL)
fig.text(0.07, 0.89, "Interceptos centrados: promedio(a_i) = 0", fontsize=10, color=GRIS)
fig.text(0.07, 0.025, "Nota: Son estimaciones descriptivas; no identifican atributos concretos ni prueban eficiencia.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.subplots_adjust(top=0.81, bottom=0.16, left=0.14, right=0.97)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S04_02_efectos.png", dpi=300)
plt.show()
plt.close(fig)

# Gráfico 3: residuos y gráfico Q-Q ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
axes[0].scatter(Panel.Profit_estimado_FE, Panel.Residuo_FE, color=NARANJA, s=25, alpha=0.65)
axes[0].axhline(0, color=AZUL, ls="--", lw=1)
axes[0].set(xlabel="Beneficio estimado FE", ylabel="Residuo")
sm.qqplot(Panel.Residuo_FE, line="s", ax=axes[1], markerfacecolor=NARANJA,
          markeredgecolor=NARANJA, markersize=3, alpha=0.65)
axes[1].get_lines()[-1].set_color(AZUL)
axes[1].set(xlabel="Cuantiles teóricos normales", ylabel="Cuantiles de los residuos")
fig.text(0.07, 0.95, "Diagnóstico visual de los residuos", fontsize=18, weight="bold", color=AZUL)
fig.text(0.07, 0.89, "Buscar curvatura, variabilidad desigual, colas y observaciones extremas", fontsize=10, color=GRIS)
fig.text(0.07, 0.025, "Nota: Los gráficos orientan; no demuestran exogeneidad ni reemplazan los diagnósticos de panel.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.subplots_adjust(top=0.82, bottom=0.17, left=0.08, right=0.97, wspace=0.30)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S04_03_diagnosticos.png", dpi=300)
plt.show()
plt.close(fig)

# Gráfico 4: beneficio observado y ajustado por sujeto ----
bancos = sorted(Panel.Bank_ID.unique())
ncols = min(4, N)
nrows = int(np.ceil(N / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(5, nrows * 2.3)),
                         dpi=200, sharex=True, sharey=True, squeeze=False)
for ax, banco in zip(axes.flat, bancos):
    temporal = Panel.loc[Panel.Bank_ID == banco]
    ax.plot(temporal.Year, temporal.Profit, color=NARANJA, lw=1.5, marker="o", ms=3, label="Observado")
    ax.plot(temporal.Year, temporal.Profit_estimado_FE, color=AZUL, lw=1.5, ls="--", label="Estimado FE")
    ax.set_title(f"Banco {banco}", color=AZUL, fontsize=10, loc="left")
    ax.set_xticks(sorted(Panel.Year.unique())[::2])
    ax.tick_params(labelsize=8)
for ax in list(axes.flat)[N:]:
    ax.set_visible(False)
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=2,
           bbox_to_anchor=(0.97, 0.93), frameon=False)
fig.text(0.07, 0.965, "Efectos fijos: ajuste por banco", fontsize=18, weight="bold", color=AZUL)
fig.text(0.07, 0.925, f"{N} sujetos | R² within = {R2_within:.3f} | Pendientes comunes, interceptos propios", fontsize=10, color=GRIS)
fig.text(0.07, 0.02, "Nota: Ajuste dentro de la muestra; no es pronóstico. Unidades originales del archivo.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.supxlabel("Año", y=0.07, color=GRIS)
fig.supylabel("Beneficio", x=0.02, color=GRIS)
fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.97, hspace=0.55, wspace=0.15)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S04_04_ajuste_bancos.png", dpi=300)
plt.show()
plt.close(fig)




