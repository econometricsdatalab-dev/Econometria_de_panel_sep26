# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Econometría de datos panel con Python
Tema: Modelo pooled OLS (Datos de bank_panel_data)
Sesión: 03
Fecha: 03/10/2026
Asesor: Alexis Adonai Morales Alberto
"""

# Modelos a cargar ----

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import statsmodels.formula.api as smf

# Clases o submodulos a cargar

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# Configuración y lectura 

ALPHA = 0.05
EXPORTAR_GRAFICOS = False
carpeta = Path.cwd()
carpeta2 = Path("D:/Descargas/EDL/") # Cargar en caso de no tener el proyecto
                                     # Es necesario la ruta dodne se encuentra el archivo

candidatos = [carpeta2 / "Datos/bank_panel_data.csv", 
              carpeta2 / "bank_panel_data.csv"]

ruta = next((p for p in candidatos if p.is_file()), None)
if ruta is None:
    raise FileNotFoundError("Coloca bank_panel_data.csv junto al script o en Datos.")

Panel = pd.read_csv(ruta).drop(columns=["index"], errors="ignore")
print(Panel.head())
print(Panel.dtypes)

# Definción de la muestra de estimación

explicativas = ["Revenue", "Assets", "Interest_Rate", "GDP_Growth"]
necesarias = ["Bank_ID", "Year", "Profit"] + explicativas

Panel_modelo = Panel.copy().replace("r^\s*$", np.nan, regex = True)

for variable in ["Year", "Profit"] + explicativas: 
    Panel_modelo[variable] = pd.to_numeric(Panel_modelo[variable],
                                           errors = "raise")
    
Panel_modelo[["Year", "Profit"] + explicativas] = (
    Panel_modelo[["Year", "Profit"] + explicativas].replace([np.inf, -np.inf], np.nan)
    )

Panel_modelo = (Panel_modelo.dropna(subset=necesarias)
                .sort_values(["Bank_ID", "Year"]).reset_index(drop=True))


# Verificar balance respecto al universo original, no sólo al que sobrevivió.

bancos_originales = sorted(Panel["Bank_ID"].dropna().unique())
anios_originales = sorted(pd.to_numeric(Panel["Year"], errors="raise").dropna().unique())
tabla_panel = (Panel_modelo.groupby(["Bank_ID", "Year"]).size().unstack(fill_value=0)
               .reindex(index=bancos_originales, columns=anios_originales, fill_value=0))
balanceado = tabla_panel.eq(1).all().all()
N = Panel_modelo["Bank_ID"].nunique()
T = Panel_modelo["Year"].nunique()
NT = len(Panel_modelo)
print(f"N = {N} | T = {T} | NT = {NT}")
print(f"Balanceado respecto al universo original: {balanceado}")
print(tabla_panel)
if N < 2:
    raise ValueError("Se necesitan al menos dos bancos para agrupar la covarianza.")
    

# Estructura y supuestos del modelo ----
# Profit_it = beta_0 + beta_1 Revenue_it + beta_2 Assets_it
#             + beta_3 Interest_Rate_it + beta_4 GDP_Growth_it + u_it

formula = "Profit ~ Revenue + Assets + Interest_Rate + GDP_Growth"
print("\nModelo:", formula)
print(Panel_modelo[["Profit"] + explicativas].describe().round(3))

# Estimación clásica y matricial ----
modelo_clasico = smf.ols(formula, data=Panel_modelo, missing="raise").fit()
print("\nPOOLED OLS: errores estándar clásicos")
print(modelo_clasico.summary())

y = Panel_modelo["Profit"].to_numpy()
X = sm.add_constant(Panel_modelo[explicativas], has_constant="add")
print("\nPrimeras filas de la matriz de diseño:")
print(X.head())
rango = np.linalg.matrix_rank(X.to_numpy())
print(f"Rango de X: {rango}; columnas: {X.shape[1]}")
if rango < X.shape[1]:
    raise ValueError("Multicolinealidad perfecta: X no tiene rango completo.")
if len(X) <= X.shape[1]:
    raise ValueError("No hay grados de libertad residuales suficientes.")

beta_matricial = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
print("\nCoeficientes mediante álgebra matricial:")
print(pd.Series(beta_matricial, index=X.columns))
assert np.allclose(beta_matricial, modelo_clasico.params.to_numpy())

# Pooled OLS con errores estándar agrupados por banco ----
modelo_cluster = smf.ols(formula, data=Panel_modelo, missing="raise").fit(
    cov_type="cluster",
    cov_kwds={"groups": Panel_modelo["Bank_ID"],
              "use_correction": True, "df_correction": True},
    use_t=True
)
print("\nPOOLED OLS: errores estándar agrupados por banco")
print(modelo_cluster.summary())
print(f"Inferencia con t y {N - 1} grados de libertad por clusters.")
print("Con pocos clusters (aquí 20 bancos), la inferencia es aproximada; "
      "considerar wild cluster bootstrap en aplicaciones formales.")
assert np.allclose(modelo_clasico.params, modelo_cluster.params)

# Tabla e interpretación de los coeficientes ----
IC = modelo_cluster.conf_int(alpha=ALPHA)
resultados = pd.DataFrame({
    "Coeficiente": modelo_cluster.params,
    "EE_clasico": modelo_clasico.bse,
    "EE_cluster": modelo_cluster.bse,
    "t_cluster": modelo_cluster.tvalues,
    "p_cluster": modelo_cluster.pvalues,
    "IC95_inferior": IC[0], "IC95_superior": IC[1]
})
print("\nTabla comparativa:")
print(resultados.round(5).to_string())

# H0: beta_j = 0 frente a H1: beta_j != 0 (prueba bilateral).
# Ceteris paribus: mantener constantes las demás explicativas.
for variable in explicativas:
    b = modelo_cluster.params[variable]
    p = modelo_cluster.pvalues[variable]
    print(f"\n{variable}: una unidad adicional se asocia con un cambio de "
          f"{b:.5f} unidades de Profit, manteniendo las demás variables constantes.")
    print(f"H0: beta_{variable} = 0; H1: beta_{variable} != 0.")
    print(f"p = {p:.5f}: " + ("rechazar H0." if p < ALPHA else "no rechazar H0."))
print("\nEl intercepto predice Profit cuando todas las X valen cero. "
      "Ese escenario está fuera del soporte observado: interpretación limitada.")
print("Si las tasas están expresadas en porcentaje, una unidad significa "
      "un punto porcentual, no un aumento relativo del 1 %. Verificar metadatos.")

# Significancia conjunta ----
# H0: beta_1 = beta_2 = beta_3 = beta_4 = 0.
# H1: al menos una pendiente es distinta de cero.
prueba_conjunta = modelo_cluster.f_test(
    "Revenue = 0, Assets = 0, Interest_Rate = 0, GDP_Growth = 0"
)
print("\nWald expresado como F, con covarianza agrupada:")
print(prueba_conjunta)
print("Rechazar H0." if float(prueba_conjunta.pvalue) < ALPHA else "No rechazar H0.")

# Ajuste, predicciones y residuos ----
Panel_modelo["Profit_estimado"] = modelo_cluster.fittedvalues
Panel_modelo["Residuo"] = modelo_cluster.resid
R2 = modelo_cluster.rsquared
R2_ajustado = modelo_cluster.rsquared_adj
RMSE = np.sqrt(np.mean(Panel_modelo["Residuo"] ** 2))
print(f"\nR² = {R2:.4f}; R² ajustado = {R2_ajustado:.4f}; RMSE = {RMSE:.4f}")
print("R² mide ajuste dentro de la muestra; no demuestra causalidad ni exogeneidad.")

# Escenario representativo: predicción puntual, no pronóstico fuera de muestra.
escenario = Panel_modelo[explicativas].mean().to_frame().T
print("\nValores medios de las explicativas:")
print(escenario)
print("Beneficio predicho:", float(modelo_cluster.predict(escenario).iloc[0]))

# Multicolinealidad: VIF ----
# Diagnóstico, no prueba con H0/H1. VIF alto indica imprecisión potencial.
VIF = pd.DataFrame({
    "Variable": X.columns[1:],
    "VIF": [variance_inflation_factor(X.to_numpy(), j)
            for j in range(1, X.shape[1])]
})
print("\nFactores de inflación de varianza:")
print(VIF.round(3))
print("VIF > 5 o > 10 son reglas orientativas, no umbrales universales.")

# Heterocedasticidad: Breusch-Pagan, versión Koenker ----
# H0: varianza condicional constante.
# H1: la varianza depende de las variables incluidas en la regresión auxiliar.
# robust=True relaja normalidad, NO corrige dependencia intrabanco.
# Su p-value es referencial si existe correlación temporal o entre bancos.
LM_bp, p_LM_bp, F_bp, p_F_bp = het_breuschpagan(
    modelo_clasico.resid, modelo_clasico.model.exog, robust=True
)
print(f"\nBP-Koenker: LM = {LM_bp:.4f}; p = {p_LM_bp:.6f}")
print("Rechazar H0." if p_LM_bp < ALPHA else "No rechazar H0.")
print("Prueba auxiliar de heterocedasticidad, NO prueba LM de efectos aleatorios.")
print("No rechazar H0 no prueba homocedasticidad. Se conserva la inferencia cluster.")

# Limitación: interceptos comunes a todos los bancos ----
# Anticipo de efectos fijos, no selección definitiva de modelo.
# H0: todos los interceptos individuales son iguales.
# H1: al menos uno difiere. Comparación F clásica bajo errores esféricos.
modelo_individual = smf.ols(formula + " + C(Bank_ID)", data=Panel_modelo).fit()
F_individual, p_individual, restricciones = modelo_individual.compare_f_test(modelo_clasico)
print("\nF clásica para interceptos individuales:")
print(f"F = {F_individual:.4f}; p = {p_individual:.6f}; "
      f"restricciones = {int(restricciones)}")
print("Rechazar H0." if p_individual < ALPHA else "No rechazar H0.")
print("Esta F no es robusta a heterocedasticidad/autocorrelación. "
      "No se usa una Wald cluster sobre todas las dummies de banco: puede ser singular.")
print("No rechazar igualdad de interceptos NO demuestra exogeneidad ni valida pooled.")

# Limitación: shocks temporales comunes ----
modelo_tiempo = smf.ols(formula + " + C(Year)", data=Panel_modelo).fit(
    cov_type="cluster",
    cov_kwds={"groups": Panel_modelo["Bank_ID"],
              "use_correction": True, "df_correction": True},
    use_t=True
)
# Si GDP_Growth fuera idéntico por año, sería colineal con las dummies de año.
X_tiempo = modelo_tiempo.model.exog
if np.linalg.matrix_rank(X_tiempo) == X_tiempo.shape[1]:
    nombres_tiempo = [s for s in modelo_tiempo.params.index if s.startswith("C(Year)")]
    if nombres_tiempo and len(nombres_tiempo) < N:
        restriccion_tiempo = ", ".join(f"{s} = 0" for s in nombres_tiempo)
        prueba_tiempo = modelo_tiempo.f_test(restriccion_tiempo)
        print("\nH0: todos los efectos de año adicionales son cero.")
        print(prueba_tiempo)
        print("Rechazar H0." if float(prueba_tiempo.pvalue) < ALPHA else "No rechazar H0.")
else:
    print("\nModelo con años no identificado: revisar explicativas comunes por año.")

# 13. Sistema visual y gráficos ----
NARANJA = "#F04A1D"
AZUL = "#003057"
GRIS = "#4D565E"
FONDO = "#F7F7F5"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "axes.labelcolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22
})

# Gráfico 1: relación bivariada vs predicción multivariada ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
axes[0].scatter(Panel_modelo.Revenue, Panel_modelo.Profit,
                color=NARANJA, alpha=0.65, s=25)
# Predicción parcial: variar Revenue y mantener las otras X en su media.
rejilla = pd.DataFrame({v: np.repeat(Panel_modelo[v].mean(), 100) for v in explicativas})
rejilla["Revenue"] = np.linspace(Panel_modelo.Revenue.min(), Panel_modelo.Revenue.max(), 100)
axes[0].plot(rejilla.Revenue, modelo_cluster.predict(rejilla), color=AZUL, lw=2)
axes[0].set(xlabel="Ingresos (Revenue)", ylabel="Beneficio (Profit)")
axes[1].scatter(Panel_modelo.Profit_estimado, Panel_modelo.Profit,
                color=NARANJA, alpha=0.65, s=25)
limites = [min(Panel_modelo.Profit.min(), Panel_modelo.Profit_estimado.min()),
           max(Panel_modelo.Profit.max(), Panel_modelo.Profit_estimado.max())]
axes[1].plot(limites, limites, color=AZUL, lw=1.5, ls="--")
axes[1].set(xlabel="Beneficio estimado", ylabel="Beneficio observado")
fig.text(0.07, 0.95, "Pooled OLS: relación y ajuste", fontsize=19, weight="bold", color=AZUL)
fig.text(0.07, 0.89, f"{NT} observaciones | {N} bancos | R² = {R2:.3f}", color=GRIS)
fig.text(0.07, 0.025, "Nota: La recta izquierda mantiene las otras X en su media; la diagonal derecha es igualdad.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv. Unidades originales de la base.",
         fontsize=8, color=GRIS)
fig.subplots_adjust(top=0.80, bottom=0.18, left=0.08, right=0.97, wspace=0.25)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S03_01_ajuste.png", dpi=300)
plt.show()
plt.close(fig)

# Gráfico 2: residuos vs ajustados y promedios por sujeto ----
residuos_banco = Panel_modelo.groupby("Bank_ID")["Residuo"].mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
axes[0].scatter(Panel_modelo.Profit_estimado, Panel_modelo.Residuo,
                color=NARANJA, alpha=0.65, s=25)
axes[0].axhline(0, color=AZUL, lw=1)
axes[0].set(xlabel="Beneficio estimado", ylabel="Residuo")
axes[1].bar(np.arange(len(residuos_banco)), residuos_banco, color=NARANJA)
axes[1].set_xticks(np.arange(len(residuos_banco)), residuos_banco.index, fontsize=8)
axes[1].axhline(0, color=AZUL, lw=1)
axes[1].set(xlabel="Banco (ID)", ylabel="Residuo medio")
fig.text(0.07, 0.95, "Diagnóstico visual del modelo pooled", fontsize=19, weight="bold", color=AZUL)
fig.text(0.07, 0.89, "Buscar curvatura, dispersión desigual y patrones por sujeto", color=GRIS)
fig.text(0.07, 0.025, "Nota: Estos patrones orientan el diagnóstico; no prueban exogeneidad ni sustituyen inferencia.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8, color=GRIS)
fig.subplots_adjust(top=0.80, bottom=0.18, left=0.08, right=0.97, wspace=0.25)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S03_02_residuos.png", dpi=300)
plt.show()
plt.close(fig)

# Gráfico 3: observados y ajustados por banco ----
bancos = sorted(Panel_modelo.Bank_ID.unique())
columnas = min(4, len(bancos))
filas = int(np.ceil(len(bancos) / columnas))
fig, axes = plt.subplots(filas, columnas, figsize=(13, max(5, filas * 2.3)),
                         dpi=200, sharex=True, sharey=True, squeeze=False)
for ax, banco in zip(axes.flat, bancos):
    temporal = Panel_modelo.loc[Panel_modelo.Bank_ID == banco]
    ax.plot(temporal.Year, temporal.Profit, color=NARANJA, lw=1.5, marker="o",
            ms=3, label="Observado")
    ax.plot(temporal.Year, temporal.Profit_estimado, color=AZUL, lw=1.4,
            ls="--", label="Estimado")
    ax.set_title(f"Banco {banco}", fontsize=10, loc="left", color=AZUL)
    ax.set_xticks(anios_originales[::2])
    ax.tick_params(labelsize=8)
for ax in list(axes.flat)[len(bancos):]:
    ax.set_visible(False)
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right",
           bbox_to_anchor=(0.97, 0.925), frameon=False, ncol=2)
fig.text(0.07, 0.965, "Beneficio observado y estimado por banco", fontsize=19, weight="bold", color=AZUL)
fig.text(0.07, 0.925, "Una misma ecuación para todos los bancos y años", fontsize=10, color=GRIS)
fig.text(0.07, 0.02, "Nota: Ajuste dentro de la muestra, no pronóstico. Escala Y compartida.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8, color=GRIS)
fig.supxlabel("Año", y=0.07, color=GRIS)
fig.supylabel("Beneficio (unidades originales)", x=0.02, color=GRIS)
fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.97, hspace=0.55, wspace=0.15)
if EXPORTAR_GRAFICOS:
    fig.savefig(carpeta / "S03_03_bancos.png", dpi=300)
plt.show()
plt.close(fig)

