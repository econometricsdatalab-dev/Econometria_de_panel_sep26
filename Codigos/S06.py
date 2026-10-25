# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Econometría de datos panel con Python
Tema: Estimar Within
Sesión: 06
Fecha: 24/10/2026
Asesor: Alexis Adonai Morales Alberto
"""

# Módulos a cargar ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import base64

# Configuración ----

BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Datos de panel\PANEL_SEP26"
    )
SALIDA = BASE / "Resultados" / "S06_Within"
SALIDA.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05

# 1. Lectura y muestra de estimación ----

if "data_bank" in globals():
    Original = data_bank.copy()
    if not {"Bank_ID", "Year"}.issubset(Original.columns):
        Original = Original.reset_index()
else:
    candidatos = [BASE / "Datos/bank_panel_data.csv", BASE / "bank_panel_data.csv",
                  Path("Datos/bank_panel_data.csv"), Path("bank_panel_data.csv")]
    ruta = next((p for p in candidatos if p.is_file()), None)
    if ruta is None:
        raise FileNotFoundError("Coloca bank_panel_data.csv en Datos o junto al script.")
    Original = pd.read_csv(ruta)

Original = Original.drop(columns=["index"], errors="ignore")
X_nombres = ["Revenue", "Assets", "Interest_Rate", "GDP_Growth"]
variables = ["Profit"] + X_nombres
necesarias = ["Bank_ID", "Year"] + variables
if not set(necesarias).issubset(Original.columns):
    raise ValueError(f"Faltan columnas: {set(necesarias) - set(Original.columns)}")
Panel = Original.copy().replace(r"^\s*$", np.nan, regex=True)
for variable in ["Year"] + variables:
    Panel[variable] = pd.to_numeric(Panel[variable], errors="raise")
Panel[["Year"] + variables] = Panel[["Year"] + variables].replace([np.inf, -np.inf], np.nan)
Excluidas_NA = Panel.loc[Panel[necesarias].isna().any(axis=1)].copy()
Panel = Panel.dropna(subset=necesarias).sort_values(["Bank_ID", "Year"])
if Panel.empty or Panel.duplicated(["Bank_ID", "Year"]).any():
    raise ValueError("Muestra vacía o duplicados banco-año: revisar.")
conteos = Panel.groupby("Bank_ID").size()
Excluidas_singleton = Panel.loc[Panel.Bank_ID.isin(conteos[conteos.eq(1)].index)].copy()
Panel = Panel.loc[~Panel.Bank_ID.isin(conteos[conteos.eq(1)].index)].reset_index(drop=True)
N, T, NT, K = Panel.Bank_ID.nunique(), Panel.Year.nunique(), len(Panel), len(X_nombres)
if N <= K or NT <= N + K:
    raise ValueError("Pocos bancos/grados de libertad para esta especificación.")
print(f"N={N}; años distintos={T}; observaciones={NT}")
print(f"Excluidas por NA={len(Excluidas_NA)}; con un único periodo={len(Excluidas_singleton)}")
print("Los singletons no aportan variación within; la selección de muestra puede importar.")
if Panel.groupby("Year").GDP_Growth.nunique().gt(1).any():
    print("GDP_Growth varía entre bancos en el mismo año: revisar metadatos antes de llamarlo PIB nacional.")

# 2. Variación intraindividual y entre individuos ----

Medias_banco = Panel.groupby("Bank_ID")[variables].mean()
Medias_alineadas = Panel.groupby("Bank_ID")[variables].transform("mean")
print("\nMedias por banco:\n", Medias_banco.round(4))

# SS_total = SS_within + SS_between.
# SS_between pondera por T_i; no es la varianza simple de medias individuales.
descomposicion = []
for variable in variables:
    media_global = Panel[variable].mean()
    ss_total = float(np.sum((Panel[variable] - media_global) ** 2))
    ss_within = float(np.sum((Panel[variable] - Medias_alineadas[variable]) ** 2))
    ss_between = float(np.sum((Medias_alineadas[variable] - media_global) ** 2))
    assert np.isclose(ss_total, ss_within + ss_between)
    descomposicion.append({"Variable": variable, "SS_total": ss_total,
        "SS_within": ss_within, "SS_between": ss_between,
        "Porcentaje_within": 100 * ss_within / ss_total if ss_total > 0 else np.nan})
Variacion = pd.DataFrame(descomposicion).set_index("Variable")
print("\nDescomposición exacta de sumas de cuadrados:\n", Variacion.round(4))

# 3. Transformación within ----

# y_it = alpha_i + x_it' beta + u_it.
# Restar la media temporal DE CADA BANCO, no la media global.

Within = Panel[variables] - Medias_alineadas
Within.columns = [v + "_w" for v in variables]
Panel = pd.concat([Panel, Within], axis=1)
Medias_within = Panel.groupby("Bank_ID")[Within.columns].mean()
assert np.allclose(Medias_within, 0, atol=1e-9)
print("\nMedias centradas por banco (cero salvo redondeo):\n", Medias_within.round(10))
primer_banco = Panel.Bank_ID.iloc[0]
Ejemplo_banco = Panel.loc[Panel.Bank_ID.eq(primer_banco), ["Bank_ID", "Year", "Profit", "Revenue"]].copy()
Ejemplo_banco["Profit_media"] = Medias_alineadas.loc[Ejemplo_banco.index, "Profit"]
Ejemplo_banco["Revenue_media"] = Medias_alineadas.loc[Ejemplo_banco.index, "Revenue"]
Ejemplo_banco["Profit_w"] = Within.loc[Ejemplo_banco.index, "Profit_w"]
Ejemplo_banco["Revenue_w"] = Within.loc[Ejemplo_banco.index, "Revenue_w"]
print("\nTransformación observación por observación:\n", Ejemplo_banco.round(4).to_string(index=False))

# 4. Matriz de proyección y propiedades ----

# D: indicadores de banco. M_D = I - D(D'D)^(-1)D'.
# D tiene rango completo; solve evita invertir explícitamente D'D.
# M_D es simétrica, idempotente y singular. NO se invierte M_D.

D = pd.get_dummies(Panel.Bank_ID, dtype=float).to_numpy()
M = np.eye(NT) - D @ np.linalg.solve(D.T @ D, D.T)
y = Panel.Profit.to_numpy()
X = Panel[X_nombres].to_numpy()
assert np.allclose(M.T, M)
assert np.allclose(M @ M, M)
assert np.allclose(M @ D, 0, atol=1e-10)
assert np.linalg.matrix_rank(M) == NT - N
assert np.allclose(M @ y, Panel.Profit_w)
assert np.allclose(M @ X, Within[[v + "_w" for v in X_nombres]])
print(f"\nRango de M: {np.linalg.matrix_rank(M)} = NT-N = {NT - N}")

# Nota: construir M es didáctico. En bases grandes usar groupby/transform,
# que evita reservar una matriz NT por NT.

# 5. Estimador within manual y equivalencia LSDV ----

y_w = Panel.Profit_w
X_w = Panel[[v + "_w" for v in X_nombres]]
if np.linalg.matrix_rank(X_w.to_numpy()) < K:
    raise ValueError("X within no tiene rango completo: no hay identificación conjunta.")
if np.isclose(float(y_w @ y_w), 0):
    raise ValueError("Profit carece de variación within.")
    
# Teoría: beta=(X'MX)^(-1)X'My. lstsq evita formar la inversa explícita.

beta_within = pd.Series(np.linalg.lstsq(X_w, y_w, rcond=None)[0], index=X_nombres)
formula = "Profit ~ Revenue + Assets + Interest_Rate + GDP_Growth"
Pooled = smf.ols(formula, data=Panel).fit()
FE_clasico = smf.ols(formula + " + C(Bank_ID)", data=Panel).fit()
FE_cluster = smf.ols(formula + " + C(Bank_ID)", data=Panel).fit(
    cov_type="cluster", cov_kwds={"groups": Panel.Bank_ID,
    "use_correction": True, "df_correction": True}, use_t=True
)
assert np.allclose(beta_within, FE_clasico.params[X_nombres], atol=1e-9)
assert FE_clasico.df_resid == NT - N - K
Comparacion = pd.DataFrame({"Pooled": Pooled.params[X_nombres],
                            "Within_manual": beta_within,
                            "LSDV_FE": FE_cluster.params[X_nombres]})
print("\nCoeficientes:\n", Comparacion.round(6).to_string())
print("No usar los EE ingenuos de MCO sobre centrados: deben contabilizarse los N interceptos.")
