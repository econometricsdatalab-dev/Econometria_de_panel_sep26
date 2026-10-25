# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Econometría de datos panel con Python
Tema: Modelo de efectos aleatorios (Datos de bank_panel_data)
Sesión: 05
Fecha: 17
/10/2026
Asesor: Alexis Adonai Morales Alberto
"""
# Instalación de modulos (por si no se cuenta con ellos)

pip install numpy pandas scipy statsmodels linearmodels matplotlib

# Modulos a cargar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Clases de modulos

from scipy import stats
from pathlib import Path
from itertools import combinations
from linearmodels.panel import RandomEffects, PanelOLS

# Configuración 

ALPHA = 0.05
EXPORTAR = True
BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Datos de panel\PANEL_SEP26"
    )
SALIDA = BASE / "Resultados" / "S05_Efectos_Aleatorios"
if EXPORTAR:
    SALIDA.mkdir(parents=True, exist_ok=True)

# Lectura y revisión de datos 

if "data_bank" in globals():
    Panel_original = data_bank.copy()
    if "Bank_ID" not in Panel_original.columns or "Year" not in Panel_original.columns:
        Panel_original = Panel_original.reset_index()
else:
    candidatos = [BASE / "Datos/bank_panel_data.csv", BASE / "bank_panel_data.csv"]
    ruta = next((p for p in candidatos if p.is_file()), None)
    if ruta is None:
        raise FileNotFoundError("Coloca bank_panel_data.csv junto al script o en Datos.")
    Panel_original = pd.read_csv(ruta)

Panel_original = Panel_original.drop(columns=["index"], errors="ignore")
print(Panel_original.head())
print(Panel_original.dtypes)
explicativas = ["Revenue", "Assets", "Interest_Rate", "GDP_Growth"]
variables = ["Profit"] + explicativas
necesarias = ["Bank_ID", "Year"] + variables
if not set(necesarias).issubset(Panel_original.columns):
    raise ValueError(f"Columnas faltantes: {set(necesarias) - set(Panel_original.columns)}")

Panel = Panel_original.copy().replace(r"^\s*$", np.nan, regex=True)
for variable in ["Year"] + variables:
    Panel[variable] = pd.to_numeric(Panel[variable], errors="raise")
Panel[["Year"] + variables] = Panel[["Year"] + variables].replace([np.inf, -np.inf], np.nan)
print("\nAusentes en variables necesarias:\n", Panel[necesarias].isna().sum())
Filas_excluidas = Panel.loc[Panel[necesarias].isna().any(axis=1)].copy()
Panel = (Panel.dropna(subset=necesarias).sort_values(["Bank_ID", "Year"])
         .reset_index(drop=True))
if Panel.empty or Panel.duplicated(["Bank_ID", "Year"]).any():
    raise ValueError("Base vacía o duplicados banco-año: revisar antes de estimar.")
if not np.equal(Panel.Year, np.floor(Panel.Year)).all():
    raise ValueError("Year debe contener años enteros.")
Panel["Year"] = Panel.Year.astype(int)
T_i = Panel.groupby("Bank_ID").size().rename("T_i")
N, T, NT = len(T_i), Panel.Year.nunique(), len(Panel)
K = len(explicativas)             # Pendientes
P = K + 1                        # Pendientes e intercepto
if T_i.lt(2).any() or N <= 2 * K + 1 or NT <= N + K:
    raise ValueError("Se requieren >=2 años por banco y más bancos/observaciones para estos diagnósticos.")
tabla_panel = Panel.groupby(["Bank_ID", "Year"]).size().unstack(fill_value=0)
balanceado = bool(tabla_panel.eq(1).all().all())
print(f"\nN={N}; años distintos={T}; NT={NT}; balanceado en muestra={balanceado}")
print(f"Filas excluidas: {len(Filas_excluidas)}. Sólo se exigen completas las variables del modelo.")
print(T_i)
print("Si se elimina un banco/año completo, el universo original debe revisarse por separado.")
if Panel.groupby("Year").GDP_Growth.nunique().gt(1).any():
    print("GDP_Growth varía entre bancos en un mismo año: verificar su definición; "
          "no asumir que es el crecimiento de una única economía nacional.")


# Estructura y supuestos 

## Índice panel y matri X con constante explícita 

Datos = Panel.set_index(["Bank_ID", "Year"])
y = Datos["Profit"]
X = sm.add_constant(Datos[explicativas], has_constant="add")
grupos = Datos.index.get_level_values("Bank_ID")
if np.linalg.matrix_rank(X.to_numpy()) < P:
    raise ValueError("X no tiene rango completo.")
print("\nMatriz explicativa en niveles:\n", X.head())

# Componentes de varianza 

Media_X = X.groupby(level="Bank_ID").transform("mean")
Media_y = y.groupby(level="Bank_ID").transform("mean")
X_within = X[explicativas] - Media_X[explicativas]
y_within = y - Media_y
if np.linalg.matrix_rank(X_within.to_numpy()) < K:
    raise ValueError("Las X within son colineales o invariantes. Revisar especificación.")
Modelo_within = sm.OLS(y_within, X_within).fit()
sigma_e2_manual = float(np.sum(Modelo_within.resid ** 2) / (NT - N - K))

# Regresión between sobre las medias por banco 

X_between = X.groupby(level="Bank_ID").mean()
y_between = y.groupby(level="Bank_ID").mean()
if np.linalg.matrix_rank(X_between.to_numpy()) < P:
    raise ValueError("La etapa between de este ejemplo no tiene rango completo.")
Modelo_between = sm.OLS(y_between, X_between).fit()
T_armonica = float(1 / np.mean(1 / T_i.to_numpy()))
sigma_a2_sin_restringir = float(
    np.sum(Modelo_between.resid ** 2) / (N - P) - sigma_e2_manual / T_armonica
)

# Una varianza no puede ser negativa

sigma_a2_manual = max(0.0, sigma_a2_sin_restringir)
rho = sigma_a2_manual / (sigma_a2_manual + sigma_e2_manual)
print("\nComponentes estimados:")
print(f"sigma_e²={sigma_e2_manual:.6f}")
print(f"sigma_a² antes de restringir={sigma_a2_sin_restringir:.6f}")
print(f"sigma_a² admisible={sigma_a2_manual:.6f}; rho={rho:.6f}")
print("rho es la proporción de varianza compuesta atribuible a a_i y, "
      "bajo el modelo, la correlación intrabanco de v_it entre años distintos.")

# Estimación de efectos aleatorios con linearmodels 

Modelo_RE = RandomEffects(y, X)
RE_clasico = Modelo_RE.fit(cov_type="unadjusted", small_sample=False, debiased=True)
print("\nResumen RE con covarianza clásica:")
print(RE_clasico.summary)
print("\nDescomposición de varianza de la biblioteca:")
print(RE_clasico.variance_decomposition)
assert np.isclose(sigma_a2_manual, RE_clasico.variance_decomposition["Effects"])
assert np.isclose(sigma_e2_manual, RE_clasico.variance_decomposition["Residual"])

# Estimador FGLS y transformación cuasi-within

Theta = (1 - np.sqrt(sigma_e2_manual / (sigma_e2_manual + T_i * sigma_a2_manual))).rename("theta")
print("\nParámetros de cuasi-centrado por banco:\n", Theta)
theta_obs = pd.Series(grupos.map(Theta), index=Datos.index)
y_GLS = y - theta_obs * Media_y
X_GLS = X - Media_X.mul(theta_obs, axis=0)
print("\nPrimeras observaciones de X transformada:\n", X_GLS.head())
assert np.allclose(Theta.to_numpy(), RE_clasico.theta.loc[Theta.index, "theta"])
beta_GLS = np.linalg.lstsq(X_GLS.to_numpy(), y_GLS.to_numpy(), rcond=None)[0]
Coeficientes_GLS = pd.Series(beta_GLS, index=X.columns)
assert np.allclose(Coeficientes_GLS, RE_clasico.params, atol=1e-8)
print("\nVerificación de coeficientes FGLS manual vs biblioteca:")
print(pd.DataFrame({"GLS_manual": Coeficientes_GLS, "RE": RE_clasico.params}))
if np.isclose(sigma_a2_manual, 0):
    print("\nRESULTADO DE FRONTERA: sigma_a²=0 => theta=0 => RE coincide con pooled.")
    print("No se fuerza una varianza positiva. Esto no prueba ausencia de toda heterogeneidad real.")
else:
    print("theta entre 0 y 1: RE combina información within y between.")
print("theta=0: sin centrado (pooled). theta cercano a 1: cuasi-centrado cercano a within.")

RE_cluster = sm.OLS(y_GLS, X_GLS).fit(
    cov_type="cluster",
    cov_kwds={"groups": grupos, "use_correction": True, "df_correction": True},
    use_t=True
)
assert np.allclose(RE_cluster.params, RE_clasico.params)
IC = RE_cluster.conf_int(alpha=ALPHA)
Tabla_RE = pd.DataFrame({
    "Coeficiente": RE_cluster.params,
    "EE_clasico": RE_clasico.std_errors,
    "EE_cluster": RE_cluster.bse,
    "t_cluster": RE_cluster.tvalues,
    "p_cluster": RE_cluster.pvalues,
    "IC95_inf": IC[0], "IC95_sup": IC[1]
})
print(f"\nTABLA PRINCIPAL: t con {N - 1} gl entre bancos")
print(Tabla_RE.round(6).to_string())
print("El summary de linearmodels anterior tiene inferencia clásica. "
      "Para explicar significancia usar p_cluster y los IC de esta tabla.")
print("20 clusters es una muestra pequeña: considerar wild cluster bootstrap "
      "en investigación formal. Cluster por banco presupone independencia entre bancos.")
for variable in explicativas:
    b, p = RE_cluster.params[variable], RE_cluster.pvalues[variable]
    print(f"\n{variable}: una unidad adicional se asocia con {b:.6f} unidades de "
          "cambio en Profit, manteniendo las demás X constantes, bajo los supuestos RE.")
    print(f"H0: beta_{variable}=0; H1: beta_{variable}!=0; p={p:.6g}.")
    print("Rechazar H0 al 5 %." if p < ALPHA else "No rechazar H0 al 5 %.")
print("RE usa variación entre y dentro de bancos: no interpretar sus coeficientes "
      "como un contraste exclusivamente dentro del mismo banco.")
print("Si las tasas son porcentajes, una unidad equivale a un punto porcentual.")

# Efectos individuales: predicción BLUP y shrinkage 

Datos["Pred_marginal"] = X.dot(RE_cluster.params)
Datos["Residuo_compuesto"] = y - Datos.Pred_marginal
Lambda = (T_i * sigma_a2_manual / (sigma_e2_manual + T_i * sigma_a2_manual)).rename("lambda")
Residuo_medio = Datos.groupby(level="Bank_ID").Residuo_compuesto.mean()
Efectos_RE = pd.concat([T_i, Theta, Lambda, Residuo_medio.rename("Residuo_medio")], axis=1)
Efectos_RE["a_hat_BLUP"] = Lambda * Residuo_medio
Datos["a_hat"] = grupos.map(Efectos_RE.a_hat_BLUP)
Datos["Pred_condicional"] = Datos.Pred_marginal + Datos.a_hat
Datos["Residuo_condicional"] = y - Datos.Pred_condicional
print("\nEfectos individuales predichos:\n", Efectos_RE.round(5).to_string())
print("lambda reduce las diferencias individuales hacia cero (shrinkage). "
      "No son 20 interceptos estimados libremente como en FE.")
print("Si sigma_a²=0, todos los a_hat son cero aunque las medias observadas difieran.")
print("Los residuos condicionales no son observaciones directas de epsilon; "
      "incluyen incertidumbre de la predicción del efecto individual.")

R2 = pd.Series({"GLS_transformado": RE_clasico.rsquared,
                "Within": RE_clasico.rsquared_within,
                "Between": RE_clasico.rsquared_between,
                "Overall": RE_clasico.rsquared_overall}, name="R2")
print("\nMedidas de ajuste:\n", R2.round(6))
print("GLS: ajuste en datos transformados; within: variación centrada por banco; "
      "between: medias individuales; overall: datos en niveles usando X beta.")
print("No compararlas como si fueran una única medida; pueden existir R² negativos.")


# Significancia conjunta de las pendientes 

Prueba_conjunta = RE_cluster.f_test(
    "Revenue=0, Assets=0, Interest_Rate=0, GDP_Growth=0"
)
print("\nWald/F de pendientes, covarianza agrupada:\n", Prueba_conjunta)


# LM de efectos aleatorios frente a pooled 

Pooled = sm.OLS(y, X).fit()
Z_LM = LM_RE = p_LM = np.nan
if balanceado and T > 1:
    A = float(np.sum(Pooled.resid.groupby(level="Bank_ID").sum() ** 2)
              / np.sum(Pooled.resid ** 2))
    Z_LM = float(np.sqrt(N * T / (2 * (T - 1))) * (A - 1))
    LM_RE = max(0.0, Z_LM) ** 2
    p_LM = float(0.5 * stats.chi2.sf(LM_RE, 1)) if Z_LM > 0 else 1.0
    print(f"\nLM unilateral RE: score Z={Z_LM:.6f}; LM_+={LM_RE:.6f}; p={p_LM:.6g}")
else:
    print("\nLM balanceada omitida: muestra desbalanceada. Usar extensión para T_i desiguales.")
print("No rechazar sigma_a²=0 no significa que RE sea causal ni que toda heterogeneidad sea inexistente.")


# Comparación FE-RE: Hausman clásico 

FE_clasico = PanelOLS(y, X, entity_effects=True).fit(cov_type="unadjusted", debiased=True)
Comparacion = pd.DataFrame({"Pooled": Pooled.params[explicativas],
                            "FE": FE_clasico.params[explicativas],
                            "RE": RE_clasico.params[explicativas]})
print("\nComparación de pendientes:\n", Comparacion.round(6).to_string())
diferencia = FE_clasico.params[explicativas] - RE_clasico.params[explicativas]
V_diferencia = (FE_clasico.cov.loc[explicativas, explicativas]
                - RE_clasico.cov.loc[explicativas, explicativas])
V_diferencia = (V_diferencia + V_diferencia.T) / 2
autovalores = np.linalg.eigvalsh(V_diferencia)
H = p_H = np.nan
if autovalores.min() > 1e-10 * max(1.0, np.abs(autovalores).max()):
    H = float(diferencia @ np.linalg.solve(V_diferencia, diferencia))
    p_H = float(stats.chi2.sf(H, K))
    print(f"\nHausman clásico: H={H:.6f}; gl={K}; p={p_H:.6g}")
else:
    print("\nHausman no calculable con referencia chi² estándar: diferencia de covarianzas no definida positiva.")
    print("No se usa valor absoluto ni pseudoinversa para forzar un p-value.")
print("Hausman es referencial si hay heterocedasticidad o varianza RE en la frontera. "
      "No seleccionarlo automáticamente sobre esta base.")

# Mundlak/efectos aleatorios correlacionados: contraste agrupado

Medias_Mundlak = Datos[explicativas].groupby(level="Bank_ID").transform("mean").add_prefix("Media_")
X_CRE = pd.concat([X, Medias_Mundlak], axis=1)
if np.linalg.matrix_rank(X_CRE.to_numpy()) < X_CRE.shape[1]:
    raise ValueError("Mundlak no identificado: revisar medias invariantes o colineales.")
CRE = RandomEffects(y, X_CRE).fit(cov_type="unadjusted", small_sample=False)
theta_CRE = pd.Series(grupos.map(CRE.theta["theta"]), index=Datos.index)
X_CRE_GLS = X_CRE - X_CRE.groupby(level="Bank_ID").transform("mean").mul(theta_CRE, axis=0)
y_CRE_GLS = y - theta_CRE * Media_y
CRE_cluster = sm.OLS(y_CRE_GLS, X_CRE_GLS).fit(
    cov_type="cluster",
    cov_kwds={"groups": grupos, "use_correction": True, "df_correction": True}, use_t=True
)
assert np.allclose(CRE_cluster.params, CRE.params, atol=1e-8)
Prueba_Mundlak = CRE_cluster.f_test(
    ", ".join(v + "=0" for v in Medias_Mundlak.columns)
)
print("\nCoeficientes de las medias en Mundlak:")
print(pd.DataFrame({"pi": CRE_cluster.params[Medias_Mundlak.columns],
                    "EE_cluster": CRE_cluster.bse[Medias_Mundlak.columns]}))
print("\nWald/F de Mundlak, covarianza agrupada:\n", Prueba_Mundlak)
print("Un rechazo favorece considerar FE/CRE bajo los otros supuestos; "
      "no es una prueba universal de causalidad o de todo tipo de endogeneidad.")
if balanceado:
    assert np.allclose(CRE.params[explicativas], FE_clasico.params[explicativas], atol=1e-8)
    print("Verificado: en este panel balanceado, las pendientes CRE coinciden con FE.")


# Autocorrelación idiosincrática: Wooldridge-Drukker 

Diferencias = Datos[variables].groupby(level="Bank_ID").diff().add_suffix("_diff")
anios = pd.Series(Datos.index.get_level_values("Year"), index=Datos.index)
salto = anios.groupby(level="Bank_ID").diff()
Diferencias.loc[salto.ne(1), :] = np.nan
Muestra_FD = Diferencias.dropna().copy()
rho_FD = F_serial = p_serial = np.nan
if len(Muestra_FD) > K and np.linalg.matrix_rank(
    Muestra_FD[[v + "_diff" for v in explicativas]].to_numpy()
) == K:
    Modelo_FD = sm.OLS(Muestra_FD.Profit_diff,
                       Muestra_FD[[v + "_diff" for v in explicativas]]).fit()
    Diferencias["e_FD"] = Modelo_FD.resid.reindex(Datos.index)
    Diferencias["e_FD_lag"] = Diferencias.e_FD.groupby(level="Bank_ID").shift(1)
    Diferencias.loc[salto.ne(1), "e_FD_lag"] = np.nan
    Aux_serial = Diferencias.dropna(subset=["e_FD", "e_FD_lag"])
    if Aux_serial.index.get_level_values("Bank_ID").nunique() > 1:
        Modelo_serial = sm.OLS(Aux_serial.e_FD, Aux_serial[["e_FD_lag"]]).fit(
            cov_type="cluster", cov_kwds={"groups": Aux_serial.index.get_level_values("Bank_ID"),
                                          "use_correction": True, "df_correction": True}, use_t=True
        )
        Prueba_serial = Modelo_serial.f_test("e_FD_lag=-0.5")
        rho_FD = float(Modelo_serial.params.iloc[0])
        F_serial = float(np.asarray(Prueba_serial.fvalue).item())
        p_serial = float(Prueba_serial.pvalue)
print(f"\nWooldridge-Drukker: rho_FD={rho_FD:.6f}; F={F_serial:.6f}; p={p_serial:.6g}")
print("No probar rho=0. La restricción -0.5 presupone estabilidad de varianzas "
      "en el tiempo dentro del banco y regularidad del modelo estático.")

# Heterocedasticidad idiosincrática entre bancos 

e_FE = FE_clasico.resids
e2_FE = e_FE ** 2
si2 = e2_FE.groupby(level="Bank_ID").mean()
Vi = e2_FE.groupby(level="Bank_ID").var(ddof=1) / T_i
W_hetero = p_hetero = np.nan
if Vi.gt(0).all():
    W_hetero = float(np.sum((si2 - e2_FE.mean()) ** 2 / Vi))
    p_hetero = float(stats.chi2.sf(W_hetero, N))
print(f"\nWald modificado auxiliar FE: W={W_hetero:.6f}; p={p_hetero:.6g}")
print("Detecta diferencias entre bancos, no toda forma de volatilidad temporal. "
      "Referencia aproximada: T=9 y posible dependencia de cuadrados residuales.")


# Dependencia transversal y normalidad 

Residuos_ancho = e_FE.unstack(level="Bank_ID")
correlaciones = []
for i, j in combinations(Residuos_ancho.columns, 2):
    comun = Residuos_ancho[[i, j]].dropna()
    if len(comun) >= 3 and comun.std().gt(0).all():
        correlaciones.append((len(comun), comun.corr().iloc[0, 1]))
CD = p_CD = np.nan
if len(correlaciones) == N * (N - 1) // 2:
    CD = float(np.sqrt(2 / (N * (N - 1))) * sum(np.sqrt(t) * r for t, r in correlaciones))
    p_CD = float(2 * stats.norm.sf(abs(CD)))
print(f"\nPesaran CD auxiliar FE: CD={CD:.6f}; p={p_CD:.6g}")
print("T corto y autocorrelación pueden afectar el contraste; correlaciones de signo "
      "contrario pueden cancelarse. No rechazar no demuestra independencia.")
print("Cluster por banco no resuelve dependencia entre bancos. "
      "No aplicar Driscoll-Kraay como solución automática con nueve años.")

# H0 JB: normalidad marginal de residuos transformados; sólo referencial.
# Bajo la Omega clásica y errores normales, GLS blanquea la covarianza.
JB, p_JB = stats.jarque_bera(RE_cluster.resid)
print(f"\nJarque-Bera, residuos GLS (referencial): JB={JB:.6f}; p={p_JB:.6g}")
print("JB usual no ajusta por dependencia o varianzas desiguales. "
      "Normalidad no es requisito de consistencia FGLS; no se prueba normalidad de a_i con a_hat=0.")

# Resumen de diagnósticos e interpretación 

Diagnosticos = pd.DataFrame({
    "Prueba": ["Pendientes RE, cluster", "LM unilateral RE vs pooled", "Hausman clásico FE-RE",
               "Mundlak, cluster", "Wooldridge-Drukker", "Wald heterocedasticidad auxiliar FE",
               "Pesaran CD auxiliar FE", "JB GLS referencial"],
    "H0": ["Pendientes cero", "Var(a_i)=0", "Sin diferencia sistemática FE-RE",
           "Coeficientes de medias cero", "Sin correlación serial idiosincrática",
           "Varianzas idiosincráticas iguales", "Sin correlación entre bancos", "Normalidad marginal"],
    "Estadistico": [float(np.asarray(Prueba_conjunta.fvalue).item()), LM_RE, H,
                    float(np.asarray(Prueba_Mundlak.fvalue).item()), F_serial, W_hetero, CD, float(JB)],
    "p_value": [float(Prueba_conjunta.pvalue), p_LM, p_H, float(Prueba_Mundlak.pvalue),
                p_serial, p_hetero, p_CD, float(p_JB)],
    "Alcance": ["Pocos clusters", "Clásica; unilateral; balanceada", "Clásica; frontera/hetero limitan",
                "Bajo proyección Mundlak; pocos clusters", "Modelo estático; pocos clusters",
                "Auxiliar; T corto", "Auxiliar; referencia asintótica", "Descriptiva si hay dependencia"]
})
Diagnosticos["Decision_5pct"] = np.where(Diagnosticos.p_value.isna(), "No calculable",
    np.where(Diagnosticos.p_value < ALPHA, "Rechazar H0", "No rechazar H0"))
print("\nRESUMEN DE CONTRASTES:\n", Diagnosticos.to_string(index=False))
if float(Prueba_Mundlak.pvalue) < ALPHA:
    print("Mundlak rechaza: cuestiona RE convencional y favorece considerar FE/CRE, "
          "sin demostrar todos los supuestos de esos modelos.")
if np.isclose(sigma_a2_manual, 0):
    print("La estimación RE colapsa a pooled, pero eso no sustituye la evaluación "
          "de correlación a_i-X ni invalida automáticamente el diagnóstico Mundlak.")
print("No escoger modelo por mayoría de pruebas. LM, Hausman, Mundlak y los "
      "diagnósticos contrastan preguntas diferentes con supuestos distintos.")

# transformación GLS ----

NARANJA, AZUL, GRIS, FONDO = "#F04A1D", "#003057", "#4D565E", "#F7F7F5"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "axes.labelcolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18})
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
primer_banco = grupos[0]
ejemplo = Datos.xs(primer_banco, level="Bank_ID")
ejemplo_GLS = y_GLS.xs(primer_banco, level="Bank_ID")
axes[0].plot(ejemplo.index, ejemplo.Profit, color=NARANJA, lw=2, marker="o", ms=4, label="Niveles")
axes[0].plot(ejemplo_GLS.index, ejemplo_GLS, color=AZUL, lw=1.7, ls="--", label="Transformada GLS")
axes[0].set(xlabel="Año", ylabel="Beneficio: unidades originales o transformadas")
axes[0].legend(frameon=False)
rho_teorica = np.linspace(0, 0.995, 200)
theta_teorica = 1 - np.sqrt((1 - rho_teorica) / (1 - rho_teorica + T * rho_teorica))
axes[1].plot(rho_teorica, theta_teorica, color=AZUL, lw=2)
axes[1].scatter([rho], [Theta.iloc[0]], color=NARANJA, s=60, zorder=5)
axes[1].set(xlabel="rho teórico", ylabel="theta", title=f"Relación teórica para T={T}", xlim=(-0.02, 1), ylim=(-0.02, 1))
fig.text(0.07, 0.95, "Efectos aleatorios: cuasi-centrado GLS", fontsize=18, weight="bold", color=AZUL)
fig.text(0.07, 0.89, f"Banco {primer_banco} | theta estimado = {Theta.iloc[0]:.3f} | La curva derecha es teórica", color=GRIS, fontsize=10)
fig.text(0.07, 0.025, "Nota: Si theta=0, las líneas de la izquierda coinciden: no se altera la base para separarlas.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.subplots_adjust(top=0.79, bottom=0.18, left=0.09, right=0.97, wspace=0.30)
if EXPORTAR:
    fig.savefig(SALIDA / "01_transformacion_GLS.png", dpi=300)
plt.show()
plt.close(fig)

# Efectos individuales predichos ----
fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
posiciones = np.arange(N)
ax.bar(posiciones, Efectos_RE.a_hat_BLUP, color=NARANJA)
ax.scatter(posiciones, Efectos_RE.a_hat_BLUP, color=NARANJA, s=25, zorder=5)
ax.set_xticks(posiciones, Efectos_RE.index, fontsize=8)
ax.axhline(0, color=AZUL, lw=1)
ax.set(xlabel="Banco (ID)", ylabel="Efecto individual predicho, a_hat")
if np.isclose(sigma_a2_manual, 0):
    ax.set_ylim(-1, 1)
    ax.text(0.5, 0.75, "Varianza individual estimada = 0\nTodos los efectos predichos son cero",
            transform=ax.transAxes, ha="center", color=AZUL, fontsize=13)
fig.text(0.07, 0.95, "Efectos individuales aleatorios: shrinkage", fontsize=18, weight="bold", color=AZUL)
fig.text(0.07, 0.89, "Predicciones del componente individual, no interceptos libres por banco", color=GRIS, fontsize=10)
fig.text(0.07, 0.025, "Nota: Un efecto cero estimado no demuestra que todos los bancos sean iguales.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.subplots_adjust(top=0.80, bottom=0.17, left=0.10, right=0.97)
if EXPORTAR:
    fig.savefig(SALIDA / "02_efectos_individuales.png", dpi=300)
plt.show()
plt.close(fig)

# Residuos transformados y Q-Q 
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
axes[0].scatter(RE_cluster.fittedvalues, RE_cluster.resid, color=NARANJA, alpha=0.65, s=25)
axes[0].axhline(0, color=AZUL, ls="--", lw=1)
axes[0].set(xlabel="Valor ajustado en escala GLS", ylabel="Residuo en escala GLS")
sm.qqplot(RE_cluster.resid, line="s", ax=axes[1], markerfacecolor=NARANJA,
          markeredgecolor=NARANJA, markersize=3, alpha=0.65)
axes[1].get_lines()[-1].set_color(AZUL)
axes[1].set(xlabel="Cuantiles normales teóricos", ylabel="Cuantiles de residuos GLS")
fig.text(0.07, 0.95, "Diagnóstico visual del modelo de efectos aleatorios", fontsize=17, weight="bold", color=AZUL)
fig.text(0.07, 0.89, "Buscar variabilidad desigual, curvatura y observaciones extremas", color=GRIS, fontsize=10)
fig.text(0.07, 0.025, "Nota: La apariencia de los residuos no prueba la ortogonalidad entre a_i y las X.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.subplots_adjust(top=0.80, bottom=0.17, left=0.09, right=0.97, wspace=0.30)
if EXPORTAR:
    fig.savefig(SALIDA / "03_residuos.png", dpi=300)
plt.show()
plt.close(fig)

# Ajuste condicional por sujeto 
bancos = list(T_i.index)
ncols = min(4, N)
nrows = int(np.ceil(N / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(5, nrows * 2.3)), dpi=200,
                         sharex=True, sharey=True, squeeze=False)
for ax, banco in zip(axes.flat, bancos):
    temporal = Datos.xs(banco, level="Bank_ID")
    ax.plot(temporal.index, temporal.Profit, color=NARANJA, lw=1.5, marker="o", ms=3, label="Observado")
    ax.plot(temporal.index, temporal.Pred_condicional, color=AZUL, lw=1.5, ls="--", label="Estimado RE")
    ax.set_title(f"Banco {banco}", color=AZUL, fontsize=10, loc="left")
    ax.set_xticks(sorted(Panel.Year.unique())[::2])
    ax.tick_params(labelsize=8)
for ax in list(axes.flat)[N:]:
    ax.set_visible(False)
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=2,
           bbox_to_anchor=(0.97, 0.93), frameon=False)
fig.text(0.07, 0.965, "Efectos aleatorios: ajuste por banco", fontsize=18, weight="bold", color=AZUL)
fig.text(0.07, 0.925, f"{N} sujetos | rho estimado = {rho:.3f} | Ajuste dentro de la muestra", fontsize=10, color=GRIS)
fig.text(0.07, 0.02, "Nota: Predicción X beta + a_hat; no pronóstico fuera de muestra.\n"
         "Fuente. Elaboración propia con bank_panel_data.csv.", fontsize=8.5, color=GRIS)
fig.supxlabel("Año", y=0.07, color=GRIS)
fig.supylabel("Beneficio", x=0.02, color=GRIS)
fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.97, hspace=0.55, wspace=0.15)
if EXPORTAR:
    fig.savefig(SALIDA / "04_ajuste_bancos.png", dpi=300)
plt.show()
plt.close(fig)

# Exportación opcional de tablas ----
if EXPORTAR:
    Tabla_RE.to_csv(SALIDA / "coeficientes_RE.csv", encoding="utf-8-sig")
    Efectos_RE.to_csv(SALIDA / "efectos_individuales.csv", encoding="utf-8-sig")
    Comparacion.to_csv(SALIDA / "comparacion_pooled_FE_RE.csv", encoding="utf-8-sig")
    Diagnosticos.to_csv(SALIDA / "diagnosticos.csv", index=False, encoding="utf-8-sig")
    Datos.to_csv(SALIDA / "ajuste_por_observacion.csv", encoding="utf-8-sig")
    print("\nResultados exportados en:", SALIDA)














