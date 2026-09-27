# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Econometría de datos panel con Python
Tema: Introducción y análsis exploratorio de datos panel 
Sesión: 01
Fecha: 26/09/2026
Asesor: Alexis Adonai Morales Alberto
"""

# Comando para instalación de modulos 

# pip install 'nombre del modulo'

# Módulos a cargar 

import warnings 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.formula.api as smf

# Importar clases desde modulo

from pathlib import Path

# Quitar mensajes de precaución 

warnings.filterwarnings('ignore')

# Sistema visual de Econometrics Data Lab ----

NARANJA = "#F04A1D"
NARANJA_CLARO = "#F5B07C"
AZUL = "#003057"
GRIS_OSCURO = "#1F2933"
GRIS = "#4D565E"
GRIS_CLARO = "#D9DEE3"
FONDO = "#F7F7F5"
BLANCO = "#FFFFFF"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Bahnschrift", "Noto Sans", "DejaVu Sans"],
    "axes.edgecolor": GRIS_CLARO,
    "axes.labelcolor": GRIS,
    "xtick.color": GRIS,
    "ytick.color": GRIS,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.65,
    "figure.facecolor": FONDO,
    "axes.facecolor": FONDO
})

# Funciones auxiliares para gráfico inicales 

def textos_figura(fig, titulo, subtitulo, nota):
    """Agrega título, subtítulo y nota con el sistema visual del curso."""
    fig.text(
        0.06, 0.975, titulo,
        fontsize=19, fontweight="bold",
        color=GRIS_OSCURO, ha="left", va="top"
    )
    fig.text(
        0.06, 0.92, subtitulo,
        fontsize=10.5,
        color=GRIS, ha="left", va="top"
    )
    fig.text(
        0.06, 0.022, nota,
        fontsize=8.5,
        color=GRIS, ha="left", va="bottom"
    )

def formato_eje(ax):
    """Aplica el formato común a un eje."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIS_CLARO)
    ax.spines["bottom"].set_color(GRIS_CLARO)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.85)
    ax.tick_params(labelsize=8.5)
    
    
def grafico_series_por_banco(datos, variable, etiqueta, titulo):
    """Construye pequeños múltiplos: una serie temporal para cada banco."""
    bancos = sorted(datos["Bank_Name"].unique(), key=lambda x: int(x.split("_")[-1]))
    fig, axes = plt.subplots(
        5, 4,
        figsize=(13, 13), dpi=300,
        sharex=True, sharey=True
    )

    for ax, banco in zip(axes.flat, bancos):
        temporal = datos.loc[datos["Bank_Name"] == banco]
        ax.plot(
            temporal["Year"], temporal[variable],
            color=NARANJA, linewidth=1.65,
            marker="o", markersize=3.8,
            markerfacecolor=BLANCO,
            markeredgewidth=1.1
        )
        ax.fill_between(
            temporal["Year"], temporal[variable],
            color=NARANJA_CLARO, alpha=0.13
        )
        ax.set_title(
            banco.replace("_", " "),
            loc="left", fontsize=9.5,
            fontweight="bold", color=AZUL
        )
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        formato_eje(ax)

    fig.supxlabel("Año", fontsize=10, color=GRIS, y=0.065)
    fig.supylabel(etiqueta, fontsize=10, color=GRIS, x=0.018)
    textos_figura(
        fig,
        titulo,
        "Pequeños múltiplos: la misma escala permite comparar niveles y cambios entre sujetos",
        "Nota: Cada panel representa la dimensión individual del banco entre 2015 y 2023.\n"
        "Fuente. Elaboración propia con bank_panel_data.csv."
    )
    fig.subplots_adjust(
        left=0.07, right=0.98,
        top=0.875, bottom=0.10,
        hspace=0.50, wspace=0.18
    )
    plt.show()

def buscar_archivo(nombre):
    """Busca el CSV en la carpeta del proyecto o dentro de Datos."""
    carpeta_script = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidatos = [
        Path("Datos") / nombre,
        Path(nombre),
        carpeta_script / "Datos" / nombre,
        carpeta_script / nombre
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró {nombre}. Colócalo junto al script o en la carpeta Datos."
    )


# Lectura de datos 

ruta = buscar_archivo("Datos\\bank_panel_data.csv")
ruta

Panel = (
    pd.read_csv(ruta)
    .drop(columns = ["index"], errors = 'ignore')
    .sort_values(["Bank_ID", "Year"])
    .reset_index(drop = True)
    )


# Series de tiempo dividas por sujeto 

grafico_series_por_banco(
    Panel,
    variable="Profit",
    etiqueta="Beneficio",
    titulo="Evolución del beneficio por banco del 2015 al 2023"
)







