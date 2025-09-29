#!/usr/bin/env python3
"""
correlacion_github.py
Versión no interactiva del analizador para ejecutarse en GitHub Actions.
Uso ejemplo:
  python correlacion_github.py --input data/sample.csv --output output
"""

import argparse
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib
# backend sin display (importante en servidores/headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sys
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_palette("husl")

def interpretar_correlacion(corr, p_valor=None):
    if np.isnan(corr):
        return "N/A"
    if abs(corr) >= 0.7:
        fuerza = "MUY FUERTE"
    elif abs(corr) >= 0.5:
        fuerza = "FUERTE"
    elif abs(corr) >= 0.3:
        fuerza = "MODERADA"
    elif abs(corr) >= 0.1:
        fuerza = "DÉBIL"
    else:
        fuerza = "MUY DÉBIL o NULA"
    direccion = "POSITIVA" if corr > 0 else "NEGATIVA"
    interpretacion = f"Correlación {fuerza} {direccion}"
    if p_valor is not None and not np.isnan(p_valor):
        if p_valor < 0.05:
            interpretacion += " ✅ (Estadísticamente significativa)"
        else:
            interpretacion += " ❌ (No significativa estadísticamente)"
    return interpretacion

def mostrar_variables(variables, titulo="Variables"):
    print(f"\n📋 {titulo}:")
    for i, var in enumerate(variables, 1):
        print(f"   {i}. {var}")

def detectar_separador(ruta):
    with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
        primera_linea = f.readline()
    if ';' in primera_linea:
        return ';'
    if '\t' in primera_linea:
        return '\t'
    return ','

def main():
    parser = argparse.ArgumentParser(description="Analizador de correlaciones (no interactivo)")
    parser.add_argument('--input', '-i', required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument('--output', '-o', default='output', help="Carpeta donde guardar resultados")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ El archivo de entrada no existe: {input_path}")
        sys.exit(2)

    archivo = input_path.name
    print("🔧 Leyendo archivo:", input_path)

    # Intentar detectar separador y leer
    separador = detectar_separador(input_path)
    try:
        datos = pd.read_csv(input_path, sep=separador, encoding='utf-8')
        print("✅ Cargado con encoding utf-8 y separador", repr(separador))
    except Exception:
        datos = pd.read_csv(input_path, sep=separador, encoding='latin-1')
        print("✅ Cargado con encoding latin-1 y separador", repr(separador))

    print(f"📈 Filas: {len(datos)}  Columnas: {len(datos.columns)}")

    # Detectar variables numéricas
    variables_numericas = [c for c in datos.columns if pd.api.types.is_numeric_dtype(datos[c])]
    if len(variables_numericas) < 2:
        print("❌ Se necesitan al menos 2 variables numéricas.")
        sys.exit(3)

    datos_finales = datos[variables_numericas].dropna()
    print(f"✅ Variables numéricas ({len(variables_numericas)}): {variables_numericas}")
    print(f"📊 Observaciones tras dropna: {len(datos_finales)}")

    # Matriz de correlación
    matriz_correlacion = datos_finales.corr()
    # pares con p-valor
    correlaciones_todas = []
    for i in range(len(variables_numericas)):
        for j in range(i+1, len(variables_numericas)):
            v1 = variables_numericas[i]
            v2 = variables_numericas[j]
            corr = matriz_correlacion.loc[v1, v2]
            try:
                _, p_val = pearsonr(datos_finales[v1], datos_finales[v2])
            except Exception:
                p_val = float('nan')
            correlaciones_todas.append((v1, v2, float(corr), float(p_val) if not np.isnan(p_val) else np.nan))
    correlaciones_todas.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else 0, reverse=True)

    # --- Visualizaciones ---
    nombre_base = archivo.replace('.csv', '_analisis_correlacion')
    # Scatter grid
    num_correlaciones = len(correlaciones_todas)
    # elegir layout simple
    cols = 3
    filas = int(np.ceil(min(num_correlaciones, 12) / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(16, 4*filas))
    axes = np.array(axes).reshape(-1)
    mostrar = correlaciones_todas[:filas*cols]
    for idx, (v1, v2, corr, p_val) in enumerate(mostrar):
        ax = axes[idx]
        x = datos_finales[v1]
        y = datos_finales[v2]
        # color por fuerza (mantengo simplicidad)
        if abs(corr) >= 0.7:
            color = 'red'
        elif abs(corr) >= 0.5:
            color = 'orange'
        elif abs(corr) >= 0.3:
            color = 'blue'
        else:
            color = 'gray'
        ax.scatter(x, y, alpha=0.6, s=30)
        ax.set_xlabel(v1)
        ax.set_ylabel(v2)
        sig_text = "✅" if (not np.isnan(p_val) and p_val < 0.05) else "❌"
        ax.set_title(f"{v1} vs {v2}  r={corr:.3f} {sig_text}")
        if abs(corr) >= 0.1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=1.2)
            except Exception:
                pass
        ax.grid(True, alpha=0.3)
    # ocultar ejes sin uso
    for k in range(len(mostrar), len(axes)):
        axes[k].set_visible(False)
    plt.tight_layout()
    scatter_path = output_dir / f"{nombre_base}_scatters.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print("✅ Scatter guardado en:", scatter_path)

    # Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))
    sns.heatmap(matriz_correlacion, annot=True, mask=mask, cmap='RdBu_r', center=0, fmt='.3f',
                cbar_kws={'shrink': 0.8, 'label': 'Coef. correlación'}, linewidths=0.4, annot_kws={'size':9})
    plt.title(f'Mapa de Calor - {archivo}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heat_path = output_dir / f"{nombre_base}_heatmap.png"
    plt.savefig(heat_path, dpi=150)
    plt.close()
    print("✅ Heatmap guardado en:", heat_path)

    # Guardar Excel con resultados
    excel_path = output_dir / f"{nombre_base}.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            matriz_correlacion.to_excel(writer, sheet_name='Matriz_Correlacion')
            datos_finales.to_excel(writer, sheet_name='Datos_Procesados', index=False)
            df_correl = pd.DataFrame(correlaciones_todas, columns=['Variable_1','Variable_2','Correlacion','P_valor'])
            df_correl['Fuerza'] = df_correl['Correlacion'].apply(lambda x: interpretar_correlacion(x)[12:] if not np.isnan(x) else 'N/A')
            df_correl['Significativa'] = df_correl['P_valor'].apply(lambda x: 'Sí' if (not np.isnan(x) and x < 0.05) else 'No')
            df_correl.to_excel(writer, sheet_name='Todas_Correlaciones', index=False)
            df_resumen = pd.DataFrame({
                'Archivo Original': [archivo],
                'Fecha de Análisis': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Número de Variables': [len(variables_numericas)],
                'Número de Observaciones': [len(datos_finales)],
                'Variables Analizadas': [', '.join(variables_numericas)]
            })
            df_resumen.to_excel(writer, sheet_name='Info_Analisis', index=False)
        print("✅ Excel guardado en:", excel_path)
    except Exception as e:
        print("❌ Error guardando Excel:", e)

    print("\n🎉 Análisis completado. Archivos en carpeta:", output_dir)

if __name__ == "__main__":
    main()
