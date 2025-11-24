import streamlit as st
import pandas as pd
import numpy as np
import math
import time

st.title("Aplicación de Estadística – Análisis de Velocidad de Internet")

st.write("Sube el archivo CSV del dataset del MinTIC:")
archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])

# ============================================================
#           1. CARGA Y LIMPIEZA DEL ARCHIVO
# ============================================================
if archivo:

    columnas_necesarias = [
        "velocidad_bajada",
        "tecnologia",
        "departamento",
        "municipio",
        "proveedor"
    ]

    df = pd.read_csv(archivo, low_memory=False)
    columnas_encontradas = [c for c in columnas_necesarias if c in df.columns]

    if "velocidad_bajada" not in columnas_encontradas:
        st.error("El archivo no contiene 'velocidad_bajada'.")
        st.stop()

    df = df[columnas_encontradas]

    # Limpieza
    df["velocidad_bajada"] = (
        df["velocidad_bajada"]
        .astype(str)
        .str.replace(",", ".", 1)
        .str.replace(",", "")
    )

    df["velocidad_bajada"] = pd.to_numeric(df["velocidad_bajada"], errors="coerce")
    df = df.dropna(subset=["velocidad_bajada"])

    st.write(f"Total de registros válidos: {len(df)}")
    st.dataframe(df.head(200))

    # ============================================================
    #           2. GENERAR MUESTRA
    # ============================================================
    if st.button("Generar muestra de 50 datos"):

        if len(df) < 50:
            st.error("No hay suficientes datos válidos para tomar 50 datos.")
            st.stop()

        muestra = df.sample(50, random_state=int(time.time()))
        st.session_state["muestra"] = muestra

        st.success("Muestra generada correctamente.")
        st.dataframe(muestra)

# ============================================================
#           3. CÁLCULOS ESTADÍSTICOS
# ============================================================
if "muestra" in st.session_state:

    st.subheader("Cálculos Estadísticos")

    datos = st.session_state["muestra"]["velocidad_bajada"]

    # ----------------- RANGO -----------------
    rango = datos.max() - datos.min()
    st.write(f"Rango: {rango:.2f} Mbps")

    # ----------------- CLASES -----------------
    k = round(math.sqrt(len(datos)))  # regla √n
    st.write(f"Número de clases (k): {k}")

    # ----------------- AMPLITUD -----------------
    amplitud_real = rango / k
    amplitud = math.ceil(amplitud_real)
    st.write(f"Amplitud real: {amplitud_real:.2f}")
    st.write(f"Amplitud redondeada: {amplitud}")

    # ----------------- INTERVALOS DINÁMICOS -----------------
    minimo = datos.min()

    intervalos = []
    li = minimo

    for i in range(k):
        ls = li + amplitud - 1    # intervalo cerrado
        intervalos.append((round(li, 2), round(ls, 2)))
        li = ls + 1               # siguiente inicia en LS + 1

    st.write("Intervalos construidos:")
    for i, (a, b) in enumerate(intervalos, start=1):
        st.write(f"Clase {i}: {a} – {b}")

    # ----------------- FRECUENCIAS -----------------
    frecuencias = []
    puntos_medios = []

    for li, ls in intervalos:
        f = sum((datos >= li) & (datos <= ls))
        frecuencias.append(f)
        puntos_medios.append((li + ls) / 2)

    fr = [f / len(datos) for f in frecuencias]
    fa = np.cumsum(frecuencias)

    # ----------------- TABLA FINAL -----------------
    tabla = pd.DataFrame({
        "Límite inferior": [li for li, _ in intervalos],
        "Límite superior": [ls for _, ls in intervalos],
        "Punto medio": puntos_medios,
        "Frecuencia": frecuencias,
        "Frecuencia relativa": fr,
        "Frecuencia acumulada": fa
    })

    st.subheader("Tabla de Frecuencias (intervalos correctos)")
    st.dataframe(tabla)

    # ============================================================
    #           4. MEDIDAS DE TENDENCIA CENTRAL
    # ============================================================
    st.subheader("Medidas de Tendencia Central")

    media = datos.mean()
    mediana = datos.median()
    moda = datos.mode().iloc[0] if not datos.mode().empty else "Sin moda"

    st.write(f"Media: {media:.2f}")
    st.write(f"Mediana: {mediana:.2f}")
    st.write(f"Moda: {moda}")

    # ============================================================
    #           5. HISTOGRAMA
    # ============================================================
    st.subheader("Histograma de Frecuencias")

    st.bar_chart(tabla.set_index("Punto medio")["Frecuencia"])

    # ============================================================
#           6. PROBABILIDAD Y SUCESOS (A y B)
# ============================================================

    st.subheader("Probabilidad, sucesos y reglas (A y B)")

    # A: velocidad > 100 Mbps
    A = datos > 100

    # B: tecnología contiene "fibra" (FTTH/FTTB)
    if "tecnologia" in st.session_state["muestra"].columns:
        B = st.session_state["muestra"]["tecnologia"].str.contains("fibra", case=False, na=False)
    else:
        B = pd.Series([False] * len(datos))

    # Probabilidades básicas
    P_A = A.mean()
    P_B = B.mean()
    P_AyB = (A & B).mean()

    st.write("### Probabilidades básicas")
    st.write(f"P(A)  = {P_A:.4f}  →  Velocidad > 100 Mbps")
    st.write(f"P(B)  = {P_B:.4f}  →  Tecnología tipo fibra óptica")
    st.write(f"P(A ∩ B) = {P_AyB:.4f}")

    # Unión de sucesos
    P_union = P_A + P_B - P_AyB
    st.write("### Unión de sucesos")
    st.write(f"P(A ∪ B) = {P_union:.4f}")

    # Condicionales
    P_A_given_B = P_AyB / P_B if P_B > 0 else 0
    P_B_given_A = P_AyB / P_A if P_A > 0 else 0

    st.write("### Probabilidad condicional")
    st.write(f"P(A | B) = {P_A_given_B:.4f}")
    st.write(f"P(B | A) = {P_B_given_A:.4f}")

    # Bayes
    P_Bayes = (P_A_given_B * P_B) / P_A if P_A > 0 else 0

    st.write("### Teorema de Bayes")
    st.write(f"P(B | A) (vía Bayes) = {P_Bayes:.4f}")

    # Independencia
    st.write("### Independencia de eventos")
    if abs(P_AyB - (P_A * P_B)) < 0.02:
        st.success("A y B son aproximadamente independientes ( |P(A∩B) − P(A)P(B)| < 0.02 )")
    else:
        st.error("A y B NO son independientes.")

