import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="estadApp",
    page_icon="📊",
    layout="wide"
)
st.title("Aplicación de Estadística – Análisis de Velocidad de Internet")

st.write("Sube el archivo CSV del dataset del MinTIC (n48w-gutb):")

archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])

# ---------------------------
# UTIL: función para limpiar velocidad_bajada
# ---------------------------
def limpiar_velocidad(series):
    s = series.astype(str)
    # reemplazo: primera coma -> punto, luego quitar comas restantes
    s = s.str.replace(",", ".", 1).str.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

# ---------------------------
# CARGA Y PREVIEWS
# ---------------------------
if archivo:
    df = pd.read_csv(archivo, low_memory=False)

    # columnas que esperamos
    columnas_necesarias = ["velocidad_bajada", "tecnologia", "departamento", "municipio", "proveedor"]
    columnas_encontradas = [c for c in columnas_necesarias if c in df.columns]

    if "velocidad_bajada" not in df.columns:
        st.error("El archivo no contiene la columna 'velocidad_bajada'. Verifica el CSV.")
        st.stop()

    # limpiar y convertir velocidad
    df["velocidad_bajada"] = limpiar_velocidad(df["velocidad_bajada"])
    df = df.dropna(subset=["velocidad_bajada"]).reset_index(drop=True)

    st.info(f"Registros válidos en velocidad_bajada: {len(df)}")

    # opción para mostrar todo o solo primeras filas
    mostrar_todo = st.checkbox("Mostrar todo el dataset cargado (puede ser grande)", value=False)
    if mostrar_todo:
        st.dataframe(df)
    else:
        st.dataframe(df.head(200))

    st.markdown("---")

    # ---------------------------
    # BOTÓN: generar muestra
    # ---------------------------
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Generar muestra aleatoria de 50 datos"):
            if len(df) < 50:
                st.error("No hay suficientes datos válidos para tomar 50 datos.")
            else:
                # generar muestra distinta cada vez
                seed = int(time.time() * 1000) % (2**32 - 1)
                muestra = df.sample(50, random_state=seed)
                st.session_state["muestra"] = muestra
                st.success("Muestra generada correctamente.")
    with col2:
        st.write("Para regenerar la muestra pulsar el botón de la izquierda (cada vez será distinta).")

# ---------------------------
# SI HAY MUESTRA: cálculos
# ---------------------------
if "muestra" in st.session_state:

    st.header("Resultados basados en la muestra (50 observaciones)")

    muestra = st.session_state["muestra"]
    st.subheader("Muestra (50 registros)")
    st.dataframe(muestra.reset_index(drop=True))

    # datos como serie
    datos = muestra["velocidad_bajada"].astype(float)

    # ----------------- RANGO -----------------
    minimo = float(datos.min())
    maximo = float(datos.max())
    rango = maximo - minimo

    st.write(f"**Rango (max - min):** {rango:.4f} Mbps")
    st.write(f"**Valor mínimo:** {minimo:.4f} | **Valor máximo:** {maximo:.4f}")

    # ----------------- K (√n) -----------------
    n = len(datos)
    k = round(math.sqrt(n))
    st.write(f"**Número de clases (k = √n):** {k} (n={n})")

    # ----------------- AMPLITUD -----------------
    amplitud_real = rango / k
    st.write(f"**Amplitud real (rango/k):** {amplitud_real:.4f} Mbps")

    metodo = st.selectbox("Método de redondeo para la amplitud (se usará para construir intervalos):",
                          ("ceil (arriba)", "round (al entero más cercano)", "floor (abajo)"))
    if metodo == "ceil (arriba)":
        amplitud_usada = math.ceil(amplitud_real)
    elif metodo == "round (al entero más cercano)":
        amplitud_usada = int(round(amplitud_real))
    else:
        amplitud_usada = math.floor(amplitud_real)

    if amplitud_usada <= 0:
        st.error("La amplitud redondeada no puede ser 0 o negativa. Cambia el método de redondeo o revisa la muestra.")
    else:
        st.write(f"**Amplitud usada (después de redondeo):** {amplitud_usada} Mbps")

        # ----------------- CREAR INTERVALOS según TU REGLA -----------------
        # Regla: LI1 = mínimo, LS1 = LI1 + amplitud - 1; luego LI_next = LS_prev + 1, etc.
        intervalos = []
        li = round(minimo, 2)

        for i in range(k):
            ls = round(li + amplitud_usada - 1, 2)
            intervalos.append((li, ls))
            li = round(ls + 1, 2)

        # Asegurar que el último cubra el máximo
        if intervalos[-1][1] < maximo:
            last_li, last_ls = intervalos[-1]
            intervalos[-1] = (last_li, round(maximo, 2))

        st.write("**Intervalos construidos (según la muestra y la regla seleccionada):**")
        for idx, (a, b) in enumerate(intervalos, start=1):
            st.write(f"Clase {idx}: {a}  –  {b}")

        # ----------------- TABLA DE FRECUENCIAS -----------------
        frecuencias = []
        puntos_medios = []
        for li, ls in intervalos:
            f = int(((datos >= li) & (datos <= ls)).sum())
            frecuencias.append(f)
            puntos_medios.append(round((li + ls) / 2, 2))

        suma_f = sum(frecuencias)
        st.write(f"**Suma de frecuencias (debe ser 50):** {suma_f}")

        fr = [round(f / n, 4) for f in frecuencias]
        fa = np.cumsum(frecuencias)

        tabla = pd.DataFrame({
            "Límite inferior": [li for li, _ in intervalos],
            "Límite superior": [ls for _, ls in intervalos],
            "Punto medio": puntos_medios,
            "Frecuencia absoluta": frecuencias,
            "Frecuencia relativa": fr,
            "Frecuencia acumulada": fa
        })

        st.subheader("Tabla de Frecuencias")
        st.dataframe(tabla)

        # ----------------- MEDIDAS DE TENDENCIA CENTRAL -----------------
        st.subheader("Medidas de tendencia central (muestra)")
        media = datos.mean()
        mediana = datos.median()
        moda = datos.mode().iloc[0] if not datos.mode().empty else np.nan
        st.write(f"Media: {media:.2f} Mbps")
        st.write(f"Mediana: {mediana:.2f} Mbps")
        st.write(f"Moda: {moda} Mbps")

        # ----------------- HISTOGRAMA (matplotlib para controlar bins) -----------------
        st.subheader("Histograma (respeta los intervalos construidos)")
        fig1, ax1 = plt.subplots()
        # construir bins list (usar límites contiguos)
        bins = [intervalos[0][0]] + [b for (_, b) in intervalos]
        ax1.hist(datos, bins=bins)
        ax1.set_xlabel("Velocidad (Mbps)")
        ax1.set_ylabel("Frecuencia")
        ax1.set_title("Histograma")
        st.pyplot(fig1, use_container_width=True)

        # ----------------- POLÍGONO DE FRECUENCIAS -----------------
        st.subheader("Polígono de frecuencias")
        fig2, ax2 = plt.subplots()
        ax2.plot(puntos_medios, frecuencias, marker="o")
        ax2.set_xlabel("Punto medio")
        ax2.set_ylabel("Frecuencia")
        ax2.set_title("Polígono de frecuencias")
        st.pyplot(fig2, use_container_width=True)

        # ----------------- OJIVA (frecuencia acumulada) -----------------
        st.subheader("Ojiva (frecuencia acumulada)")
        fig3, ax3 = plt.subplots()
        # eje X: límites superiores, eje Y: frecuencia acumulada
        limites_sup = [b for (_, b) in intervalos]
        ax3.plot(limites_sup, fa, marker="o")
        ax3.set_xlabel("Límite superior")
        ax3.set_ylabel("Frecuencia acumulada")
        ax3.set_title("Ojiva")
        st.pyplot(fig3, use_container_width=True)

        # ----------------- CONCLUSIONES AUTOMÁTICAS (básicas) -----------------
        st.subheader("Conclusiones automáticas (interpretación básica)")

        conclusions = []
        # sesgo
        if media > mediana:
            conclusions.append("La media es mayor que la mediana → posible sesgo a la derecha (colas a la derecha).")
        elif media < mediana:
            conclusions.append("La media es menor que la mediana → posible sesgo a la izquierda.")
        else:
            conclusions.append("Media y mediana son iguales → distribución aproximadamente simétrica.")

        # concentración
        max_fr_index = np.argmax(frecuencias)
        if max_fr_index == 0:
            conclusions.append("La mayor concentración de la muestra está en el primer intervalo → velocidades relativamente bajas en la muestra.")
        else:
            conclusions.append(f"La mayor concentración de la muestra está en la clase {max_fr_index+1} (punto medio {puntos_medios[max_fr_index]} Mbps).")

        # dispersión
        conclusions.append(f"El rango de la muestra es de {rango:.2f} Mbps, indicador de la dispersión entre valores extremos.")

        # añadir interpretación sobre moda
        if not np.isnan(moda):
            conclusions.append(f"La moda es {moda} Mbps, indica el valor más frecuente en la muestra.")

        # mostrar
        for c in conclusions:
            st.write("- " + c)

        # ----------------- PROBABILIDAD Y SUCESOS (A y B) -----------------
        st.subheader("Probabilidad, eventos y reglas (A y B)")

        # A: velocidad > 100 Mbps
        A = datos > 100

        # B: tecnologia contiene "fibra"
        if "tecnologia" in muestra.columns:
            B = muestra["tecnologia"].str.contains("fibra", case=False, na=False)
        else:
            B = pd.Series([False] * len(muestra), index=muestra.index)

        P_A = A.mean()
        P_B = B.mean()
        P_AyB = (A & B).mean()

        st.write(f"P(A) = P(Velocidad > 100) = {P_A:.4f}")
        st.write(f"P(B) = P(Tecnología = fibra) = {P_B:.4f}")
        st.write(f"P(A ∩ B) = {P_AyB:.4f}")

        # Unión
        P_union = P_A + P_B - P_AyB
        st.write(f"P(A ∪ B) = {P_union:.4f} (P(A)+P(B)-P(A∩B))")

        # Condicionales
        P_A_given_B = P_AyB / P_B if P_B > 0 else np.nan
        P_B_given_A = P_AyB / P_A if P_A > 0 else np.nan

        st.write(f"P(A | B) = {P_A_given_B:.4f}" if not np.isnan(P_A_given_B) else "P(A | B) indefinido (P(B)=0)")
        st.write(f"P(B | A) = {P_B_given_A:.4f}" if not np.isnan(P_B_given_A) else "P(B | A) indefinido (P(A)=0)")

        # Bayes (comprobación)
        if P_A > 0:
            bayes_check = (P_A_given_B * P_B) / P_A if (not np.isnan(P_A_given_B)) else np.nan
            st.write(f"Teorema de Bayes (comprobación): P(B|A) = {bayes_check:.4f}" if not np.isnan(bayes_check) else "Bayes indefinido")
        else:
            st.write("Bayes indefinido (P(A)=0)")

        # Independencia (tolerancia)
        tol = 0.02
        st.write("Independencia (tolerancia): |P(A∩B) − P(A)P(B)| < 0.02")
        if abs(P_AyB - (P_A * P_B)) < tol:
            st.success("A y B son aproximadamente independientes.")
        else:
            st.error("A y B NO son independientes.")

        st.markdown("---")
        st.info("La app realizó todos los cálculos sobre la muestra de 50 datos. Si quieres otra muestra pulsa 'Generar muestra de 50 datos' para recalcular.")
