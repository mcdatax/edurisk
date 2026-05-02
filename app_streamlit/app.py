"""
app.py
Dashboard EduRisk — Predicción de abandono universitario.
Dos modos: predicción individual y predicción masiva por CSV.
"""

import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Rutas — DEBE ir antes de cualquier import de src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Imports de src — después del sys.path
from src.data_processing import DataProcessor
from src.utils import (decode_target, get_risk_level,
                       generate_random_student, COURSES,
                       detect_csv_type, REQUIRED_COLS,
                       ENGINEERED_COLS)

# resto del código...
# Configuración de la página
st.set_page_config(
    page_title="EduRisk – Predicción de Abandono",
    page_icon="🎓",
    layout="wide"
)

# Cargar modelo
@st.cache_resource
def load_model():
    model_path = ROOT / 'models' / 'final_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()
processor = DataProcessor()

# Título
st.title("🎓 EduRisk")
st.markdown("**Predicción temprana de abandono universitario**")
st.markdown("Modelo: RandomForest · Dataset: UCI #697 · Métrica principal: Recall Dropout")
st.divider()

# Selector de modo
modo = st.radio(
    "Selecciona el modo de predicción:",
    ["Predicción individual", "Predicción masiva (CSV)"],
    horizontal=True
)


# ─────────────────────────────────────────
# MODO 1 — Predicción individual
# ─────────────────────────────────────────
if modo == "Predicción individual":

    st.subheader("Datos del estudiante")

    # Botón aleatorio — antes de los inputs
    if st.button("🎲 Generar estudiante aleatorio"):
        st.session_state['random_student'] = generate_random_student()

    # Leer valores del session_state si existen
    s = st.session_state.get('random_student', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Edad en el momento de matrícula", 17, 70,
                              s.get('age', 20))
        gender = st.selectbox("Género", [0, 1],
                              index=s.get('gender', 0),
                              format_func=lambda x: "Femenino" if x==0 else "Masculino")
        marital = st.selectbox("Estado civil", [1,2,3,4,5,6],
                              index=[1,2,3,4,5,6].index(s.get('marital', 1)),
                              format_func=lambda x: {1:"Soltero",2:"Casado",3:"Viudo",4:"Divorciado",5:"Unión de hecho",6:"Separado"}[x])
        displaced = st.selectbox("Estudiante desplazado", [0,1],
                              index=s.get('displaced', 0),
                              format_func=lambda x: "No" if x==0 else "Sí")
        international = st.selectbox("Internacional", [0,1],
                              index=s.get('international', 0),
                              format_func=lambda x: "No" if x==0 else "Sí")
        special_needs = st.selectbox("Necesidades educativas especiales", [0,1],
                              index=s.get('special_needs', 0),
                              format_func=lambda x: "No" if x==0 else "Sí")

    with col2:
        scholarship = st.selectbox("Beca", [0,1],
                              index=s.get('scholarship', 0),
                              format_func=lambda x: "No" if x==0 else "Sí")
        debtor = st.selectbox("Deudor", [0,1],
                              index=s.get('debtor', 0),
                              format_func=lambda x: "No" if x==0 else "Sí")
        tuition = st.selectbox("Matrícula al día", [0,1],
                              index=s.get('tuition', 0),
                              format_func=lambda x: "No" if x==0 else "Sí")
        admission_grade = st.slider("Nota de admisión", 95.0, 190.0,
                              float(s.get('admission_grade', 127.0)))
        prev_grade = st.slider("Nota titulación previa", 95.0, 190.0,
                              float(s.get('prev_grade', 133.0)))
        prev_qual = st.selectbox("Tipo de titulación previa", list(range(1, 44)),
                              index=s.get('prev_qual', 1) - 1)

    with col3:
        course = st.selectbox("Curso", COURSES,
                              index=COURSES.index(s.get('course', COURSES[0])))
        app_mode = st.selectbox("Vía de acceso", list(range(1, 58)),
                              index=s.get('app_mode', 1) - 1)
        app_order = st.selectbox("Orden de preferencia", list(range(0, 10)),
                              index=s.get('app_order', 0))
        attendance = st.selectbox("Turno", [0,1],
                              index=s.get('attendance', 1),
                              format_func=lambda x: "Nocturno" if x==0 else "Diurno")
        nationality = st.selectbox("Nacionalidad (código)", list(range(1, 110)),
                              index=s.get('nationality', 1) - 1)
        mother_qual = st.selectbox("Nivel educativo madre", list(range(1, 44)),
                              index=s.get('mother_qual', 1) - 1)
        father_qual = st.selectbox("Nivel educativo padre", list(range(1, 44)),
                              index=s.get('father_qual', 1) - 1)
        mother_occ = st.selectbox("Ocupación madre", list(range(0, 195)),
                              index=s.get('mother_occ', 0))
        father_occ = st.selectbox("Ocupación padre", list(range(0, 195)),
                              index=s.get('father_occ', 0))

    st.subheader("Contexto macroeconómico")
    col4, col5, col6 = st.columns(3)
    with col4:
        unemployment = st.slider("Tasa de desempleo (%)", 7.0, 17.0,
                              float(s.get('unemployment', 11.0)))
    with col5:
        inflation = st.slider("Tasa de inflación (%)", -1.0, 4.0,
                              float(s.get('inflation', 1.0)))
    with col6:
        gdp = st.slider("PIB", -5.0, 4.0,
                              float(s.get('gdp', 0.0)))
        
    st.divider()

    if st.button("🎯 Predecir riesgo de abandono", type="primary", use_container_width=True):

        input_data = {
            'Marital Status': marital,
            'Application mode': app_mode,
            'Application order': app_order,
            'Course': course,
            'Daytime/evening attendance': attendance,
            'Previous qualification': prev_qual,
            'Previous qualification (grade)': prev_grade,
            'Nacionality': nationality,
            "Mother's qualification": mother_qual,
            "Father's qualification": father_qual,
            "Mother's occupation": mother_occ,
            "Father's occupation": father_occ,
            'Admission grade': admission_grade,
            'Displaced': displaced,
            'Educational special needs': special_needs,
            'Debtor': debtor,
            'Tuition fees up to date': tuition,
            'Gender': gender,
            'Scholarship holder': scholarship,
            'Age at enrollment': age,
            'International': international,
            'Unemployment rate': unemployment,
            'Inflation rate': inflation,
            'GDP': gdp,
        }

        df_input = pd.DataFrame([input_data])
        X = processor.process(df_input)
        probas = model.predict_proba(X)[0]
        pred   = model.predict(X)[0]

        proba_dropout  = probas[0]
        proba_enrolled = probas[1]
        proba_graduate = probas[2]
        risk           = get_risk_level(proba_dropout)
        label          = decode_target(pred)

        color = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}[risk]
        st.markdown(f"### {color} Predicción: **{label}** · Riesgo: **{risk}**")

        col7, col8, col9 = st.columns(3)
        col7.metric("P(Dropout)",  f"{proba_dropout:.1%}")
        col8.metric("P(Enrolled)", f"{proba_enrolled:.1%}")
        col9.metric("P(Graduate)", f"{proba_graduate:.1%}")

        st.bar_chart(
            pd.DataFrame({
                'Probabilidad': [proba_dropout, proba_enrolled, proba_graduate]
            }, index=['Dropout', 'Enrolled', 'Graduate'])
        )
# ─────────────────────────────────────────
# MODO 2 — Predicción masiva por CSV
# ─────────────────────────────────────────
else:
    st.subheader("Carga un CSV con datos de estudiantes")

    st.info("""
    **Formato esperado:** CSV con las mismas columnas del dataset raw UCI #697.
    El sistema detecta automáticamente si el CSV es crudo o ya preprocesado
    y aplica el tratamiento correspondiente.
    """)

    # Generador de CSV sintético
    st.subheader("¿No tienes un CSV? Genera datos sintéticos")

    col_gen1, col_gen2 = st.columns([1, 2])
    with col_gen1:
        n_students = st.number_input(
            "Número de estudiantes a generar",
            min_value=1, max_value=10000, value=100, step=10
        )
    with col_gen2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Generar CSV sintético", use_container_width=True):
            students = [generate_random_student() for _ in range(n_students)]
            df_synthetic = pd.DataFrame([{
                'Marital Status':                   s['marital'],
                'Application mode':                 s['app_mode'],
                'Application order':                s['app_order'],
                'Course':                           s['course'],
                'Daytime/evening attendance':       s['attendance'],
                'Previous qualification':           s['prev_qual'],
                'Previous qualification (grade)':   s['prev_grade'],
                'Nacionality':                      s['nationality'],
                "Mother's qualification":           s['mother_qual'],
                "Father's qualification":           s['father_qual'],
                "Mother's occupation":              s['mother_occ'],
                "Father's occupation":              s['father_occ'],
                'Admission grade':                  s['admission_grade'],
                'Displaced':                        s['displaced'],
                'Educational special needs':        s['special_needs'],
                'Debtor':                           s['debtor'],
                'Tuition fees up to date':          s['tuition'],
                'Gender':                           s['gender'],
                'Scholarship holder':               s['scholarship'],
                'Age at enrollment':                s['age'],
                'International':                    s['international'],
                'Unemployment rate':                s['unemployment'],
                'Inflation rate':                   s['inflation'],
                'GDP':                              s['gdp'],
            } for s in students])

            csv_synthetic = df_synthetic.to_csv(index=False).encode('utf-8')
            st.success(f"{n_students} estudiantes generados correctamente.")
            st.download_button(
                label="⬇️ Descargar CSV sintético",
                data=csv_synthetic,
                file_name=f"edurisk_sintetico_{n_students}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.divider()

    uploaded_file = st.file_uploader(
        "Selecciona el archivo CSV", type=['csv']
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(
                f"Archivo cargado: {df_raw.shape[0]} "
                f"estudiantes · {df_raw.shape[1]} columnas"
            )
            st.dataframe(df_raw.head(5), use_container_width=True)

            # Validación y detección de tipo
            csv_type, missing_cols = detect_csv_type(df_raw)

            if csv_type == 'invalid':
                st.error(
                    "CSV inválido. Faltan las siguientes "
                    "columnas obligatorias:"
                )
                st.code('\n'.join(missing_cols))
                st.stop()

            elif csv_type == 'processed':
                st.info(
                    "CSV ya preprocesado detectado. "
                    "Se omite el preprocesamiento."
                )
                X = df_raw[REQUIRED_COLS + ENGINEERED_COLS]

            else:
                st.info(
                    "CSV raw detectado. "
                    "Aplicando preprocesamiento automático."
                )
                X = processor.process(df_raw)

            if st.button(
                "🚀 Predecir para todos",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Procesando y prediciendo..."):
                    probas = model.predict_proba(X)
                    preds  = model.predict(X)

                    df_result = df_raw.copy()
                    df_result['predicted_class'] = [
                        decode_target(p) for p in preds
                    ]
                    df_result['proba_dropout']  = probas[:, 0].round(3)
                    df_result['proba_enrolled'] = probas[:, 1].round(3)
                    df_result['proba_graduate'] = probas[:, 2].round(3)
                    df_result['risk_level']     = [
                        get_risk_level(p) for p in probas[:, 0]
                    ]

                st.success(
                    f"Predicciones completadas "
                    f"para {len(df_result)} estudiantes"
                )

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Alto riesgo",
                    f"{(df_result['risk_level']=='Alto').sum()}",
                    f"{(df_result['risk_level']=='Alto').mean():.1%}"
                )
                col2.metric(
                    "Riesgo medio",
                    f"{(df_result['risk_level']=='Medio').sum()}",
                    f"{(df_result['risk_level']=='Medio').mean():.1%}"
                )
                col3.metric(
                    "Bajo riesgo",
                    f"{(df_result['risk_level']=='Bajo').sum()}",
                    f"{(df_result['risk_level']=='Bajo').mean():.1%}"
                )

                st.subheader("Resultados")
                st.dataframe(
                    df_result[[
                        'predicted_class', 'proba_dropout',
                        'proba_enrolled', 'proba_graduate',
                        'risk_level'
                    ]],
                    use_container_width=True
                )

                csv_out = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar resultados como CSV",
                    data=csv_out,
                    file_name="edurisk_predicciones.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")