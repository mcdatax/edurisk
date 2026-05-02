# 🎓 EduRisk — Early Student Dropout Prediction

> **Predicción temprana de abandono universitario con Machine Learning**  
> Detecta estudiantes en riesgo de abandono el día de la matrícula — antes de que ocurra.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189A3A?style=flat)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![UCI Dataset](https://img.shields.io/badge/Dataset-UCI%20%23697-blue?style=flat)](https://archive.ics.uci.edu/dataset/697)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 🎯 El problema

Las universidades detectan el abandono cuando ya es tarde — cuando el estudiante lleva semestres con bajo rendimiento o simplemente deja de aparecer. **EduRisk predice el riesgo de abandono el día de la matrícula**, con solo los datos administrativos disponibles en ese momento, dando tiempo real para intervenir.

---

## 🚀 Demo

```bash
git clone https://github.com/mcdatax/edurisk.git
cd edurisk
pip install -r app_streamlit/requirements.txt
streamlit run app_streamlit/app.py
```

La app ofrece dos modos:
- **Predicción individual** — introduce los datos de un estudiante y obtén P(Dropout) al instante
- **Predicción masiva** — sube un CSV con N estudiantes y descarga los resultados con nivel de riesgo

---

## 🏗️ Arquitectura

```
CSV raw (datos de matrícula)
        ↓
src/data_processing.py      ← Drop Grupo B + Feature Engineering
        ↓
Pipeline sklearn             ← StandardScaler + OneHotEncoder
        ↓
RandomForestClassifier       ← Modelo final serializado
        ↓
P(Dropout) · P(Enrolled) · P(Graduate) + Risk Level
```

---

## 📊 Resultados del modelo

| Métrica | Valor |
|---|---|
| **Recall Dropout** | **0.6479** ← métrica principal |
| F1-weighted | 0.5973 |
| ROC-AUC macro | 0.7558 |
| Overfitting (diff) | 0.0683 — MODERADO |

> **¿Por qué Recall Dropout como métrica principal?**  
> Un Falso Negativo (predecir "no abandona" cuando sí abandona) tiene coste alto: el tutor no interviene y se pierde un estudiante. Un Falso Positivo (intervenir innecesariamente) tiene coste bajo. Por eso optimizamos Recall de la clase Dropout.

---

## 🤖 Modelos entrenados y comparados

| Modelo | Recall Dropout | F1-weighted | ROC-AUC | Overfitting |
|---|---|---|---|---|
| **RandomForest ✅** | **0.6479** | 0.5973 | 0.7558 | MODERADO |
| ExtraTrees | 0.6197 | 0.5886 | 0.7507 | MODERADO |
| XGBoost | 0.5986 | 0.6183 | 0.7689 | ALTO |
| LogisticRegression | 0.5915 | 0.6026 | 0.7606 | MODERADO |
| GradientBoosting | 0.5915 | 0.6012 | 0.7593 | ALTO |
| KMeans (no supervisado) | — | Silhouette=0.378 | k=3 | — |

---

## 🔬 Top features más importantes

| # | Feature | Importancia | Por qué importa |
|---|---|---|---|
| 1 | `Tuition fees up to date` | 0.156 | 86.6% de los que no están al día abandonan |
| 2 | `financial_risk` ⭐ | 0.106 | Feature engineered — combina deuda + pago |
| 3 | `Scholarship holder` | 0.077 | Solo el 12.2% de los becados abandona |
| 4 | `Age at enrollment` | 0.069 | Estudiantes mayores tienen más presión laboral |
| 5 | `age_group` ⭐ | 0.063 | Feature engineered — categorización no lineal |

⭐ = features creadas en el proceso de feature engineering

---

## 🗂️ Estructura del proyecto

```
edurisk/
├── app_streamlit/
│   ├── app.py                    ← Dashboard Streamlit (predicción individual + masiva)
│   └── requirements.txt
│
├── data/
│   ├── raw/                      ← Dataset original UCI #697
│   ├── processed/                ← Datos preprocesados (.parquet)
│   ├── train/                    ← Split de entrenamiento
│   └── test/                     ← Split de evaluación
│
├── models/
│   ├── final_model.pkl           ← Modelo final (RandomForest)
│   ├── final_model.pkl.sha256    ← Checksum de integridad
│   └── final_model_metadata.json ← Métricas y parámetros del modelo
│
├── notebooks/
│   ├── 01_Fuentes.ipynb          ← Adquisición y exploración inicial
│   ├── 02_LimpiezaEDA.ipynb      ← EDA, limpieza y feature engineering
│   └── 03_Entrenamiento_Evaluacion.ipynb ← Modelado, evaluación, SHAP, KMeans
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py        ← Clase DataProcessor
│   └── utils.py                  ← Constantes, helpers, validaciones
│
└── README.md
```

---

## ⚙️ Feature Engineering

| Feature nueva | Basada en | Justificación |
|---|---|---|
| `financial_risk` | Debtor + Tuition fees | Score 0-2 de riesgo financiero combinado |
| `parents_qualification_avg` | Mother + Father qualification | Nivel educativo familiar como concepto único |
| `age_group` | Age at enrollment | Captura distribución no lineal (skew=2.05) |
| `is_first_choice` | Application order | Proxy de motivación del estudiante |

---

## 🧱 Tech Stack

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| ML | scikit-learn · XGBoost |
| Preprocesamiento | ColumnTransformer · StandardScaler · OneHotEncoder |
| Búsqueda HP | GridSearchCV · RandomizedSearchCV |
| No supervisado | KMeans · PCA |
| Dashboard | Streamlit |
| Serialización | pickle |
| Gestión de proyecto | uv |

---

## 📁 Dataset

**UCI Machine Learning Repository #697**  
[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

- **Instancias:** 4.424 estudiantes
- **Features:** 36 (usamos 24 del día de matrícula)
- **Target:** Dropout · Enrolled · Graduate
- **Missing values:** 0
- **Origen:** Universidad portuguesa (Politécnico de Portalegre)

> **Decisión clave de diseño:** el dataset tiene 36 features divididas en dos grupos. Las 12 features de rendimiento académico (semestres 1 y 2) **no se usan** en el modelo porque solo están disponibles meses después de la matrícula. El modelo opera con las 24 features disponibles en el día 0.

---

## 🔄 Próximos pasos (v2.0)

- [ ] SHAP values para explicabilidad individual por estudiante
- [ ] Airflow DAG para ingesta automática de nuevas matrículas
- [ ] Migración de SQLite a PostgreSQL para producción
- [ ] Optimización bayesiana de hiperparámetros con Optuna
- [ ] API REST con FastAPI para integración con sistemas universitarios
- [ ] Contenedorización con Docker

---

## 👩‍💻 Autor

**Mane** · Data Scientist & Data Engineer  
📍 Madrid, España  
🔗 [GitHub](https://github.com/mcdatax) · [LinkedIn](https://linkedin.com/in/tu-perfil)

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE) para más detalles.

---

*Proyecto desarrollado durante el Bootcamp de Data Science en The Bridge (2025)*  
*Dataset: UCI ML Repository #697 — Realinho et al. (2022)*