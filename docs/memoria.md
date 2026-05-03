# Memoria Técnica, EduRisk
## Early Student Dropout Prediction System

**Autor:** Manuel Correa  
**Bootcamp:** Data Science - The Bridge  
**Fecha:** Mayo 2025  
**Dataset:** UCI Machine Learning Repository #697  
**Repositorio:** https://github.com/mcdatax/edurisk  
**Demo:** https://eduriskml.streamlit.app  

---

## 1. Problema y objetivo

Las universidades detectan el abandono cuando ya es tarde.
**EduRisk** predice si un estudiante abandonará, seguirá cursando
o se graduará usando únicamente los datos disponibles el día
de la matrícula, sin esperar resultados académicos.

**Objetivo de negocio:** Detectar estudiantes en riesgo de
abandono antes de que ocurra para permitir intervención temprana.

**Métrica principal:** Recall de Dropout, minimiza los
estudiantes en riesgo que el sistema no detecta.

---

## 2. Dataset

- **Fuente:** UCI ML Repository #697
- **Instancias:** 4.424 estudiantes
- **Features:** 36 originales → 24 usadas (día de matrícula)
- **Target:** Dropout (32.1%) · Enrolled (17.9%) · Graduate (50%)
- **Missing values:** 0

**Decisión clave:** Las 12 features de rendimiento académico
(semestres 1 y 2) se descartan porque no están disponibles
en el momento de la matrícula. El modelo opera solo con datos
del día 0.

---

## 3. Preprocesamiento

### 3.1 Limpieza
- Sin missing values ni duplicados, dataset limpio
- Variables nominales codificadas como enteros por los autores

### 3.2 Feature Engineering

| Feature creada | Basada en | Justificación |
|---|---|---|
| `financial_risk` | Debtor + Tuition fees | Score 0-2 de riesgo financiero |
| `parents_qualification_avg` | Mother + Father qualification | Nivel educativo familiar |
| `age_group` | Age at enrollment | Captura distribución no lineal (skew=2.05) |
| `is_first_choice` | Application order | Proxy de motivación |

### 3.3 Preprocesamiento en Pipeline

| Grupo | Columnas | Transformación |
|---|---|---|
| Continuas (7) | Notas, edad, variables macro | StandardScaler |
| Nominales (9) | Course, Nationality, ocupaciones | OneHotEncoder(drop='first') |
| Binarias/ordinales (12) | Debtor, Gender, features engineered | Passthrough |

---

## 4. Modelado

### 4.1 Modelos entrenados

| Modelo | Búsqueda | F1-CV | F1 Test | Recall Dropout | Overfitting |
|---|---|---|---|---|---|
| **RandomForest** | GridSearch | 0.6184 | 0.5973 | **0.6479** | MODERADO |
| XGBoost | RandomizedSearch | 0.6359 | 0.6183 | 0.5986 | ALTO |
| LogisticRegression | GridSearch | 0.6216 | 0.6026 | 0.5915 | MODERADO |
| GradientBoosting | RandomizedSearch | 0.6434 | 0.6012 | 0.5915 | ALTO |
| ExtraTrees | GridSearch | 0.6160 | 0.5886 | 0.6197 | MODERADO |
| KMeans | No supervisado | Silhouette=0.378 | k=3 |, |, |

### 4.2 Modelo final: RandomForest

**Justificación:** Mayor Recall Dropout (0.6479) con overfitting
moderado y controlado (diff=0.068). XGBoost tiene mejor F1 y
ROC-AUC pero overfitting alto (diff=0.248).

### 4.3 Resultados del modelo final

**Matriz de confusión:**

| | Predijo Dropout | Predijo Enrolled | Predijo Graduate | Total real |
|---|---|---|---|---|
| **Dropout real** | **184** ✅ | 55 ❌ | 45 ❌ | 284 |
| **Enrolled real** | 41 ❌ | **62** ✅ | 56 ❌ | 159 |
| **Graduate real** | 82 ❌ | 85 ❌ | **275** ✅ | 442 |

**Métricas:**

| Métrica | Valor |
|---|---|
| F1-weighted | 0.5973 |
| **Recall Dropout** | **0.6479** ← métrica principal |
| ROC-AUC macro | 0.7558 |
| Overfitting diff | 0.0683 — MODERADO |

**Interpretación:** De 284 estudiantes que realmente abandonan,
el modelo detecta 184 correctamente (65%). Los 100 restantes
son falsos negativos, es decir, estudiantes en riesgo que no se detectan.

---

## 5. Modelo no supervisado - KMeans

Segmentación de estudiantes en 3 perfiles sin usar el target:

| Cluster | Estudiantes | % Dropout | Perfil |
|---|---|---|---|
| 0 | 2.770 | 20% | Bajo riesgo: Jóvenes, sin deudas y beca |
| 1 | 1.003 | 45% | Riesgo moderado: Estudiantes maduros (~33 años) |
| 2 | 651 | 64% | Alto riesgo: Grave riesgo financiero |

---

## 6. Interpretación - Feature Importance

| # | Feature | Importancia | Descripción |
|---|---|---|---|
| 1 | `Tuition fees up to date` | 0.156 | Si el estudiante tiene la matrícula al día en sus pagos |
| 2 | `financial_risk` | 0.106 | Score combinado de riesgo financiero (deuda + pago) |
| 3 | `Scholarship holder` | 0.077 | Si el estudiante tiene beca |
| 4 | `Age at enrollment` | 0.069 | Edad en el momento de la matrícula |
| 5 | `age_group` | 0.063 | Grupo de edad: joven, adulto joven, adulto, maduro |

---

## 7. Despliegue

- **App:** Streamlit con dos modos (individual + masivo CSV)
- **URL:** https://eduriskml.streamlit.app
- **Modelo:** RandomForest serializado con pickle
- **Preprocesamiento:** `src/data_processing.py`

---

## 8. Limitaciones y próximos pasos

- Dataset pequeño (4.424 registros), limita la generalización
- Overfitting **moderado** en RandomForest (diff=0.068)
- Clase Enrolled ambigua,  Recall=0.39
- **Próximos pasos:** SHAP values, Optuna, Docker, Airflow DAG,
  API REST con FastAPI