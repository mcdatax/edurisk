"""
training.py
Entrena los modelos a partir de data/processed/, guarda los splits
en data/train/ y data/test/ y serializa los modelos en models/.

Ejecución desde la raíz del proyecto:
    python -m src.training
"""
# Utilidades estándar
import warnings
warnings.filterwarnings('ignore')

import pickle
import time
from pathlib import Path

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / 'data' / 'processed' / 'data_processed.parquet'
TRAIN_DIR = ROOT / 'data' / 'train'
TEST_DIR  = ROOT / 'data' / 'test'
MODELS_DIR = ROOT / 'models'

COLS_SCALE = [
    'Previous qualification (grade)', 'Admission grade',
    'Age at enrollment', 'Unemployment rate',
    'Inflation rate', 'GDP', 'parents_qualification_avg'
]
COLS_OHE = [
    'Marital Status', 'Application mode', 'Course',
    'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
COLS_PASSTHROUGH = [
    'Application order', 'Daytime/evening attendance',
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'International', 'financial_risk', 'age_group', 'is_first_choice'
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('scaler',      StandardScaler(),                          COLS_SCALE),
        ('ohe',         OneHotEncoder(drop='first',
                                      sparse_output=False,
                                      handle_unknown='ignore'),   COLS_OHE),
        ('passthrough', 'passthrough',                             COLS_PASSTHROUGH),
    ])


def main():
    print("EduRisk Training Pipeline=\n")

    # Cargar datos procesados
    df = pd.read_parquet(PROCESSED)
    X  = df.drop(columns=['Target'])
    y  = df['Target']
    print(f"Dataset cargado: {df.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Guardar splits
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy(); train_df['Target'] = y_train
    test_df  = X_test.copy();  test_df['Target']  = y_test
    train_df.to_parquet(TRAIN_DIR / 'train.parquet', index=False)
    test_df.to_parquet(TEST_DIR   / 'test.parquet',  index=False)
    print("Splits guardados.\n")

    preprocessor = build_preprocessor()

    # Modelos y grids
    configs = {
        'LogisticRegression': (
            Pipeline([('preprocessor', preprocessor),
                      ('model', LogisticRegression(max_iter=1000,
                                                    class_weight='balanced',
                                                    random_state=42))]),
            'grid',
            {'model__C': [0.01, 0.1, 1, 10],
             'model__solver': ['lbfgs', 'saga']}
        ),
        'RandomForest': (
            Pipeline([('preprocessor', preprocessor),
                      ('model', RandomForestClassifier(class_weight='balanced',
                                                        random_state=42,
                                                        n_jobs=-1))]),
            'grid',
            {'model__n_estimators': [100, 200, 300],
             'model__max_depth': [3, 5, 7],
             'model__min_samples_split': [5, 10],
             'model__min_samples_leaf': [2, 4]}
        ),
        'XGBoost': (
            Pipeline([('preprocessor', preprocessor),
                      ('model', XGBClassifier(objective='multi:softprob',
                                               num_class=3,
                                               eval_metric='mlogloss',
                                               random_state=42,
                                               n_jobs=-1))]),
            'random',
            {'model__n_estimators': randint(100, 300),
             'model__max_depth': randint(2, 5),
             'model__learning_rate': uniform(0.01, 0.1),
             'model__subsample': uniform(0.5, 0.4),
             'model__colsample_bytree': uniform(0.5, 0.4),
             'model__min_child_weight': randint(5, 15)}
        ),
        'ExtraTrees': (
            Pipeline([('preprocessor', preprocessor),
                      ('model', ExtraTreesClassifier(class_weight='balanced',
                                                      random_state=42,
                                                      n_jobs=-1))]),
            'grid',
            {'model__n_estimators': [100, 200, 300],
             'model__max_depth': [3, 5, 7],
             'model__min_samples_split': [5, 10],
             'model__min_samples_leaf': [2, 4]}
        ),
        'GradientBoosting': (
            Pipeline([('preprocessor', preprocessor),
                      ('model', GradientBoostingClassifier(random_state=42))]),
            'random',
            {'model__n_estimators': randint(150, 350),
             'model__max_depth': randint(3, 5),
             'model__learning_rate': uniform(0.05, 0.15),
             'model__subsample': uniform(0.7, 0.25),
             'model__min_samples_leaf': randint(2, 5)}
        ),
    }

    best_models = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, (pipeline, search_type, params) in configs.items():
        print(f"Entrenando {name}...")
        start = time.time()

        if search_type == 'grid':
            search = GridSearchCV(pipeline, params, cv=5,
                                  scoring='f1_weighted', n_jobs=-1)
        else:
            search = RandomizedSearchCV(pipeline, params, n_iter=30,
                                        cv=5, scoring='f1_weighted',
                                        n_jobs=-1, random_state=42)

        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        model_path = MODELS_DIR / f'trained_model_{name.lower()}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(search.best_estimator_, f)

        elapsed = time.time() - start
        print(f"  F1-CV: {search.best_score_:.4f} | "
              f"Params: {search.best_params_} | "
              f"Tiempo: {elapsed:.1f}s")

    # Modelo final — RandomForest
    final_path = MODELS_DIR / 'final_model.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(best_models['RandomForest'], f)
    print(f"\nModelo final guardado: {final_path.name}")

    # KMeans
    scaler_km = StandardScaler()
    X_scaled  = scaler_km.fit_transform(X)
    pca_km    = PCA(n_components=3, random_state=42)
    X_pca     = pca_km.fit_transform(X_scaled)
    kmeans    = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    km_path = MODELS_DIR / 'trained_model_kmeans.pkl'
    with open(km_path, 'wb') as f:
        pickle.dump({'scaler': scaler_km, 'pca': pca_km, 'kmeans': kmeans}, f)
    print(f"KMeans guardado: {km_path.name}")

    print("\n=== Training completado ===")


if __name__ == '__main__':
    main()