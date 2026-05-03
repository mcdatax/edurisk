"""
evaluation.py
Evalúa el modelo final usando data/test/ y genera métricas.

Ejecución desde la raíz:
    python -m src.evaluation
"""
import warnings
warnings.filterwarnings('ignore')
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT      = Path(__file__).resolve().parent.parent
TEST_DIR  = ROOT / 'data' / 'test'
MODEL_PATH = ROOT / 'models' / 'final_model.pkl'


def main():
    print("EduRisk Evaluation Pipeline\n")

    # Cargar test set
    test_df = pd.read_parquet(TEST_DIR / 'test.parquet')
    X_test  = test_df.drop(columns=['Target'])
    y_test  = test_df['Target']
    print(f"Test set cargado: {X_test.shape}")

    # Cargar modelo final
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo cargado: {MODEL_PATH.name}\n")

    # Predicciones
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Métricas
    f1      = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba,
                             multi_class='ovr', average='macro')

    # Overfitting check
    train_df = pd.read_parquet(ROOT / 'data' / 'train' / 'train.parquet')
    X_train  = train_df.drop(columns=['Target'])
    y_train  = train_df['Target']
    f1_train = f1_score(y_train, model.predict(X_train), average='weighted')

    print("=== Métricas del modelo final (RandomForest) ===\n")
    print(f"F1-weighted Test:  {f1:.4f}")
    print(f"F1-weighted Train: {f1_train:.4f}")
    print(f"Overfitting diff:  {f1_train - f1:.4f}")
    print(f"ROC-AUC macro:     {roc_auc:.4f}")

    print("\n=== Classification Report ===\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Dropout', 'Enrolled', 'Graduate']
    ))

    print("=== Confusion Matrix ===\n")
    print(confusion_matrix(y_test, y_pred))

    print("\n Evaluation completado con éxito.")


if __name__ == '__main__':
    main()