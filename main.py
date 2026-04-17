import os
import joblib
from src.data_extraction import load_data
from src.eda import run_eda
from src.data_preparation import prepare_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.validate import validate_model


def main():
    print("MLOps Pipeline — Vehicle Insurance (Ethiopian IC)")

    df = load_data(data_dir="data")
    run_eda(df, output_dir="reports")
    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = prepare_data(df)
    models = train_models(X_train, y_train)
    results_df, best_name = evaluate_models(models, X_test, y_test)
    best_model = models[best_name]
    validate_model(best_model, X_train, y_train, X_test)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    print(f"\n[Этап 7] Артефакты сохранены в models/")
    print(f"  Лучшая модель: {best_name}")
    print(f"  Для запуска API: python src/serve.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
