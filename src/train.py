import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def train_models(X_train, y_train):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=8, learning_rate=0.1,
            random_seed=42, verbose=0
        ),
    }

    trained = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        trained[name] = model
        print(f"[Этап 4] {name} — обучена за {elapsed:.1f} с")

    return trained


if __name__ == "__main__":
    from data_extraction import load_data
    from data_preparation import prepare_data
    df = load_data()
    X_train, X_test, y_train, y_test, *_ = prepare_data(df)
    models = train_models(X_train, y_train)
