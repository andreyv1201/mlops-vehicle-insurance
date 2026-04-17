import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df = df.copy()
    df = df.dropna(subset=["PREMIUM"])
    df = df[df["PREMIUM"] > 0]
    df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
    df["INSR_END"] = pd.to_datetime(df["INSR_END"], format="%d-%b-%y", errors="coerce")
    df["DURATION_DAYS"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
    df["DURATION_DAYS"] = df["DURATION_DAYS"].fillna(365).clip(lower=0, upper=730)
    df["PROD_YEAR"] = df["PROD_YEAR"].fillna(df["PROD_YEAR"].median())
    df["SEATS_NUM"] = df["SEATS_NUM"].fillna(df["SEATS_NUM"].median())
    df["CARRYING_CAPACITY"] = df["CARRYING_CAPACITY"].fillna(0)
    df["CCM_TON"] = df["CCM_TON"].fillna(df["CCM_TON"].median())
    df["MAKE"] = df["MAKE"].fillna("UNKNOWN")
    df["VEHICLE_AGE"] = 2018 - df["PROD_YEAR"]
    df["VEHICLE_AGE"] = df["VEHICLE_AGE"].clip(lower=0, upper=60)

    cat_cols = ["TYPE_VEHICLE", "MAKE", "USAGE"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "SEX", "INSR_TYPE", "INSURED_VALUE", "PROD_YEAR", "SEATS_NUM",
        "CARRYING_CAPACITY", "CCM_TON", "TYPE_VEHICLE", "MAKE", "USAGE",
        "DURATION_DAYS", "VEHICLE_AGE"
    ]

    X = df[feature_cols].values
    y = df["PREMIUM"].values
    p995 = np.percentile(y, 99.5)
    mask = y <= p995
    X, y = X[mask], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[Этап 3] Подготовка данных завершена")
    print(f"  Train: {X_train.shape[0]} объектов")
    print(f"  Test: {X_test.shape[0]} объектов")
    print(f"  Количество признаков: {X_train.shape[1]}")
    print(
        f"  PREMIUM (train): min={y_train.min():.2f}, "
        f"max={y_train.max():.2f}, mean={y_train.mean():.2f}"
    )

    return X_train, X_test, y_train, y_test, scaler, encoders, feature_cols


if __name__ == "__main__":
    from data_extraction import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, enc, cols = prepare_data(df)
