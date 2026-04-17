import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")


def evaluate_models(models: dict, X_test, y_test, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1, None))) * 100

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_%": mape,
            }
        )

        print(
            f"[Этап 5] {name}: "
            f"MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}, "
            f"R2={r2:.4f}, "
            f"MAPE={mape:.1f}%"
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["MAE", "RMSE", "R2"]

    for ax, metric in zip(axes, metrics):
        df_results.plot.bar(x="Model", y=metric, ax=ax, legend=False, color="steelblue")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close()

    best_name = df_results.loc[df_results["R2"].idxmax(), "Model"]
    print(f"\n[Этап 5] Лучшая модель по R2: {best_name}")

    return df_results, best_name